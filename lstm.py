"""Bidirectional LSTM over whole breaths (80 steps) for ventilator pressure.

Inputs per step: u_in, u_out, R, C, time_step, plus u_in_cumsum with --u-in-cumsum
(standardized with train-split stats).
Output: pressure at every step.
Loss: L1, computed only on inspiratory steps (u_out == 0).
Holdout: the same GroupShuffleSplit(test_size=0.2, random_state=42) on breath_id as train.py.
Reported metric: MAE on held-out rows with u_out == 0, at the final epoch
(fixed schedule, no early stopping on the holdout). The best epoch is logged too,
but picking it uses the holdout, so it is optimistic.

Each run appends one JSON line to lstm_trials.jsonl.

Usage:
    python lstm.py --epochs 60 --trial-name bilstm_h256_l3
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

SEED = 42
SEQ_LEN = 80
INPUTS = ["u_in", "u_out", "R", "C", "time_step"]


def load_breaths(path):
    df = pd.read_csv(path).sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    sizes = df.groupby("breath_id").size()
    assert (sizes == SEQ_LEN).all(), "every breath must have 80 steps"
    return df


def to_arrays(df, inputs):
    n = len(df) // SEQ_LEN
    X = df[inputs].to_numpy(np.float32).reshape(n, SEQ_LEN, len(inputs))
    y = df["pressure"].to_numpy(np.float32).reshape(n, SEQ_LEN) if "pressure" in df else None
    mask = (df["u_out"].to_numpy() == 0).reshape(n, SEQ_LEN)
    return X, y, mask


class BiLSTM(nn.Module):
    def __init__(self, n_in, hidden, layers, dropout):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU())
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.lstm(self.inp(x))
        return self.head(h).squeeze(-1)


def mem_str(device):
    if device.type != "mps":
        return ""
    return f"  mps mem {torch.mps.driver_allocated_memory() / 2**30:.1f} GiB"


def masked_l1(pred, target, mask):
    return (pred - target).abs()[mask].mean()


@torch.no_grad()
def predict(model, X, device, bs=512):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + bs]).to(device)).float().cpu().numpy()
           for i in range(0, len(X), bs)]
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--trial-name", default="bilstm")
    ap.add_argument("--log", default="lstm_trials.jsonl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--u-in-cumsum", action="store_true", help="add running total of u_in per breath as an input")
    ap.add_argument("--refit-all", action="store_true",
                    help="train on all of train.csv (no holdout), predict test.csv, write --out")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--max-train-breaths", type=int, default=0, help="subsample for smoke tests")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)

    df = load_breaths(f"{args.data_dir}/train.csv")
    inputs = list(INPUTS)
    if args.u_in_cumsum:
        df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
        inputs.append("u_in_cumsum")
    print(f"inputs: {inputs}")
    if args.refit_all:
        tr_df, va_df = df, None
    else:
        # same split as train.py: GroupShuffleSplit on breath_id, 20% holdout, seed 42
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr_idx, va_idx = next(gss.split(df, groups=df["breath_id"]))
        tr_df, va_df = df.iloc[tr_idx], df.iloc[va_idx]
        assert not set(tr_df.breath_id) & set(va_df.breath_id)

    Xtr, ytr, mtr = to_arrays(tr_df, inputs)
    Xva, yva, mva = to_arrays(va_df, inputs) if va_df is not None else (None, None, None)
    if args.max_train_breaths:
        Xtr, ytr, mtr = Xtr[:args.max_train_breaths], ytr[:args.max_train_breaths], mtr[:args.max_train_breaths]

    mu = Xtr.reshape(-1, len(inputs)).mean(0)
    sd = Xtr.reshape(-1, len(inputs)).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    if Xva is not None:
        Xva = (Xva - mu) / sd
    print(f"train breaths {len(Xtr)}, holdout breaths {0 if Xva is None else len(Xva)}, device {device}")

    model = BiLSTM(len(inputs), args.hidden, args.layers, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(len(Xtr) / args.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.05)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    mtr_t = torch.from_numpy(mtr).to(device)

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for i in range(0, len(Xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = masked_l1(model(Xtr_t[idx]), ytr_t[idx], mtr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if Xva is not None:
            va_pred = predict(model, Xva, device)
            va_mae = float(np.abs(va_pred - yva)[mva].mean())
            history.append(va_mae)
            va_str = f"  holdout insp MAE {va_mae:.4f}"
        else:
            va_str = ""
        print(f"epoch {epoch:3d}  train L1 {tot / len(Xtr):.4f}{va_str}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s{mem_str(device)}", flush=True)

    if args.refit_all:
        test = load_breaths(f"{args.data_dir}/test.csv")
        if args.u_in_cumsum:
            test["u_in_cumsum"] = test.groupby("breath_id")["u_in"].cumsum()
        Xte, _, _ = to_arrays(test, inputs)
        Xte = (Xte - mu) / sd
        check_X, check_pred = Xte, predict(model, Xte, device)
        sub = pd.DataFrame({"id": test["id"].to_numpy(), "pressure": check_pred.reshape(-1)})
        sub = sub.sort_values("id").reset_index(drop=True)
        sample = pd.read_csv(f"{args.data_dir}/sample_submission.csv")
        raw_test_ids = pd.read_csv(f"{args.data_dir}/test.csv", usecols=["id"])["id"]
        assert list(sub.columns) == ["id", "pressure"]
        assert sub["id"].is_unique, "duplicate ids"
        assert set(sub["id"]) == set(raw_test_ids), "submission ids differ from test.csv ids"
        assert sub["id"].equals(sample["id"].sort_values().reset_index(drop=True)), "ids differ from sample_submission"
        assert np.isfinite(sub["pressure"]).all(), "non-finite predictions"
        sub.to_csv(args.out, index=False)
        print(f"wrote {args.out}: {len(sub)} rows; all {raw_test_ids.nunique()} test ids present, no NaN/inf")
    else:
        check_X, check_pred = Xva, va_pred

    if device.type == "mps":
        # MPS sanity check: the same weights on CPU must give the same predictions
        cpu_model = BiLSTM(len(inputs), args.hidden, args.layers, args.dropout)
        cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
        n = min(2000, len(check_X))
        cpu_pred = predict(cpu_model, check_X[:n], torch.device("cpu"))
        diff = float(np.abs(cpu_pred - check_pred[:n]).max())
        print(f"MPS vs CPU prediction max |diff| on {n} breaths: {diff:.2e}")
        assert diff < 1e-2, "MPS predictions disagree with CPU"

    runtime = time.time() - t_start
    final_mae = history[-1] if history else None
    rec = {
        "trial": args.trial_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "holdout_insp_mae_final": round(final_mae, 5) if history else None,
        "holdout_insp_mae_best": round(min(history), 5) if history else None,
        "best_epoch": int(np.argmin(history)) + 1 if history else None,
        "runtime_sec": round(runtime, 1),
        "n_params": n_params,
        "train_breaths": len(Xtr),
        "inputs": inputs,
        "config": {k: v for k, v in vars(args).items() if k not in ("data_dir", "log")},
    }
    with open(args.log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if history:
        print(f"HOLDOUT inspiratory MAE (u_out==0), final epoch: {final_mae:.4f}  "
              f"(best {min(history):.4f} @ epoch {rec['best_epoch']})  runtime {runtime / 60:.1f} min")
    else:
        print(f"refit on all data done (no holdout)  runtime {runtime / 60:.1f} min")


if __name__ == "__main__":
    main()
