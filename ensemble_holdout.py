"""Average holdout predictions from several lstm.py runs and report inspiratory MAE.

Usage:
    python ensemble_holdout.py --name seeds1-5 holdout_preds/seed1.npz ... holdout_preds/seed5.npz
Appends one record to lstm_trials.jsonl. Never writes submission.csv.
"""
import argparse
import json
from datetime import datetime

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--log", default="lstm_trials.jsonl")
    args = ap.parse_args()

    runs = [np.load(f) for f in args.files]
    ref = runs[0]
    for f, r in zip(args.files, runs):
        # every run must cover exactly the same holdout breaths, in the same order
        assert np.array_equal(r["breath_id"], ref["breath_id"]), f"{f}: holdout breaths differ"
        assert np.array_equal(r["y"], ref["y"]) and np.array_equal(r["mask"], ref["mask"]), f
    y, mask = ref["y"], ref["mask"]

    per_run = {}
    for f, r in zip(args.files, runs):
        per_run[f] = float(np.abs(r["pred"] - y)[mask].mean())
        print(f"{f}: holdout insp MAE {per_run[f]:.4f}")
    mean_pred = np.mean([r["pred"] for r in runs], axis=0)
    median_pred = np.median([r["pred"] for r in runs], axis=0)
    mae_mean = float(np.abs(mean_pred - y)[mask].mean())
    mae_median = float(np.abs(median_pred - y)[mask].mean())
    print(f"individual MAE mean {np.mean(list(per_run.values())):.4f} +/- {np.std(list(per_run.values())):.4f}")
    print(f"ENSEMBLE (mean of {len(runs)}) holdout insp MAE: {mae_mean:.4f}")
    print(f"ensemble (median of {len(runs)}) holdout insp MAE: {mae_median:.4f}")

    rec = {
        "trial": args.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": "holdout_ensemble",
        "holdout_insp_mae_final": round(mae_mean, 5),
        "holdout_insp_mae_median_ensemble": round(mae_median, 5),
        "member_maes": {k: round(v, 5) for k, v in per_run.items()},
        "n_holdout_breaths": int(len(ref["breath_id"])),
    }
    with open(args.log, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
