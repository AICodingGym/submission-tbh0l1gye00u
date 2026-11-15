"""Baseline for Google Brain Ventilator Pressure Prediction.

Model: sklearn HistGradientBoostingRegressor on per-breath engineered features
(cumulative u_in, lags/leads, diffs, breath-level stats, R/C interactions).
It is trained only on inspiratory rows (u_out == 0), which are the only rows
scored. Lag/lead features still come from the whole breath.

Validation: a fraction of whole breaths (GroupShuffleSplit on breath_id) is
held out. The reported metric is MAE on held-out rows where u_out == 0.
After validation the model is refit on all of train.csv and predicts test.csv.

Usage:
    python train.py                     # validate, refit on all data, write submission.csv
    python train.py --no-refit          # predict test with the validation model (faster)
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

SEED = 42


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
    g = df.groupby("breath_id")

    df["step"] = g.cumcount()
    df["dt"] = g["time_step"].diff().fillna(0)
    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"]).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / (df["step"] + 1)

    for k in (1, 2, 3, 4):
        df[f"u_in_lag{k}"] = g["u_in"].shift(k).fillna(0)
        df[f"u_in_lead{k}"] = g["u_in"].shift(-k).fillna(0)
        df[f"u_in_diff{k}"] = df["u_in"] - df[f"u_in_lag{k}"]
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0)
    df["u_out_lead1"] = g["u_out"].shift(-1).fillna(1)

    df["u_in_roll_mean5"] = g["u_in"].transform(lambda s: s.rolling(5, min_periods=1).mean())
    df["u_in_roll_max5"] = g["u_in"].transform(lambda s: s.rolling(5, min_periods=1).max())
    df["u_in_ewm"] = g["u_in"].transform(lambda s: s.ewm(halflife=3).mean())

    # breath-level stats over the inspiratory phase
    insp_u = df["u_in"].where(df["u_out"] == 0)
    df["breath_u_in_mean"] = insp_u.groupby(df["breath_id"]).transform("mean")
    df["breath_u_in_max"] = insp_u.groupby(df["breath_id"]).transform("max")
    df["breath_u_in_last"] = g["u_in"].transform("last")
    df["u_in_minus_max"] = df["u_in"] - df["breath_u_in_max"]

    df["R_C"] = df["R"] * 100 + df["C"]  # categorical lung setting
    df["RC"] = df["R"] * df["C"]  # time constant (scaled)
    df["area_over_C"] = df["area"] / df["C"]
    df["u_in_times_R"] = df["u_in"] * df["R"]
    df["cumsum_over_C"] = df["u_in_cumsum"] / df["C"]
    return df




def feature_cols(df):
    drop = {"id", "breath_id", "pressure"}
    return [c for c in df.columns if c not in drop]


def make_model():
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.1,
        max_iter=1500,
        max_leaf_nodes=255,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features=None,
        early_stopping=False,
        random_state=SEED,
    )


def insp_mae(y_true, y_pred, u_out):
    m = u_out == 0
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--no-refit", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    train = add_features(pd.read_csv(f"{args.data_dir}/train.csv"))
    test = add_features(pd.read_csv(f"{args.data_dir}/test.csv"))
    feats = feature_cols(train)
    print(f"features ({len(feats)}): {feats}")
    print(f"loaded + features in {time.time() - t0:.0f}s")

    # hold out whole breaths
    gss = GroupShuffleSplit(n_splits=1, test_size=args.holdout_frac, random_state=SEED)
    tr_idx, va_idx = next(gss.split(train, groups=train["breath_id"]))
    tr, va = train.iloc[tr_idx], train.iloc[va_idx]
    assert not set(tr.breath_id) & set(va.breath_id)
    print(f"train breaths {tr.breath_id.nunique()}, holdout breaths {va.breath_id.nunique()}")

    tr_in = tr[tr.u_out == 0]
    t0 = time.time()
    model = make_model().fit(tr_in[feats], tr_in["pressure"])
    print(f"fit on {len(tr_in)} inspiratory rows in {time.time() - t0:.0f}s")

    va_pred = model.predict(va[feats])
    mae = insp_mae(va["pressure"].values, va_pred, va["u_out"].values)
    # reference: per-(R,C,step) median from train split
    ref = tr_in.groupby(["R", "C", "step"])["pressure"].median().rename("ref")
    va_ref = va.join(ref, on=["R", "C", "step"])["ref"].fillna(tr_in.pressure.median()).values
    ref_mae = insp_mae(va["pressure"].values, va_ref, va["u_out"].values)
    print(f"HOLDOUT inspiratory MAE (u_out==0): {mae:.4f}   [per-(R,C,step) median reference: {ref_mae:.4f}]")

    if not args.no_refit:
        all_in = train[train.u_out == 0]
        t0 = time.time()
        model = make_model().fit(all_in[feats], all_in["pressure"])
        print(f"refit on all {len(all_in)} inspiratory rows in {time.time() - t0:.0f}s")

    # expiratory rows (u_out==1) are not scored; the model still emits a value for them
    test["pressure"] = model.predict(test[feats])
    sub = test[["id", "pressure"]].sort_values("id")
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(sub)} rows")


if __name__ == "__main__":
    main()
