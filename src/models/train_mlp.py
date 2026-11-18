#!/usr/bin/env python
"""
Train an MLP that scores each candidate arc (label 1 = true arc).

Usage (from the folder that holds data/processed):
    python train_mlp.py --data data/processed --out artifacts

Rules followed:
  * train + CV only on partition == 'train', using the 'fold' column (0..4)
  * student_test is only used for the final development check
  * scaler is fitted inside each training fold (no leakage)
  * negatives are subsampled for training only; scoring uses ALL candidates
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, average_precision_score

FEATURES = ["distance", "rank", "dist_rel_min", "frac_along", "arc_length",
            "n_cand", "heading_diff", "heading_diff_undirected",
            "heading_valid", "speed", "d_prev", "d_next"]
KEY = ["trip_id", "obs_idx"]


def subsample(df, neg_per_pos=5, seed=0):
    """Keep every positive row and a random subset of negatives."""
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    n = min(len(neg), neg_per_pos * len(pos))
    return pd.concat([pos, neg.sample(n, random_state=seed)]).sample(frac=1, random_state=seed)


def make_model(seed=0):
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                      alpha=1e-4, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=10, max_iter=200, random_state=seed))


def evaluate(df, prob):
    """Row-level metrics + per-point top-1 accuracy."""
    d = df[KEY + ["arc_id", "label"]].copy()
    d["prob"] = prob
    has_true = d.groupby(KEY).label.transform("max") == 1      # true arc is a candidate
    top = d.loc[d.groupby(KEY).prob.idxmax()]
    top1_all = top.label.mean()                                 # counts misses as wrong
    top1_cov = top[top.set_index(KEY).index.isin(
        d[has_true].set_index(KEY).index)].label.mean()         # only coverable points
    return dict(logloss=log_loss(d.label, np.clip(prob, 1e-6, 1 - 1e-6)),
                pr_auc=average_precision_score(d.label, prob),
                top1_all_points=top1_all, top1_covered_points=top1_cov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="artifacts")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(data / "features.parquet")
    folds = pd.read_parquet(data / "folds.parquet")
    df = feats.merge(folds, on="trip_id", how="inner")
    df = df[df.label.notna()].copy()
    df["label"] = df.label.astype(int)
    train = df[df.partition == "train"].reset_index(drop=True)
    stest = df[df.partition == "student_test"].reset_index(drop=True)
    print(f"train rows {len(train):,}  student_test rows {len(stest):,}")

    # ---- 5-fold CV over train trips ----
    rows = []
    for f in sorted(train.fold.unique()):
        tr, va = train[train.fold != f], train[train.fold == f]
        m = make_model().fit(subsample(tr)[FEATURES], subsample(tr).label)
        p = m.predict_proba(va[FEATURES])[:, 1]
        r = evaluate(va, p)
        r["fold"] = int(f)
        rows.append(r)
        print(f"fold {f}: " + "  ".join(f"{k}={v:.4f}" for k, v in r.items() if k != "fold"))
    cv = pd.DataFrame(rows)
    print("\nCV mean:\n", cv.drop(columns="fold").mean().round(4))
    cv.to_csv(out / "mlp_cv_results.csv", index=False)

    # ---- final model on all train trips, check on student_test ----
    s = subsample(train)
    final = make_model().fit(s[FEATURES], s.label)
    if len(stest):
        p = final.predict_proba(stest[FEATURES])[:, 1]
        res = evaluate(stest, p)
        print("\nstudent_test:", {k: round(v, 4) for k, v in res.items()})
        pd.DataFrame([res]).to_csv(out / "mlp_student_test.csv", index=False)
        pr = stest[KEY + ["arc_id"]].copy()
        pr["prob"] = p
        pr.to_parquet(out / "mlp_student_test_probs.parquet")   # for Member 3
    joblib.dump(dict(model=final, features=FEATURES), out / "mlp_model.joblib")
    print(f"\nsaved model + results in {out}")


if __name__ == "__main__":
    main()
