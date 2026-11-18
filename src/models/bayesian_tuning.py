#!/usr/bin/env python
"""
PathSnap - Member 2 - Bayesian hyperparameter tuning (standalone step)

Run this ONLY after the 5-model comparison table has been merged from
everyone's parts (LogisticRegression, KNN, RandomForest, ExtraTrees, MLP)
and you know which model actually won on student-test performance
(not accuracy alone -- weigh stability/log loss/inference time too).

Usage:
    python scripts/member2_bayesian_tuning.py --model RandomForest --n-trials 25
    python scripts/member2_bayesian_tuning.py --model MLP --n-trials 20 --opt-folds 3

Output (in artifacts/):
    optuna_trials.csv       every trial tried + its score
    tuning_summary.json     best params, untuned vs tuned comparison
    best_model.pkl          final tuned pipeline + feature list, ready for Member 3/4
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from member2_train_models import (  # noqa: E402
    FEATURES, RANDOM_STATE, cross_validate, eval_split, fit_full_and_eval_student,
    load_data, make_pipeline,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)


def build_model(trial, name):
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 6, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            max_depth=trial.suggest_int("max_depth", 6, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
    if name == "LogisticRegression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 1e2, log=True),
            max_iter=400, class_weight="balanced", random_state=RANDOM_STATE)
    if name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 5, 50),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]), n_jobs=-1)
    if name == "MLP":
        n1 = trial.suggest_int("n1", 16, 128, step=16)
        n2 = trial.suggest_int("n2", 8, 64, step=8)
        return MLPClassifier(
            hidden_layer_sizes=(n1, n2),
            alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            learning_rate_init=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            batch_size=4096, early_stopping=True, n_iter_no_change=8, max_iter=150,
            random_state=RANDOM_STATE)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                     choices=["LogisticRegression", "KNN", "RandomForest", "ExtraTrees", "MLP"],
                     help="The model that won the 5-model comparison -- tune THIS one.")
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--opt-folds", type=int, default=3,
                     help="Folds used *inside* the search for speed (full 5-fold "
                          "CV is still used for the final untuned-vs-tuned comparison).")
    ap.add_argument("--comparison-csv", type=str, default=None,
                     help="Optional: path to the merged 5-model comparison_table.csv, "
                          "so the untuned baseline row for --model is reused instead of "
                          "being recomputed.")
    args = ap.parse_args()

    print("loading data ...")
    import member2_train_models as m
    train, student, labels = load_data()
    m.LABELS = labels
    print(f"train candidate rows: {len(train):,} ({train.trip_id.nunique()} trips)")

    best_name = args.model
    raw_dir = ROOT / "data" / "raw"

    # untuned baseline: reuse from the merged comparison table if given, else recompute
    untuned_row = None
    if args.comparison_csv:
        comp = pd.read_csv(args.comparison_csv)
        row = comp[comp.model == best_name]
        if len(row):
            untuned_row = row.iloc[0].to_dict()
    if untuned_row is None:
        print(f"(no matching row in --comparison-csv; recomputing untuned {best_name} baseline)")
        from member2_train_models import get_models
        family, model = get_models()[best_name]
        cv = cross_validate(best_name, model, train)
        _, stu = fit_full_and_eval_student(best_name, model, train, student, raw_dir)
        untuned_row = {**{"cv_" + k: v for k, v in cv.items()},
                        **{"student_" + k: v for k, v in stu.items()}}

    print(f"\n=== Bayesian hyperparameter optimisation for {best_name} "
          f"({args.n_trials} trials, {args.opt_folds}-fold CV inside search) ===")

    def objective(trial):
        model = build_model(trial, best_name)
        scores = []
        for f in range(args.opt_folds):
            tr = train[train.fold != f]
            va = train[train.fold == f]
            pipe = make_pipeline(model)
            pipe.fit(tr[FEATURES].values, tr["label"].values)
            metrics, _ = eval_split(pipe, va, labels=labels)
            scores.append(metrics["top1_acc"])
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials)
    opt_time = time.time() - t0

    study.trials_dataframe().to_csv(ART / "optuna_trials.csv", index=False)
    print(f"optimisation done in {opt_time:.1f}s. best params: {study.best_params}")
    print(f"best CV top1 (mean of {args.opt_folds} folds): {study.best_value:.4f}")

    tuned_model = build_model(study.best_trial, best_name)
    tuned_cv = cross_validate(best_name + "_tuned", tuned_model, train)
    tuned_pipe, tuned_stu = fit_full_and_eval_student(
        best_name + "_tuned", tuned_model, train, student, raw_dir)

    print("\n================ TUNED vs UNTUNED ================")
    print(f"{'metric':<24}{'untuned':>12}{'tuned':>12}")
    pairs = [("cv_top1_acc", "top1_acc"), ("cv_macro_f1", "macro_f1"), ("cv_log_loss", "log_loss")]
    for k, kk in pairs:
        u = untuned_row.get(k, float("nan"))
        print(f"{k:<24}{u:>12.4f}{tuned_cv[kk]:>12.4f}")
    for k in ["top1_acc", "macro_f1", "log_loss", "route_continuity", "inference_ms_per_row"]:
        u = untuned_row.get("student_" + k, float("nan"))
        print(f"student_{k:<16}{u:>12.4f}{tuned_stu[k]:>12.4f}")

    summary = dict(best_model_family=best_name, best_params=study.best_params,
                    n_trials=args.n_trials, opt_folds=args.opt_folds, optuna_time_s=opt_time,
                    untuned=untuned_row, tuned_cv=tuned_cv, tuned_student=tuned_stu,
                    feature_columns=FEATURES)
    with open(ART / "tuning_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(ART / "best_model.pkl", "wb") as fh:
        pickle.dump({"pipeline": tuned_pipe, "feature_columns": FEATURES,
                     "model_family": best_name}, fh)

    print(f"\nSaved: {ART/'best_model.pkl'}, {ART/'optuna_trials.csv'}, "
          f"{ART/'tuning_summary.json'}")
    print("Send these 3 files back -- that's the whole Bayesian tuning deliverable.")


if __name__ == "__main__":
    main()
