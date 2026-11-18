#!/usr/bin/env python
"""
PathSnap - Member 2 (Models and Tuning)

Trains & compares 5+ candidate-classification models, then runs Bayesian
hyperparameter optimisation (Optuna) on the best one.

Input: data/processed_local/{features.parquet, folds.parquet, point_labels.parquet}
  (features.parquet is gitignored upstream as "large generated"; regenerate it
   with `python src/data_processing/member1_pipeline.py --raw data/raw --out data/processed_local`
   if it's missing -- it must come from the SAME run as folds.parquet.)

Output: artifacts/best_model.pkl, artifacts/comparison_table.csv,
        artifacts/optuna_trials.csv, artifacts/model_comparison.png
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
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed_local"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

FEATURES = ["distance", "rank", "frac_along", "arc_length", "dist_rel_min",
            "n_cand", "heading_diff", "heading_diff_undirected", "heading_valid",
            "speed", "d_prev", "d_next"]

N_FOLDS = 5
RANDOM_STATE = 42


# --------------------------------------------------------------- data
def load_data():
    feat = pd.read_parquet(PROC / "features.parquet")
    folds = pd.read_parquet(PROC / "folds.parquet")
    labels = pd.read_parquet(PROC / "point_labels.parquet")[["trip_id", "obs_idx", "true_arc_id"]]

    feat = feat.merge(folds[["trip_id", "partition", "fold"]], on="trip_id", how="left")
    train = feat[feat.partition == "train"].reset_index(drop=True)
    student = feat[feat.partition == "student_test"].reset_index(drop=True)
    return train, student, labels


# --------------------------------------------------------- candidate-level helpers
def make_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("clf", model)])


def eval_split(pipe, df, labels):
    """Row-level (log loss, macro F1) + observation-level Top-1 accuracy."""
    X, y = df[FEATURES].values, df["label"].values
    t0 = time.time()
    proba = pipe.predict_proba(X)[:, 1]
    inf_ms = (time.time() - t0) * 1000.0 / max(len(df), 1)

    pred = (proba >= 0.5).astype(int)
    f1_macro = f1_score(y, pred, average="macro")
    try:
        ll = log_loss(y, proba, labels=[0, 1])
    except ValueError:
        ll = np.nan

    tmp = df[["trip_id", "obs_idx", "arc_id"]].copy()
    tmp["score"] = proba
    top1 = tmp.loc[tmp.groupby(["trip_id", "obs_idx"])["score"].idxmax()]
    top1 = top1.merge(labels, on=["trip_id", "obs_idx"], how="left")
    top1_acc = float((top1.arc_id == top1.true_arc_id).mean())

    return dict(macro_f1=f1_macro, log_loss=ll, top1_acc=top1_acc, inference_ms_per_row=inf_ms), top1


def route_continuity(top1_df, raw_dir):
    """Fraction of consecutive predicted arcs per trip that are the same arc
    or share an endpoint node (i.e. the predicted route doesn't 'teleport')."""
    scores = []
    for tid, g in top1_df.groupby("trip_id"):
        g = g.sort_values("obs_idx")
        parts = tid.split("__")
        arcs_path = raw_dir / parts[0] / parts[1] / f"{parts[-1]}.arcs"
        if not arcs_path.exists():
            continue
        arcs = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"])
        endpoints = {i: (int(r.u), int(r.v)) for i, r in arcs.iterrows()}
        seq = g.arc_id.values
        ok, tot = 0, 0
        for a, b in zip(seq[:-1], seq[1:]):
            tot += 1
            if a == b:
                ok += 1
                continue
            ea, eb = endpoints.get(a), endpoints.get(b)
            if ea and eb and set(ea) & set(eb):
                ok += 1
        if tot:
            scores.append(ok / tot)
    return float(np.mean(scores)) if scores else np.nan


# --------------------------------------------------------------- models
def get_models():
    return {
        "LogisticRegression": ("Linear/Probabilistic",
                                LogisticRegression(max_iter=300, class_weight="balanced",
                                                    random_state=RANDOM_STATE)),
        "KNN": ("Instance-based",
                KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
        "RandomForest": ("Ensemble (bagging)",
                         RandomForestClassifier(n_estimators=200, max_depth=18, n_jobs=-1,
                                                 class_weight="balanced_subsample",
                                                 random_state=RANDOM_STATE)),
        "ExtraTrees": ("Ensemble (extremely randomised)",
                       ExtraTreesClassifier(n_estimators=200, max_depth=18, n_jobs=-1,
                                             class_weight="balanced_subsample",
                                             random_state=RANDOM_STATE)),
        "MLP": ("Neural network",
                MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=4096,
                               early_stopping=True, n_iter_no_change=8, max_iter=150,
                               random_state=RANDOM_STATE)),
    }


def cross_validate(name, model, train, n_folds=N_FOLDS):
    fold_metrics = []
    t0 = time.time()
    for f in range(n_folds):
        tr = train[train.fold != f]
        va = train[train.fold == f]
        pipe = make_pipeline(model)
        pipe.fit(tr[FEATURES].values, tr["label"].values)
        m, _ = eval_split(pipe, va, labels=LABELS)
        fold_metrics.append(m)
    train_time = time.time() - t0
    agg = pd.DataFrame(fold_metrics).mean().to_dict()
    agg["cv_fold_std_top1"] = pd.DataFrame(fold_metrics).top1_acc.std()
    agg["train_time_s"] = train_time
    return agg


def fit_full_and_eval_student(name, model, train, student, raw_dir):
    pipe = make_pipeline(model)
    t0 = time.time()
    pipe.fit(train[FEATURES].values, train["label"].values)
    fit_time = time.time() - t0
    m, top1 = eval_split(pipe, student, labels=LABELS)
    rc = route_continuity(top1, raw_dir)
    m["route_continuity"] = rc
    m["fit_time_s"] = fit_time
    return pipe, m


LABELS = None  # set in main()


def parse_args():
    ap = argparse.ArgumentParser(description="PathSnap Member 2: train & compare models")
    ap.add_argument("--models", type=str, default=None,
                     help="Comma-separated subset to run, e.g. "
                          "'RandomForest,ExtraTrees'. Default: all 5. "
                          f"Choices: {','.join(get_models().keys())}")
    ap.add_argument("--skip-tuning", action="store_true",
                     help="Only run the 5-model comparison; skip Bayesian optimisation "
                          "(do this when running a subset of models in parallel -- "
                          "tuning should happen once, after everyone's results are merged).")
    ap.add_argument("--ckpt-suffix", type=str, default="",
                     help="Suffix for the checkpoint/output files, e.g. '_teammateA', "
                          "so two people running in parallel don't overwrite each other.")
    return ap.parse_args()


def main():
    global LABELS
    args = parse_args()
    print("loading data ...")
    train, student, labels = load_data()
    LABELS = labels
    print(f"train candidate rows: {len(train):,} ({train.trip_id.nunique()} trips)")
    print(f"student candidate rows: {len(student):,} ({student.trip_id.nunique()} trips)")
    print(f"positive rate (train): {train.label.mean():.4f}")

    raw_dir = ROOT / "data" / "raw"
    results = []
    fitted = {}

    all_models = get_models()
    # preserve the ORDER given in --models (so e.g. "ExtraTrees,KNN" really
    # runs ExtraTrees first) instead of always following get_models()'s order
    wanted_order = ([m.strip() for m in args.models.split(",")]
                     if args.models else list(all_models.keys()))
    unknown = set(wanted_order) - set(all_models.keys())
    if unknown:
        raise SystemExit(f"Unknown model name(s): {unknown}. "
                          f"Choices: {list(all_models.keys())}")
    wanted = set(wanted_order)
    all_models = {name: all_models[name] for name in wanted_order}

    # resume support: skip models already checkpointed from a previous (killed) run
    ckpt_path = ART / f"comparison_table_partial{args.ckpt_suffix}.csv"
    done_models = set()
    if ckpt_path.exists():
        prev = pd.read_csv(ckpt_path)
        results = prev.to_dict("records")
        done_models = set(prev["model"])
        print(f"resuming: already have results for {sorted(done_models)}")

    for name, (family, model) in all_models.items():
        if name not in wanted:
            continue
        if name in done_models:
            print(f"\n=== {name} ({family}) === [skipping, already checkpointed]")
            continue
        print(f"\n=== {name} ({family}) ===", flush=True)
        cv = cross_validate(name, model, train)
        print(f"  CV: top1={cv['top1_acc']:.4f} macroF1={cv['macro_f1']:.4f} "
              f"logloss={cv['log_loss']:.4f} train_time={cv['train_time_s']:.1f}s", flush=True)
        pipe, stu = fit_full_and_eval_student(name, model, train, student, raw_dir)
        print(f"  Student-test: top1={stu['top1_acc']:.4f} macroF1={stu['macro_f1']:.4f} "
              f"logloss={stu['log_loss']:.4f} route_continuity={stu['route_continuity']:.4f} "
              f"inference={stu['inference_ms_per_row']:.4f} ms/row", flush=True)
        fitted[name] = pipe
        results.append(dict(
            model=name, family=family,
            cv_macro_f1=cv["macro_f1"], cv_top1_acc=cv["top1_acc"], cv_log_loss=cv["log_loss"],
            cv_fold_std_top1=cv["cv_fold_std_top1"], train_time_s=cv["train_time_s"],
            student_macro_f1=stu["macro_f1"], student_top1_acc=stu["top1_acc"],
            student_log_loss=stu["log_loss"], route_continuity=stu["route_continuity"],
            inference_ms_per_row=stu["inference_ms_per_row"],
        ))
        pd.DataFrame(results).to_csv(ckpt_path, index=False)  # checkpoint after every model

    comp = pd.DataFrame(results).sort_values("student_top1_acc", ascending=False)
    comp.to_csv(ART / f"comparison_table{args.ckpt_suffix}.csv", index=False)
    print("\n================ COMPARISON TABLE (this run's models only) ================")
    print(comp.to_string(index=False))

    with open(ART / f"untuned_models{args.ckpt_suffix}.pkl", "wb") as fh:
        pickle.dump(fitted, fh)

    if args.skip_tuning:
        print(f"\n--skip-tuning set: stopping here. Send back "
              f"artifacts/comparison_table_partial{args.ckpt_suffix}.csv "
              f"(and untuned_models{args.ckpt_suffix}.pkl if useful) "
              f"so the full 5-model table + Bayesian tuning can be run once "
              f"everyone's results are merged.")
        return

    if len(results) < 5:
        print("\nWARNING: fewer than 5 models in this run's table -- only pick/tune "
              "the best model once ALL 5 models' results have been merged together, "
              "not from this partial table.")

    best_name = comp.iloc[0]["model"]
    print(f"\nBest by student-test Top-1 accuracy (one of several criteria): {best_name}")
    print("NOTE: final pick should also weigh stability (cv_fold_std_top1), "
          "log loss / probability quality, inference time and route_continuity -- "
          "not Top-1 accuracy alone. Edit BEST_MODEL_OVERRIDE below if you disagree.")

    BEST_MODEL_OVERRIDE = None  # e.g. "RandomForest" to force a choice
    best_name = BEST_MODEL_OVERRIDE or best_name

    # ---------------------------------------------------------- Bayesian tuning
    print(f"\n=== Bayesian hyperparameter optimisation for {best_name} ===")

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

    # subsample of folds for speed inside the search (objective = mean CV top-1 acc)
    N_OPT_FOLDS = 3

    def objective(trial):
        model = build_model(trial, best_name)
        scores = []
        for f in range(N_OPT_FOLDS):
            tr = train[train.fold != f]
            va = train[train.fold == f]
            pipe = make_pipeline(model)
            pipe.fit(tr[FEATURES].values, tr["label"].values)
            m, _ = eval_split(pipe, va, labels=LABELS)
            scores.append(m["top1_acc"])
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    N_TRIALS = 25
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    opt_time = time.time() - t0

    trials_df = study.trials_dataframe()
    trials_df.to_csv(ART / "optuna_trials.csv", index=False)
    print(f"optimisation done in {opt_time:.1f}s over {N_TRIALS} trials")
    print("best params:", study.best_params)
    print("best CV top1 (mean of 3 folds):", study.best_value)

    tuned_model = build_model(study.best_trial, best_name)
    tuned_cv = cross_validate(best_name + "_tuned", tuned_model, train)
    tuned_pipe, tuned_stu = fit_full_and_eval_student(best_name + "_tuned", tuned_model,
                                                        train, student, raw_dir)

    untuned_row = comp[comp.model == best_name].iloc[0]
    print("\n================ TUNED vs UNTUNED ================")
    print(f"{'metric':<22}{'untuned':>12}{'tuned':>12}")
    for k, kk in [("cv_top1_acc", "top1_acc"), ("cv_macro_f1", "macro_f1"),
                  ("cv_log_loss", "log_loss")]:
        print(f"{k:<22}{untuned_row[k]:>12.4f}{tuned_cv[kk]:>12.4f}")
    for k in ["top1_acc", "macro_f1", "log_loss", "route_continuity", "inference_ms_per_row"]:
        print(f"student_{k:<14}{untuned_row.get('student_' + k, float('nan')):>12.4f}"
              f"{tuned_stu[k]:>12.4f}")

    summary = dict(
        best_model_family=best_name, best_params=study.best_params,
        n_trials=N_TRIALS, optuna_time_s=opt_time,
        untuned=untuned_row.to_dict(), tuned_cv=tuned_cv, tuned_student=tuned_stu,
        feature_columns=FEATURES,
    )
    with open(ART / "tuning_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    with open(ART / "best_model.pkl", "wb") as fh:
        pickle.dump({"pipeline": tuned_pipe, "feature_columns": FEATURES,
                     "model_family": best_name}, fh)

    print(f"\nSaved: {ART/'best_model.pkl'}, {ART/'comparison_table.csv'}, "
          f"{ART/'optuna_trials.csv'}, {ART/'tuning_summary.json'}")


if __name__ == "__main__":
    main()
