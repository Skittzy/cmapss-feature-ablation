"""Run the experiments and record one row per run in results/runs.csv.

Each row identifies the configuration, dataset, model and seed. Changing the
initial seed of the random-number generator changes the model's starting
weights and produces slightly different results. The log is the source of
every number reported in the paper.

After an interruption, such as a crash or system sleep,
restarting the runner skips runs already recorded in results/runs.csv.

Examples
--------
    # pilot: one config, one dataset, three seeds
    python run_experiment.py --experiment exp1 --configs C6_expose \\
        --datasets FD001 --seeds 0 1 2

    # the full ablation grid
    python run_experiment.py --experiment exp1 --seeds 0 1 2

    # the architecture comparison, once you know which config won
    python run_experiment.py --experiment exp2 --best-config C4_trend --seeds 0 1 2
"""

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ALL_DATASETS, FEATURE_COLS, SENSOR_COLS, SEQUENCE_LENGTH, BATCH_SIZE,
    RUL_CAPS, N_CLUSTERS,
)
from src.data.loader import load_train_data, load_test_data, load_rul_labels
from src.data.preprocessor import prepare_train_data, prepare_test_data
from features_v2 import (CONFIGS, ANCHOR_CONFIGS, LOO_CONFIGS, DIAG_CONFIGS,
                         build_features, group_sizes, count_dead)

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
RUNS_CSV = RESULTS / "runs.csv"
PREDS_DIR = RESULTS / "predictions"
PREDS_DIR.mkdir(exist_ok=True)

# PROTOCOL.md fixes these settings for the duration of the experimental grid.
EPOCHS = 60          # maximum passes over the training data
PATIENCE = 10        # epochs without validation-loss improvement before stopping
VAL_FRACTION = 0.2   # fraction of engines held out for early stopping


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def nasa_score(y_true, y_pred):
    """Compute the PHM08 score used in the C-MAPSS literature.

    Late predictions overestimate remaining life and carry greater risk, so
    they incur a larger penalty than early predictions. Lower scores are
    better. Reviewers in this field expect the score alongside RMSE.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0,
                                 np.exp(d / 10.0) - 1.0)))


def metrics(y_true, y_pred):
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {
        # Root mean squared error (RMSE) in cycles gives greater weight to
        # large errors. Lower values indicate better predictions.
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        # Mean absolute error (MAE) gives the average absolute error in cycles.
        "mae": float(np.mean(np.abs(err))),
        "nasa_score": nasa_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------

def sequences_from(df, feature_cols, seq_len, stride=1):
    """Cut each engine's history into overlapping windows of `seq_len` cycles.

    The label is the remaining useful life (RUL) at the final cycle of each
    window. With `stride` set to 1, all windows are retained. A stride of 2
    keeps every second window, halving the data and roughly halving training time.
    """
    X, y, units = [], [], []
    df = df.sort_values(["unit_id", "time_cycles"])
    for uid, g in df.groupby("unit_id"):
        f, lab = g[feature_cols].values, g["RUL"].values
        n = len(g)
        if n < seq_len:
            continue
        for s in range(0, n - seq_len + 1, stride):
            X.append(f[s:s + seq_len])
            y.append(lab[s + seq_len - 1])
            units.append(uid)
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(units)


def last_windows(df, feature_cols, seq_len):
    """Return the final `seq_len` cycles of each test engine.

    Each test engine has one true RUL, measured at its final recorded cycle.
    Only the final window therefore has a label. Engines with fewer than
    `seq_len` cycles receive leading zeros so the observed readings stay
    closest to the prediction point.
    """
    X, y, units = [], [], []
    df = df.sort_values(["unit_id", "time_cycles"])
    for uid, g in df.groupby("unit_id"):
        f = g[feature_cols].values
        if len(g) >= seq_len:
            w = f[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(g), len(feature_cols)), np.float32)
            w = np.vstack([pad, f])
        X.append(w)
        y.append(g["RUL"].iloc[-1])
        units.append(uid)
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(units)


def prepare(dataset, config_name, groups, stride=1):
    """Load and preprocess the data, build the requested features and form windows."""
    legacy = config_name in ANCHOR_CONFIGS   # C6_legacy and C6_expose

    tr = load_train_data(dataset)
    tr, scalers, kmeans, base_cols = prepare_train_data(
        tr, FEATURE_COLS.copy(), rul_cap=RUL_CAPS[dataset],
        n_clusters=N_CLUSTERS[dataset])

    te = prepare_test_data(load_test_data(dataset), scalers, kmeans, base_cols,
                           load_rul_labels(dataset))

    if legacy:
        # The original pipeline removed constant columns from its feature list,
        # then read all 24 settings and sensors back from the table. This branch
        # reproduces that input exactly, including rolling, lag and trend
        # features for all 21 sensors, even those with constant readings.
        sensors, feat_base = SENSOR_COLS, FEATURE_COLS.copy()
    else:
        sensors, feat_base = None, base_cols

    tr, cols = build_features(tr, groups, feat_base, sensor_cols=sensors,
                              drop_dead=not legacy)
    te, _ = build_features(te, groups, feat_base, sensor_cols=sensors,
                           keep_cols=cols)

    Xtr, ytr, utr = sequences_from(tr, cols, SEQUENCE_LENGTH, stride)
    Xte, yte, ute = last_windows(te, cols, SEQUENCE_LENGTH)
    return dict(Xtr=Xtr, ytr=ytr, utr=utr, Xte=Xte, yte=yte, ute=ute,
                cols=cols, n_features=len(cols), groups=group_sizes(cols),
                n_dead=count_dead(tr, cols))


def split_by_engine(units, seed, val_fraction=VAL_FRACTION):
    """Hold out entire engines for validation.

    Validation engines are excluded from model fitting and used only for early
    stopping. Two windows from one engine can share up to 29 of their 30
    cycles. A window-level split would put near-copies of validation data in
    training and invalidate the stopping decision.
    """
    uniq = np.unique(units)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_fraction)))
    val_ids = set(uniq[:n_val].tolist())
    mask = np.array([u in val_ids for u in units])
    return ~mask, mask


def standardise(Xtr, Xte):
    """Standardise each feature to mean 0 and standard deviation 1.

    Without scaling, the larger numerical values of the cycle counter (0-300+)
    dominate the sensor readings (0-1) in the model. Fit the scaler on
    training data only and apply the fitted transformation to test data.
    """
    from sklearn.preprocessing import StandardScaler
    n, t, f = Xtr.shape
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr.reshape(-1, f)).reshape(n, t, f)
    Xte = sc.transform(Xte.reshape(-1, f)).reshape(Xte.shape)
    return Xtr, Xte


def set_seed(seed):
    import tensorflow as tf
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def fit_lstm(d, seed):
    """Fit the LSTM, carrying its recurrent state through the 30 cycles in order."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping

    set_seed(seed)
    Xtr, Xte = standardise(d["Xtr"], d["Xte"])
    tr_mask, val_mask = split_by_engine(d["utr"], seed)

    model = Sequential([
        Input(shape=(Xtr.shape[1], Xtr.shape[2])),
        LSTM(128, return_sequences=True), Dropout(0.3),
        LSTM(64), Dropout(0.3),
        Dense(32, activation="relu"), Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    t0 = time.time()
    hist = model.fit(
        Xtr[tr_mask], d["ytr"][tr_mask],
        validation_data=(Xtr[val_mask], d["ytr"][val_mask]),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                 restore_best_weights=True)])
    train_s = time.time() - t0

    t0 = time.time()
    pred = model.predict(Xte, verbose=0).flatten()
    return pred, dict(train_seconds=round(train_s, 1),
                      predict_seconds=round(time.time() - t0, 3),
                      epochs_run=len(hist.history["loss"]),
                      n_params=int(model.count_params()))


def fit_flat(d, seed, kind, flatten_window=False):
    """Fit a tree model without a notion of temporal order.

    By default, the input is the final cycle, whose engineered features encode
    recent history. With flatten_window=True, the model receives the same
    values as the LSTM across all 30 cycles in a single row, without any
    notion of their order.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV, GroupKFold

    Xtr, Xte = standardise(d["Xtr"], d["Xte"])
    if flatten_window:
        Xtr2, Xte2 = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    else:
        Xtr2, Xte2 = Xtr[:, -1, :], Xte[:, -1, :]

    fit_kw = {}
    if kind == "rf":
        # Random Forest: hundreds of decision trees that vote on the answer.
        m = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
    elif kind == "dt":
        # Select the depth of the single tree by repeated fitting and scoring
        # on separate folds. GroupKFold keeps all observations from each
        # engine together in every split.
        m = GridSearchCV(DecisionTreeRegressor(random_state=seed),
                         {"max_depth": [4, 6, 8, 10, 14, None]},
                         scoring="neg_root_mean_squared_error",
                         cv=GroupKFold(n_splits=3), n_jobs=-1)
        fit_kw["groups"] = d["utr"]
    else:
        raise ValueError(kind)

    t0 = time.time()
    m.fit(Xtr2, d["ytr"], **fit_kw)
    train_s = time.time() - t0

    t0 = time.time()
    pred = m.predict(Xte2)
    extra = dict(train_seconds=round(train_s, 1),
                 predict_seconds=round(time.time() - t0, 3),
                 epochs_run=None, n_params=None)
    if kind == "dt":
        extra["best_depth"] = m.best_params_["max_depth"]
    return pred, extra


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------

def done_jobs():
    if not RUNS_CSV.exists():
        return set()
    df = pd.read_csv(RUNS_CSV)
    return set(zip(df.experiment, df.config, df.dataset, df.model, df.seed))


def run_job(experiment, config_name, groups, dataset, model, seed, stride):
    d = prepare(dataset, config_name, groups, stride=stride)

    if model == "lstm":
        pred, extra = fit_lstm(d, seed)
    elif model in ("rf", "dt"):
        pred, extra = fit_flat(d, seed, model)
    elif model == "rf_flat":
        pred, extra = fit_flat(d, seed, "rf", flatten_window=True)
    else:
        raise ValueError(model)

    m = metrics(d["yte"], pred)

    # Save each engine's prediction for the paired significance test.
    tag = f"{experiment}_{config_name}_{dataset}_{model}_s{seed}"
    pd.DataFrame({"unit_id": d["ute"], "y_true": d["yte"], "y_pred": pred}) \
        .to_csv(PREDS_DIR / f"{tag}.csv", index=False)

    row = dict(experiment=experiment, config=config_name, dataset=dataset,
               model=model, seed=seed, stride=stride,
               n_features=d["n_features"], n_dead=d["n_dead"],
               **d["groups"], **m, **extra)
    # All appended CSV rows require the same columns. Only the decision tree
    # reports best_depth, so reserve the field for the other models as well.
    row.setdefault("best_depth", None)
    row["timestamp"] = pd.Timestamp.now().isoformat(timespec="seconds")
    pd.DataFrame([row]).to_csv(RUNS_CSV, mode="a",
                               header=not RUNS_CSV.exists(), index=False)

    print(f"  -> RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}  "
          f"score {m['nasa_score']:.0f}  ({d['n_features']} features, "
          f"{d['n_dead']} of them dead)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="exp1",
                    choices=["exp1", "exp1_loo", "exp2"])
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--best-config", default=None,
                    help="which config exp2 uses; pick it from the exp1 results")
    ap.add_argument("--stride", type=int, default=1,
                    help="window step; 2 halves the training set")
    ap.add_argument("--models", nargs="+", default=None,
                    help="override the model list, e.g. --models rf dt")
    args = ap.parse_args()

    all_configs = {**CONFIGS, **ANCHOR_CONFIGS, **DIAG_CONFIGS}

    if args.experiment == "exp2":
        if not args.best_config:
            raise SystemExit("exp2 needs --best-config (choose it from exp1)")
        space = {args.best_config: all_configs[args.best_config]}
        models = ["lstm", "rf", "dt", "rf_flat"]
    else:
        pool = LOO_CONFIGS if args.experiment == "exp1_loo" else all_configs
        keys = args.configs or list(pool)
        space = {k: pool[k] for k in keys}
        models = ["lstm"]

    if args.models:
        models = args.models

    with open(RESULTS / "environment.json", "w") as f:
        json.dump({"python": platform.python_version(),
                   "platform": platform.platform(),
                   "numpy": np.__version__, "pandas": pd.__version__}, f, indent=2)

    already = done_jobs()
    todo = [(c, g, ds, m, s)
            for c, g in space.items() for ds in args.datasets
            for m in models for s in args.seeds
            if (args.experiment, c, ds, m, s) not in already]

    print(f"{len(todo)} runs to do ({len(already)} already finished)\n")
    for i, (c, g, ds, m, s) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {args.experiment} | {c} | {ds} | {m} | seed {s}",
              flush=True)
        run_job(args.experiment, c, g, ds, m, s, args.stride)

    print(f"\nFinished. Results are in {RUNS_CSV}")


if __name__ == "__main__":
    main()
