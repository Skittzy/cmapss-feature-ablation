"""Runs the experiments and writes one row of results per run.

WHAT IT DOES
------------
Every run gets one line in results/runs.csv, tagged with the config, dataset,
model, and seed (the starting number for the random-number generator -- change
it and the model starts from different random weights, so you get a slightly
different answer). That file is the raw evidence behind every number in the
paper.

It is resumable: re-running skips anything already in runs.csv. If your laptop
sleeps or the run crashes, just start it again and it picks up where it left
off.

EXAMPLES
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
from features_v2 import (CONFIGS, ANCHOR_CONFIGS, LOO_CONFIGS,
                         build_features, group_sizes, count_dead)

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
RUNS_CSV = RESULTS / "runs.csv"
PREDS_DIR = RESULTS / "predictions"
PREDS_DIR.mkdir(exist_ok=True)

# These are fixed in PROTOCOL.md. Do not change them once the grid is running.
EPOCHS = 60          # max passes over the training data
PATIENCE = 10        # stop if validation loss hasn't improved for this many
VAL_FRACTION = 0.2   # share of ENGINES held back to decide when to stop


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def nasa_score(y_true, y_pred):
    """The PHM08 score used across the C-MAPSS literature.

    It punishes late predictions (saying an engine has more life left than it
    does) harder than early ones, because that is the dangerous direction.
    Lower is better. Reviewers in this field expect to see it next to RMSE.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0,
                                 np.exp(d / 10.0) - 1.0)))


def metrics(y_true, y_pred):
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return {
        # RMSE = root mean squared error: average error size in cycles, with
        # big misses counted extra heavily. Lower is better.
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        # MAE = mean absolute error: plain average error size in cycles.
        "mae": float(np.mean(np.abs(err))),
        "nasa_score": nasa_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------

def sequences_from(df, feature_cols, seq_len, stride=1):
    """Cut each engine's history into overlapping windows of `seq_len` cycles.

    The label for a window is the RUL at its last cycle. `stride` is how far the
    window jumps each time: 1 means every possible window, 2 means every second
    one (half the training data, roughly half the training time).
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
    """One window per engine: its final `seq_len` cycles.

    The test files give one true RUL per engine, measured at its last recorded
    cycle, so this is the only window that has a label. Engines with fewer than
    `seq_len` cycles get zeros added at the front, keeping the real readings
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
    """Load -> preprocess -> build the requested features -> cut into windows."""
    legacy = config_name in ANCHOR_CONFIGS   # C6_legacy and C6_expose

    tr = load_train_data(dataset)
    tr, scalers, kmeans, base_cols = prepare_train_data(
        tr, FEATURE_COLS.copy(), rul_cap=RUL_CAPS[dataset],
        n_clusters=N_CLUSTERS[dataset])

    te = prepare_test_data(load_test_data(dataset), scalers, kmeans, base_cols,
                           load_rul_labels(dataset))

    if legacy:
        # The old code dropped constant columns from the LIST but not from the
        # table, then read every column back in. So all 24 settings/sensors came
        # back, and rolling/lag/trend were built on all 21 sensors including the
        # flat ones. Reproduced here exactly, dead columns and all.
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
    """Hold back whole ENGINES for validation, never individual windows.

    Validation set = data the model never trains on, used only to decide when to
    stop training. Two windows from the same engine overlap by up to 29 of their
    30 cycles, so splitting by window would put near-copies of the validation
    data into training and make the stopping decision meaningless.
    """
    uniq = np.unique(units)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_fraction)))
    val_ids = set(uniq[:n_val].tolist())
    mask = np.array([u in val_ids for u in units])
    return ~mask, mask


def standardise(Xtr, Xte):
    """Rescale every feature to mean 0, spread 1.

    Without this the cycle counter (0-300+) drowns out the sensor readings
    (0-1), because the model pays most attention to the biggest numbers.
    Fitted on training data only, then applied to test.
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
    """The LSTM (a network that reads the 30 cycles in order and remembers)."""
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
    """Models with no sense of time.

    By default they see only the engine's LAST cycle -- the engineered features
    already contain the recent history as numbers. With flatten_window=True they
    instead see all 30 cycles laid out side by side, which is the same
    information the LSTM gets, just without any notion of order.
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
        # A single decision tree. Depth is picked by cross-validation (train on
        # part of the data, test on the rest, repeat). GroupKFold keeps all of
        # one engine's rows on the same side of every split.
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

    # save every engine's prediction so the significance test can pair them up
    tag = f"{experiment}_{config_name}_{dataset}_{model}_s{seed}"
    pd.DataFrame({"unit_id": d["ute"], "y_true": d["yte"], "y_pred": pred}) \
        .to_csv(PREDS_DIR / f"{tag}.csv", index=False)

    row = dict(experiment=experiment, config=config_name, dataset=dataset,
               model=model, seed=seed, stride=stride,
               n_features=d["n_features"], n_dead=d["n_dead"],
               **d["groups"], **m, **extra)
    # every row must have identical columns, or the appended CSV goes ragged.
    # only the decision tree reports best_depth, so fill it in for everyone else.
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

    all_configs = {**CONFIGS, **ANCHOR_CONFIGS}

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
