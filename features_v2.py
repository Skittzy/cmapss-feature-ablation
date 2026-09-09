"""Builds the input features for the C-MAPSS ablation study.

WHY THIS FILE EXISTS
--------------------
The original `create_all_features()` always builds every feature group at once.
That makes an ablation study (running the model with some features removed, to
see which ones actually matter) impossible. This version lets you say exactly
which groups you want.

THE SEVEN GROUPS
----------------
    base     the settings and sensors that survive constant-removal
    rolling  rolling mean and standard deviation, windows 5 and 10
             (the average and the spread of the last 5 and last 10 cycles)
    lag      the value 1 cycle ago and 3 cycles ago
    trend    value now minus value 10 cycles ago (how fast it is moving)
    agg      5 numbers summarising all sensors at once (mean, sd, min, max, range)
    cycle    cycle number, log of it, square root of it
    opcond   the operating-condition cluster number (only used by C6_expose)

THE LADDER (Experiment 1)
-------------------------
Each rung adds one group on top of the one before:

    C1_base -> C2_rolling -> C3_lag -> C4_trend -> C5_agg -> C6_full

THREE ANCHOR RUNS
-----------------
    C6_full     the clean pipeline this paper reports (129-145 features)
    C6_legacy   the code currently on GitHub, warts and all (179 features)
    C6_expose   the code that produced the exposé's RMSE 17.1 / 23.7 / 14.2 /
                24.0 (180 features -- C6_legacy plus `op_condition`, the
                k-means cluster label, fed in as an ordinary number)

Running all three answers a question for free: does cleaning up the feature set
change the accuracy at all?

USAGE
-----
    train_df, cols = build_features(train_df, CONFIGS["C3_lag"], base_cols)
    test_df,  _    = build_features(test_df,  CONFIGS["C3_lag"], base_cols,
                                    keep_cols=cols)

Always pass `keep_cols` from the train call when you build the test set. That
guarantees both sets have the same columns in the same order.
"""

import numpy as np
import pandas as pd

# columns that are never model inputs: ids, the target, and bookkeeping
METADATA = ["unit_id", "time_cycles", "RUL", "max_cycle", "op_condition"]

GROUP_ORDER = ["base", "rolling", "lag", "trend", "agg", "cycle", "opcond"]

CONFIGS = {
    "C1_base":    ["base"],
    "C2_rolling": ["base", "rolling"],
    "C3_lag":     ["base", "rolling", "lag"],
    "C4_trend":   ["base", "rolling", "lag", "trend"],
    "C5_agg":     ["base", "rolling", "lag", "trend", "agg"],
    "C6_full":    ["base", "rolling", "lag", "trend", "agg", "cycle"],
}

# anchors -- see the module docstring. run_experiment.py handles them specially.
ANCHOR_CONFIGS = {
    "C6_legacy": ["base", "rolling", "lag", "trend", "agg", "cycle"],
    "C6_expose": ["base", "rolling", "lag", "trend", "agg", "cycle", "opcond"],
}

# leave-one-out: everything except one group. Answers a different question from
# the ladder -- see PROTOCOL.md.
LOO_CONFIGS = {
    f"L_no_{g}": [x for x in GROUP_ORDER if x not in (g, "opcond")]
    for g in ["rolling", "lag", "trend", "agg", "cycle"]
}


def _rolling(df, cols, windows):
    out = []
    g = df.groupby("unit_id")          # never mix data across engines
    for c in cols:
        for w in windows:
            out.append(g[c].transform(lambda x: x.rolling(w, min_periods=1).mean())
                       .rename(f"{c}_rolling_mean_{w}"))
            out.append(g[c].transform(lambda x: x.rolling(w, min_periods=1).std())
                       .rename(f"{c}_rolling_std_{w}"))
    return out


def _lag(df, cols, lags):
    g = df.groupby("unit_id")
    return [g[c].shift(l).rename(f"{c}_lag_{l}") for c in cols for l in lags]


def _trend(df, cols, window):
    g = df.groupby("unit_id")
    return [g[c].transform(lambda x: x - x.shift(window)).rename(f"{c}_trend_{window}")
            for c in cols]


def _agg(df, sensor_cols):
    s = df[sensor_cols]
    mn, mx = s.min(axis=1), s.max(axis=1)
    return [
        s.mean(axis=1).rename("sensors_mean"),
        s.std(axis=1).rename("sensors_std"),
        mn.rename("sensors_min"),
        mx.rename("sensors_max"),
        (mx - mn).rename("sensors_range"),
    ]


def _cycle(df):
    return [
        df["time_cycles"].rename("cycle"),
        np.log1p(df["time_cycles"]).rename("cycle_log"),
        np.sqrt(df["time_cycles"]).rename("cycle_sqrt"),
    ]


def build_features(df, groups, base_cols, sensor_cols=None,
                   rolling_windows=(5, 10), lags=(1, 3), trend_window=10,
                   drop_dead=True, keep_cols=None):
    """Build only the feature groups you ask for.

    Parameters
    ----------
    df : DataFrame
        Output of prepare_train_data or prepare_test_data. Must already have
        unit_id, time_cycles, RUL and op_condition.
    groups : list of str
        Which groups to build. "base" is always included.
    base_cols : list of str
        Settings + sensors that survived constant-removal. This is the
        `feature_cols` list that prepare_train_data returns.
    sensor_cols : list of str, optional
        Which sensors get rolling / lag / trend built on top of them. Defaults
        to the sensors inside base_cols, so dead sensors don't spawn dead
        copies. Pass all 21 sensors to reproduce the old behaviour.
    drop_dead : bool
        Remove columns whose value never changes (a "dead" feature -- it carries
        no information, so the model cannot learn anything from it). Only ever
        use this on the TRAINING frame; for the test frame pass keep_cols.
    keep_cols : list of str, optional
        Return exactly these columns, in this order. Use for the test set.

    Returns
    -------
    (DataFrame, list of str)
        The frame with the new columns added, and the ordered list of model
        inputs.
    """
    groups = list(dict.fromkeys(["base"] + list(groups)))
    unknown = set(groups) - set(GROUP_ORDER)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")

    if sensor_cols is None:
        sensor_cols = [c for c in base_cols if c.startswith("sensor_")]

    # rolling / lag / trend all assume the rows are in time order per engine
    df = df.sort_values(["unit_id", "time_cycles"]).reset_index(drop=True)

    new, names = [], {"base": list(base_cols)}

    if "rolling" in groups:
        cols = _rolling(df, sensor_cols, list(rolling_windows))
        new += cols
        names["rolling"] = [c.name for c in cols]
    if "lag" in groups:
        cols = _lag(df, sensor_cols, list(lags))
        new += cols
        names["lag"] = [c.name for c in cols]
    if "trend" in groups:
        cols = _trend(df, sensor_cols, trend_window)
        new += cols
        names["trend"] = [c.name for c in cols]
    if "agg" in groups:
        cols = _agg(df, sensor_cols)
        new += cols
        names["agg"] = [c.name for c in cols]
    if "cycle" in groups:
        cols = _cycle(df)
        new += cols
        names["cycle"] = [c.name for c in cols]
    if "opcond" in groups:
        # op_condition already exists as a column; it is normally excluded as
        # metadata. Only C6_expose puts it back, to match the exposé run.
        names["opcond"] = ["op_condition"]

    if new:
        df = pd.concat([df] + new, axis=1)

    # rolling and lag leave gaps at the start of each engine's history.
    # ffill copies the nearest earlier value forward; anything still missing
    # (the very first rows) becomes 0.
    df = df.ffill().fillna(0)

    if keep_cols is not None:
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise ValueError(f"columns missing from test frame: {missing[:5]}")
        return df, list(keep_cols)

    feature_cols = [c for g in GROUP_ORDER if g in names for c in names[g]]

    if drop_dead:
        var = df[feature_cols].var()
        feature_cols = [c for c in feature_cols if var[c] > 1e-12]

    return df, feature_cols


def group_sizes(feature_cols):
    """Count how many surviving columns came from each group (for the paper)."""
    counts = {g: 0 for g in GROUP_ORDER}
    for c in feature_cols:
        if "_rolling_" in c:
            counts["rolling"] += 1
        elif "_lag_" in c:
            counts["lag"] += 1
        elif "_trend_" in c:
            counts["trend"] += 1
        elif c.startswith("sensors_"):
            counts["agg"] += 1
        elif c in ("cycle", "cycle_log", "cycle_sqrt"):
            counts["cycle"] += 1
        elif c == "op_condition":
            counts["opcond"] += 1
        else:
            counts["base"] += 1
    return counts


def count_dead(df, feature_cols):
    """How many of these columns never change? Reported in the paper."""
    var = df[feature_cols].var()
    return int((var <= 1e-12).sum())
