"""feature engineering for rul prediction.

raw per-cycle sensor values give the model a snapshot of system state but no
temporal context — it cant tell whether a sensor reading is stable, trending, or
accelerating toward failure. this module adds that context by computing rolling
statistics, lag values, rate-of-change trends, and cross-sensor aggregations.

all rolling/lag/trend operations are grouped by unit_id so they never bleed
across engine boundaries. nan values from the first few cycles of each engine
(before the window fills) are handled with forward-fill then zero at the end.
"""

import numpy as np
import pandas as pd


def add_rolling_statistics(df, feature_cols, windows=[5, 10, 20]):
    """add rolling mean and std for each sensor column.

    rolling mean smooths out measurement noise and reveals the underlying trend.
    rolling std captures how erratic a sensor is becoming — increasing variability
    is often an early sign of degradation, especially on fd002/fd004.

    using two window sizes (5 and 10 cycles) lets the model distinguish short-term
    fluctuations from longer-term drift patterns.

    args:
        df: dataframe sorted by unit_id, time_cycles
        feature_cols: columns to compute rolling stats for
        windows: list of window sizes in cycles

    returns:
        dataframe with added rolling features
    """
    new_features = []

    for col in feature_cols:
        for window in windows:
            rolling_mean = (
                df.groupby('unit_id')[col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            new_features.append(rolling_mean.rename(f'{col}_rolling_mean_{window}'))

            rolling_std = (
                df.groupby('unit_id')[col]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
            new_features.append(rolling_std.rename(f'{col}_rolling_std_{window}'))

    # min_periods=1 ensures we get values from cycle 1 rather than nans for the
    # first (window-1) rows of each engine. the std for a single observation is nan
    # but that gets handled by the fillna step at the end of create_all_features.
    return pd.concat([df] + new_features, axis=1)


def add_lag_features(df, feature_cols, lags=[1, 3, 5]):
    """add lagged values of sensor columns.

    lag features give the lstm explicit access to recent history. without them, the
    lstm would have to reconstruct past values from its hidden state alone, which
    is harder to learn. with them, the model can directly compare the current reading
    to where the sensor was 1 or 3 cycles ago — a simple but strong degradation signal.

    args:
        df: dataframe with sensor data
        feature_cols: columns to create lags for
        lags: list of lag steps (in cycles)

    returns:
        dataframe with lagged features
    """
    new_features = []

    for col in feature_cols:
        for lag in lags:
            lagged = df.groupby('unit_id')[col].shift(lag)
            new_features.append(lagged.rename(f'{col}_lag_{lag}'))

    return pd.concat([df] + new_features, axis=1)


def add_degradation_trends(df, feature_cols, window=10):
    """add rate-of-change (trend) features over a fixed lookback window.

    captures how fast each sensor is moving. a sensor that jumped 0.4 units over
    10 cycles is a much stronger degradation signal than one that moved 0.02 units.
    the discrete derivative (current - value_10_cycles_ago) is simple but effective
    and less noisy than a single-cycle difference.

    args:
        df: dataframe with sensor data
        feature_cols: columns to compute trends for
        window: lookback window size in cycles

    returns:
        dataframe with trend features
    """
    new_features = []

    for col in feature_cols:
        trend = (
            df.groupby('unit_id')[col]
            .transform(lambda x: x - x.shift(window))
        )
        new_features.append(trend.rename(f'{col}_trend_{window}'))

    return pd.concat([df] + new_features, axis=1)


def add_cycle_features(df):
    """add transformed cycle-number features.

    the raw cycle counter is linear, but its relationship to degradation is not —
    a step from cycle 10 to 20 means something very different than a step from 200
    to 210. log and sqrt compressions give the model three different "views" of time:
    linear (cycle), compressed early life (log), and intermediate (sqrt).

    args:
        df: dataframe with 'time_cycles' column

    returns:
        dataframe with cycle, cycle_log, and cycle_sqrt columns
    """
    df = df.copy()
    df['cycle']      = df['time_cycles']
    df['cycle_log']  = np.log1p(df['time_cycles'])   # log1p avoids log(0) at cycle 0
    df['cycle_sqrt'] = np.sqrt(df['time_cycles'])
    return df


def add_sensor_aggregations(df, sensor_cols):
    """add aggregate statistics across all sensors at each timestep.

    cross-sensor features capture overall system state rather than individual
    sensor behaviour. sensors_std is particularly useful — as an engine degrades,
    individual sensors become more erratic at different rates, so the spread
    across all sensors tends to increase before failure.

    args:
        df: dataframe with sensor columns
        sensor_cols: list of sensor column names

    returns:
        dataframe with aggregated features added
    """
    df = df.copy()
    df['sensors_mean']  = df[sensor_cols].mean(axis=1)
    df['sensors_std']   = df[sensor_cols].std(axis=1)
    df['sensors_min']   = df[sensor_cols].min(axis=1)
    df['sensors_max']   = df[sensor_cols].max(axis=1)
    df['sensors_range'] = df['sensors_max'] - df['sensors_min']
    return df


def create_all_features(df, feature_cols, sensor_cols,
                        rolling_windows=[5, 10],
                        lags=[1, 3],
                        trend_window=10):
    """apply all feature engineering steps in sequence.

    order: cycle transforms → sensor aggregations → rolling stats → lags → trends.
    cycle and aggregation features are computed first because they're simple row-wise
    operations. rolling, lag, and trend features depend on the sort order, so we
    sort by unit_id and time_cycles before starting.

    total features after engineering: ~100+ per timestep (up from ~17 raw columns).
    the exact count varies by dataset because constant-feature removal upstream
    changes how many sensor columns we start with.

    args:
        df: preprocessed dataframe (rul labelled, normalised, constant features removed)
        feature_cols: list of columns to apply rolling/lag/trend to
        sensor_cols: list of sensor columns for aggregations
        rolling_windows: window sizes for rolling statistics
        lags: lag steps for lag features
        trend_window: lookback window for trend (rate of change) features

    returns:
        dataframe with all engineered features
    """
    print("creating features...")

    # sort is critical — rolling and lag operations assume chronological order within each engine
    df = df.sort_values(['unit_id', 'time_cycles']).reset_index(drop=True)

    print("  - cycle features")
    df = add_cycle_features(df)

    print("  - sensor aggregations")
    df = add_sensor_aggregations(df, sensor_cols)

    print(f"  - rolling statistics (windows: {rolling_windows})")
    df = add_rolling_statistics(df, sensor_cols, windows=rolling_windows)

    print(f"  - lag features (lags: {lags})")
    df = add_lag_features(df, sensor_cols, lags=lags)

    print(f"  - degradation trends (window: {trend_window})")
    df = add_degradation_trends(df, sensor_cols, window=trend_window)

    # fill nans from rolling/lag at the start of each engine's history.
    # forward-fill first (propagates the earliest valid value backward into the nans),
    # then zero for any remaining nans at the very beginning of the series.
    df = df.ffill().fillna(0)

    print("✓ feature engineering complete")
    return df


def get_feature_columns(df):
    """return the list of engineered feature columns, excluding metadata.

    args:
        df: dataframe after feature engineering

    returns:
        list of column names that should be used as model inputs
    """
    exclude = ['unit_id', 'time_cycles', 'RUL', 'max_cycle', 'op_condition']
    return [col for col in df.columns if col not in exclude]
