"""preprocessing pipeline for the nasa c-mapss dataset.

four steps run in sequence: rul labelling → operating condition clustering →
constant feature removal → per-cluster normalisation. the order matters because
clustering needs raw sensor values, and normalisation needs the cluster labels.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from src.config import TARGET_COL


def add_rul_column(df, rul_cap=125):
    """calculate rul for training data and apply the piecewise-linear cap.

    rul = max_cycles_for_this_engine - current_cycle. at the last cycle before
    failure, rul = 0. values above the cap are clipped because early-life sensor
    readings are essentially flat — there's no useful degradation signal to learn
    from, and training on high arbitrary rul values just adds noise.

    args:
        df: dataframe with 'unit_id' and 'time_cycles' columns
        rul_cap: maximum rul value to clip to (125 for fd001/fd003, 150 for fd002/fd004)

    returns:
        dataframe with added 'RUL' column
    """
    max_cycles = df.groupby('unit_id')['time_cycles'].max()
    df = df.copy()
    df['max_cycle'] = df['unit_id'].map(max_cycles)
    df[TARGET_COL] = (df['max_cycle'] - df['time_cycles']).clip(upper=rul_cap)
    df.drop('max_cycle', axis=1, inplace=True)
    # result: rul column added. the distribution will show a flat plateau at the
    # cap value (healthy phase) followed by a linear decline to zero at failure.
    return df


def assign_operating_conditions(df, kmeans=None, n_clusters=6):
    """assign operating condition cluster labels using kmeans on the settings columns.

    fd002 and fd004 have 6 distinct operating conditions (altitude/throttle combos).
    a jump between conditions causes all sensors to shift simultaneously — without
    labelling these, the model would confuse condition switches with degradation.
    we cluster the three settings columns to identify each condition, then normalise
    sensors within each cluster so the model only sees degradation-driven variation.

    args:
        df: dataframe with setting_1, setting_2, setting_3 columns
        kmeans: pre-fitted kmeans model for test data (pass none to fit on training data)
        n_clusters: number of clusters — should match the dataset's operating conditions

    returns:
        dataframe with added 'op_condition' column, and the fitted kmeans model
    """
    settings = df[['setting_1', 'setting_2', 'setting_3']]
    if kmeans is None:
        # fit on training data only
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(settings)
    df = df.copy()
    df['op_condition'] = kmeans.predict(settings)
    # result: each row now has a cluster label (0 to n_clusters-1).
    # for fd001/fd003 all rows land in one cluster since the settings are constant —
    # the normaliser will then behave like a standard global minmax scaler, which is fine.
    return df, kmeans


def normalize_data(df, feature_cols, scalers=None):
    """normalise feature columns per operating condition cluster.

    each cluster gets its own minmax scaler so that sensor baselines, which
    differ between conditions, dont get mixed together. a global scaler would
    map the same physical sensor reading to different normalised values depending
    on which condition it came from, which would look like a degradation signal
    to the model even when the engine is perfectly healthy.

    args:
        df: dataframe with 'op_condition' column and feature columns
        feature_cols: list of column names to normalise
        scalers: dict of {cluster_id: fitted_scaler} for test data — pass none to fit on training

    returns:
        normalised dataframe, dict of fitted scalers
    """
    df = df.copy()
    df[feature_cols] = df[feature_cols].astype(float)

    if scalers is None:
        # fit one scaler per cluster on training data
        scalers = {}
        for condition in df['op_condition'].unique():
            mask = df['op_condition'] == condition
            scaler = MinMaxScaler()
            df.loc[mask, feature_cols] = scaler.fit_transform(df.loc[mask, feature_cols])
            scalers[condition] = scaler
        # result: each operating condition has its own minmax scaler fitted on training data only
    else:
        # apply pre-fitted scalers to test data — no fitting here to avoid leakage
        for condition, scaler in scalers.items():
            mask = df['op_condition'] == condition
            if mask.any():
                df.loc[mask, feature_cols] = scaler.transform(df.loc[mask, feature_cols])
        # result: test data normalised on the same scale as training, no data leakage

    return df, scalers


def remove_constant_features(df, feature_cols):
    """remove features with zero variance across the dataset.

    constant features are useless for prediction and would cause division-by-zero
    inside scalers. on fd001 the removed sensors (1, 5, 6, 10, 16, 18, 19) match
    exactly what the eda correlation heatmap flagged as uninformative. fd002/fd004
    retain more sensors because the six conditions introduce cross-condition variance
    even for sensors that appear constant within any single condition.

    args:
        df: dataframe with feature columns
        feature_cols: list of column names to check

    returns:
        list of non-constant feature column names
    """
    variances = df[feature_cols].var()
    non_constant_cols = variances[variances > 0].index.tolist()

    removed = set(feature_cols) - set(non_constant_cols)
    if removed:
        print(f"  removed {len(removed)} constant features: {sorted(removed)}")
        # finding: on fd001 this removes 7 sensors. on fd002/fd004 fewer sensors are removed
        # because operating condition switches create variance even in otherwise flat sensors.

    return non_constant_cols


def prepare_train_data(df, feature_cols, rul_cap=125, n_clusters=6):
    """full preprocessing pipeline for training data.

    runs all four steps in the correct order:
    1. add rul column (clipped at rul_cap)
    2. assign operating condition clusters (fit kmeans here)
    3. remove constant features
    4. normalise features per cluster (fit scalers here)

    args:
        df: raw training dataframe from loader
        feature_cols: list of feature column names
        rul_cap: maximum rul value to clip to
        n_clusters: number of kmeans clusters

    returns:
        preprocessed dataframe, scalers dict, fitted kmeans, final feature column list
    """
    df = add_rul_column(df, rul_cap=rul_cap)
    df, kmeans = assign_operating_conditions(df, n_clusters=n_clusters)
    feature_cols = remove_constant_features(df, feature_cols)
    df, scalers = normalize_data(df, feature_cols)
    # result: training data is fully preprocessed. rul is labelled and capped,
    # conditions are clustered, uninformative sensors are dropped, and all remaining
    # features are minmax-normalised within each operating condition.
    return df, scalers, kmeans, feature_cols


def prepare_test_data(df, scalers, kmeans, feature_cols, rul_values):
    """full preprocessing pipeline for test data.

    uses the kmeans model and scalers fitted on training data — nothing is
    re-fitted here. the true rul labels are attached before normalisation so
    the sequences module can pick them up as targets later.

    args:
        df: raw test dataframe from loader
        scalers: dict of {cluster_id: scaler} from training
        kmeans: fitted kmeans model from training
        feature_cols: feature columns to use (same list as training, post constant-removal)
        rul_values: dataframe with one 'RUL' value per test engine

    returns:
        preprocessed test dataframe with rul column attached
    """
    df = df.copy()

    # the rul file has one value per engine (the rul at the last observed cycle).
    # we merge it onto every row of each engine so the sequences module can access
    # it when building the test window — the value is the same for all rows of a given engine.
    last_cycles = df.groupby('unit_id').tail(1).reset_index(drop=True)
    last_cycles[TARGET_COL] = rul_values['RUL'].values
    df = df.merge(last_cycles[['unit_id', TARGET_COL]], on='unit_id', how='left')

    df, _ = assign_operating_conditions(df, kmeans)
    df, _ = normalize_data(df, feature_cols, scalers)
    # result: test data processed using same cluster model and scalers as training.
    # no information from the test set was used during fitting, so there's no leakage.
    return df
