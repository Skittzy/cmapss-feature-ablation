"""sequence creation for lstm — converts per-row dataframes into sliding windows.

the lstm expects input of shape (samples, timesteps, features). we use two
different strategies for training and test because the labeling structure differs:
  - training: many overlapping windows per engine, label at window end
  - test: one window per engine (the last 30 cycles), label from the rul file
"""

import numpy as np


def create_train_sequences(df, feature_cols, sequence_length):
    """create sliding window sequences from training data.

    for each engine, slides a window of `sequence_length` cycles across its
    full history. each window becomes one training sample; the rul at the
    last cycle of the window is the label. this gives us many samples per engine
    and preserves the temporal ordering the lstm needs.

    args:
        df: dataframe with engineered features, must contain 'unit_id',
            'time_cycles', and 'RUL' columns.
        feature_cols: ordered list of feature column names to include in X.
        sequence_length: number of timesteps per sequence.

    returns:
        X: np.ndarray of shape (N, sequence_length, len(feature_cols))
        y: np.ndarray of shape (N,)
    """
    X_list, y_list = [], []

    df = df.sort_values(['unit_id', 'time_cycles'])

    for _, engine_df in df.groupby('unit_id'):
        features = engine_df[feature_cols].values
        labels = engine_df['RUL'].values
        n_cycles = len(engine_df)

        # skip engines that are shorter than the window — cant build even one sequence
        if n_cycles < sequence_length:
            continue

        # slide the window one cycle at a time across the engine's full history.
        # a 200-cycle engine contributes 171 sequences (200 - 30 + 1), giving the
        # model plenty of exposure to different points in the degradation trajectory.
        for start in range(n_cycles - sequence_length + 1):
            end = start + sequence_length
            X_list.append(features[start:end])
            y_list.append(labels[end - 1])  # label is rul at the window's last timestep

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def create_test_sequences(df, feature_cols, sequence_length):
    """create one sequence per engine from test data.

    the ground-truth rul label is defined at the last observed cycle, so we
    only need the final `sequence_length` cycles for each engine. engines with
    fewer recorded cycles than the window are zero-padded at the front rather
    than the back — padding at the front means the real data is always at the
    end of the sequence, closest to the prediction point.

    args:
        df: dataframe with engineered features, must contain 'unit_id',
            'time_cycles', and 'RUL' columns.
        feature_cols: ordered list of feature column names to include in X.
        sequence_length: number of timesteps per sequence.

    returns:
        X: np.ndarray of shape (num_engines, sequence_length, len(feature_cols))
        y: np.ndarray of shape (num_engines,)
    """
    X_list, y_list = [], []

    df = df.sort_values(['unit_id', 'time_cycles'])

    for _, engine_df in df.groupby('unit_id'):
        features = engine_df[feature_cols].values
        n_cycles = len(engine_df)

        if n_cycles >= sequence_length:
            # take the last 30 cycles directly
            window = features[-sequence_length:]
        else:
            # engine has fewer cycles than the window length — pad with zeros at the front.
            # front-padding keeps the real observations at the tail, which is where the
            # degradation signal is most relevant for the final rul estimate.
            pad = np.zeros((sequence_length - n_cycles, len(feature_cols)), dtype=np.float32)
            window = np.vstack([pad, features])

        X_list.append(window)
        # all rows in the test df for this engine have the same rul value (set in preprocessing),
        # so taking the last row's value is equivalent to taking any row
        y_list.append(engine_df['RUL'].iloc[-1])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
