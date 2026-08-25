"""Verify the full pipeline from raw data to LSTM-ready sequences."""

import numpy as np
from src.config import SELECTED_DATASET, FEATURE_COLS, SENSOR_COLS, SEQUENCE_LENGTH, RUL_CAPS, N_CLUSTERS
from src.data.loader import load_train_data, load_test_data, load_rul_labels
from src.data.preprocessor import prepare_train_data, prepare_test_data
from src.features.engineer import create_all_features, get_feature_columns
from src.data.sequences import create_train_sequences, create_test_sequences


def main(dataset=None):
    if dataset is None:
        dataset = SELECTED_DATASET

    rul_cap = RUL_CAPS[dataset]
    n_clusters = N_CLUSTERS[dataset]

    print("=" * 60)
    print(f"SEQUENCE CREATION TEST — {dataset}")
    print("=" * 60)

    # --- Training data ---
    print("\n1. Loading and preprocessing training data...")
    train_df = load_train_data(dataset)
    train_df, scalers, kmeans, feature_cols = prepare_train_data(
        train_df, FEATURE_COLS, rul_cap=rul_cap, n_clusters=n_clusters
    )
    print(f"   Preprocessed shape: {train_df.shape}")

    print("\n2. Feature engineering...")
    train_df = create_all_features(train_df, feature_cols, SENSOR_COLS)
    all_feature_cols = get_feature_columns(train_df)
    print(f"   Features after engineering: {len(all_feature_cols)}")

    print("\n3. Creating train sequences...")
    X_train, y_train = create_train_sequences(train_df, all_feature_cols, SEQUENCE_LENGTH)
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")

    # --- Test data ---
    print("\n4. Loading and preprocessing test data...")
    test_df = load_test_data(dataset)
    rul_labels = load_rul_labels(dataset)
    test_df = prepare_test_data(test_df, scalers, kmeans, feature_cols, rul_labels)
    test_df = create_all_features(test_df, feature_cols, SENSOR_COLS)
    print(f"   Preprocessed shape: {test_df.shape}")

    print("\n5. Creating test sequences...")
    X_test, y_test = create_test_sequences(test_df, all_feature_cols, SEQUENCE_LENGTH)
    print(f"   X_test shape:  {X_test.shape}")
    print(f"   y_test shape:  {y_test.shape}")

    # --- Sanity checks ---
    print("\n6. Sanity checks...")
    assert not np.isnan(X_train).any(), "NaN in X_train!"
    assert not np.isnan(y_train).any(), "NaN in y_train!"
    assert not np.isnan(X_test).any(),  "NaN in X_test!"
    assert not np.isnan(y_test).any(),  "NaN in y_test!"
    print("   NaN check:     OK")

    assert X_train.shape[1] == SEQUENCE_LENGTH, "Wrong sequence length in X_train!"
    assert X_test.shape[1] == SEQUENCE_LENGTH,  "Wrong sequence length in X_test!"
    print("   Sequence length check: OK")

    print(f"\n   y_train range: [{y_train.min():.0f}, {y_train.max():.0f}]")
    print(f"   y_test  range: [{y_test.min():.0f}, {y_test.max():.0f}]")

    print("\n" + "=" * 60)
    print("Sequence creation OK")
    print("=" * 60)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = main()
