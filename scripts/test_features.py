"""Test feature engineering pipeline."""

from src.data.loader import load_train_data
from src.data.preprocessor import add_rul_column
from src.features.engineer import create_all_features, get_feature_columns
from src.config import SELECTED_DATASET, FEATURE_COLS, SENSOR_COLS


def main():
    """Test feature engineering on training data."""
    print("=" * 60)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    df = load_train_data(SELECTED_DATASET)
    df = add_rul_column(df)
    print(f"   Original shape: {df.shape}")
    print(f"   Original columns: {len(df.columns)}")

    # Create features
    print("\n2. Creating features...")
    df_features = create_all_features(
        df,
        feature_cols=FEATURE_COLS,
        sensor_cols=SENSOR_COLS,
        rolling_windows=[5, 10, 20],
        lags=[1, 3],
        trend_window=10
    )

    print(f"\n   New shape: {df_features.shape}")
    print(f"   New columns: {len(df_features.columns)}")

    # Show example
    print("\n3. Example for one engine (unit_id=1):")
    sample = df_features[df_features['unit_id'] == 1].head(15)
    print(sample[['unit_id', 'time_cycles', 'RUL', 'cycle_log',
                  'sensors_mean', 'sensors_std']].to_string())

    # Show all feature columns
    print("\n4. All engineered feature columns:")
    feature_cols = get_feature_columns(df_features)
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Sample features: {feature_cols[:10]}")

    print("\n" + "=" * 60)
    print("✓ Feature engineering test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
