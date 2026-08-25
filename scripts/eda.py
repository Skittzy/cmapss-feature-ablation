"""standalone exploratory data analysis script.

runs the same eda that is embedded in main.py but as a standalone script,
useful for quick inspection without running the full training pipeline.
all analysis is on fd001 (selected_dataset in config.py) — the simplest
sub-dataset with the cleanest degradation signals.

outputs saved to results/:
  eda_lifecycle_distribution.png
  eda_rul_distribution.png
  eda_sensor_degradation.png
  eda_kmeans_clusters.png
  eda_correlation_heatmap.png
"""

import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import load_train_data
from src.data.preprocessor import add_rul_column
from src.config import SELECTED_DATASET, SENSOR_COLS, FEATURE_COLS

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_prepare_data():
    """load training data and attach the rul column for correlation analysis."""
    df = load_train_data(SELECTED_DATASET)
    df = add_rul_column(df)
    return df


def basic_info(df):
    """print a summary of the raw dataset structure."""
    print("=" * 60)
    print("BASIC DATASET INFO")
    print("=" * 60)
    print(f"\nDataset: {SELECTED_DATASET}")
    print(f"Total rows: {len(df):,}")
    print(f"Number of engines: {df['unit_id'].nunique()}")
    print(f"Number of features: {len(FEATURE_COLS)}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nShape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())


def analyze_engine_lifecycles(df):
    """analyse and plot the distribution of engine lifecycle lengths.

    this tells us how much variance exists in the rul values we'll be predicting.
    high variance means the model must cover a wide prediction range.
    """
    print("\n" + "=" * 60)
    print("ENGINE LIFECYCLE ANALYSIS")
    print("=" * 60)

    lifecycle_lengths = df.groupby('unit_id')['time_cycles'].max()

    print(f"\nLifecycle statistics (in cycles):")
    print(lifecycle_lengths.describe())
    # finding: the range of lifecycle lengths (roughly 130–360 cycles on fd001) directly
    # motivates the rul cap. engines at the same early cycle could have very different
    # true ruls, so we group them all under the cap value rather than predicting noise.

    plt.figure(figsize=(10, 5))
    plt.hist(lifecycle_lengths, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Lifecycle Length (cycles)')
    plt.ylabel('Number of Engines')
    plt.title(f'Distribution of Engine Lifecycles - {SELECTED_DATASET}')
    plt.axvline(lifecycle_lengths.mean(), color='red', linestyle='--',
                label=f'Mean: {lifecycle_lengths.mean():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/eda_lifecycle_distribution.png', dpi=100)
    print(f"\n✓ Saved: results/eda_lifecycle_distribution.png")
    plt.close()


def analyze_rul_distribution(df):
    """plot the distribution of rul values after applying the cap.

    the flat plateau on the left is the cap in action — all early-life observations
    are assigned the maximum rul value. the declining tail is the degradation region
    where the model actually has a signal to learn from.
    """
    print("\n" + "=" * 60)
    print("RUL DISTRIBUTION")
    print("=" * 60)

    print(f"\nRUL statistics:")
    print(df['RUL'].describe())

    plt.figure(figsize=(10, 5))
    plt.hist(df['RUL'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('RUL (cycles)')
    plt.ylabel('Frequency')
    plt.title(f'RUL Distribution - {SELECTED_DATASET}')
    plt.tight_layout()
    plt.savefig(f'results/eda_rul_distribution.png', dpi=100)
    print(f"\n✓ Saved: results/eda_rul_distribution.png")
    plt.close()


def check_constant_features(df):
    """identify zero-variance and low-variance features.

    zero-variance features will be removed in preprocessing because they carry
    no information and would cause division-by-zero in the scalers.
    low-variance features might still be worth keeping — the threshold is hard zero.
    """
    print("\n" + "=" * 60)
    print("FEATURE VARIANCE ANALYSIS")
    print("=" * 60)

    variances = df[FEATURE_COLS].var().sort_values()

    print("\nFeature variances (lowest to highest):")
    print(variances)

    constant_features = variances[variances == 0].index.tolist()
    low_variance = variances[variances < 0.01].index.tolist()

    if constant_features:
        print(f"\n⚠ Constant features (variance = 0): {constant_features}")
        # finding: on fd001 the constant sensors (1, 5, 6, 10, 16, 18, 19) match exactly
        # what the correlation heatmap shows as uninformative — consistent signal
    else:
        print(f"\n✓ No constant features found")

    if low_variance:
        print(f"\n⚠ Low variance features (< 0.01): {low_variance}")

    return variances


def plot_sensor_degradation(df, sample_engines=3):
    """plot sensor readings over time for a sample of engines.

    gives a visual intuition for which sensors change monotonically toward failure
    and which stay flat. flat sensors are candidates for removal.
    """
    print("\n" + "=" * 60)
    print("SENSOR DEGRADATION PATTERNS")
    print("=" * 60)

    sample_units = df['unit_id'].unique()[:sample_engines]
    key_sensors = SENSOR_COLS[:6]  # first 6 sensors as a representative sample

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, sensor in enumerate(key_sensors):
        ax = axes[idx]
        for unit in sample_units:
            unit_data = df[df['unit_id'] == unit]
            ax.plot(unit_data['time_cycles'], unit_data[sensor],
                   label=f'Engine {unit}', alpha=0.7)

        ax.set_xlabel('Time Cycles')
        ax.set_ylabel(sensor)
        ax.set_title(f'{sensor} Degradation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/eda_sensor_degradation.png', dpi=100)
    print(f"\n✓ Saved: results/eda_sensor_degradation.png")
    plt.close()


def plot_kmeans_clusters(df, n_clusters=6):
    """visualise kmeans operating condition clusters in 3d settings-space.

    on fd001 (single condition) all points collapse into one region, confirming
    the pipeline will behave like a global normaliser for this dataset.
    on fd002/fd004 this same plot would show 6 clearly separated point clouds.
    """
    from sklearn.cluster import KMeans

    settings = df[['setting_1', 'setting_2', 'setting_3']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(settings)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['setting_1'], df['setting_2'], df['setting_3'],
               c=labels, cmap='tab10', alpha=0.3, s=5)
    ax.scatter(*kmeans.cluster_centers_.T, c='black', s=100, marker='X', label='Centroids')
    ax.set_xlabel('setting_1')
    ax.set_ylabel('setting_2')
    ax.set_zlabel('setting_3')
    ax.set_title(f'KMeans Clustering (k={n_clusters})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/eda_kmeans_clusters.png', dpi=100)
    print(f"\n✓ Saved: results/eda_kmeans_clusters.png")
    plt.close()


def plot_correlation_heatmap(df):
    """plot sensor-to-rul and sensor-to-sensor correlation matrix.

    strong negative correlation with rul means the sensor rises as the engine
    ages. near-zero correlation means the sensor is probably not useful for prediction.
    finding: sensors 2, 3, 4, 7, 11, 12, 15 are the primary degradation indicators on fd001.
    """
    plt.figure(figsize=(14, 10))
    corr = df[SENSOR_COLS + ['RUL']].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                linewidths=0.5)
    plt.title(f'Sensor Correlation Heatmap (incl. RUL) — {SELECTED_DATASET}')
    plt.tight_layout()
    plt.savefig('results/eda_correlation_heatmap.png', dpi=100)
    plt.close()
    print(f"✓ Saved: results/eda_correlation_heatmap.png")


def main():
    """run complete eda pipeline and print a summary of findings."""
    print("\n" + "=" * 60)
    print("STARTING EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = load_and_prepare_data()

    basic_info(df)
    analyze_engine_lifecycles(df)
    analyze_rul_distribution(df)
    check_constant_features(df)
    plot_sensor_degradation(df)
    plot_kmeans_clusters(df)
    plot_correlation_heatmap(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETE ✓")
    print("=" * 60)
    # summary of key findings:
    # - lifecycle variance (130–360 cycles) justifies the rul cap at 125
    # - sensors 2, 3, 4, 7, 11, 12, 15 are the strongest predictors
    # - sensors 1, 5, 6, 10, 16, 18, 19 are constant and will be removed
    # - kmeans correctly identifies one cluster on fd001 (single operating condition)


if __name__ == "__main__":
    main()
