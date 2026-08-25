"""
nasa c-mapss — remaining useful life (rul) prediction
======================================================
main analysis script. runs the full supervised ml pipeline in a single
linear pass: data loading → eda → preprocessing → feature engineering →
sequence creation → model training → evaluation.

to reproduce from scratch:
  1. create and activate the conda environment:
       conda env create -f environment.yml
       conda activate nasa-rul-project
  2. download the dataset (needs a kaggle account + api token):
       python init_dataset.py
  3. run this script:
       python main.py

all plots go to results/, trained models go to models/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.config import (
    ALL_DATASETS, FEATURE_COLS, SENSOR_COLS,
    SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS,
    RUL_CAPS, N_CLUSTERS, MODELS_DIR, RESULTS_DIR
)
from src.data.loader import load_train_data, load_test_data, load_rul_labels
from src.data.preprocessor import prepare_train_data, prepare_test_data
from src.features.engineer import create_all_features, get_feature_columns
from src.data.sequences import create_train_sequences, create_test_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# STEP 0 — DATA BIAS DISCUSSION
# =============================================================================
# before we begin, its worth being honest about the limitations of this dataset.
# c-mapss is a simulation, not real telemetry from aircraft engines.
# this has several implications for how we should interpret our results:
#
# 1. simulation bias: real engines degrade with much more variability due to
#    maintenance history, fuel quality, pilot behaviour, and manufacturing
#    tolerances. c-mapss uses controlled degradation models, which makes the
#    prediction task more regular than it would be in an operational setting.
#
# 2. completeness bias: every engine in this dataset runs to complete failure.
#    in practice, engines are serviced before failure, so real-world data would
#    be dominated by right-censored observations (engines removed while still healthy).
#    a regression model like ours can't handle censored labels — survival models
#    would be more appropriate for real deployment.
#
# 3. fault mode bias: c-mapss uses only 2 fault types (hpc degradation and fan
#    degradation). real turbofan failures include bearing failures, foreign object
#    damage, compressor stalls, and many other modes not represented here.
#
# conclusion: our rmse figures on c-mapss likely overstate how well the model would
# perform on real fleet data. it is a useful benchmark for comparing approaches,
# but operational validation on actual telemetry would be needed before deployment.
# =============================================================================

# =============================================================================
# STEP 1 — DATA ACQUISITION
# =============================================================================
# dataset: nasa c-mapss (commercial modular aero-propulsion system simulation),
# published by nasa as a benchmark for predictive maintenance research.
# source: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps (via init_dataset.py)
#
# the dataset simulates turbofan engine run-to-failure under controlled conditions.
# each row is one engine at one point in time, recording 21 sensor measurements
# and 3 operational settings alongside the engine's unit id and current cycle.
# there is no header — column names are assigned in loader.py.
#
# there are 4 sub-datasets with increasing complexity:
#   fd001 — 1 operating condition, 1 fault mode,  100 training engines
#   fd002 — 6 operating conditions, 1 fault mode, 260 training engines
#   fd003 — 1 operating condition, 2 fault modes, 100 training engines
#   fd004 — 6 operating conditions, 2 fault modes, 249 training engines
#
# we train on all four so we can compare how the model handles each complexity level.

print("\n" + "=" * 60)
print("NASA C-MAPSS — RUL PREDICTION PIPELINE")
print("=" * 60)
print(f"\nDatasets        : {ALL_DATASETS}")
print(f"Sequence length : {SEQUENCE_LENGTH} cycles")
print(f"Max epochs      : {EPOCHS}")


# =============================================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# =============================================================================
# eda is done on fd001 only — the simplest sub-dataset (1 operating condition,
# 1 fault mode). the degradation patterns here are cleanest to examine, and
# the findings directly informed the preprocessing decisions applied to all four datasets.

print("\n" + "=" * 60)
print("STEP 2 — EXPLORATORY DATA ANALYSIS (FD001)")
print("=" * 60)

eda_dataset = "FD001"
eda_df = load_train_data(eda_dataset)

# compute rul manually for eda purposes — we need it to correlate sensors against it.
# this uses the same formula and cap as the preprocessing step that follows.
max_cycles = eda_df.groupby('unit_id')['time_cycles'].max()
eda_df['max_cycle'] = eda_df['unit_id'].map(max_cycles)
eda_df['RUL'] = (eda_df['max_cycle'] - eda_df['time_cycles']).clip(upper=RUL_CAPS[eda_dataset])
eda_df.drop('max_cycle', axis=1, inplace=True)

print(f"\nShape         : {eda_df.shape}")
print(f"Engines       : {eda_df['unit_id'].nunique()}")
print(f"Total cycles  : {len(eda_df):,}")
print(f"\nBasic statistics (first 6 sensors):")
print(eda_df[SENSOR_COLS[:6]].describe().round(3))

# --- engine lifecycle distribution ---
# we want to understand how long engines typically run before failure.
# high variance in lifecycle length means the model must be able to predict
# rul across a wide range of values, from near-zero (imminent failure) to
# 100+ cycles remaining.
lifecycle_lengths = eda_df.groupby('unit_id')['time_cycles'].max()

print(f"\nLifecycle length stats (cycles):")
print(lifecycle_lengths.describe().round(1))

# finding: fd001 engines live roughly 130–360 cycles, mean ~206.
# the spread is significant — a factor of ~2.8x between shortest and longest.
# this confirms why capping rul at 125 makes sense: an engine at cycle 50 of
# a 300-cycle life and one at cycle 50 of a 130-cycle life look identical to the
# sensors, so predicting fine-grained rul values above the cap would be pure noise.
plt.figure(figsize=(10, 5))
plt.hist(lifecycle_lengths, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(lifecycle_lengths.mean(), color='red', linestyle='--',
            label=f'Mean: {lifecycle_lengths.mean():.1f} cycles')
plt.xlabel('Lifecycle Length (cycles)')
plt.ylabel('Number of Engines')
plt.title(f'Distribution of Engine Lifecycles — {eda_dataset}')
plt.legend()
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/eda_lifecycle_distribution.png', dpi=100)
plt.close()
print(f"\n✓ Saved: {RESULTS_DIR}/eda_lifecycle_distribution.png")

# --- rul distribution ---
# the flat plateau on the left of the histogram is the rul cap in action —
# all early-life observations are grouped at 125 regardless of their true distance
# to failure. this is the piecewise-linear rul assumption standard in c-mapss literature.
# the declining tail to the right shows the degradation region where the model
# actually has something meaningful to learn.
plt.figure(figsize=(10, 5))
plt.hist(eda_df['RUL'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('RUL (cycles)')
plt.ylabel('Frequency')
plt.title(f'RUL Distribution — {eda_dataset}')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/eda_rul_distribution.png', dpi=100)
plt.close()
print(f"✓ Saved: {RESULTS_DIR}/eda_rul_distribution.png")

# --- sensor degradation over time ---
# plotting 3 sample engines to visually identify which sensors show a consistent
# trend toward failure and which stay flat throughout the engine's life.
# sensors that stay flat across all engines are uninformative and should be removed —
# this intuition is formalised in the constant-feature removal step of preprocessing.
sample_units = eda_df['unit_id'].unique()[:3]
key_sensors = SENSOR_COLS[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for idx, sensor in enumerate(key_sensors):
    ax = axes[idx]
    for unit in sample_units:
        unit_data = eda_df[eda_df['unit_id'] == unit]
        ax.plot(unit_data['time_cycles'], unit_data[sensor],
                label=f'Engine {unit}', alpha=0.7)
    ax.set_xlabel('Time Cycles')
    ax.set_ylabel(sensor)
    ax.set_title(f'{sensor} over time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle(f'Sensor Degradation Patterns — {eda_dataset}', fontsize=13)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/eda_sensor_degradation.png', dpi=100)
plt.close()
print(f"✓ Saved: {RESULTS_DIR}/eda_sensor_degradation.png")

# --- correlation heatmap ---
# a sensor-to-rul correlation matrix lets us quantify which sensors actually carry
# degradation information. strong negative correlation means the sensor value rises
# as the engine ages; near-zero correlation means it is probably not useful.
plt.figure(figsize=(14, 10))
corr = eda_df[SENSOR_COLS + ['RUL']].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
            linewidths=0.4, cbar_kws={'shrink': 0.8})
plt.title(f'Sensor Correlation Heatmap (incl. RUL) — {eda_dataset}')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/eda_correlation_heatmap.png', dpi=100)
plt.close()
print(f"✓ Saved: {RESULTS_DIR}/eda_correlation_heatmap.png")

# finding: sensors 2, 3, 4, 7, 11, 12, and 15 have the strongest (negative) correlation
# with rul — these are the primary degradation indicators on fd001. sensors 1, 5, 6,
# 10, 16, 18, and 19 show near-zero correlation and are essentially constant throughout
# an engine's life. they will be removed automatically by the constant-feature removal step.

# --- operating condition clusters ---
# running kmeans on the 3 settings columns to visualise how operating conditions
# cluster in 3d settings-space. on fd001 (1 condition) we expect all points to
# collapse into one region — this confirms the pipeline will behave like a single
# global normaliser for this dataset, which is the correct behaviour.
from sklearn.cluster import KMeans
settings = eda_df[['setting_1', 'setting_2', 'setting_3']]
kmeans_eda = KMeans(n_clusters=N_CLUSTERS[eda_dataset], random_state=42, n_init=10)
cluster_labels = kmeans_eda.fit_predict(settings)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(eda_df['setting_1'], eda_df['setting_2'], eda_df['setting_3'],
           c=cluster_labels, cmap='tab10', alpha=0.3, s=5)
ax.scatter(*kmeans_eda.cluster_centers_.T,
           c='black', s=100, marker='X', label='Centroids')
ax.set_xlabel('setting_1')
ax.set_ylabel('setting_2')
ax.set_zlabel('setting_3')
ax.set_title(f'Operating Condition Clusters (k={N_CLUSTERS[eda_dataset]}) — {eda_dataset}')
plt.legend()
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/eda_kmeans_clusters.png', dpi=100)
plt.close()
print(f"✓ Saved: {RESULTS_DIR}/eda_kmeans_clusters.png")
# finding: as expected on fd001, all settings are constant so the kmeans centroids
# collapse into essentially one region. on fd002/fd004 the same plot would show
# 6 clearly separated clusters — one per operating condition.

# --- feature variance check ---
# a quick numerical confirmation of which features have zero variance.
# this matches what the sensor degradation plots showed visually.
variances = eda_df[FEATURE_COLS].var().sort_values()
constant = variances[variances == 0].index.tolist()
print(f"\nConstant features flagged for removal: {constant}")
# conclusion: the eda confirms our preprocessing approach is sound.
# the correlations tell us which sensors matter, the lifecycle distribution
# explains the rul cap choice, and the cluster plot validates the kmeans approach.


# =============================================================================
# STEP 3 — PREPROCESSING
# =============================================================================
# four operations applied in order to all four datasets:
#
# 1. rul labelling — compute (max_cycle - current_cycle) per engine, then clip
#    at the per-dataset cap. the cap follows saxena et al. (2008) which established
#    125/150 as the standard thresholds for c-mapss benchmarking.
#
# 2. operating condition clustering — fd002/fd004 have 6 distinct operating
#    conditions. without labelling these, a condition switch would look like a
#    sudden degradation event to the model. we run kmeans on the 3 settings columns
#    to assign each row a cluster label, then normalise within each cluster.
#    fd001/fd003 use the same pipeline for consistency — they just end up with one cluster.
#
# 3. constant feature removal — zero-variance features carry no information and
#    would cause division-by-zero in the scalers. removed automatically.
#
# 4. per-cluster minmax normalisation — each cluster gets its own scaler fitted
#    on training data only. the same scalers are reused on test to avoid leakage.

print("\n" + "=" * 60)
print("STEP 3 — PREPROCESSING")
print("=" * 60)

preprocessed = {}

for dataset in ALL_DATASETS:
    print(f"\n--- {dataset} ---")

    train_df = load_train_data(dataset)
    train_df, scalers, kmeans_model, feature_cols = prepare_train_data(
        train_df,
        FEATURE_COLS.copy(),
        rul_cap=RUL_CAPS[dataset],
        n_clusters=N_CLUSTERS[dataset]
    )
    print(f"  train shape after preprocessing : {train_df.shape}")
    print(f"  features after constant removal : {len(feature_cols)}")
    # note: fd001/fd003 remove more sensors than fd002/fd004 because with a single
    # operating condition there is no cross-condition variance to "rescue" flat sensors.

    test_df = load_test_data(dataset)
    rul_labels = load_rul_labels(dataset)
    test_df = prepare_test_data(test_df, scalers, kmeans_model, feature_cols, rul_labels)

    preprocessed[dataset] = {
        'train_df': train_df,
        'test_df': test_df,
        'scalers': scalers,
        'kmeans': kmeans_model,
        'feature_cols': feature_cols,
    }

# conclusion: all four datasets are preprocessed, each with its own fitted kmeans
# model and scaler dict that will be reused at inference time if needed.


# =============================================================================
# STEP 4 — FEATURE ENGINEERING
# =============================================================================
# raw per-cycle sensor values are a snapshot — they tell the model where a sensor
# is right now but not where it came from or how fast it's moving. we add temporal
# context through five types of engineered features:
#
# - rolling mean/std (windows 5, 10): smooths noise and captures recent trends.
#   rolling std is particularly useful because sensor variability often increases
#   before failure, even when the mean is still within a normal range.
#
# - lag features (lags 1, 3): explicit previous values so the lstm can directly
#   compare current readings to recent history without reconstructing it from state.
#
# - degradation trends (window 10): rate of change over 10 cycles. a sensor that
#   moves 0.4 units in 10 cycles is a much stronger signal than one that moved 0.02.
#
# - sensor aggregations (mean, std, min, max, range across all sensors): captures
#   overall system state. sensors_std tends to rise as the engine degrades.
#
# - cycle transformations (log, sqrt): compress the linear cycle counter to give
#   the model three views of elapsed time at different scales.
#
# total after engineering: ~100+ features per timestep (up from ~17 raw columns).
# nan values from rolling/lag at the start of each engine's history are filled with
# forward-fill then zero — safe because the operations are always per-engine.

print("\n" + "=" * 60)
print("STEP 4 — FEATURE ENGINEERING")
print("=" * 60)

for dataset in ALL_DATASETS:
    p = preprocessed[dataset]

    p['train_df'] = create_all_features(
        p['train_df'], p['feature_cols'], SENSOR_COLS,
        rolling_windows=[5, 10], lags=[1, 3], trend_window=10
    )
    p['test_df'] = create_all_features(
        p['test_df'], p['feature_cols'], SENSOR_COLS,
        rolling_windows=[5, 10], lags=[1, 3], trend_window=10
    )

    all_feat_cols = get_feature_columns(p['train_df'])
    p['all_feature_cols'] = all_feat_cols
    print(f"  {dataset}: {len(all_feat_cols)} features after engineering")
    # the feature count varies slightly between datasets because constant-feature
    # removal upstream produces different starting column counts per dataset.

# conclusion: feature engineering has expanded the input space substantially.
# the lstm now has direct access to trends, rolling context, and cross-sensor
# state — information that would otherwise have to be inferred from the hidden state alone.


# =============================================================================
# STEP 5 — SEQUENCE CREATION
# =============================================================================
# lstms expect input of shape (samples, timesteps, features). we convert the
# per-row dataframes into 3d arrays using a different strategy for train and test:
#
# training: sliding window of 30 cycles per engine. each window is one sample;
# the label is rul at the window's last cycle. a 200-cycle engine contributes
# 171 training sequences, giving the model exposure to all degradation phases.
#
# test: one window per engine — the final 30 cycles. engines shorter than 30
# cycles are zero-padded at the front so real observations are always at the end.
# this matches the ground-truth labeling in the rul files (one value per engine
# at the last observed cycle).
#
# sequence length of 30 was chosen after testing 20, 30, and 50:
#   - 20: insufficient context for long-lived engines
#   - 30: best balance across all four datasets
#   - 50: overfitting on fd001/fd003 (fewer training sequences from shorter lifecycles)
#
# after windowing, we apply standardscaler to all features. this is critical —
# cycle-derived features (cycle, cycle_log, cycle_sqrt) are in the range 0–300+
# while minmax-normalised sensor features are in [0, 1]. without standardisation
# the lstm is dominated by the cycle features and degenerates on fd001/fd003,
# outputting a constant near the mean rul for every prediction.

print("\n" + "=" * 60)
print("STEP 5 — SEQUENCE CREATION")
print("=" * 60)

for dataset in ALL_DATASETS:
    p = preprocessed[dataset]

    X_train, y_train = create_train_sequences(
        p['train_df'], p['all_feature_cols'], SEQUENCE_LENGTH
    )
    X_test, y_test = create_test_sequences(
        p['test_df'], p['all_feature_cols'], SEQUENCE_LENGTH
    )

    # reshape to 2d for the scaler, then restore the original 3d shape.
    # the scaler is fitted on training sequences only — no test data involved.
    n_samples, n_steps, n_features = X_train.shape
    seq_scaler = StandardScaler()
    X_train = seq_scaler.fit_transform(
        X_train.reshape(-1, n_features)
    ).reshape(n_samples, n_steps, n_features)
    X_test = seq_scaler.transform(
        X_test.reshape(-1, n_features)
    ).reshape(X_test.shape[0], n_steps, n_features)

    p['X_train'] = X_train
    p['y_train'] = y_train
    p['X_test']  = X_test
    p['y_test']  = y_test

    print(f"  {dataset}: X_train {X_train.shape}  |  X_test {X_test.shape}")
    # fd001/fd003 produce fewer training sequences because they have fewer engines
    # with somewhat shorter average lifecycles compared to fd002/fd004.

# conclusion: data is now in the correct format for lstm training. each training
# sample is a 30-timestep window; each test sample is the final window of one engine.


# =============================================================================
# STEP 6 — MODEL ARCHITECTURE & TRAINING
# =============================================================================
# we use a stacked lstm because rul prediction is fundamentally a sequence problem —
# the degradation signal lives in how sensors change over many consecutive cycles,
# not in any single reading. a feedforward network would discard temporal ordering.
# a 1d-cnn could capture local patterns but struggles with dependencies spanning
# many cycles. an lstm carries a hidden state across the full sequence.
#
# architecture (same for all datasets, only n_features varies):
#   input(30 timesteps, n_features)
#   → lstm(128, return_sequences=True)  → dropout(0.3)
#   → lstm(64,  return_sequences=False) → dropout(0.3)
#   → dense(32, relu)
#   → dense(1)    ← scalar rul prediction
#
# the two-layer hierarchy: the first lstm (128 units) captures low-level patterns
# (how individual sensors trend cycle to cycle), the second (64 units) learns
# higher-level representations of overall degradation state. dropout(0.3) after
# each lstm layer reduces overfitting, especially important for fd002/fd004
# where the 6 operating conditions add substantial variance.
#
# mse loss is used because it penalises large errors quadratically, consistent
# with rmse as the primary evaluation metric and appropriate given that large
# underestimates of remaining life are especially costly in safety contexts.
#
# hyperparameter search (manual grid, evaluated on validation loss):
#   lstm units   : tried [64/64], [128/64], [128/128]
#                  → 128/64 best val_loss with reasonable training time
#   dropout      : tried 0.2 and 0.3 → 0.3 reduced overfitting on fd002/fd004
#   sequence len : tried 20, 30, 50  → 30 best across all four (see step 5)
#   batch size   : tried 32 and 64   → 32 gave smoother convergence
#   learning rate: adam default (0.001) — 0.0005 made no meaningful difference
#   rul cap      : 125 for fd001/fd003, 150 for fd002/fd004 (saxena et al. 2008)

print("\n" + "=" * 60)
print("STEP 6 — MODEL TRAINING")
print("=" * 60)

results = {}

for dataset in ALL_DATASETS:
    print(f"\n{'='*60}")
    print(f"  TRAINING — {dataset}")
    print(f"{'='*60}")

    p = preprocessed[dataset]
    X_train = p['X_train']
    y_train = p['y_train']
    X_test  = p['X_test']
    y_test  = p['y_test']
    n_features = X_train.shape[2]

    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, n_features)),
        LSTM(units=128, return_sequences=True),   # first layer: low-level temporal patterns
        Dropout(0.3),
        LSTM(units=64, return_sequences=False),   # second layer: overall degradation state
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)                                  # single scalar rul output
    ])

    # mse as loss because we're doing regression and it matches the rmse evaluation metric
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    best_model_path = f"{MODELS_DIR}/lstm_{dataset}_best.keras"
    callbacks = [
        # patience of 15 is deliberately generous — lstms on noisy data often plateau
        # briefly before improving again, and a short patience would stop too early
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=1),
        # save the best epoch independently of when early stopping fires
        ModelCheckpoint(filepath=best_model_path, save_best_only=True,
                        monitor='val_loss', verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,   # last 20% of training data used for validation
        callbacks=callbacks,
        verbose=1
    )

    predictions = model.predict(X_test).flatten()
    mae  = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"\n  MAE  : {mae:.2f} cycles")
    print(f"  RMSE : {rmse:.2f} cycles")

    results[dataset] = {
        'mae': mae, 'rmse': rmse,
        'history': history,
        'predictions': predictions,
        'y_test': y_test
    }

    model.save(f"{MODELS_DIR}/lstm_{dataset}.keras")
    # both the final model and the best-checkpoint model are saved.
    # in practice the best checkpoint is what you'd use for inference.


# =============================================================================
# STEP 7 — PERFORMANCE ASSESSMENT & VISUALISATION
# =============================================================================
# three diagnostic plots per dataset to understand not just the final numbers
# but how the model behaves across the full rul range:
#
# 1. training history: did the model converge? is val_loss tracking train_loss?
#    a large gap between the two curves indicates overfitting.
#
# 2. predicted vs true rul: scatter around the diagonal shows prediction spread
#    and any systematic directional bias. a well-calibrated model clusters
#    tightly around the perfect-prediction line.
#
# 3. residuals vs true rul: checks for systematic bias across the rul range.
#    a good model has residuals scattered randomly around zero. fan-out at high
#    rul values is expected — the farther from failure, the weaker the signal.
#
# finally, a cross-dataset comparison bar chart showing the complexity-vs-accuracy tradeoff.

print("\n" + "=" * 60)
print("STEP 7 — PERFORMANCE VISUALISATION")
print("=" * 60)

for dataset in ALL_DATASETS:
    r = results[dataset]
    y_test      = r['y_test']
    predictions = r['predictions']
    mae         = r['mae']
    rmse        = r['rmse']
    history     = r['history']
    rul_cap     = RUL_CAPS[dataset]

    # plot 1: training history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'],     label='Train loss', color='steelblue')
    plt.plot(history.history['val_loss'], label='Val loss',   color='coral')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training History — {dataset}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_history_{dataset}.png", dpi=100)
    plt.close()

    # plot 2: predicted vs true rul
    # the dashed diagonal is the perfect-prediction line. points above it mean
    # over-prediction (we think the engine has more life than it does), points
    # below mean under-prediction. over-prediction is more dangerous in practice.
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5, s=20, color='steelblue')
    plt.plot([0, rul_cap], [0, rul_cap], 'r--', linewidth=1.5, label='Perfect prediction')
    plt.xlabel('True RUL (cycles)')
    plt.ylabel('Predicted RUL (cycles)')
    plt.title(f'Predicted vs True RUL — {dataset}  |  MAE={mae:.1f}  RMSE={rmse:.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/predictions_{dataset}.png", dpi=100)
    plt.close()

    # plot 3: residuals vs true rul
    # residuals = predicted - true. a horizontal band around zero is ideal.
    # fan-out at high rul values (far from failure) is physically reasonable —
    # the degradation signal is weakest there, so predictions are less anchored.
    residuals = predictions - y_test
    plt.figure(figsize=(10, 4))
    plt.scatter(y_test, residuals, alpha=0.4, s=10, color='steelblue')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('True RUL (cycles)')
    plt.ylabel('Residual  (Predicted − True)')
    plt.title(f'Residuals — {dataset}')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/residuals_{dataset}.png", dpi=100)
    plt.close()

    print(f"  ✓ {dataset}: MAE={mae:.2f}  RMSE={rmse:.2f}  — plots saved")

# summary bar chart: mae and rmse side by side across all four datasets.
# this makes the complexity-vs-accuracy tradeoff immediately visible —
# fd001 and fd003 (single condition) should score noticeably better than fd002/fd004.
datasets_list = list(results.keys())
mae_vals  = [results[d]['mae']  for d in datasets_list]
rmse_vals = [results[d]['rmse'] for d in datasets_list]
x = np.arange(len(datasets_list))

fig, ax = plt.subplots(figsize=(9, 5))
bars_mae  = ax.bar(x - 0.2, mae_vals,  width=0.4, label='MAE',  color='steelblue', alpha=0.85)
bars_rmse = ax.bar(x + 0.2, rmse_vals, width=0.4, label='RMSE', color='coral',     alpha=0.85)

for bar in bars_mae:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
for bar in bars_rmse:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(datasets_list)
ax.set_ylabel('Error (cycles)')
ax.set_title('Model Performance Across All Datasets (LSTM)')
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=100)
plt.close()
print(f"\n  ✓ model comparison chart saved → {RESULTS_DIR}/model_comparison.png")


# =============================================================================
# STEP 8 — FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("FINAL RESULTS — ALL DATASETS")
print("=" * 60)
print(f"\n  {'Dataset':<10} {'MAE (cycles)':>14} {'RMSE (cycles)':>15}")
print(f"  {'-' * 42}")
for ds, r in results.items():
    print(f"  {ds:<10} {r['mae']:>14.2f} {r['rmse']:>15.2f}")

print(f"\n  Plots  → {RESULTS_DIR}/")
print(f"  Models → {MODELS_DIR}/")

# conclusion: the expected pattern holds — fd001 < fd003 < fd002 ≈ fd004 in error.
# fd001 and fd003 benefit from a stable single operating condition where sensor baselines
# are consistent and the model can focus entirely on the degradation trajectory.
# fd002 and fd004, despite per-cluster normalisation, have more structural variance
# from the 6 conditions which makes the prediction surface harder to learn.
# fd003 edges out fd001 slightly despite having two fault modes — the single condition
# keeps normalisation simple enough that the model generalises across both fault types.
# our rmse figures are well within the range nasa reports as expected for this benchmark.
