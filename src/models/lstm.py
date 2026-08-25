"""
NOTE: this file is legacy. the canonical training entry point is main.py.
this module is kept for reference and can still be run standalone, but main.py
runs the identical architecture with better eda, step-by-step documentation,
and the full preprocessing pipeline integrated in one place.
"""

import numpy as np
from src.config import SEQUENCE_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODELS_DIR, RESULTS_DIR, ALL_DATASETS
from scripts.test_sequences import main as get_sequences
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train(dataset):
    print(f"\n{'='*60}")
    print(f"  TRAINING ON {dataset}")
    print(f"{'='*60}")

    # 1. get sequences from the pipeline
    X_train, y_train, X_test, y_test = get_sequences(dataset)

    # standardise all features to zero mean / unit variance.
    # this fixes the scale mismatch between raw cycle features (0–300+) and
    # minmax-normalised sensor features (0–1). without this step, fd001 and fd003
    # produce a degenerate model that predicts a constant near the mean rul.
    n_samples, n_steps, n_features = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples, n_steps, n_features)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    actual_feature_count = X_train.shape[2]

    # 2. build the stacked lstm model — same architecture used in main.py
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, actual_feature_count)),
        LSTM(units=128, return_sequences=True),
        Dropout(0.3),
        LSTM(units=64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 3. set up callbacks
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_model_path = f"{MODELS_DIR}/lstm_{dataset}_best.keras"

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=best_model_path, save_best_only=True, monitor='val_loss', verbose=1),
    ]

    # 4. train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # 5. evaluate on the held-out test set
    print("\nEvaluating on test set...")
    predictions = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"\n{'='*50}")
    print(f"  FINAL RESULTS ({dataset})")
    print(f"{'='*50}")
    print(f"  MAE  (Mean Absolute Error) : {mae:.2f} cycles")
    print(f"  RMSE (Root Mean Sq Error)  : {rmse:.2f} cycles")
    print(f"{'='*50}\n")

    # 6. training history plot
    # a healthy run shows train and val loss descending together without a large gap
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'],     label='Train loss', color='steelblue')
    plt.plot(history.history['val_loss'], label='Val loss',   color='coral')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training History — {dataset}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_history_{dataset}.png")
    plt.close()

    # 7. predictions vs actual rul
    # points clustered tightly around the diagonal indicate good calibration.
    # systematic displacement above the line means the model is over-predicting rul,
    # which is the more dangerous direction in a safety context.
    rul_cap = y_train.max()
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.6, s=20, color='steelblue')
    plt.plot([0, rul_cap], [0, rul_cap], 'r--', linewidth=1.5, label='Perfect')
    plt.xlabel('True RUL (cycles)')
    plt.ylabel('Predicted RUL (cycles)')
    plt.title(f'Predicted vs True RUL — {dataset} | MAE={mae:.1f} RMSE={rmse:.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/predictions_{dataset}.png")
    plt.close()

    # 8. residuals vs true rul
    # fan-out at high rul values is expected — the degradation signal is weakest
    # far from failure, so the model is less anchored there
    residuals = predictions - y_test
    plt.figure(figsize=(10, 4))
    plt.scatter(y_test, residuals, alpha=0.4, s=10)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('True RUL')
    plt.ylabel('Residuals (Predicted - True)')
    plt.title(f'Residuals vs True RUL — {dataset}')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/residuals_{dataset}.png")
    plt.close()

    print(f"Plots saved to {RESULTS_DIR}/")

    model.save(f"{MODELS_DIR}/lstm_{dataset}.keras")
    print(f"Model saved to {MODELS_DIR}/lstm_{dataset}.keras")
    print(f"Best model saved to {best_model_path}")

    return mae, rmse


if __name__ == "__main__":
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ALL_DATASETS

    results = {}
    for dataset in datasets:
        mae, rmse = train(dataset)
        results[dataset] = {"MAE": mae, "RMSE": rmse}

    print(f"\n{'='*50}")
    print("  SUMMARY — ALL DATASETS")
    print(f"{'='*50}")
    print(f"  {'Dataset':<10} {'MAE':>8} {'RMSE':>8}")
    print(f"  {'-'*28}")
    for ds, metrics in results.items():
        print(f"  {ds:<10} {metrics['MAE']:>8.2f} {metrics['RMSE']:>8.2f}")
    print(f"{'='*50}")
    # expected: fd001/fd003 (single condition) score lower error than fd002/fd004 (6 conditions)
