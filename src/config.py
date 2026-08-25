"""project configuration — all constants in one place so changing a hyperparameter
never requires hunting through multiple files."""

# output directories — created at runtime if they don't exist
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# dataset identifiers — ALL_DATASETS drives the main training loop
SELECTED_DATASET = "FD002"  # used by standalone scripts (eda.py, test_features.py)
ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]

# rul caps per dataset — values follow saxena et al. (2008), the standard c-mapss benchmark paper.
# the idea is that engines with many cycles remaining are still in their "healthy" phase and
# their sensors are flat. asking the model to predict precise values like 280 from a flat signal
# is just learning noise, so we cap everything above the threshold at the cap value.
# fd001/fd003 use 125 (single operating condition, shorter typical lifecycles),
# fd002/fd004 use 150 (six conditions, longer average engine lives).
RUL_CAPS = {
    "FD001": 125,
    "FD002": 150,
    "FD003": 125,
    "FD004": 150,
}

# number of kmeans clusters for operating condition assignment.
# all four datasets use k=6 for pipeline consistency. fd001/fd003 only have one real
# condition so all rows land in a single cluster, which is fine — the scaler just
# sees one group and behaves like a standard global minmax scaler.
N_CLUSTERS = {
    "FD001": 6,
    "FD002": 6,
    "FD003": 6,
    "FD004": 6,
}

# raw column names — the text files have no header so we assign names manually.
# order matches the dataset exactly: id, cycle, 3 settings, 21 sensors.
COLUMNS = [
    "unit_id",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]

# feature columns used for modelling — time_cycles is excluded here because it goes
# through its own transformation (log, sqrt) in feature engineering rather than
# being fed raw, which would introduce scale issues
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLS = SETTING_COLS + SENSOR_COLS

# regression target
TARGET_COL = "RUL"

# sequence length of 30 was chosen after testing 20, 30, and 50 cycles.
# 20 gave insufficient context for long-lived engines; 50 caused overfitting
# on fd001/fd003 where shorter lifecycles mean fewer distinct training windows.
SEQUENCE_LENGTH = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42

# training hyperparameters — batch 32 gave smoother convergence than 64.
# lr 0.001 (adam default) performed the same as 0.0005 so we kept the default.
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
