"""data loading functions for the nasa c-mapss dataset.

the raw files are space-separated plain text with no header row. the separator
is inconsistent (sometimes one space, sometimes two), so we use a regex instead
of a fixed delimiter. each file has a phantom empty column from trailing whitespace
that we drop immediately after loading.
"""

import pandas as pd
from pathlib import Path
from src.config import DATA_RAW, COLUMNS


def load_train_data(dataset_name):
    """load training data for the specified dataset.

    training files run each engine all the way to failure, so every unit_id
    has a clear maximum cycle we can use to calculate rul in preprocessing.

    args:
        dataset_name: dataset identifier e.g. 'FD001', 'FD002'

    returns:
        dataframe with raw sensor + settings data, no rul column yet
    """
    file_path = Path(DATA_RAW) / f"train_{dataset_name}.txt"

    # sep=r"\s+" handles the variable whitespace between columns.
    # engine="python" is required for regex separators in pandas.
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")

    # the raw files have a trailing space on every line which pandas reads as
    # an extra empty column — drop it before assigning our column names
    df.dropna(axis=1, inplace=True)
    df.columns = COLUMNS
    return df


def load_test_data(dataset_name):
    """load test data for the specified dataset.

    test files stop before failure — the true rul at the last observed cycle
    is in a separate file (loaded by load_rul_labels). the structure is otherwise
    identical to the training files.

    args:
        dataset_name: dataset identifier e.g. 'FD001', 'FD002'

    returns:
        dataframe with raw sensor + settings data, no rul column
    """
    file_path = Path(DATA_RAW) / f"test_{dataset_name}.txt"
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df.dropna(axis=1, inplace=True)
    df.columns = COLUMNS
    return df


def load_rul_labels(dataset_name):
    """load the ground-truth rul values for the test set.

    each line in RUL_FD00X.txt is the rul for one test engine at its last
    observed cycle — one value per engine, in the same order as unit_ids
    appear in the test file. these are merged back onto the test dataframe
    in preprocessor.prepare_test_data().

    args:
        dataset_name: dataset identifier e.g. 'FD001', 'FD002'

    returns:
        dataframe with a single 'RUL' column, one row per test engine
    """
    file_path = Path(DATA_RAW) / f"RUL_{dataset_name}.txt"
    rul = pd.read_csv(file_path, header=None, names=["RUL"])
    return rul
