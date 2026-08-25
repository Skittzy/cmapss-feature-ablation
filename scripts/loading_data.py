from src.data.loader import load_train_data, load_test_data, load_rul_labels
from src.data.preprocessor import add_rul_column, normalize_data, prepare_test_data, prepare_train_data, remove_constant_features
from src.config import SELECTED_DATASET

test_df = load_test_data(SELECTED_DATASET)
train_df = load_train_data(SELECTED_DATASET)
rul_labels = load_rul_labels(SELECTED_DATASET)



print(test_df.head(5))
print(train_df.head(5))
print(rul_labels.head(5))