# Data paths
TRAIN_FEATURES_PATH: str = "/kaggle/input/drw-all-auxiliary-datasets/top500_train_dataset.csv"
TRAIN_REGIMES_PATH: str = "/kaggle/input/drw-all-auxiliary-datasets/train_regimes.csv"
TEST_FEATURES_PATH: str = "/kaggle/input/drw-all-auxiliary-datasets/top500_test_dataset.csv"
TEST_REGIMES_PATH: str = "/kaggle/input/drw-all-auxiliary-datasets/test_regimes.csv"
    
# Column names - Updated to match actual data structure
TARGET_COLUMN: str = "label"
REGIME_COLUMN: str = "predicted_regime_Est"  # Fixed: Use actual column name
ID_COLUMN: str = "ID"
TIMESTAMP_COLUMN: str = "timestamp"
Feature_Columns: list = []