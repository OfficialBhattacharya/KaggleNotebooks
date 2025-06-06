import pandas as pd

def load_and_merge_datasets():
    """
    Load and merge train and test datasets.
    
    Returns:
        tuple: (merged_train_df, merged_test_df)
    """
    # Dataset paths
    train_top500Features = "/kaggle/input/drw-all-auxiliary-datasets/top500_train_dataset.csv"
    train_regimes = "/kaggle/input/drw-all-auxiliary-datasets/train_regimes.csv"
    test_top500Features = "/kaggle/input/drw-all-auxiliary-datasets/top500_test_dataset.csv"
    test_regimes = "/kaggle/input/drw-all-auxiliary-datasets/test_regimes.csv"
    
    # Load datasets
    print("Loading datasets...")
    train_features_df = pd.read_csv(train_top500Features)
    train_regimes_df = pd.read_csv(train_regimes)
    test_features_df = pd.read_csv(test_top500Features)
    test_regimes_df = pd.read_csv(test_regimes)
    
    print(f"Train features shape: {train_features_df.shape}")
    print(f"Train regimes shape: {train_regimes_df.shape}")
    print(f"Test features shape: {test_features_df.shape}")
    print(f"Test regimes shape: {test_regimes_df.shape}")
    
    # Merge train datasets on 'timestamp'
    print("Merging train datasets on 'timestamp'...")
    merged_train_df = pd.merge(train_features_df, train_regimes_df, on='timestamp', how='inner')
    print(f"Merged train dataset shape: {merged_train_df.shape}")
    
    # Merge test datasets on 'ID'
    print("Merging test datasets on 'ID'...")
    merged_test_df = pd.merge(test_features_df, test_regimes_df, on='ID', how='inner')
    print(f"Merged test dataset shape: {merged_test_df.shape}")
    
    return merged_train_df, merged_test_df

# Example usage:
if __name__ == "__main__":
    train_df, test_df = load_and_merge_datasets()
    print("\nMerged datasets loaded successfully!")
    print(f"Train columns: {list(train_df.columns)[:10]}...")  # Show first 10 columns
    print(f"Test columns: {list(test_df.columns)[:10]}...")   # Show first 10 columns
