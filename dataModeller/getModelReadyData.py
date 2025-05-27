import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.special import logit # logit and its inverse (expit)
from scipy.stats import boxcox
import pandas as pd


def getModelReadyData(train_dataset, test_dataset, xTransform, yTransform, target):
    """
    Processes the train and test data by applying specified transformations to X and y.

    Args:
        train_dataset (pd.DataFrame): The training dataset.
        test_dataset (pd.DataFrame): The testing dataset.
        xTransform (str): The transformation to apply to X.
                          Options: 'minmax', 'standardized', 'robust', 'boxcox', 'logit', 'none'.
        yTransform (str): The transformation to apply to y.
                          Options: 'logit', 'standardized', 'robust', 'boxcox', 'log1p', 'none'.
        target (str): The name of the target column in the train_dataset.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray or pd.DataFrame): Transformed training features.
            - y_train (np.ndarray): Transformed training target.
            - X_test (np.ndarray or pd.DataFrame): Transformed testing features.
    """

    # Make copies to avoid modifying original datasets
    X_train = train_dataset.copy().drop(columns=[target])
    y_train = train_dataset[target].copy()
    X_test = test_dataset.copy()

    # Apply X transformations
    if xTransform == 'minmax':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif xTransform == 'standardized':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif xTransform == 'robust':
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif xTransform == 'boxcox':
        # Box-Cox requires positive data. Add a small constant if not already positive.
        # It's typically applied column-wise.
        X_train_transformed = np.zeros_like(X_train, dtype=float)
        X_test_transformed = np.zeros_like(X_test, dtype=float)

        # Store lambda values from training data to apply to test data
        # It's good practice to store these if you need to inverse transform later or for consistency
        boxcox_lambdas_X = {}

        for i, col in enumerate(X_train.columns):
            # Handle non-positive values for Box-Cox
            offset = 0
            if (X_train[col] <= 0).any():
                print(f"Warning: X_train column '{col}' contains non-positive values. "
                      f"Adding a small constant for Box-Cox transformation.")
                offset = abs(X_train[col].min()) + 1e-6 if X_train[col].min() <= 0 else 1e-6 # Ensure positive
            
            # Fit and transform X_train and get the lambda value
            transformed_col_train, lambda_val = boxcox(X_train[col] + offset)
            boxcox_lambdas_X[col] = lambda_val # Store lambda for this column

            # Transform X_test using the lambda value learned from X_train
            # If X_test also has non-positive values, apply the same offset
            transformed_col_test = boxcox(X_test[col] + offset, lmbda=lambda_val)[0] # [0] to get just the transformed data

            X_train_transformed[:, i] = transformed_col_train
            X_test_transformed[:, i] = transformed_col_test

        X_train = X_train_transformed
        X_test = X_test_transformed
    elif xTransform == 'logit':
        # Logit requires input values to be in (0, 1). Scale if necessary.
        if not ((X_train > 0).all().all() and (X_train < 1).all().all()):
            print("Warning: X_train values are not strictly between 0 and 1. Applying MinMaxScaler before logit.")
            scaler_logit = MinMaxScaler()
            X_train = scaler_logit.fit_transform(X_train)
            X_test = scaler_logit.transform(X_test)
        X_train = logit(X_train)
        X_test = logit(X_test)
    elif xTransform == 'none':
        # No transformation applied to X
        pass
    else:
        raise ValueError(f"Unknown xTransform: {xTransform}")

    # Apply y transformations
    if yTransform == 'logit':
        # Logit requires input values to be in (0, 1). Scale if necessary.
        if not ((y_train > 0).all() and (y_train < 1).all()):
            print("Warning: y_train values are not strictly between 0 and 1. Applying MinMax scaling to y before logit.")
            # Reshape for scaler
            y_train_reshaped = y_train.values.reshape(-1, 1)
            scaler_logit_y = MinMaxScaler()
            y_train = scaler_logit_y.fit_transform(y_train_reshaped).flatten()
        y_train = logit(y_train)
    elif yTransform == 'standardized':
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    elif yTransform == 'robust':
        scaler_y = RobustScaler()
        y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    elif yTransform == 'boxcox':
        # Box-Cox requires positive data.
        offset_y = 0
        if (y_train <= 0).any():
            print("Warning: y_train contains non-positive values. Adding a small constant for Box-Cox transformation.")
            offset_y = abs(y_train.min()) + 1e-6 if y_train.min() <= 0 else 1e-6 # Ensure positive
            y_train_transformed, _ = boxcox(y_train + offset_y)
        else:
            y_train_transformed, _ = boxcox(y_train)
        y_train = y_train_transformed
        # Note: For inverse transformation of predictions, you would need to store lambda_val_y.
    elif yTransform == 'log1p':
        # log1p requires x > -1. Add a small constant if not already meeting this.
        if (y_train <= -1).any():
            print("Warning: y_train contains values <= -1. Adding a constant to ensure y > -1 for log1p transformation.")
            offset_log1p_y = abs(y_train.min()) + 1e-6 if y_train.min() <= -1 else 0
            y_train = np.log1p(y_train + offset_log1p_y)
        else:
            y_train = np.log1p(y_train)
    elif yTransform == 'none':
        # No transformation applied to y
        pass
    else:
        raise ValueError(f"Unknown yTransform: {yTransform}")

    return X_train, y_train, X_test


# Backward compatibility function
def getModelReadyData_legacy(train_dataset, test_dataset, xTransform, yTransform, target):
    """
    Legacy version of getModelReadyData for backward compatibility.
    This is the original function signature.
    """
    result = getModelReadyData(train_dataset, test_dataset, xTransform, yTransform, target)
    return result[:2]  # Return only X_train, y_train


# Example Usage
if __name__ == "__main__":
    np.random.seed(42)

    # Dataset 1: General example
    train_data1 = {
        'feature_A': np.random.rand(100) * 10,
        'feature_B': np.random.randint(0, 50, 100),
        'my_target_variable': np.random.rand(100) * 100
    }
    test_data1 = {
        'feature_A': np.random.rand(50) * 10,
        'feature_B': np.random.randint(0, 50, 50),
    }
    train_dataset1 = pd.DataFrame(train_data1)
    test_dataset1 = pd.DataFrame(test_data1)
    my_target_col = 'my_target_variable'

    print("--- Running with xTransform = 'boxcox', yTransform = 'log1p' ---")
    # This is the test case that caused the error
    xTransform1 = 'boxcox'
    yTransform1 = 'log1p'
    X_train_bc_log1p, y_train_bc_log1p, X_test_bc_log1p = getModelReadyData(train_dataset1, test_dataset1, xTransform1, yTransform1, my_target_col)
    print("X_train_bc_log1p shape:", X_train_bc_log1p.shape)
    print("y_train_bc_log1p shape:", y_train_bc_log1p.shape)
    print("X_test_bc_log1p shape:", X_test_bc_log1p.shape)
    print("y_train_bc_log1p (first 5 values):\n", y_train_bc_log1p[:5])
    print(f"Original y_train min: {train_dataset1[my_target_col].min()}, max: {train_dataset1[my_target_col].max()}")
    print(f"Transformed y_train_bc_log1p min: {y_train_bc_log1p.min()}, max: {y_train_bc_log1p.max()}")


    # Dataset 2: With zeros in target to specifically test log1p handling
    train_data2 = {
        'f1': np.random.rand(100),
        'f2': np.random.randint(1, 10, 100),
        'sales': np.random.randint(0, 100, 100) # Can contain zeros
    }
    test_data2 = {
        'f1': np.random.rand(50),
        'f2': np.random.randint(1, 10, 50),
    }
    train_dataset2 = pd.DataFrame(train_data2)
    test_dataset2 = pd.DataFrame(test_data2)
    target_sales = 'sales'

    print("\n--- Running with yTransform = 'log1p' on data with zeros ---")
    X_train_sales, y_train_sales, X_test_sales = getModelReadyData(train_dataset2, test_dataset2, 'standardized', 'log1p', target_sales)
    print("y_train_sales (first 5 values):\n", y_train_sales[:5])
    print(f"Original y_train_sales min: {train_dataset2[target_sales].min()}, max: {train_dataset2[target_sales].max()}")
    print(f"Transformed y_train_sales min: {y_train_sales.min()}, max: {y_train_sales.max()}")

    # Dataset 3: With negative values for y (will trigger warning for log1p)
    train_data3 = {
        'fA': np.random.rand(100),
        'fB': np.random.randint(1, 10, 100),
        'net_profit': np.random.uniform(-50, 100, 100) # Can contain negative values
    }
    test_data3 = {
        'fA': np.random.rand(50),
        'fB': np.random.randint(1, 10, 50),
    }
    train_dataset3 = pd.DataFrame(train_data3)
    test_dataset3 = pd.DataFrame(test_data3)
    target_profit = 'net_profit'

    print("\n--- Running with yTransform = 'log1p' on data with negative values ---")
    X_train_profit, y_train_profit, X_test_profit = getModelReadyData(train_dataset3, test_dataset3, 'standardized', 'log1p', target_profit)
    print("y_train_profit (first 5 values):\n", y_train_profit[:5])
    print(f"Original y_train_profit min: {train_dataset3[target_profit].min()}, max: {train_dataset3[target_profit].max()}")
    print(f"Transformed y_train_profit min: {y_train_profit.min()}, max: {y_train_profit.max()}")

    # Add back the previous examples to ensure nothing is broken
    train_data_orig = {
        'feature1': np.random.rand(100) * 10,
        'feature2': np.random.randint(0, 50, 100),
        'actual_y': np.random.rand(100) * 100
    }
    test_data_orig = {
        'feature1': np.random.rand(50) * 10,
        'feature2': np.random.randint(0, 50, 50),
    }
    train_dataset_orig = pd.DataFrame(train_data_orig)
    test_dataset_orig = pd.DataFrame(test_data_orig)
    target_orig = 'actual_y'


    print("\n--- Running with xTransform = 'logit', yTransform = 'boxcox' (Original Test Case) ---")
    xTransform_orig1 = 'logit'
    yTransform_orig1 = 'boxcox'
    X_train_orig1, y_train_orig1, X_test_orig1 = getModelReadyData(train_dataset_orig, test_dataset_orig, xTransform_orig1, yTransform_orig1, target_orig)
    print("X_train_orig1 shape:", X_train_orig1.shape)
    print("y_train_orig1 shape:", y_train_orig1.shape)

    print("\n--- Running with xTransform = 'standardized', yTransform = 'robust' (Original Test Case) ---")
    xTransform_orig2 = 'standardized'
    yTransform_orig2 = 'robust'
    X_train_orig2, y_train_orig2, X_test_orig2 = getModelReadyData(train_dataset_orig, test_dataset_orig, xTransform_orig2, yTransform_orig2, target_orig)
    print("X_train_orig2 shape:", X_train_orig2.shape)
    print("y_train_orig2 shape:", y_train_orig2.shape)

    print("\n--- Testing Box-Cox with non-positive values (Original Test Case) ---")
    train_data_neg_orig = {
        'feature1': np.random.rand(100) * 10 - 5,
        'feature2': np.random.randint(-10, 40, 100),
        'y_col_neg': np.random.rand(100) * 100 - 20
    }
    test_data_neg_orig = {
        'feature1': np.random.rand(50) * 10 - 5,
        'feature2': np.random.randint(-10, 40, 50),
    }
    train_dataset_neg_orig = pd.DataFrame(train_data_neg_orig)
    test_dataset_neg_orig = pd.DataFrame(test_data_neg_orig)

    xTransform_bc_orig = 'boxcox'
    yTransform_bc_orig = 'boxcox'
    target_neg_orig = 'y_col_neg'
    X_train_bc_orig, y_train_bc_orig, X_test_bc_orig = getModelReadyData(train_dataset_neg_orig, test_dataset_neg_orig, xTransform_bc_orig, yTransform_bc_orig, target_neg_orig)
    print("X_train_bc_orig shape:", X_train_bc_orig.shape)
    print("y_train_bc_orig shape:", y_train_bc_orig.shape) 