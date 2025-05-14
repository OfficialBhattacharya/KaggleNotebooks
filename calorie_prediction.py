"""
Calorie Prediction Model

This script implements a machine learning solution for predicting calories burned during exercise
based on various physiological and activity parameters. It uses both CatBoost and XGBoost models
and blends their predictions for the final output.

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from scipy.stats import skew
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

def load_data():
    """
    Load training, testing, and submission data.
    
    Returns:
        tuple: (train_df, test_df, submission_df)
    """
    logger.info("Loading data...")
    train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
    test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
    submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
    return train, test, submission

def preprocess_data(train, test):
    """
    Preprocess the data by encoding categorical variables and handling duplicates.
    
    Args:
        train (pd.DataFrame): Training data
        test (pd.DataFrame): Test data
        
    Returns:
        tuple: (processed_train, processed_test)
    """
    logger.info("Preprocessing data...")
    # Convert sex to binary
    train['Sex'] = train['Sex'].map({'male': 1, 'female': 0})
    test['Sex'] = test['Sex'].map({'male': 1, 'female': 0})
    
    # Remove duplicates and get minimum calories for same features
    train = train.drop_duplicates(subset=train.columns).reset_index(drop=True)
    train = train.groupby(['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])['Calories'].min().reset_index()
    
    return train, test

def add_features(df, train):
    """
    Add engineered features to the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        train (pd.DataFrame): Training dataframe for reference
        
    Returns:
        pd.DataFrame: DataFrame with added features
    """
    logger.info("Adding engineered features...")
    # Basic features
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Intensity'] = df['Heart_Rate'] / df['Duration']
    
    # Duration-based features - use only durations present in training data
    df['Sex_Reversed'] = 1 - df['Sex']
    train_durations = sorted(train['Duration'].unique())
    for dur in train_durations:
        df[f'HR_Dur_{int(dur)}'] = np.where(df['Duration'] == dur, df['Heart_Rate'], 0)
        df[f'Temp_Dur_{int(dur)}'] = np.where(df['Duration'] == dur, df['Body_Temp'], 0)
    
    # Age-based features - use only ages present in training data
    train_ages = sorted(train['Age'].unique())
    for age in train_ages:
        df[f'HR_Age_{int(age)}'] = np.where(df['Age'] == age, df['Heart_Rate'], 0)
        df[f'Temp_Age_{int(age)}'] = np.where(df['Age'] == age, df['Body_Temp'], 0)
    
    # Interaction features
    for f1 in ['Duration', 'Heart_Rate', 'Body_Temp']:
        for f2 in ['Sex', 'Sex_Reversed']:
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    
    # Statistical features
    for col in ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']:
        for agg in ['min', 'max']:
            agg_val = train.groupby('Sex')[col].agg(agg).rename(f'Sex_{col}_{agg}')
            df = df.merge(agg_val, on='Sex', how='left')
    
    df.drop(columns=['Sex_Reversed'], inplace=True)
    return df

def add_advanced_features(train_df, test_df):
    """
    Add advanced physiological features to the dataset.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        
    Returns:
        tuple: (train_df, test_df) with added features
    """
    logger.info("Adding advanced features...")
    
    # 1. Basal Metabolic Rate (BMR)
    train_df['BMR'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
    test_df['BMR'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2)
    
    # 2. Metabolic Efficiency Index
    train_df['Metabolic_Efficiency'] = train_df['BMR'] * (train_df['Heart_Rate'] / train_df['BMR'].median())
    test_df['Metabolic_Efficiency'] = test_df['BMR'] * (test_df['Heart_Rate'] / test_df['BMR'].median())
    
    # 3. Cardiovascular Stress
    train_df['Cardio_Stress'] = (train_df['Heart_Rate'] / (220 - train_df['Age'])) * train_df['Duration']
    test_df['Cardio_Stress'] = (test_df['Heart_Rate'] / (220 - test_df['Age'])) * test_df['Duration']
    
    # 4. Thermic Effect Ratio
    train_df['Thermic_Effect'] = (train_df['Body_Temp'] * 100) / (train_df['Weight'] ** 0.5)
    test_df['Thermic_Effect'] = (test_df['Body_Temp'] * 100) / (test_df['Weight'] ** 0.5)
    
    # 5. Power Output Estimate
    train_df['Power_Output'] = train_df['Weight'] * train_df['Duration'] * (train_df['Heart_Rate'] / 1000)
    test_df['Power_Output'] = test_df['Weight'] * test_df['Duration'] * (test_df['Heart_Rate'] / 1000)
    
    # 6. Body Volume Index
    train_df['BVI'] = train_df['Weight'] / ((train_df['Height']/100) ** 3)
    test_df['BVI'] = test_df['Weight'] / ((test_df['Height']/100) ** 3)
    
    # 7. Age-Adjusted Intensity
    bins = [18, 25, 35, 45, 55, 65, 100]
    train_df['Age_Adj_Intensity'] = train_df['Duration'] * pd.cut(train_df['Age'], bins).cat.codes
    test_df['Age_Adj_Intensity'] = test_df['Duration'] * pd.cut(test_df['Age'], bins).cat.codes
    
    # 8. Gender-Specific Metabolic Rate
    gender_coeff = {'male': 1.67, 'female': 1.55}
    train_df['Gender_Metabolic'] = train_df['Sex'].map(gender_coeff) * train_df['BMR']
    test_df['Gender_Metabolic'] = test_df['Sex'].map(gender_coeff) * test_df['BMR']
    
    # 9. Cardiovascular Drift
    train_df['HR_Drift'] = train_df.groupby('Age')['Heart_Rate'].diff() / train_df['Duration']
    test_df['HR_Drift'] = test_df.groupby('Age')['Heart_Rate'].diff() / test_df['Duration']
    
    # 10. Body Composition Index
    train_df['BCI'] = (train_df['Weight'] * 1000) / (train_df['Height'] ** 1.5) * (1 / (train_df['Age'] ** 0.2))
    test_df['BCI'] = (test_df['Weight'] * 1000) / (test_df['Height'] ** 1.5) * (1 / (test_df['Age'] ** 0.2))
    
    # 11. Thermal Work Capacity
    train_df['Thermal_Work'] = (train_df['Body_Temp'] ** 2) * np.log1p(train_df['Duration'])
    test_df['Thermal_Work'] = (test_df['Body_Temp'] ** 2) * np.log1p(test_df['Duration'])
    
    # 12. Binary Features
    train_df['Temp_Binary'] = np.where(train_df['Body_Temp'] <= 39.5, 0, 1)
    test_df['Temp_Binary'] = np.where(test_df['Body_Temp'] <= 39.5, 0, 1)
    
    train_df['HeartRate_Binary'] = np.where(train_df['Heart_Rate'] <= 99.5, 0, 1)
    test_df['HeartRate_Binary'] = np.where(test_df['Heart_Rate'] <= 99.5, 0, 1)
    
    return train_df, test_df

def transform_features(train_df, test_df, numerical_features):
    """
    Transform features to handle skewness and outliers.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        numerical_features (list): List of numerical feature names
        
    Returns:
        tuple: (transformed_train, transformed_test)
    """
    logger.info("Transforming features...")
    numeric_cols = [col for col in numerical_features if col != "Calories"]
    
    # Calculate original skewness
    original_skewness = train_df[numeric_cols].skew().sort_values(ascending=False)
    
    # Initialize transformed DataFrames
    train_df_transformed = train_df.copy()
    test_df_transformed = test_df.copy()
    
    # Store transformers for each column
    transformers = {}
    
    # Apply skewness correction
    for col in numeric_cols:
        if train_df[col].nunique() <= 1:
            continue
        
        if original_skewness[col] > 0.5:  # Right skew
            if (train_df[col] > 0).all():
                # Log transform
                train_df_transformed[col] = np.log1p(train_df[col])
                test_df_transformed[col] = np.log1p(test_df[col])
            else:
                # Yeo-Johnson transform
                pt = PowerTransformer(method='yeo-johnson')
                train_df_transformed[col] = pt.fit_transform(train_df[[col]])
                test_df_transformed[col] = pt.transform(test_df[[col]])
                transformers[col] = pt
        elif original_skewness[col] < -0.5:  # Left skew
            pt = PowerTransformer(method='yeo-johnson')
            train_df_transformed[col] = pt.fit_transform(train_df[[col]])
            test_df_transformed[col] = pt.transform(test_df[[col]])
            transformers[col] = pt
    
    return train_df_transformed, test_df_transformed

def remove_outliers(df, numeric_cols):
    """
    Remove only the most extreme 1% of outliers using percentile-based method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_cols (list): List of numerical column names
        
    Returns:
        pd.DataFrame: DataFrame with only the most extreme 1% of outliers removed
    """
    logger.info("Removing extreme outliers (top and bottom 0.5%)...")
    df_cleaned = df.copy()
    
    # Calculate total rows to remove (1% of data)
    total_rows = len(df_cleaned)
    rows_to_remove = int(total_rows * 0.01)
    
    # Calculate outlier scores for each row
    outlier_scores = np.zeros(len(df_cleaned))
    
    for col in numeric_cols:
        # Calculate z-scores for each column
        z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
        # Add to total outlier score
        outlier_scores += z_scores
    
    # Get indices of rows with highest outlier scores
    outlier_indices = np.argsort(outlier_scores)[-rows_to_remove:]
    
    # Remove only the most extreme outliers
    df_cleaned = df_cleaned.drop(df_cleaned.index[outlier_indices])
    
    logger.info(f"Removed {rows_to_remove} rows ({rows_to_remove/total_rows*100:.2f}% of data)")
    return df_cleaned

def train_catboost(X, y, X_test):
    """
    Train CatBoost model with cross-validation.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (predictions, out-of-fold predictions)
    """
    logger.info("Starting CatBoost training...")
    print("\n=== Starting CatBoost Training ===")
    start_time = time.time()
    
    # Prepare data
    logger.info("Preparing data for CatBoost...")
    print("Preparing data for CatBoost...")
    
    # Ensure feature alignment
    logger.info("Aligning features between train and test...")
    print("Aligning features between train and test...")
    common_features = list(set(X.columns) & set(X_test.columns))
    X_cat = X[common_features].copy()
    X_test_cat = X_test[common_features].copy()
    
    # Create duration bins for stratified splitting
    bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    duration_bins = bins.fit_transform(X_cat[['Duration']]).astype(int).flatten()

    # CatBoost parameters
    cat_params = {
        'iterations': 2500,
        'learning_rate': 0.02,
        'depth': 10,
        'loss_function': 'RMSE',
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 200,
        'cat_features': ['Sex'],
        'verbose': 100,  # Show progress every 100 iterations
        'task_type': 'GPU',
        'thread_count': -1,  # Use all available threads
        'gpu_ram_part': 0.8,  # Use 80% of GPU memory
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 0.8,
        'random_strength': 0.8,
        'min_data_in_leaf': 20,
        'max_leaves': 31,
        'feature_border_type': 'UniformAndQuantiles',
        'leaf_estimation_iterations': 10,
        'boosting_type': 'Plain',
        'grow_policy': 'Lossguide',
        'max_bin': 256
    }

    # Initialize prediction arrays
    cat_preds = np.zeros(len(X_test_cat))
    cat_oof = np.zeros(len(X_cat))
    cat_scores = []
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    total_folds = skf.n_splits
    
    logger.info(f"Starting {total_folds}-fold cross-validation...")
    print(f"\nStarting {total_folds}-fold cross-validation...")
    logger.info(f"Training with {len(common_features)} features: {', '.join(common_features)}")
    print(f"Training with {len(common_features)} features")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cat, duration_bins), 1):
        fold_start_time = time.time()
        logger.info(f"\nFold {fold}/{total_folds} - Starting training...")
        print(f"\nFold {fold}/{total_folds} - Starting training...")
        
        # Create and train model
        model = CatBoostRegressor(**cat_params)
        
        # Train model with progress logging
        model.fit(
            X_cat.iloc[train_idx], 
            y.iloc[train_idx],
            eval_set=(X_cat.iloc[val_idx], y.iloc[val_idx]),
            use_best_model=True,
            verbose=100
        )
        
        # Make predictions
        logger.info(f"Fold {fold} - Making predictions...")
        print(f"Fold {fold} - Making predictions...")
        cat_oof[val_idx] = model.predict(X_cat.iloc[val_idx])
        cat_preds += model.predict(X_test_cat) / skf.n_splits
        
        # Calculate fold score
        fold_score = np.sqrt(mean_squared_log_error(
            np.expm1(y.iloc[val_idx]), 
            np.expm1(cat_oof[val_idx])
        ))
        
        # Calculate fold timing
        fold_time = time.time() - fold_start_time
        
        logger.info(f"Fold {fold} completed:")
        logger.info(f"  - RMSLE Score: {fold_score:.5f}")
        logger.info(f"  - Time taken: {fold_time:.2f} seconds")
        print(f"\nFold {fold} completed:")
        print(f"  - RMSLE Score: {fold_score:.5f}")
        print(f"  - Time taken: {fold_time:.2f} seconds")
        
        cat_scores.append(fold_score)
        
        # Estimate remaining time
        if fold < total_folds:
            avg_fold_time = (time.time() - start_time) / fold
            remaining_folds = total_folds - fold
            estimated_time = avg_fold_time * remaining_folds
            logger.info(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
            print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
    
    # Calculate and log final metrics
    total_time = time.time() - start_time
    mean_score = np.mean(cat_scores)
    std_score = np.std(cat_scores)
    
    logger.info("\nCatBoost Training Summary:")
    logger.info(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    logger.info(f"  - Total training time: {total_time/60:.1f} minutes")
    logger.info(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    print("\n=== CatBoost Training Summary ===")
    print(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    print(f"  - Total training time: {total_time/60:.1f} minutes")
    print(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    
    return cat_preds, cat_oof

def train_xgboost(X, y, X_test):
    """
    Train XGBoost model with cross-validation.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (predictions, out-of-fold predictions)
    """
    logger.info("Starting XGBoost training...")
    print("\n=== Starting XGBoost Training ===")
    start_time = time.time()
    
    # Prepare data
    logger.info("Preparing data for XGBoost...")
    print("Preparing data for XGBoost...")
    
    # Ensure feature alignment
    logger.info("Aligning features between train and test...")
    print("Aligning features between train and test...")
    common_features = list(set(X.columns) & set(X_test.columns))
    X_xgb = X[common_features].copy()
    X_test_xgb = X_test[common_features].copy()
    
    # Convert categorical features
    X_xgb['Sex'] = X_xgb['Sex'].astype(int)
    X_test_xgb['Sex'] = X_test_xgb['Sex'].astype(int)

    # Initialize prediction arrays
    xgb_oof = np.zeros(len(X))
    xgb_preds = np.zeros(len(X_test))
    xgb_scores = []

    # XGBoost parameters
    xgb_params = {
        'max_depth': 9,
        'colsample_bytree': 0.7,
        'subsample': 0.9,
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'gamma': 0.01,
        'max_delta_step': 2,
        'eval_metric': 'rmse',
        'enable_categorical': False,
        'random_state': 42,
        'early_stopping_rounds': 100,
        'tree_method': 'gpu_hist',
        'n_jobs': -1,  # Use all available CPU cores
        'gpu_id': 0,   # Use first GPU
        'predictor': 'gpu_predictor',
        'sampling_method': 'gradient_based',
        'max_bin': 256,
        'grow_policy': 'lossguide'
    }

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    total_folds = kf.n_splits
    
    logger.info(f"Starting {total_folds}-fold cross-validation...")
    print(f"\nStarting {total_folds}-fold cross-validation...")
    logger.info(f"Training with {len(common_features)} features: {', '.join(common_features)}")
    print(f"Training with {len(common_features)} features")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_xgb), 1):
        fold_start_time = time.time()
        logger.info(f"\nFold {fold}/{total_folds} - Starting training...")
        print(f"\nFold {fold}/{total_folds} - Starting training...")
        
        # Create and train model
        model = XGBRegressor(**xgb_params)
        
        # Train model with progress logging
        model.fit(
            X_xgb.iloc[train_idx], 
            y.iloc[train_idx],
            eval_set=[(X_xgb.iloc[val_idx], y.iloc[val_idx])],
            verbose=100  # Show progress every 100 iterations
        )
        
        # Make predictions
        logger.info(f"Fold {fold} - Making predictions...")
        print(f"Fold {fold} - Making predictions...")
        xgb_oof[val_idx] = model.predict(X_xgb.iloc[val_idx])
        xgb_preds += model.predict(X_test_xgb) / kf.n_splits
        
        # Calculate fold score
        fold_score = np.sqrt(mean_squared_log_error(
            np.expm1(y.iloc[val_idx]), 
            np.expm1(xgb_oof[val_idx])
        ))
        
        # Calculate fold timing
        fold_time = time.time() - fold_start_time
        
        logger.info(f"Fold {fold} completed:")
        logger.info(f"  - RMSLE Score: {fold_score:.5f}")
        logger.info(f"  - Time taken: {fold_time:.2f} seconds")
        print(f"\nFold {fold} completed:")
        print(f"  - RMSLE Score: {fold_score:.5f}")
        print(f"  - Time taken: {fold_time:.2f} seconds")
        
        xgb_scores.append(fold_score)
        
        # Estimate remaining time
        if fold < total_folds:
            avg_fold_time = (time.time() - start_time) / fold
            remaining_folds = total_folds - fold
            estimated_time = avg_fold_time * remaining_folds
            logger.info(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
            print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
    
    # Calculate and log final metrics
    total_time = time.time() - start_time
    mean_score = np.mean(xgb_scores)
    std_score = np.std(xgb_scores)
    
    logger.info("\nXGBoost Training Summary:")
    logger.info(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    logger.info(f"  - Total training time: {total_time/60:.1f} minutes")
    logger.info(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    print("\n=== XGBoost Training Summary ===")
    print(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    print(f"  - Total training time: {total_time/60:.1f} minutes")
    print(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    
    return xgb_preds, xgb_oof

class CalorieDataset(Dataset):
    """Custom Dataset for loading calorie prediction data."""
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class CalorieNet(nn.Module):
    """Neural Network for calorie prediction."""
    def __init__(self, input_dim):
        super(CalorieNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def train_neural_network(X, y, X_test):
    """
    Train Neural Network model with cross-validation.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (predictions, out-of-fold predictions)
    """
    logger.info("Starting Neural Network training...")
    print("\n=== Starting Neural Network Training ===")
    start_time = time.time()
    
    # Prepare data
    logger.info("Preparing data for Neural Network...")
    print("Preparing data for Neural Network...")
    
    # Ensure feature alignment
    logger.info("Aligning features between train and test...")
    print("Aligning features between train and test...")
    common_features = list(set(X.columns) & set(X_test.columns))
    X_nn = X[common_features].copy()
    X_test_nn = X_test[common_features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_nn)
    X_test_scaled = scaler.transform(X_test_nn)
    
    # Convert target to numpy array
    y_np = y.values
    
    # Initialize prediction arrays
    nn_preds = np.zeros(len(X_test_nn))
    nn_oof = np.zeros(len(X_nn))
    nn_scores = []
    
    # Training parameters
    BATCH_SIZE = 256
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    total_folds = kf.n_splits
    
    logger.info(f"Starting {total_folds}-fold cross-validation...")
    print(f"\nStarting {total_folds}-fold cross-validation...")
    logger.info(f"Training with {len(common_features)} features: {', '.join(common_features)}")
    print(f"Training with {len(common_features)} features")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    print(f"Using device: {device}")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        fold_start_time = time.time()
        logger.info(f"\nFold {fold}/{total_folds} - Starting training...")
        print(f"\nFold {fold}/{total_folds} - Starting training...")
        
        # Create data loaders
        train_dataset = CalorieDataset(X_scaled[train_idx], y_np[train_idx])
        val_dataset = CalorieDataset(X_scaled[val_idx], y_np[val_idx])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = CalorieNet(input_dim=X_scaled.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs.squeeze(), batch_y).item()
                    val_preds.extend(outputs.cpu().numpy())
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model predictions
                nn_oof[val_idx] = np.array(val_preds).flatten()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f}")
                print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Make predictions on test set
        test_dataset = CalorieDataset(X_test_scaled)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model.eval()
        test_preds = []
        with torch.no_grad():
            for batch_X in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                test_preds.extend(outputs.cpu().numpy())
        
        nn_preds += np.array(test_preds).flatten() / kf.n_splits
        
        # Calculate fold score
        fold_score = np.sqrt(mean_squared_log_error(
            np.expm1(y_np[val_idx]), 
            np.expm1(nn_oof[val_idx])
        ))
        
        # Calculate fold timing
        fold_time = time.time() - fold_start_time
        
        logger.info(f"Fold {fold} completed:")
        logger.info(f"  - RMSLE Score: {fold_score:.5f}")
        logger.info(f"  - Time taken: {fold_time:.2f} seconds")
        print(f"\nFold {fold} completed:")
        print(f"  - RMSLE Score: {fold_score:.5f}")
        print(f"  - Time taken: {fold_time:.2f} seconds")
        
        nn_scores.append(fold_score)
        
        # Estimate remaining time
        if fold < total_folds:
            avg_fold_time = (time.time() - start_time) / fold
            remaining_folds = total_folds - fold
            estimated_time = avg_fold_time * remaining_folds
            logger.info(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
            print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
    
    # Calculate and log final metrics
    total_time = time.time() - start_time
    mean_score = np.mean(nn_scores)
    std_score = np.std(nn_scores)
    
    logger.info("\nNeural Network Training Summary:")
    logger.info(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    logger.info(f"  - Total training time: {total_time/60:.1f} minutes")
    logger.info(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    print("\n=== Neural Network Training Summary ===")
    print(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    print(f"  - Total training time: {total_time/60:.1f} minutes")
    print(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    
    return nn_preds, nn_oof

def main():
    """Main execution function."""
    # Define numerical features
    numerical_features = [
        "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp",
        "Calories", "BMR", 'Metabolic_Efficiency', 'Cardio_Stress',
        'Thermic_Effect', 'Power_Output', 'BVI', 'Age_Adj_Intensity',
        'Gender_Metabolic', 'HR_Drift', 'BCI', 'Thermal_Work',
        'Temp_Binary', 'HeartRate_Binary', 'Sex'
    ]
    
    # Load and preprocess data
    train_df, test_df, submission_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Feature engineering
    train_df = add_features(train_df, train_df)
    test_df = add_features(test_df, train_df)
    train_df, test_df = add_advanced_features(train_df, test_df)
    
    # Transform features
    train_df_transformed, test_df_transformed = transform_features(
        train_df, test_df, numerical_features
    )
    
    # Remove outliers
    cleaned_train_df = remove_outliers(train_df_transformed, numerical_features)
    cleaned_test_df = test_df_transformed.copy()
    
    # Prepare data for modeling
    X = cleaned_train_df.drop(columns=['Calories', 'id'])
    y = np.log1p(cleaned_train_df['Calories'])
    X_test = cleaned_test_df.drop(columns=['id'])
    
    # Train models
    cat_preds, cat_oof = train_catboost(X, y, X_test)
    xgb_preds, xgb_oof = train_xgboost(X, y, X_test)
    nn_preds, nn_oof = train_neural_network(X, y, X_test)
    
    # Create submission with all three models
    final_preds = 0.35 * np.expm1(xgb_preds) + 0.35 * np.expm1(cat_preds) + 0.30 * np.expm1(nn_preds)
    submission_df['Calories'] = np.clip(final_preds, 1, 314)
    submission_df.to_csv('submission.csv', index=False)
    
    # Plot predictions
    plot_predictions(cat_oof, xgb_oof, nn_oof)
    
    # Print final statistics
    logger.info("\nFinal Submission Preview:")
    logger.info(submission_df.describe())
    logger.info("\nFirst few predictions:")
    logger.info(submission_df.head())

def plot_predictions(cat_oof, xgb_oof, nn_oof):
    """
    Plot prediction distributions.
    
    Args:
        cat_oof (np.array): CatBoost out-of-fold predictions
        xgb_oof (np.array): XGBoost out-of-fold predictions
        nn_oof (np.array): Neural Network out-of-fold predictions
    """
    plt.figure(figsize=(12, 6))
    plt.hist(np.expm1(cat_oof), bins=50, alpha=0.4, label='CatBoost OOF')
    plt.hist(np.expm1(xgb_oof), bins=50, alpha=0.4, label='XGBoost OOF')
    plt.hist(np.expm1(nn_oof), bins=50, alpha=0.4, label='Neural Network OOF')
    plt.title("OOF Prediction Distribution")
    plt.xlabel("Calories")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main() 