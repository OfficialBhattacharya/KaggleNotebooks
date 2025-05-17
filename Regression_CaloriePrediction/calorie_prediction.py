"""
Calorie Prediction Model

This script implements a machine learning solution for predicting calories burned during exercise
based on various physiological and activity parameters. It uses both CatBoost and XGBoost models
and blends their predictions for the final output.

Author: [Diganta]
Date: [Current Date]
"""

# Standard library imports
import os
import time
import logging
import warnings
from datetime import datetime
from collections import defaultdict
from itertools import combinations

# Third-party imports for data manipulation and analysis
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import boxcox, skew, spearmanr, pearsonr
from scipy.interpolate import LSQUnivariateSpline
from scipy.io import arff
from minepy import MINE
import ppscore as pps
import Levenshtein

# Machine learning imports
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, 
    GridSearchCV, cross_val_score
)
from sklearn.preprocessing import (
    LabelEncoder, StandardScaler, MinMaxScaler,
    KBinsDiscretizer, PowerTransformer, RobustScaler
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    mean_squared_error, accuracy_score, roc_auc_score,
    roc_curve, mean_squared_log_error
)
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Model-specific imports
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import statsmodels.api as sm

# Bayesian network imports
import pgmpy.estimators as ests
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
from pgmpy.metrics import structure_score
from pgmpy.inference import BeliefPropagation, VariableElimination

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Utility imports
import joblib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def create_features(train_df, test_df):
    """
    Comprehensive feature engineering function that combines all feature creation steps
    and adds polynomial features.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        
    Returns:
        tuple: (processed_train_df, processed_test_df)
    """
    logger.info("Starting comprehensive feature engineering...")
    print("\n=== Starting Feature Engineering ===")
    
    # Create copies to avoid modifying original dataframes
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # 1. Basic Preprocessing
    logger.info("Performing basic preprocessing...")
    print("Performing basic preprocessing...")
    
    # Convert sex to binary
    train_processed['Sex'] = train_processed['Sex'].map({'male': 1, 'female': 0})
    test_processed['Sex'] = test_processed['Sex'].map({'male': 1, 'female': 0})
    
    # Remove duplicates and get minimum calories for same features
    train_processed = train_processed.drop_duplicates(subset=train_processed.columns).reset_index(drop=True)
    train_processed = train_processed.groupby(['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])['Calories'].min().reset_index()
    
    # 2. Basic Features
    logger.info("Creating basic features...")
    print("Creating basic features...")
    
    # BMI and Intensity
    train_processed['BMI'] = train_processed['Weight'] / (train_processed['Height'] / 100) ** 2
    test_processed['BMI'] = test_processed['Weight'] / (test_processed['Height'] / 100) ** 2
    
    train_processed['Intensity'] = train_processed['Heart_Rate'] / train_processed['Duration']
    test_processed['Intensity'] = test_processed['Heart_Rate'] / test_processed['Duration']
    
    # 3. Polynomial Features
    logger.info("Creating polynomial features...")
    print("Creating polynomial features...")
    
    # Define features for polynomial combinations
    poly_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']
    
    # Create polynomial features
    for i in range(len(poly_features)):
        for j in range(i+1, len(poly_features)):
            feat1, feat2 = poly_features[i], poly_features[j]
            
            # Multiplication
            train_processed[f'{feat1}_x_{feat2}'] = train_processed[feat1] * train_processed[feat2]
            test_processed[f'{feat1}_x_{feat2}'] = test_processed[feat1] * test_processed[feat2]
            
            # Division (avoid division by zero)
            train_processed[f'{feat1}_div_{feat2}'] = train_processed[feat1] / (train_processed[feat2] + 1e-6)
            test_processed[f'{feat1}_div_{feat2}'] = test_processed[feat1] / (test_processed[feat2] + 1e-6)
            
            # Square of each feature
            train_processed[f'{feat1}_squared'] = train_processed[feat1] ** 2
            test_processed[f'{feat1}_squared'] = test_processed[feat1] ** 2
    
    # 4. Advanced Physiological Features
    logger.info("Creating advanced physiological features...")
    print("Creating advanced physiological features...")
    
    # Basal Metabolic Rate (BMR)
    train_processed['BMR'] = train_processed['Weight'] / ((train_processed['Height'] / 100) ** 2)
    test_processed['BMR'] = test_processed['Weight'] / ((test_processed['Height'] / 100) ** 2)
    
    # Metabolic Efficiency Index
    train_processed['Metabolic_Efficiency'] = train_processed['BMR'] * (train_processed['Heart_Rate'] / train_processed['BMR'].median())
    test_processed['Metabolic_Efficiency'] = test_processed['BMR'] * (test_processed['Heart_Rate'] / test_processed['BMR'].median())
    
    # Cardiovascular Stress
    train_processed['Cardio_Stress'] = (train_processed['Heart_Rate'] / (220 - train_processed['Age'])) * train_processed['Duration']
    test_processed['Cardio_Stress'] = (test_processed['Heart_Rate'] / (220 - test_processed['Age'])) * test_processed['Duration']
    
    # Thermic Effect Ratio
    train_processed['Thermic_Effect'] = (train_processed['Body_Temp'] * 100) / (train_processed['Weight'] ** 0.5)
    test_processed['Thermic_Effect'] = (test_processed['Body_Temp'] * 100) / (test_processed['Weight'] ** 0.5)
    
    # Power Output Estimate
    train_processed['Power_Output'] = train_processed['Weight'] * train_processed['Duration'] * (train_processed['Heart_Rate'] / 1000)
    test_processed['Power_Output'] = test_processed['Weight'] * test_processed['Duration'] * (test_processed['Heart_Rate'] / 1000)
    
    # 5. Interaction Features
    logger.info("Creating interaction features...")
    print("Creating interaction features...")
    
    # Duration-based features
    train_durations = sorted(train_processed['Duration'].unique())
    for dur in train_durations:
        train_processed[f'HR_Dur_{int(dur)}'] = np.where(train_processed['Duration'] == dur, train_processed['Heart_Rate'], 0)
        test_processed[f'HR_Dur_{int(dur)}'] = np.where(test_processed['Duration'] == dur, test_processed['Heart_Rate'], 0)
        
        train_processed[f'Temp_Dur_{int(dur)}'] = np.where(train_processed['Duration'] == dur, train_processed['Body_Temp'], 0)
        test_processed[f'Temp_Dur_{int(dur)}'] = np.where(test_processed['Duration'] == dur, test_processed['Body_Temp'], 0)
    
    # Age-based features
    train_ages = sorted(train_processed['Age'].unique())
    for age in train_ages:
        train_processed[f'HR_Age_{int(age)}'] = np.where(train_processed['Age'] == age, train_processed['Heart_Rate'], 0)
        test_processed[f'HR_Age_{int(age)}'] = np.where(test_processed['Age'] == age, test_processed['Heart_Rate'], 0)
        
        train_processed[f'Temp_Age_{int(age)}'] = np.where(train_processed['Age'] == age, train_processed['Body_Temp'], 0)
        test_processed[f'Temp_Age_{int(age)}'] = np.where(test_processed['Age'] == age, test_processed['Body_Temp'], 0)
    
    # 6. Statistical Features
    logger.info("Creating statistical features...")
    print("Creating statistical features...")
    
    for col in ['Height', 'Weight', 'Heart_Rate', 'Body_Temp']:
        for agg in ['min', 'max']:
            agg_val = train_processed.groupby('Sex')[col].agg(agg).rename(f'Sex_{col}_{agg}')
            train_processed = train_processed.merge(agg_val, on='Sex', how='left')
            test_processed = test_processed.merge(agg_val, on='Sex', how='left')
    
    # 7. Additional Derived Features
    logger.info("Creating additional derived features...")
    print("Creating additional derived features...")
    
    # Body Volume Index
    train_processed['BVI'] = train_processed['Weight'] / ((train_processed['Height']/100) ** 3)
    test_processed['BVI'] = test_processed['Weight'] / ((test_processed['Height']/100) ** 3)
    
    # Age-Adjusted Intensity
    bins = [18, 25, 35, 45, 55, 65, 100]
    train_processed['Age_Adj_Intensity'] = train_processed['Duration'] * pd.cut(train_processed['Age'], bins).cat.codes
    test_processed['Age_Adj_Intensity'] = test_processed['Duration'] * pd.cut(test_processed['Age'], bins).cat.codes
    
    # Gender-Specific Metabolic Rate
    gender_coeff = {'male': 1.67, 'female': 1.55}
    train_processed['Gender_Metabolic'] = train_processed['Sex'].map(gender_coeff) * train_processed['BMR']
    test_processed['Gender_Metabolic'] = test_processed['Sex'].map(gender_coeff) * test_processed['BMR']
    
    # Cardiovascular Drift
    train_processed['HR_Drift'] = train_processed.groupby('Age')['Heart_Rate'].diff() / train_processed['Duration']
    test_processed['HR_Drift'] = test_processed.groupby('Age')['Heart_Rate'].diff() / test_processed['Duration']
    
    # Body Composition Index
    train_processed['BCI'] = (train_processed['Weight'] * 1000) / (train_processed['Height'] ** 1.5) * (1 / (train_processed['Age'] ** 0.2))
    test_processed['BCI'] = (test_processed['Weight'] * 1000) / (test_processed['Height'] ** 1.5) * (1 / (test_processed['Age'] ** 0.2))
    
    # Thermal Work Capacity
    train_processed['Thermal_Work'] = (train_processed['Body_Temp'] ** 2) * np.log1p(train_processed['Duration'])
    test_processed['Thermal_Work'] = (test_processed['Body_Temp'] ** 2) * np.log1p(test_processed['Duration'])
    
    # Binary Features
    train_processed['Temp_Binary'] = np.where(train_processed['Body_Temp'] <= 39.5, 0, 1)
    test_processed['Temp_Binary'] = np.where(test_processed['Body_Temp'] <= 39.5, 0, 1)
    
    train_processed['HeartRate_Binary'] = np.where(train_processed['Heart_Rate'] <= 99.5, 0, 1)
    test_processed['HeartRate_Binary'] = np.where(test_processed['Heart_Rate'] <= 99.5, 0, 1)
    
    # Log feature creation summary
    logger.info(f"Created {len(train_processed.columns)} features")
    print(f"Created {len(train_processed.columns)} features")
    
    return train_processed, test_processed

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
        tuple: (predictions, out-of-fold predictions, scores, model, total_time, params)
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
    
    return cat_preds, cat_oof, cat_scores, model, total_time, cat_params

def train_xgboost_parallel(X, y, X_test):
    """
    Train XGBoost model with parallel cross-validation across multiple GPUs.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (predictions, out-of-fold predictions, scores, model, total_time)
    """
    logger.info("Starting parallel XGBoost training...")
    print("\n=== Starting Parallel XGBoost Training ===")
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

    # Base XGBoost parameters
    base_params = {
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
        'predictor': 'gpu_predictor',
        'sampling_method': 'gradient_based',
        'max_bin': 256,
        'grow_policy': 'lossguide'
    }

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    total_folds = kf.n_splits
    
    logger.info(f"Starting {total_folds}-fold cross-validation in parallel...")
    print(f"\nStarting {total_folds}-fold cross-validation in parallel...")
    logger.info(f"Training with {len(common_features)} features: {', '.join(common_features)}")
    print(f"Training with {len(common_features)} features")

    def train_fold(fold_idx, train_idx, val_idx, gpu_id):
        """Train a single fold on specified GPU"""
        fold_start_time = time.time()
        logger.info(f"\nFold {fold_idx}/{total_folds} - Starting training on GPU {gpu_id}...")
        print(f"\nFold {fold_idx}/{total_folds} - Starting training on GPU {gpu_id}...")
        
        # Create fold-specific parameters
        fold_params = base_params.copy()
        fold_params['gpu_id'] = gpu_id
        
        # Create and train model
        model = XGBRegressor(**fold_params)
        
        # Train model with progress logging
        model.fit(
            X_xgb.iloc[train_idx], 
            y.iloc[train_idx],
            eval_set=[(X_xgb.iloc[val_idx], y.iloc[val_idx])],
            verbose=100
        )
        
        # Make predictions
        logger.info(f"Fold {fold_idx} - Making predictions...")
        print(f"Fold {fold_idx} - Making predictions...")
        fold_oof = model.predict(X_xgb.iloc[val_idx])
        fold_test_preds = model.predict(X_test_xgb)
        
        # Calculate fold score
        fold_score = np.sqrt(mean_squared_log_error(
            np.expm1(y.iloc[val_idx]), 
            np.expm1(fold_oof)
        ))
        
        # Calculate fold timing
        fold_time = time.time() - fold_start_time
        
        logger.info(f"Fold {fold_idx} completed on GPU {gpu_id}:")
        logger.info(f"  - RMSLE Score: {fold_score:.5f}")
        logger.info(f"  - Time taken: {fold_time:.2f} seconds")
        print(f"\nFold {fold_idx} completed on GPU {gpu_id}:")
        print(f"  - RMSLE Score: {fold_score:.5f}")
        print(f"  - Time taken: {fold_time:.2f} seconds")
        
        return fold_idx, fold_oof, fold_test_preds, fold_score, model, fold_time

    # Get number of available GPUs
    n_gpus = torch.cuda.device_count()
    logger.info(f"Found {n_gpus} GPUs")
    print(f"Found {n_gpus} GPUs")
    
    # Create process pool for parallel execution
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        # Submit all folds for parallel execution
        future_to_fold = {
            executor.submit(
                train_fold, 
                fold_idx + 1, 
                train_idx, 
                val_idx, 
                fold_idx % n_gpus
            ): fold_idx 
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_xgb))
        }
        
        # Collect results as they complete
        fold_times = []
        for future in as_completed(future_to_fold):
            fold_idx, fold_oof, fold_test_preds, fold_score, model, fold_time = future.result()
            xgb_oof[kf.split(X_xgb)[fold_idx][1]] = fold_oof
            xgb_preds += fold_test_preds / total_folds
            xgb_scores.append(fold_score)
            fold_times.append(fold_time)
            
            # Estimate remaining time
            if len(fold_times) < total_folds:
                avg_fold_time = sum(fold_times) / len(fold_times)
                remaining_folds = total_folds - len(fold_times)
                estimated_time = avg_fold_time * remaining_folds
                logger.info(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
                print(f"Estimated time remaining: {estimated_time/60:.1f} minutes")
    
    # Calculate and log final metrics
    total_time = time.time() - start_time
    mean_score = np.mean(xgb_scores)
    std_score = np.std(xgb_scores)
    
    logger.info("\nParallel XGBoost Training Summary:")
    logger.info(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    logger.info(f"  - Total training time: {total_time/60:.1f} minutes")
    logger.info(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    print("\n=== Parallel XGBoost Training Summary ===")
    print(f"  - Mean RMSLE: {mean_score:.5f} ± {std_score:.5f}")
    print(f"  - Total training time: {total_time/60:.1f} minutes")
    print(f"  - Average fold time: {total_time/total_folds:.1f} seconds")
    
    return xgb_preds, xgb_oof, xgb_scores, model, total_time

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

class ResBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        return F.gelu(self.block(x) + self.shortcut(x))

class CalorieNet(nn.Module):
    """Enhanced Neural Network for calorie prediction."""
    def __init__(self, input_dim):
        super(CalorieNet, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.5),
            
            ResBlock(1024, 512),
            ResBlock(512, 256),
            ResBlock(256, 128),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.base(x)
        return self.head(x)

def train_neural_network(X, y, X_test):
    """
    Train Neural Network model with cross-validation.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (predictions, out-of-fold predictions, scores, model, total_time)
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
    
    # Scale features using RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_nn)
    X_test_scaled = scaler.transform(X_test_nn)
    
    # Convert target to numpy array and ensure it's positive
    y_np = np.clip(y.values, 0, None)
    
    # Initialize prediction arrays
    nn_preds = np.zeros(len(X_test_nn))
    nn_oof = np.zeros(len(X_nn))
    nn_scores = []
    
    # Enhanced training parameters
    BATCH_SIZE = 32  # Smaller batch size for better stability
    EPOCHS = 300     # More epochs with better early stopping
    LEARNING_RATE = 5e-4  # Lower learning rate for stability
    WEIGHT_DECAY = 1e-6   # Reduced weight decay
    PATIENCE = 25         # Increased patience
    MIN_DELTA = 1e-4      # Minimum change in validation loss
    
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
        
        # Enhanced model configuration
        model = CalorieNet(input_dim=X_scaled.shape[1]).to(device)
        criterion = nn.SmoothL1Loss()  # Using Huber loss for better stability
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                               weight_decay=WEIGHT_DECAY, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, 
            min_lr=1e-6, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            train_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                # Ensure outputs are valid
                outputs = torch.clamp(outputs.squeeze(), min=0.0)
                
                # Calculate loss
                loss = criterion(outputs, batch_y)
                
                # Add L2 regularization manually
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += 1e-6 * l2_reg  # Reduced L2 regularization
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                train_batches += 1
            
            # Calculate average training loss
            avg_train_loss = train_loss / train_batches
            
            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            val_preds = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    outputs = torch.clamp(outputs.squeeze(), min=0.0)
                    val_loss += criterion(outputs, batch_y).item()
                    val_preds.extend(outputs.cpu().numpy())
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            scheduler.step(avg_val_loss)
            
            # Enhanced early stopping with dynamic patience
            if avg_val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                nn_oof[val_idx] = np.array(val_preds).flatten()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
                print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Make predictions on test set
        test_dataset = CalorieDataset(X_test_scaled)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model.eval()
        test_preds = []
        with torch.no_grad():
            for batch_X in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                outputs = torch.clamp(outputs.squeeze(), min=0.0)
                test_preds.extend(outputs.cpu().numpy())
        
        # Ensure predictions are positive and within bounds
        test_preds = np.clip(np.array(test_preds).flatten(), 0, 314)
        nn_preds += test_preds / kf.n_splits
        
        # Calculate fold score using RMSLE
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
    
    return nn_preds, nn_oof, nn_scores, model, total_time

def save_model_and_metrics(model, model_name, metrics, model_dir='saved_models'):
    """
    Save model and its metrics to disk.
    
    Args:
        model: Trained model object
        model_name (str): Name of the model (e.g., 'catboost', 'xgboost', 'neural_net')
        metrics (dict): Dictionary containing model metrics
        model_dir (str): Directory to save models
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate timestamp for unique model version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.joblib"
    model_path = os.path.join(model_dir, model_filename)
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_filename = f"{model_name}_{timestamp}_metrics.joblib"
    metrics_path = os.path.join(model_dir, metrics_filename)
    joblib.dump(metrics, metrics_path)
    
    logger.info(f"Saved model and metrics to {model_dir}")
    print(f"Saved model and metrics to {model_dir}")
    
    return model_path, metrics_path

def load_latest_model(model_name, model_dir='saved_models'):
    """
    Load the latest saved model and its metrics.
    
    Args:
        model_name (str): Name of the model to load
        model_dir (str): Directory containing saved models
        
    Returns:
        tuple: (model, metrics, model_path, metrics_path)
    """
    # Get all files for the model
    model_files = [f for f in os.listdir(model_dir) if f.startswith(model_name) and f.endswith('.joblib')]
    if not model_files:
        return None, None, None, None
    
    # Get latest model file
    latest_model_file = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model_file)
    
    # Get corresponding metrics file
    metrics_file = latest_model_file.replace('.joblib', '_metrics.joblib')
    metrics_path = os.path.join(model_dir, metrics_file)
    
    # Load model and metrics
    model = joblib.load(model_path)
    metrics = joblib.load(metrics_path)
    
    logger.info(f"Loaded model from {model_path}")
    print(f"Loaded model from {model_path}")
    
    return model, metrics, model_path, metrics_path

def create_model_comparison_df(model_dir='saved_models'):
    """
    Create a DataFrame comparing all saved models.
    
    Args:
        model_dir (str): Directory containing saved models
        
    Returns:
        pd.DataFrame: Comparison DataFrame
    """
    comparison_data = []
    
    # Get all metrics files
    metrics_files = [f for f in os.listdir(model_dir) if f.endswith('_metrics.joblib')]
    
    for metrics_file in metrics_files:
        metrics_path = os.path.join(model_dir, metrics_file)
        metrics = joblib.load(metrics_path)
        
        # Extract model name and timestamp
        model_name = metrics_file.split('_')[0]
        timestamp = '_'.join(metrics_file.split('_')[1:-1])
        
        # Add to comparison data
        comparison_data.append({
            'model_name': model_name,
            'timestamp': timestamp,
            **metrics
        })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['timestamp'] = pd.to_datetime(comparison_df['timestamp'], format='%Y%m%d_%H%M%S')
    comparison_df = comparison_df.sort_values('timestamp', ascending=False)
    
    return comparison_df

def warm_start_training(X, y, X_test, model_name, model_dir='saved_models'):
    """
    Warm start training using the latest saved model.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        model_name (str): Name of the model to warm start
        model_dir (str): Directory containing saved models
        
    Returns:
        tuple: (predictions, out-of-fold predictions, metrics)
    """
    # Load latest model
    model, old_metrics, model_path, metrics_path = load_latest_model(model_name, model_dir)
    
    if model is None:
        logger.info(f"No saved model found for {model_name}, starting fresh training")
        print(f"No saved model found for {model_name}, starting fresh training")
        return None, None, None
    
    logger.info(f"Warm starting {model_name} from {model_path}")
    print(f"Warm starting {model_name} from {model_path}")
    
    # Train model based on type
    if model_name == 'catboost':
        preds, oof, scores, model, total_time, params = train_catboost(X, y, X_test)
    elif model_name == 'xgboost':
        preds, oof, scores, model, total_time = train_xgboost_parallel(X, y, X_test)
    elif model_name == 'neural_net':
        preds, oof, scores, model, total_time = train_neural_network(X, y, X_test)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Calculate metrics
    metrics = {
        'train_rmsle': old_metrics.get('train_rmsle', 0),
        'val_rmsle': old_metrics.get('val_rmsle', 0),
        'test_rmsle': None,  # We don't have true test values
        'improvement': old_metrics.get('improvement', 0),
        'training_time': old_metrics.get('training_time', 0),
        'model_specs': old_metrics.get('model_specs', {})
    }
    
    return preds, oof, metrics

def update_model_comparison(model_name, new_metrics, model_dir='saved_models'):
    """
    Update the model comparison DataFrame with new metrics.
    
    Args:
        model_name (str): Name of the model
        new_metrics (dict): New metrics to add
        model_dir (str): Directory containing saved models
        
    Returns:
        pd.DataFrame: Updated comparison DataFrame
    """
    # Load existing comparison
    comparison_df = create_model_comparison_df(model_dir)
    
    # Add new metrics
    new_row = {
        'model_name': model_name,
        'timestamp': datetime.now(),
        **new_metrics
    }
    
    comparison_df = pd.concat([pd.DataFrame([new_row]), comparison_df], ignore_index=True)
    
    # Save updated comparison
    comparison_path = os.path.join(model_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    return comparison_df

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
    
    # Load data
    train_df, test_df, submission_df = load_data()
    
    # Create features
    train_processed, test_processed = create_features(train_df, test_df)
    
    # Transform features
    train_df_transformed, test_df_transformed = transform_features(
        train_processed, test_processed, numerical_features
    )
    
    # Remove outliers
    cleaned_train_df = remove_outliers(train_df_transformed, numerical_features)
    cleaned_test_df = test_df_transformed.copy()
    
    # Prepare data for modeling
    X = cleaned_train_df.drop(columns=['Calories', 'id'])
    y = np.log1p(cleaned_train_df['Calories'])
    X_test = cleaned_test_df.drop(columns=['id'])
    
    # Train models and save them
    cat_preds, cat_oof, cat_scores, cat_model, cat_total_time, cat_params = train_catboost(X, y, X_test)
    cat_metrics = {
        'train_rmsle': np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(cat_oof))),
        'val_rmsle': np.mean(cat_scores),
        'test_rmsle': None,  # We don't have true test values
        'improvement': 0,  # Will be updated in future runs
        'training_time': cat_total_time,
        'model_specs': cat_params
    }
    save_model_and_metrics(cat_model, 'catboost', cat_metrics)
    
    xgb_preds, xgb_oof, xgb_scores, xgb_model, xgb_total_time = train_xgboost_parallel(X, y, X_test)
    xgb_metrics = {
        'train_rmsle': np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(xgb_oof))),
        'val_rmsle': np.mean(xgb_scores),
        'test_rmsle': None,  # We don't have true test values
        'improvement': 0,  # Will be updated in future runs
        'training_time': xgb_total_time,
        'model_specs': base_params
    }
    save_model_and_metrics(xgb_model, 'xgboost', xgb_metrics)
    
    nn_preds, nn_oof, nn_scores, nn_model, nn_total_time = train_neural_network(X, y, X_test)
    nn_metrics = {
        'train_rmsle': np.sqrt(mean_squared_log_error(np.expm1(y), np.expm1(nn_oof))),
        'val_rmsle': np.mean(nn_scores),
        'test_rmsle': None,  # We don't have true test values
        'improvement': 0,
        'training_time': nn_total_time,
        'model_specs': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY
        }
    }
    save_model_and_metrics(nn_model, 'neural_net', nn_metrics)
    
    # Create and display model comparison
    comparison_df = create_model_comparison_df()
    logger.info("\nModel Comparison:")
    logger.info(comparison_df)
    print("\nModel Comparison:")
    print(comparison_df)
    
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
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Distribution of predictions
    ax1.hist(np.expm1(cat_oof), bins=50, alpha=0.4, label='CatBoost OOF', density=True)
    ax1.hist(np.expm1(xgb_oof), bins=50, alpha=0.4, label='XGBoost OOF', density=True)
    ax1.hist(np.expm1(nn_oof), bins=50, alpha=0.4, label='Neural Network OOF', density=True)
    ax1.set_title("OOF Prediction Distribution")
    ax1.set_xlabel("Calories")
    ax1.set_ylabel("Density")
    ax1.legend()
    
    # Plot 2: Scatter plot of predictions
    ax2.scatter(np.expm1(cat_oof), np.expm1(xgb_oof), alpha=0.5, label='CatBoost vs XGBoost')
    ax2.scatter(np.expm1(cat_oof), np.expm1(nn_oof), alpha=0.5, label='CatBoost vs Neural Network')
    ax2.scatter(np.expm1(xgb_oof), np.expm1(nn_oof), alpha=0.5, label='XGBoost vs Neural Network')
    
    # Add diagonal line
    min_val = min(np.min(np.expm1(cat_oof)), np.min(np.expm1(xgb_oof)), np.min(np.expm1(nn_oof)))
    max_val = max(np.max(np.expm1(cat_oof)), np.max(np.expm1(xgb_oof)), np.max(np.expm1(nn_oof)))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Correlation')
    
    ax2.set_title("Model Predictions Comparison")
    ax2.set_xlabel("Predictions from Model 1")
    ax2.set_ylabel("Predictions from Model 2")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 