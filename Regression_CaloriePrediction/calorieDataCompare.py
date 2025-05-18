import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureType(Enum):
    BASIC = "basic"
    POLYNOMIAL = "polynomial"
    PHYSIOLOGICAL = "physiological"
    INTERACTION = "interaction"
    SPECIFIC_INTERACTION = "specific_interaction"
    STATISTICAL = "statistical"
    DERIVED = "derived"

@dataclass
class FeatureConfig:
    """Configuration for feature generation"""
    feature_types: List[FeatureType]
    poly_features: Optional[List[str]] = None
    interaction_features: Optional[List[str]] = None
    specific_interactions: Optional[List[List[str]]] = None
    statistical_features: Optional[List[str]] = None

class FeatureManager:
    """Manages feature generation and tracking"""
    def __init__(self):
        self.feature_pipelines: Dict[str, Dict] = {}
        self.datasets: Dict[str, pd.DataFrame] = {}
        
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic features like BMI and Intensity"""
        df_processed = df.copy()
        
        # BMI
        df_processed['BMI'] = df_processed['Weight'] / (df_processed['Height'] / 100) ** 2
        
        # Intensity
        df_processed['Intensity'] = df_processed['Heart_Rate'] / df_processed['Duration']
        
        return df_processed
    
    def create_polynomial_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Create polynomial features from specified columns"""
        df_processed = df.copy()
        
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                feat1, feat2 = features[i], features[j]
                
                # Multiplication
                df_processed[f'{feat1}_x_{feat2}'] = df_processed[feat1] * df_processed[feat2]
                
                # Division (avoid division by zero)
                df_processed[f'{feat1}_div_{feat2}'] = df_processed[feat1] / (df_processed[feat2] + 1e-6)
                
                # Square of each feature
                df_processed[f'{feat1}_squared'] = df_processed[feat1] ** 2
        
        return df_processed
    
    def create_physiological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced physiological features"""
        df_processed = df.copy()
        
        # Basal Metabolic Rate (BMR)
        df_processed['BMR'] = df_processed['Weight'] / ((df_processed['Height'] / 100) ** 2)
        
        # Metabolic Efficiency Index
        df_processed['Metabolic_Efficiency'] = df_processed['BMR'] * (df_processed['Heart_Rate'] / df_processed['BMR'].median())
        
        # Cardiovascular Stress
        df_processed['Cardio_Stress'] = (df_processed['Heart_Rate'] / (220 - df_processed['Age'])) * df_processed['Duration']
        
        # Thermic Effect Ratio
        df_processed['Thermic_Effect'] = (df_processed['Body_Temp'] * 100) / (df_processed['Weight'] ** 0.5)
        
        # Power Output Estimate
        df_processed['Power_Output'] = df_processed['Weight'] * df_processed['Duration'] * (df_processed['Heart_Rate'] / 1000)
        
        return df_processed
    
    def create_interaction_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Create interaction features between specified columns"""
        df_processed = df.copy()
        
        # Duration-based features
        durations = sorted(df_processed['Duration'].unique())
        for dur in durations:
            df_processed[f'HR_Dur_{int(dur)}'] = np.where(df_processed['Duration'] == dur, df_processed['Heart_Rate'], 0)
            df_processed[f'Temp_Dur_{int(dur)}'] = np.where(df_processed['Duration'] == dur, df_processed['Body_Temp'], 0)
        
        # Age-based features
        ages = sorted(df_processed['Age'].unique())
        for age in ages:
            df_processed[f'HR_Age_{int(age)}'] = np.where(df_processed['Age'] == age, df_processed['Heart_Rate'], 0)
            df_processed[f'Temp_Age_{int(age)}'] = np.where(df_processed['Age'] == age, df_processed['Body_Temp'], 0)
        
        return df_processed

    def create_specific_interactions(self, df: pd.DataFrame, interaction_pairs: List[List[str]]) -> pd.DataFrame:
        """
        Create specific interaction features based on provided pairs of features
        
        Args:
            df: Input dataframe
            interaction_pairs: List of feature pairs to create interactions for
                             Each pair should be a list of two feature names
        
        Returns:
            DataFrame with added interaction features
        """
        df_processed = df.copy()
        
        for pair in interaction_pairs:
            if len(pair) != 2:
                logger.warning(f"Skipping invalid interaction pair: {pair}. Must contain exactly 2 features.")
                continue
                
            feat1, feat2 = pair
            if feat1 not in df.columns or feat2 not in df.columns:
                logger.warning(f"Skipping interaction pair {pair}. One or both features not found in dataset.")
                continue
            
            # Create interaction name
            interaction_name = f"{feat1}_interact_{feat2}"
            
            # Create interaction (multiplication)
            df_processed[interaction_name] = df_processed[feat1] * df_processed[feat2]
            
            # Create ratio interaction (avoid division by zero)
            ratio_name = f"{feat1}_ratio_{feat2}"
            df_processed[ratio_name] = df_processed[feat1] / (df_processed[feat2] + 1e-6)
            
            logger.info(f"Created interactions for pair: {pair}")
        
        return df_processed
    
    def create_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, config: FeatureConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create features based on specified configuration
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            config: Feature configuration object
            
        Returns:
            Tuple of processed training and test dataframes
        """
        logger.info("Starting feature engineering with specified configuration...")
        
        # Create copies
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        # Basic preprocessing
        train_processed['Sex'] = train_processed['Sex'].map({'male': 1, 'female': 0})
        test_processed['Sex'] = test_processed['Sex'].map({'male': 1, 'female': 0})
        
        # Apply feature generation based on configuration
        if FeatureType.BASIC in config.feature_types:
            train_processed = self.create_basic_features(train_processed)
            test_processed = self.create_basic_features(test_processed)
        
        if FeatureType.POLYNOMIAL in config.feature_types and config.poly_features:
            train_processed = self.create_polynomial_features(train_processed, config.poly_features)
            test_processed = self.create_polynomial_features(test_processed, config.poly_features)
        
        if FeatureType.PHYSIOLOGICAL in config.feature_types:
            train_processed = self.create_physiological_features(train_processed)
            test_processed = self.create_physiological_features(test_processed)
        
        if FeatureType.INTERACTION in config.feature_types and config.interaction_features:
            train_processed = self.create_interaction_features(train_processed, config.interaction_features)
            test_processed = self.create_interaction_features(test_processed, config.interaction_features)
            
        if FeatureType.SPECIFIC_INTERACTION in config.feature_types and config.specific_interactions:
            train_processed = self.create_specific_interactions(train_processed, config.specific_interactions)
            test_processed = self.create_specific_interactions(test_processed, config.specific_interactions)
        
        return train_processed, test_processed
    
    def create_final_dataset(self, df: pd.DataFrame, selected_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create final dataset with selected features and return both selected and complete feature datasets
        
        Args:
            df: Input dataframe
            selected_features: List of feature names to include
            
        Returns:
            Tuple of (selected_features_df, complete_features_df)
            - selected_features_df: DataFrame with only selected features
            - complete_features_df: DataFrame with all available features
        """
        available_features = [col for col in df.columns if col in selected_features]
        if not available_features:
            raise ValueError("None of the selected features are available in the dataset")
        
        selected_df = df[available_features]
        complete_df = df.copy()
        
        logger.info(f"Created final dataset with {len(available_features)} selected features out of {len(df.columns)} total features")
        
        return selected_df, complete_df
    
    def track_pipeline(self, pipeline_name: str, config: FeatureConfig, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Track feature pipeline and datasets
        
        Args:
            pipeline_name: Name of the pipeline
            config: Feature configuration used
            train_df: Training dataframe
            test_df: Test dataframe
        """
        self.feature_pipelines[pipeline_name] = {
            'config': config,
            'feature_count': len(train_df.columns),
            'feature_names': list(train_df.columns)
        }
        
        self.datasets[f"{pipeline_name}_train"] = train_df
        self.datasets[f"{pipeline_name}_test"] = test_df
        
        logger.info(f"Pipeline '{pipeline_name}' tracked with {len(train_df.columns)} features")

def process_features_with_config(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_types: List[str],
    poly_features: Optional[List[str]] = None,
    interaction_features: Optional[List[str]] = None,
    specific_interactions: Optional[List[List[str]]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process features based on provided configuration
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        feature_types: List of feature types to include (e.g., ['basic', 'polynomial', 'physiological'])
        poly_features: List of features for polynomial combinations
        interaction_features: List of features for general interactions
        specific_interactions: List of feature pairs for specific interactions
        
    Returns:
        Tuple of (processed_train_df, processed_test_df)
    """
    # Convert string feature types to enum
    selected_feature_types = [FeatureType(ft.lower()) for ft in feature_types]
    
    # Create feature configuration
    config = FeatureConfig(
        feature_types=selected_feature_types,
        poly_features=poly_features,
        interaction_features=interaction_features,
        specific_interactions=specific_interactions
    )
    
    # Create feature manager and process features
    feature_manager = FeatureManager()
    train_processed, test_processed = feature_manager.create_features(train_df, test_df, config)
    
    # Track the pipeline
    feature_manager.track_pipeline("configured_pipeline", config, train_processed, test_processed)
    
    print(f"\nCreated processed datasets:")
    print(f"Training data shape: {train_processed.shape}")
    print(f"Test data shape: {test_processed.shape}")
    
    return train_processed, test_processed


def train_xgboost_parallel(X, y, X_test, base_params=None):
    """
    Train XGBoost model with parallel cross-validation across multiple GPUs if available,
    otherwise falls back to CPU training.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        base_params (dict, optional): Base XGBoost parameters
        
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

    # Check for GPU availability
    try:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            logger.info(f"Found {n_gpus} GPUs")
            print(f"Found {n_gpus} GPUs")
            device = "cuda"
            tree_method = "hist"
        else:
            logger.info("No GPUs found, using CPU")
            print("No GPUs found, using CPU")
            device = "cpu"
            tree_method = "hist"
    except:
        logger.info("GPU check failed, using CPU")
        print("GPU check failed, using CPU")
        device = "cpu"
        tree_method = "hist"

    # Use provided parameters or defaults
    params = base_params or {
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
        'tree_method': tree_method,
        'device': device,
        'sampling_method': 'gradient_based',
        'max_bin': 256,
        'grow_policy': 'lossguide'
    }

    # Cross-validation
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    total_folds = kf.n_splits
    
    # Store all splits
    splits = list(kf.split(X_xgb))
    
    logger.info(f"Starting {total_folds}-fold cross-validation in parallel...")
    print(f"\nStarting {total_folds}-fold cross-validation in parallel...")
    logger.info(f"Training with {len(common_features)} features: {', '.join(common_features)}")
    print(f"Training with {len(common_features)} features")

    def train_fold(fold_idx, train_idx, val_idx, device_id):
        """Train a single fold on specified device"""
        fold_start_time = time.time()
        logger.info(f"\nFold {fold_idx}/{total_folds} - Starting training on {device}...")
        print(f"\nFold {fold_idx}/{total_folds} - Starting training on {device}...")
        
        # Create fold-specific parameters
        fold_params = params.copy()
        if device == "cuda":
            fold_params['device'] = f"cuda:{device_id}"
        
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
        
        logger.info(f"Fold {fold_idx} completed on {device}:")
        logger.info(f"  - RMSLE Score: {fold_score:.5f}")
        logger.info(f"  - Time taken: {fold_time:.2f} seconds")
        print(f"\nFold {fold_idx} completed on {device}:")
        print(f"  - RMSLE Score: {fold_score:.5f}")
        print(f"  - Time taken: {fold_time:.2f} seconds")
        
        return fold_idx, fold_oof, fold_test_preds, fold_score, model, fold_time, val_idx

    # Create process pool for parallel execution
    max_workers = n_gpus if device == "cuda" else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all folds for parallel execution
        future_to_fold = {
            executor.submit(
                train_fold, 
                fold_idx + 1, 
                train_idx, 
                val_idx, 
                fold_idx % max_workers
            ): fold_idx 
            for fold_idx, (train_idx, val_idx) in enumerate(splits)
        }
        
        # Collect results as they complete
        fold_times = []
        for future in as_completed(future_to_fold):
            fold_idx, fold_oof, fold_test_preds, fold_score, model, fold_time, val_idx = future.result()
            xgb_oof[val_idx] = fold_oof
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

def train_gender_specific_models(X, y, X_test, test_ids, male_params=None, female_params=None):
    """
    Train separate XGBoost models for males and females and combine predictions.
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        test_ids (pd.Series): Test IDs for final submission
        male_params (dict, optional): XGBoost parameters for male model
        female_params (dict, optional): XGBoost parameters for female model
        
    Returns:
        tuple: (combined_predictions, male_model, female_model, total_time)
    """
    logger.info("Starting gender-specific model training...")
    print("\n=== Starting Gender-Specific Model Training ===")
    start_time = time.time()
    
    # Default parameters if none provided
    default_params = {
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
        'tree_method': 'hist',
        'sampling_method': 'gradient_based',
        'max_bin': 256,
        'grow_policy': 'lossguide'
    }
    
    # Use provided parameters or defaults
    male_params = male_params or default_params.copy()
    female_params = female_params or default_params.copy()
    
    # Split data by gender
    male_mask = X['Sex'] == 1
    female_mask = X['Sex'] == 0
    
    X_male = X[male_mask].copy()
    y_male = y[male_mask].copy()
    X_female = X[female_mask].copy()
    y_female = y[female_mask].copy()
    
    # Split test data
    X_test_male = X_test[X_test['Sex'] == 1].copy()
    X_test_female = X_test[X_test['Sex'] == 0].copy()
    
    logger.info(f"Male training samples: {len(X_male)}")
    logger.info(f"Female training samples: {len(X_female)}")
    print(f"Male training samples: {len(X_male)}")
    print(f"Female training samples: {len(X_female)}")
    
    # Train male model
    logger.info("\nTraining male model...")
    print("\nTraining male model...")
    male_preds, male_oof, male_scores, male_model, male_time = train_xgboost_parallel(
        X_male, y_male, X_test_male, base_params=male_params
    )
    
    # Train female model
    logger.info("\nTraining female model...")
    print("\nTraining female model...")
    female_preds, female_oof, female_scores, female_model, female_time = train_xgboost_parallel(
        X_female, y_female, X_test_female, base_params=female_params
    )
    
    # Combine predictions
    combined_predictions = pd.DataFrame({
        'id': test_ids,
        'Calories': np.zeros(len(test_ids))
    })
    
    # Fill predictions based on gender
    male_test_mask = X_test['Sex'] == 1
    female_test_mask = X_test['Sex'] == 0
    
    combined_predictions.loc[male_test_mask, 'Calories'] = np.expm1(male_preds)
    combined_predictions.loc[female_test_mask, 'Calories'] = np.expm1(female_preds)
    
    # Calculate and log final metrics
    total_time = time.time() - start_time
    male_mean_score = np.mean(male_scores)
    female_mean_score = np.mean(female_scores)
    
    logger.info("\nGender-Specific Model Training Summary:")
    logger.info(f"  - Male Model Mean RMSLE: {male_mean_score:.5f}")
    logger.info(f"  - Female Model Mean RMSLE: {female_mean_score:.5f}")
    logger.info(f"  - Total training time: {total_time/60:.1f} minutes")
    print("\n=== Gender-Specific Model Training Summary ===")
    print(f"  - Male Model Mean RMSLE: {male_mean_score:.5f}")
    print(f"  - Female Model Mean RMSLE: {female_mean_score:.5f}")
    print(f"  - Total training time: {total_time/60:.1f} minutes")
    
    return combined_predictions, male_model, female_model, total_time

def train_age_gender_specific_models(X, y, X_test, test_ids, group_params=None):
    """
    Train separate XGBoost models for different age and gender groups.
    Age groups:
    - Male: 18-30, 31-45, 46+
    - Female: 18-35, 36+
    
    Args:
        X (pd.DataFrame): Training features
        y (pd.Series): Target variable
        X_test (pd.DataFrame): Test features
        test_ids (pd.Series): Test IDs for final submission
        group_params (dict, optional): Dictionary of XGBoost parameters for each group
        
    Returns:
        tuple: (combined_predictions, models_dict, total_time)
    """
    logger.info("Starting age and gender-specific model training...")
    print("\n=== Starting Age and Gender-Specific Model Training ===")
    start_time = time.time()
    
    # Default parameters if none provided
    default_params = {
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
        'tree_method': 'hist',
        'sampling_method': 'gradient_based',
        'max_bin': 256,
        'grow_policy': 'lossguide'
    }
    
    # Define age groups
    age_groups = {
        'male_18_30': {'gender': 1, 'min_age': 18, 'max_age': 30},
        'male_31_45': {'gender': 1, 'min_age': 31, 'max_age': 45},
        'male_46_plus': {'gender': 1, 'min_age': 46, 'max_age': 100},
        'female_18_35': {'gender': 0, 'min_age': 18, 'max_age': 35},
        'female_36_plus': {'gender': 0, 'min_age': 36, 'max_age': 100}
    }
    
    # Initialize storage for models and predictions
    models_dict = {}
    test_predictions = {}
    
    # Train models for each group
    for group_name, criteria in age_groups.items():
        logger.info(f"\nTraining model for {group_name}...")
        print(f"\nTraining model for {group_name}...")
        
        # Get parameters for this group
        group_specific_params = group_params.get(group_name, default_params.copy()) if group_params else default_params.copy()
        
        # Create masks for training data
        train_mask = (
            (X['Sex'] == criteria['gender']) & 
            (X['Age'] >= criteria['min_age']) & 
            (X['Age'] <= criteria['max_age'])
        )
        
        # Create masks for test data
        test_mask = (
            (X_test['Sex'] == criteria['gender']) & 
            (X_test['Age'] >= criteria['min_age']) & 
            (X_test['Age'] <= criteria['max_age'])
        )
        
        # Get group data
        X_group = X[train_mask].copy()
        y_group = y[train_mask].copy()
        X_test_group = X_test[test_mask].copy()
        
        # Log group sizes
        logger.info(f"{group_name} training samples: {len(X_group)}")
        logger.info(f"{group_name} test samples: {len(X_test_group)}")
        print(f"{group_name} training samples: {len(X_group)}")
        print(f"{group_name} test samples: {len(X_test_group)}")
        
        # Skip if no training samples
        if len(X_group) == 0:
            logger.warning(f"No training samples for {group_name}, skipping...")
            print(f"No training samples for {group_name}, skipping...")
            continue
        
        # Train model for this group
        group_preds, group_oof, group_scores, group_model, group_time = train_xgboost_parallel(
            X_group, y_group, X_test_group, base_params=group_specific_params
        )
        
        # Store model and predictions
        models_dict[group_name] = {
            'model': group_model,
            'scores': group_scores,
            'mean_score': np.mean(group_scores),
            'time': group_time,
            'params': group_specific_params
        }
        
        # Store test predictions with their indices
        test_predictions[group_name] = {
            'predictions': np.expm1(group_preds),
            'test_indices': X_test[test_mask].index
        }
    
    # Combine predictions
    combined_predictions = pd.DataFrame({
        'id': test_ids,
        'Calories': np.zeros(len(test_ids))
    })
    
    # Fill predictions for each group
    for group_name, pred_data in test_predictions.items():
        combined_predictions.loc[pred_data['test_indices'], 'Calories'] = pred_data['predictions']
    
    # Calculate and log final metrics
    total_time = time.time() - start_time
    
    logger.info("\nAge and Gender-Specific Model Training Summary:")
    print("\n=== Age and Gender-Specific Model Training Summary ===")
    
    for group_name, model_data in models_dict.items():
        logger.info(f"  - {group_name} Mean RMSLE: {model_data['mean_score']:.5f}")
        print(f"  - {group_name} Mean RMSLE: {model_data['mean_score']:.5f}")
    
    logger.info(f"  - Total training time: {total_time/60:.1f} minutes")
    print(f"  - Total training time: {total_time/60:.1f} minutes")
    
    return combined_predictions, models_dict, total_time

# Example usage in Jupyter notebook:
"""
# Define feature configuration
feature_types = ['basic', 'polynomial', 'physiological', 'interaction', 'specific_interaction']

poly_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI']

interaction_features = ['Duration', 'Age', 'Heart_Rate', 'Body_Temp']

specific_interactions = [
    ['Heart_Rate', 'Duration'],
    ['Body_Temp', 'Weight'],
    ['Age', 'BMI']
]

# Process features
train_processed, test_processed = process_features_with_config(
    train_df=train_df,
    test_df=test_df,
    feature_types=feature_types,
    poly_features=poly_features,
    interaction_features=interaction_features,
    specific_interactions=specific_interactions
)

# Prepare data for modeling
X = train_processed.drop(columns=['id','Calories'])
y = np.log1p(train_processed['Calories'])
X_test = test_processed.drop(columns=['id'])
test_ids = test_processed['id']

# Example 1: Gender-specific models with custom parameters
male_params = {
    'max_depth': 8,
    'n_estimators': 2500,
    'learning_rate': 0.015,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.02,
    'tree_method': 'hist',
    'random_state': 42
}

female_params = {
    'max_depth': 10,
    'n_estimators': 3000,
    'learning_rate': 0.01,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'gamma': 0.01,
    'tree_method': 'hist',
    'random_state': 42
}

# Train gender-specific models
predictions, male_model, female_model, total_time = train_gender_specific_models(
    X, y, X_test, test_ids,
    male_params=male_params,
    female_params=female_params
)

# Save predictions
predictions.to_csv('gender_specific_predictions.csv', index=False)

# Example 2: Age and gender-specific models with custom parameters
group_params = {
    'male_18_30': {
        'max_depth': 7,
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    'male_31_45': {
        'max_depth': 8,
        'n_estimators': 2500,
        'learning_rate': 0.015,
        'subsample': 0.85,
        'colsample_bytree': 0.75
    },
    'male_46_plus': {
        'max_depth': 9,
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'subsample': 0.9,
        'colsample_bytree': 0.7
    },
    'female_18_35': {
        'max_depth': 8,
        'n_estimators': 2500,
        'learning_rate': 0.015,
        'subsample': 0.85,
        'colsample_bytree': 0.75
    },
    'female_36_plus': {
        'max_depth': 10,
        'n_estimators': 3000,
        'learning_rate': 0.01,
        'subsample': 0.9,
        'colsample_bytree': 0.7
    }
}

# Train age and gender-specific models
predictions, models_dict, total_time = train_age_gender_specific_models(
    X, y, X_test, test_ids,
    group_params=group_params
)

# Save predictions
predictions.to_csv('age_gender_specific_predictions.csv', index=False)

# Print detailed model performance
print("\nDetailed Model Performance:")
for group_name, model_data in models_dict.items():
    print(f"\n{group_name}:")
    print(f"  Mean RMSLE: {model_data['mean_score']:.5f}")
    print(f"  Training time: {model_data['time']/60:.1f} minutes")
    print("  Parameters:", model_data['params'])
"""

