"""
Fertilizer Classification Model using XGBoost with Optuna Optimization
====================================================================

This script implements a multi-class classification model for fertilizer prediction using XGBoost
with Optuna for hyperparameter optimization. The model uses GPU acceleration and implements
cross-validation with MAP@3 as the evaluation metric.

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import optuna
import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Tuple, List, Any

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'fertilizer_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FertilizerClassifier:
    """Main class for fertilizer classification model."""
    
    def __init__(self, 
                 train_path: str,
                 test_path: str,
                 external_data_path: str,
                 output_dir: str = "outputs"):
        """
        Initialize the classifier.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            external_data_path: Path to external data
            output_dir: Directory for saving outputs
        """
        self.train_path = train_path
        self.test_path = test_path
        self.external_data_path = external_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.encoder = LabelEncoder()
        self.target_encoder = LabelEncoder()
        self.best_params = None
        self.model = None
        
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess the data.
        
        Returns:
            Tuple containing processed train, test, and target data
        """
        logger.info("Loading and preprocessing data...")
        
        # Load datasets
        train = pd.read_csv(self.train_path, index_col="id")
        test = pd.read_csv(self.test_path, index_col="id")
        external = pd.read_csv(self.external_data_path)
        
        # Combine and deduplicate
        overall_train = pd.concat([train, external], ignore_index=True)
        overall_train = overall_train.drop_duplicates()
        
        # Prepare features and target
        X = overall_train.drop(columns=["Fertilizer Name"])
        y = overall_train["Fertilizer Name"]
        
        # Encode categorical variables
        categorical = X.select_dtypes(include=['object']).columns
        y = self.target_encoder.fit_transform(y)
        
        for cat in categorical:
            X[cat] = self.encoder.fit_transform(X[cat])
            test[cat] = self.encoder.transform(test[cat])
            
        # Calculate feature importance using mutual information
        mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
        mi_df = pd.DataFrame({
            "feature": X.columns,
            "mi_score": mi_scores
        }).sort_values(by="mi_score", ascending=False)
        
        logger.info(f"Feature importance (top 5):\n{mi_df.head()}")
        
        return X, test, y
    
    @staticmethod
    def map_at_3(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3) -> float:
        """
        Calculate MAP@3 metric.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            k: Number of top predictions to consider
            
        Returns:
            MAP@3 score
        """
        map_score = 0.0
        y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
        
        for i in range(len(y_true)):
            top_k_preds = np.argsort(y_pred_proba[i])[-k:][::-1]
            if y_true[i] in top_k_preds:
                rank = np.where(top_k_preds == y_true[i])[0][0] + 1
                map_score += 1.0 / rank
                
        return map_score / len(y_true)
    
    def objective(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: np.ndarray,
                 X_val: pd.DataFrame, y_val: np.ndarray) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            MAP@3 score
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-5, 10, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100, log=True),
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "enable_categorical": False,
            "device": "cuda"
        }
        
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=0
        )
        
        preds = model.predict_proba(X_val)
        return self.map_at_3(y_val, preds)
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: np.ndarray,
                               n_trials: int = 100,
                               previous_trial_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Features
            y: Target
            n_trials: Number of optimization trials
            previous_trial_params: Dictionary of parameters from a previous trial to start from
            
        Returns:
            Dictionary of best parameters
        """
        logger.info("Starting hyperparameter optimization...")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
        )
        
        # If previous trial parameters are provided, add them as a trial
        if previous_trial_params is not None:
            logger.info("Adding previous trial parameters as starting point...")
            study.enqueue_trial(previous_trial_params)
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            timeout=3600
        )
        
        self.best_params = study.best_params
        logger.info(f"Best parameters found: {self.best_params}")
        
        return self.best_params
    
    def train_final_model(self, X: pd.DataFrame, y: np.ndarray,
                         n_splits: int = 5, n_repeats: int = 2) -> Tuple[float, float, np.ndarray]:
        """
        Train final model using cross-validation.
        
        Args:
            X: Features
            y: Target
            n_splits: Number of CV splits
            n_repeats: Number of CV repeats
            
        Returns:
            Tuple of (CV MAP@3 score, OOF Log Loss, test predictions)
        """
        logger.info("Training final model with cross-validation...")
        
        n_classes = len(np.unique(y))
        oof_proba = np.zeros((len(X), n_classes))
        test_preds = np.zeros((len(self.test), n_classes))
        fold_counter = np.zeros(len(X))
        
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=42
        )
        
        fold_scores = []
        
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Training fold {fold+1}/{n_splits * n_repeats}")
            
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            
            model = XGBClassifier(**self.best_params)
            model.fit(
                X_train,
                y_train,
                early_stopping_rounds=50,
                eval_set=[(X_valid, y_valid)],
                verbose=False
            )
            
            valid_proba = model.predict_proba(X_valid)
            oof_proba[valid_idx] += valid_proba
            fold_counter[valid_idx] += 1
            
            fold_map = self.map_at_3(y_valid, valid_proba)
            fold_scores.append(fold_map)
            logger.info(f"Fold {fold+1} MAP@3: {fold_map:.5f}")
            
            test_preds += model.predict_proba(self.test) / (n_splits * n_repeats)
        
        oof_proba /= fold_counter[:, np.newaxis]
        cv_map = np.mean(fold_scores)
        oof_logloss = log_loss(y, oof_proba)
        
        logger.info(f"CV MAP@3: {cv_map:.5f}")
        logger.info(f"OOF Log Loss: {oof_logloss:.5f}")
        
        return cv_map, oof_logloss, test_preds
    
    def generate_submission(self, test_preds: np.ndarray) -> None:
        """
        Generate submission file.
        
        Args:
            test_preds: Test predictions
        """
        logger.info("Generating submission file...")
        
        top3_indices = np.argsort(-test_preds, axis=1)[:, :3]
        top3_labels = self.target_encoder.inverse_transform(top3_indices.ravel())
        top3_labels = top3_labels.reshape(len(self.test), 3)
        
        id = pd.read_csv(self.test_path)["id"]
        
        submission = pd.DataFrame({
            "id": id,
            "Fertilizer Name": [" ".join(row) for row in top3_labels]
        })
        
        submission_path = self.output_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission file saved to {submission_path}")
    
    def save_model_report(self, cv_map: float, oof_logloss: float) -> None:
        """
        Save model report with parameters and metrics.
        
        Args:
            cv_map: CV MAP@3 score
            oof_logloss: OOF Log Loss
        """
        report = {
            "best_parameters": self.best_params,
            "cv_map3": cv_map,
            "oof_logloss": oof_logloss,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        report_path = self.output_dir / "model_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Model report saved to {report_path}")

def main():
    """Main execution function."""
    # Initialize classifier
    classifier = FertilizerClassifier(
        train_path="path/to/train.csv",
        test_path="path/to/test.csv",
        external_data_path="path/to/external.csv"
    )
    
    # Load and preprocess data
    X, test, y = classifier.load_and_preprocess_data()
    classifier.test = test
    
    # Example of previous trial parameters (can be loaded from a file or passed as argument)
    previous_trial_params = {
        'n_estimators': 957,
        'learning_rate': 0.006852545490110178,
        'max_depth': 5,
        'min_child_weight': 9.275294835907687e-05,
        'subsample': 0.9718790609370292,
        'colsample_bytree': 0.9232481518257668,
        'gamma': 6.334037565104235,
        'reg_alpha': 5.1832947024610565,
        'reg_lambda': 1.0882301873180538
    }
    
    # Optimize hyperparameters with previous trial parameters
    best_params = classifier.optimize_hyperparameters(
        X, y, 
        n_trials=100,
        previous_trial_params=previous_trial_params
    )
    
    # Train final model and get test predictions
    cv_map, oof_logloss, test_preds = classifier.train_final_model(X, y)
    
    # Generate submission and save report
    classifier.generate_submission(test_preds)
    classifier.save_model_report(cv_map, oof_logloss)

if __name__ == "__main__":
    main()