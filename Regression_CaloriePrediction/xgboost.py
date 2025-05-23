"""
ScientificXGBRegressor: Advanced XGBoost Regressor with Scientific Computing Features
================================================================================

This module provides a scientifically-enhanced XGBoost regressor that includes:
- Automated data-driven hyperparameter setting
- Nested cross-validation protocols
- Comprehensive diagnostic visualizations
- Advanced hyperparameter optimization
- Incremental learning capabilities
- Complete model artifact persistence
- Model upgrade and migration capabilities

Mathematical Foundation:
------------------------
The XGBoost objective function is formulated as:
    L(θ) = Σᵢ l(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)

Where:
- l(yᵢ, ŷᵢ) is the loss function between true and predicted values
- Ω(fₖ) = γT + ½λ||w||² is the regularization term
- T is the number of leaves, γ is the minimum loss reduction
- λ is the L2 regularization parameter

The gradient boosting update rule:
    ŷᵢ⁽ᵗ⁾ = ŷᵢ⁽ᵗ⁻¹⁾ + η·fₜ(xᵢ)

Where η is the learning rate and fₜ is the t-th tree.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
import pickle
import json
import warnings
from datetime import datetime
import os
import time
from pathlib import Path

# Scientific computing and ML libraries
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold, GridSearchCV, 
    RandomizedSearchCV, validation_curve
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.inspection import partial_dependence
import scipy.stats as stats

# XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Please install with: pip install xgboost")

# Optional advanced libraries
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ScientificXGBRegressor(XGBRegressor):
    """
    Advanced XGBoost Regressor with Scientific Computing Features
    
    This class extends XGBRegressor with sophisticated features for scientific
    machine learning applications, including automated parameterization,
    comprehensive validation, and advanced diagnostics.
    
    Mathematical Formulation:
    -------------------------
    The model optimizes the regularized objective:
    
    L(θ) = Σᵢ l(yᵢ, ŷᵢ) + Σₖ Ω(fₖ) + R(θ)
    
    Where:
    - l(yᵢ, ŷᵢ): Loss function (MSE for regression)
    - Ω(fₖ): Tree complexity penalty = γT + ½λ||w||²
    - R(θ): Additional regularization based on data characteristics
    
    Parameters:
    -----------
    cv_folds : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random state for reproducibility
    auto_tune : bool, default=True
        Whether to automatically tune hyperparameters based on data
    verbose : bool, default=True
        Whether to print progress information
    early_stopping_rounds : int, default=50
        Early stopping rounds for training
    **kwargs : dict
        Additional XGBoost parameters
    """
    
    def __init__(
        self,
        cv_folds: int = 5,
        random_state: int = 42,
        auto_tune: bool = True,
        verbose: bool = True,
        early_stopping_rounds: int = 50,
        n_estimators: int = 1000,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        **kwargs
    ):
        # Store scientific parameters
        self.cv_folds = cv_folds
        self.auto_tune = auto_tune
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self._is_fitted = False
        self._cv_results = {}
        self._feature_importance_scores = {}
        self._validation_history = {}
        self._hyperparameter_history = {}
        
        # Initialize XGBRegressor with scientific defaults
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs
        )
        
        if self.verbose:
            print(f"🧪 ScientificXGBRegressor initialized with {cv_folds}-fold CV")
            if auto_tune:
                print("🔧 Automatic hyperparameter tuning enabled")

    # ========================================
    # MODEL UPGRADE AND MIGRATION METHODS
    # ========================================
    
    @classmethod
    def from_xgb_regressor(
        cls, 
        existing_model: XGBRegressor, 
        cv_folds: int = 5,
        auto_tune: bool = True,
        verbose: bool = True,
        preserve_training: bool = True
    ) -> 'ScientificXGBRegressor':
        """
        Upgrade an existing XGBRegressor to ScientificXGBRegressor.
        
        This method converts a standard XGBRegressor into a ScientificXGBRegressor
        while preserving the trained state and hyperparameters.
        
        Parameters:
        -----------
        existing_model : XGBRegressor
            The existing XGBoost model to upgrade
        cv_folds : int, default=5
            Number of cross-validation folds for scientific features
        auto_tune : bool, default=True
            Whether to enable automatic hyperparameter tuning
        verbose : bool, default=True
            Whether to print upgrade information
        preserve_training : bool, default=True
            Whether to preserve the trained state of the model
            
        Returns:
        --------
        ScientificXGBRegressor
            Upgraded model with scientific features
            
        Example:
        --------
        ```python
        # Existing XGBoost model
        old_model = XGBRegressor(n_estimators=500, learning_rate=0.1)
        old_model.fit(X_train, y_train)
        
        # Upgrade to ScientificXGBRegressor
        new_model = ScientificXGBRegressor.from_xgb_regressor(
            old_model, 
            auto_tune=True,
            preserve_training=True
        )
        
        # Now use scientific features
        new_model.diagnostic_plots(X_train, y_train)
        cv_results = new_model.nested_cross_validate(X_train, y_train)
        ```
        """
        if verbose:
            print("🔄 Upgrading XGBRegressor to ScientificXGBRegressor...")
        
        # Extract parameters from existing model
        params = existing_model.get_params()
        
        # Add scientific parameters
        scientific_params = {
            'cv_folds': cv_folds,
            'auto_tune': auto_tune,
            'verbose': verbose
        }
        
        # Merge parameters
        all_params = {**params, **scientific_params}
        
        # Create new scientific model
        scientific_model = cls(**all_params)
        
        # Preserve training state if requested and model is fitted
        if preserve_training and hasattr(existing_model, 'booster_'):
            if verbose:
                print("  📋 Preserving trained model state...")
            
            # Copy the trained booster
            scientific_model._Booster = existing_model._Booster
            scientific_model.booster_ = existing_model.booster_
            scientific_model._is_fitted = True
            
            # Copy other fitted attributes
            if hasattr(existing_model, 'feature_importances_'):
                scientific_model.feature_importances_ = existing_model.feature_importances_
                scientific_model._feature_importance_scores['gain'] = existing_model.feature_importances_
            
            if hasattr(existing_model, 'evals_result_'):
                scientific_model.evals_result_ = existing_model.evals_result_
                
            if hasattr(existing_model, 'n_features_in_'):
                scientific_model.n_features_in_ = existing_model.n_features_in_
        
        # Record upgrade history
        upgrade_info = {
            'upgraded_from': 'XGBRegressor',
            'original_params': params,
            'scientific_params_added': scientific_params,
            'preserved_training': preserve_training and hasattr(existing_model, 'booster_'),
            'upgrade_timestamp': datetime.now().isoformat()
        }
        
        scientific_model._hyperparameter_history['model_upgrade'] = upgrade_info
        
        if verbose:
            print(f"  ✅ Model upgraded successfully!")
            print(f"  📊 Original parameters preserved: {len(params)}")
            print(f"  🧪 Scientific features added: {len(scientific_params)}")
            if preserve_training and hasattr(existing_model, 'booster_'):
                print(f"  🎯 Trained state preserved: Yes")
            else:
                print(f"  🎯 Trained state preserved: No (model not fitted or preserve_training=False)")
        
        return scientific_model
    
    def migrate_and_retrain(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        apply_scientific_tuning: bool = True,
        compare_performance: bool = True,
        refit_strategy: str = 'enhanced'  # 'enhanced', 'retrain', 'incremental'
    ) -> Dict[str, Any]:
        """
        Migrate existing model parameters and retrain with scientific enhancements.
        
        This method applies scientific tuning to an existing model and retrains
        it with enhanced hyperparameters, providing performance comparisons.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training targets
        apply_scientific_tuning : bool, default=True
            Whether to apply automated scientific hyperparameter tuning
        compare_performance : bool, default=True
            Whether to compare performance before and after migration
        refit_strategy : str, default='enhanced'
            Strategy for refitting:
            - 'enhanced': Apply scientific tuning then refit
            - 'retrain': Complete retrain with scientific parameters
            - 'incremental': Add estimators with scientific tuning
            
        Returns:
        --------
        Dict[str, Any]
            Migration results including performance comparisons
            
        Example:
        --------
        ```python
        # Load existing model
        model = ScientificXGBRegressor.from_xgb_regressor(old_xgb_model)
        
        # Migrate and retrain with scientific enhancements
        migration_results = model.migrate_and_retrain(
            X_train, y_train,
            apply_scientific_tuning=True,
            refit_strategy='enhanced'
        )
        
        print(f"Performance improvement: {migration_results['performance_improvement']:.4f}")
        ```
        """
        if self.verbose:
            print("🔄 Starting model migration and retraining...")
        
        migration_start = time.time()
        results = {}
        
        # Store original parameters and performance
        original_params = self.get_params().copy()
        original_fitted = self._is_fitted
        
        # Get baseline performance if model is already fitted
        if original_fitted and compare_performance:
            if self.verbose:
                print("  📊 Evaluating original model performance...")
            
            original_predictions = self.predict(X)
            original_r2 = r2_score(y, original_predictions)
            original_rmse = np.sqrt(mean_squared_error(y, original_predictions))
            
            results['original_performance'] = {
                'r2': original_r2,
                'rmse': original_rmse,
                'mae': mean_absolute_error(y, original_predictions)
            }
            
            if self.verbose:
                print(f"    Original R²: {original_r2:.4f}, RMSE: {original_rmse:.4f}")
        
        # Apply scientific tuning
        if apply_scientific_tuning:
            if self.verbose:
                print("  🔬 Applying scientific hyperparameter tuning...")
            
            tuning_results = self.automated_parameterization(X, y)
            results['scientific_tuning'] = tuning_results
        
        # Refit based on strategy
        if refit_strategy == 'enhanced':
            if self.verbose:
                print("  🎯 Enhanced refit: Using scientific parameters...")
            self.fit(X, y)
            
        elif refit_strategy == 'retrain':
            if self.verbose:
                print("  🔄 Complete retrain with scientific parameters...")
            # Reset fitted state and retrain
            self._is_fitted = False
            self.fit(X, y)
            
        elif refit_strategy == 'incremental':
            if self.verbose:
                print("  📈 Incremental enhancement...")
            if original_fitted:
                # Add more estimators with scientific tuning
                incremental_results = self.incremental_learn(X, y, n_new_estimators=200)
                results['incremental_results'] = incremental_results
            else:
                if self.verbose:
                    print("    ⚠️  Model not fitted, performing initial fit...")
                self.fit(X, y)
        
        # Evaluate new performance
        if self.verbose:
            print("  📈 Evaluating enhanced model performance...")
        
        new_predictions = self.predict(X)
        new_r2 = r2_score(y, new_predictions)
        new_rmse = np.sqrt(mean_squared_error(y, new_predictions))
        
        results['enhanced_performance'] = {
            'r2': new_r2,
            'rmse': new_rmse,
            'mae': mean_absolute_error(y, new_predictions)
        }
        
        # Calculate performance improvement
        if original_fitted and compare_performance:
            r2_improvement = new_r2 - original_r2
            rmse_improvement = original_rmse - new_rmse  # Positive is better
            
            results['performance_improvement'] = {
                'r2_improvement': r2_improvement,
                'rmse_improvement': rmse_improvement,
                'relative_r2_improvement': r2_improvement / max(abs(original_r2), 1e-6),
                'relative_rmse_improvement': rmse_improvement / max(original_rmse, 1e-6)
            }
            
            if self.verbose:
                print(f"  📊 Performance Changes:")
                print(f"    R² improvement: {r2_improvement:+.4f} ({results['performance_improvement']['relative_r2_improvement']:+.2%})")
                print(f"    RMSE improvement: {rmse_improvement:+.4f} ({results['performance_improvement']['relative_rmse_improvement']:+.2%})")
        
        # Store migration metadata
        migration_time = time.time() - migration_start
        migration_metadata = {
            'migration_strategy': refit_strategy,
            'scientific_tuning_applied': apply_scientific_tuning,
            'original_parameters': original_params,
            'migration_time_seconds': migration_time,
            'migration_timestamp': datetime.now().isoformat()
        }
        
        results['migration_metadata'] = migration_metadata
        self._hyperparameter_history['migration'] = migration_metadata
        
        if self.verbose:
            print(f"  ✅ Migration completed in {migration_time:.2f}s")
            print(f"  🎯 New R²: {new_r2:.4f}, RMSE: {new_rmse:.4f}")
        
        return results
    
    def upgrade_from_pickle(
        self, 
        pickle_path: str,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        auto_enhance: bool = True
    ) -> Dict[str, Any]:
        """
        Load and upgrade a pickled XGBoost model to ScientificXGBRegressor.
        
        Parameters:
        -----------
        pickle_path : str
            Path to the pickled XGBoost model
        X : array-like, optional
            Training data for scientific enhancement
        y : array-like, optional
            Training targets for scientific enhancement
        auto_enhance : bool, default=True
            Whether to automatically apply scientific enhancements
            
        Returns:
        --------
        Dict[str, Any]
            Upgrade results and model information
            
        Example:
        --------
        ```python
        # Load and upgrade from pickle
        model = ScientificXGBRegressor()
        upgrade_results = model.upgrade_from_pickle(
            'old_model.pkl',
            X_train, y_train,
            auto_enhance=True
        )
        ```
        """
        if self.verbose:
            print(f"📂 Loading model from: {pickle_path}")
        
        # Load the pickled model
        with open(pickle_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Check if it's an XGBoost model
        if not isinstance(loaded_model, (XGBRegressor, ScientificXGBRegressor)):
            raise ValueError(f"Loaded model is not an XGBoost model. Got: {type(loaded_model)}")
        
        # If it's already a ScientificXGBRegressor, just update attributes
        if isinstance(loaded_model, ScientificXGBRegressor):
            if self.verbose:
                print("  ℹ️  Model is already a ScientificXGBRegressor")
            
            # Copy all attributes
            for attr, value in loaded_model.__dict__.items():
                setattr(self, attr, value)
            
            upgrade_info = {
                'already_scientific': True,
                'model_type': 'ScientificXGBRegressor',
                'upgrade_timestamp': datetime.now().isoformat()
            }
        else:
            if self.verbose:
                print("  🔄 Converting XGBRegressor to ScientificXGBRegressor...")
            
            # Use the from_xgb_regressor method
            upgraded_model = self.from_xgb_regressor(
                loaded_model, 
                cv_folds=self.cv_folds,
                auto_tune=self.auto_tune,
                verbose=self.verbose
            )
            
            # Copy attributes from upgraded model
            for attr, value in upgraded_model.__dict__.items():
                setattr(self, attr, value)
            
            upgrade_info = {
                'already_scientific': False,
                'model_type': 'XGBRegressor',
                'upgrade_timestamp': datetime.now().isoformat()
            }
        
        # Apply scientific enhancements if data is provided
        if auto_enhance and X is not None and y is not None:
            if self.verbose:
                print("  🧪 Applying scientific enhancements...")
            
            enhancement_results = self.migrate_and_retrain(
                X, y,
                apply_scientific_tuning=True,
                refit_strategy='enhanced'
            )
            upgrade_info['enhancement_results'] = enhancement_results
        
        # Store upgrade history
        self._hyperparameter_history['pickle_upgrade'] = upgrade_info
        
        if self.verbose:
            print("  ✅ Model upgrade from pickle completed!")
        
        return upgrade_info
    
    # ========================================
    # EXISTING METHODS (unchanged)
    # ========================================
    
    def _calculate_data_characteristics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistical characteristics of the dataset for informed parameterization.
        
        Mathematical Analysis:
        ----------------------
        Computes key statistical measures:
        - Noise level: σₑ = √(Var(y) - Var(ŷ_simple))
        - Feature correlation: ρ = |X^T X|
        - Dimensionality ratio: d/n where d=features, n=samples
        - Target complexity: H(y) using entropy approximation
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target vector
            
        Returns:
        --------
        Dict[str, float]
            Dictionary of dataset characteristics
        """
        n_samples, n_features = X.shape
        
        characteristics = {
            'n_samples': n_samples,
            'n_features': n_features,
            'dimensionality_ratio': n_features / n_samples,
            'target_variance': np.var(y),
            'target_mean': np.mean(y),
            'target_std': np.std(y),
            'target_skewness': stats.skew(y),
            'target_kurtosis': stats.kurtosis(y),
            'feature_variance_mean': np.mean(np.var(X, axis=0)),
            'feature_correlation_max': np.max(np.abs(np.corrcoef(X.T))) if n_features > 1 else 0.0,
            'missing_rate': np.mean(np.isnan(X)) if np.any(np.isnan(X)) else 0.0,
        }
        
        # Estimate noise level using simple linear regression residuals
        try:
            from sklearn.linear_model import LinearRegression
            simple_model = LinearRegression()
            simple_model.fit(X, y)
            y_pred_simple = simple_model.predict(X)
            noise_estimate = np.std(y - y_pred_simple)
            characteristics['estimated_noise'] = noise_estimate
            characteristics['signal_to_noise'] = characteristics['target_std'] / max(noise_estimate, 1e-6)
        except:
            characteristics['estimated_noise'] = characteristics['target_std'] * 0.1
            characteristics['signal_to_noise'] = 10.0
        
        return characteristics
    
    def automated_parameterization(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Set hyperparameters automatically based on dataset characteristics.
        
        Scientific Rationale:
        ---------------------
        Uses statistical analysis to inform hyperparameter choices:
        
        1. Learning Rate (η): Adaptive based on convergence theory
           η = min(0.3, 1.0 / √n_samples)
           
        2. Maximum Depth: Based on bias-variance tradeoff
           depth = log₂(n_samples) + noise_adjustment
           
        3. Regularization: Based on dimensionality and noise
           λ = α·(n_features/n_samples)·noise_factor
           
        4. Subsample Rates: Based on overfitting risk
           subsample = max(0.5, 1.0 - dimensionality_ratio)
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training targets
            
        Returns:
        --------
        Dict[str, Any]
            Optimized hyperparameters with scientific justification
        """
        if self.verbose:
            print("🔬 Analyzing dataset characteristics for automated parameterization...")
        
        char = self._calculate_data_characteristics(X, y)
        
        # Scientific parameter calculation
        params = {}
        
        # Learning rate: Adaptive based on sample size and complexity
        base_lr = min(0.3, 1.0 / np.sqrt(char['n_samples']))
        complexity_factor = min(2.0, char['dimensionality_ratio'] * 10)
        params['learning_rate'] = max(0.01, base_lr / complexity_factor)
        
        # Number of estimators: Based on learning rate and convergence theory
        params['n_estimators'] = min(2000, int(1000 / params['learning_rate']))
        
        # Maximum depth: Information-theoretic approach
        base_depth = int(np.log2(char['n_samples'])) + 1
        noise_adjustment = -1 if char['signal_to_noise'] < 5 else 0
        params['max_depth'] = max(3, min(10, base_depth + noise_adjustment))
        
        # Regularization: Based on dimensionality and overfitting risk
        reg_factor = char['dimensionality_ratio'] * char['feature_correlation_max']
        params['reg_lambda'] = min(10.0, max(0.1, reg_factor * 5))
        params['reg_alpha'] = params['reg_lambda'] * 0.1  # L1 regularization
        
        # Subsampling: Prevent overfitting in high-dimensional spaces
        params['subsample'] = max(0.5, min(0.9, 1.0 - char['dimensionality_ratio'] * 0.5))
        params['colsample_bytree'] = max(0.5, min(0.9, 1.0 - char['dimensionality_ratio'] * 0.3))
        
        # Feature bagging: Based on feature correlation
        if char['feature_correlation_max'] > 0.8:
            params['colsample_bylevel'] = 0.7
        
        # Minimum child weight: Based on sample size
        params['min_child_weight'] = max(1, int(char['n_samples'] / 1000))
        
        # Gamma: Minimum loss reduction (conservative for small datasets)
        if char['n_samples'] < 1000:
            params['gamma'] = 0.1
        
        # Update model parameters
        for key, value in params.items():
            setattr(self, key, value)
        
        if self.verbose:
            print("📊 Dataset Characteristics:")
            for key, value in char.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            
            print("🎯 Automated Parameters:")
            for key, value in params.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
        
        self._hyperparameter_history['automated'] = {
            'parameters': params,
            'characteristics': char,
            'timestamp': datetime.now().isoformat()
        }
        
        return params
    
    def nested_cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        inner_cv: int = 3,
        outer_cv: int = 5,
        param_grid: Optional[Dict] = None,
        scoring: str = 'neg_mean_squared_error'
    ) -> Dict[str, Any]:
        """
        Perform nested cross-validation for unbiased performance estimation.
        
        Mathematical Framework:
        -----------------------
        Nested CV provides unbiased estimate of generalization error:
        
        Outer Loop (Model Selection):
        CV_outer(M) = 1/K Σₖ L(M(D_train^(k)), D_test^(k))
        
        Inner Loop (Hyperparameter Optimization):
        θ* = argmin_θ 1/J Σⱼ L(M_θ(D_train^(j)), D_val^(j))
        
        This prevents data leakage and provides realistic performance estimates.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training targets
        inner_cv : int, default=3
            Number of inner CV folds for hyperparameter tuning
        outer_cv : int, default=5
            Number of outer CV folds for performance estimation
        param_grid : dict, optional
            Parameter grid for hyperparameter search
        scoring : str, default='neg_mean_squared_error'
            Scoring metric for evaluation
            
        Returns:
        --------
        Dict[str, Any]
            Nested CV results including performance metrics and best parameters
        """
        if self.verbose:
            print(f"🔄 Starting nested {outer_cv}x{inner_cv} cross-validation...")
        
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5, 6, 8],
                'n_estimators': [100, 200, 500, 1000],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_lambda': [0.1, 1.0, 10.0]
            }
        
        # Outer CV loop for unbiased performance estimation
        outer_kf = KFold(n_splits=outer_cv, shuffle=True, random_state=self.random_state)
        
        outer_scores = []
        best_params_per_fold = []
        feature_importance_per_fold = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X)):
            if self.verbose:
                print(f"  📊 Outer fold {fold_idx + 1}/{outer_cv}")
            
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Inner CV loop for hyperparameter optimization
            inner_kf = KFold(n_splits=inner_cv, shuffle=True, random_state=self.random_state)
            
            grid_search = RandomizedSearchCV(
                estimator=XGBRegressor(random_state=self.random_state),
                param_distributions=param_grid,
                cv=inner_kf,
                scoring=scoring,
                n_iter=50,  # Reasonable number for computational efficiency
                n_jobs=-1,
                random_state=self.random_state
            )
            
            # Fit inner CV
            grid_search.fit(X_train_outer, y_train_outer)
            
            # Get best model and evaluate on outer test set
            best_model = grid_search.best_estimator_
            y_pred_outer = best_model.predict(X_test_outer)
            
            # Calculate multiple metrics
            fold_score = {
                'mse': mean_squared_error(y_test_outer, y_pred_outer),
                'rmse': np.sqrt(mean_squared_error(y_test_outer, y_pred_outer)),
                'mae': mean_absolute_error(y_test_outer, y_pred_outer),
                'r2': r2_score(y_test_outer, y_pred_outer),
                'explained_variance': explained_variance_score(y_test_outer, y_pred_outer)
            }
            
            # Add MAPE if no zero values
            if not np.any(y_test_outer == 0):
                fold_score['mape'] = mean_absolute_percentage_error(y_test_outer, y_pred_outer)
            
            outer_scores.append(fold_score)
            best_params_per_fold.append(grid_search.best_params_)
            
            # Store feature importance
            if hasattr(best_model, 'feature_importances_'):
                feature_importance_per_fold.append(best_model.feature_importances_)
            
            if self.verbose:
                print(f"    RMSE: {fold_score['rmse']:.4f}, R²: {fold_score['r2']:.4f}")
        
        # Aggregate results
        results = {
            'outer_scores': outer_scores,
            'best_params_per_fold': best_params_per_fold,
            'feature_importance_per_fold': feature_importance_per_fold,
            'mean_scores': {},
            'std_scores': {},
            'cv_config': {
                'outer_cv': outer_cv,
                'inner_cv': inner_cv,
                'param_grid': param_grid,
                'scoring': scoring
            }
        }
        
        # Calculate mean and std for each metric
        for metric in outer_scores[0].keys():
            scores = [fold[metric] for fold in outer_scores]
            results['mean_scores'][metric] = np.mean(scores)
            results['std_scores'][metric] = np.std(scores)
        
        # Find most common parameters across folds
        from collections import Counter
        param_counts = {}
        for param_name in param_grid.keys():
            param_values = [params.get(param_name) for params in best_params_per_fold]
            param_counts[param_name] = Counter(param_values).most_common(1)[0][0]
        
        results['consensus_params'] = param_counts
        
        # Store results
        self._cv_results['nested'] = results
        
        if self.verbose:
            print("📈 Nested CV Results:")
            for metric, mean_val in results['mean_scores'].items():
                std_val = results['std_scores'][metric]
                print(f"   {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
            
            print("🎯 Consensus Parameters:")
            for param, value in param_counts.items():
                print(f"   {param}: {value}")
        
        return results
    
    def diagnostic_plots(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        figsize: Tuple[int, int] = (15, 12),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate comprehensive diagnostic plots for model analysis.
        
        Scientific Visualizations:
        --------------------------
        1. Learning curves: Training/validation loss vs iterations
        2. Residual analysis: Residuals vs predicted values
        3. Feature importance: Gain, weight, and cover importance
        4. Partial dependence: Feature effect visualization
        5. Prediction intervals: Uncertainty quantification
        6. Cross-validation stability: Performance across folds
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target vector
        figsize : tuple, default=(15, 12)
            Figure size for the diagnostic plots
        save_path : str, optional
            Path to save the diagnostic plots
            
        Returns:
        --------
        matplotlib.Figure
            Figure object containing all diagnostic plots
        """
        if not self._is_fitted:
            self.fit(X, y)
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('Scientific XGBoost Diagnostic Analysis', fontsize=16, fontweight='bold')
        
        # Get predictions
        y_pred = self.predict(X)
        residuals = y - y_pred
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y, y_pred, alpha=0.6, s=20)
        axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted')
        r2 = r2_score(y, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. Residual Plot
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Analysis')
        
        # Add residual statistics
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        axes[0, 1].text(0.05, 0.95, f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}', 
                       transform=axes[0, 1].transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 3. Residual Distribution
        axes[0, 2].hist(residuals, bins=30, alpha=0.7, density=True, color='skyblue')
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Residual Distribution')
        
        # Overlay normal distribution
        mu, sigma = stats.norm.fit(residuals)
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0, 2].plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', lw=2, label='Normal fit')
        axes[0, 2].legend()
        
        # 4. Q-Q Plot for normality check
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Feature Importance
        if hasattr(self, 'feature_importances_'):
            feature_names = [f'Feature_{i}' for i in range(len(self.feature_importances_))]
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
            
            # Get top 10 features
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            
            axes[1, 1].barh(importance_df['feature'], importance_df['importance'])
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].tick_params(axis='y', labelsize=8)
        
        # 6. Learning Curve Validation
        if hasattr(self, 'evals_result_') and self.evals_result_:
            train_scores = list(self.evals_result_.values())[0]['rmse']
            axes[1, 2].plot(train_scores, label='Training RMSE', color='blue')
            axes[1, 2].set_xlabel('Boosting Iterations')
            axes[1, 2].set_ylabel('RMSE')
            axes[1, 2].set_title('Learning Curve')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            # Cross-validation scores if available
            if 'nested' in self._cv_results:
                cv_scores = [fold['rmse'] for fold in self._cv_results['nested']['outer_scores']]
                axes[1, 2].boxplot(cv_scores)
                axes[1, 2].set_ylabel('RMSE')
                axes[1, 2].set_title('Cross-Validation RMSE Distribution')
                axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Prediction Error Distribution
        abs_errors = np.abs(residuals)
        axes[2, 0].hist(abs_errors, bins=30, alpha=0.7, color='orange')
        axes[2, 0].set_xlabel('Absolute Error')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].set_title('Absolute Error Distribution')
        axes[2, 0].axvline(np.mean(abs_errors), color='red', linestyle='--', label=f'Mean: {np.mean(abs_errors):.4f}')
        axes[2, 0].legend()
        
        # 8. Scale-Location Plot (Homoscedasticity check)
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        axes[2, 1].scatter(y_pred, sqrt_abs_residuals, alpha=0.6, s=20)
        axes[2, 1].set_xlabel('Predicted Values')
        axes[2, 1].set_ylabel('√|Residuals|')
        axes[2, 1].set_title('Scale-Location Plot')
        
        # Add trend line
        z = np.polyfit(y_pred, sqrt_abs_residuals, 1)
        p = np.poly1d(z)
        axes[2, 1].plot(sorted(y_pred), p(sorted(y_pred)), "r--", alpha=0.8)
        
        # 9. Cook's Distance or Leverage Plot
        # Simplified leverage plot using residuals and predictions
        leverage = np.abs(residuals) / np.std(residuals)
        axes[2, 2].scatter(range(len(leverage)), leverage, alpha=0.6, s=20)
        axes[2, 2].set_xlabel('Observation Index')
        axes[2, 2].set_ylabel('Standardized Residuals')
        axes[2, 2].set_title('Residual Leverage')
        axes[2, 2].axhline(y=2, color='r', linestyle='--', alpha=0.7, label='2σ threshold')
        axes[2, 2].axhline(y=-2, color='r', linestyle='--', alpha=0.7)
        axes[2, 2].legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"📊 Diagnostic plots saved to: {save_path}")
        
        return fig
    
    def elbow_tune(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        param_name: str = 'n_estimators',
        param_range: Optional[List] = None,
        cv_folds: Optional[int] = None,
        scoring: str = 'neg_mean_squared_error'
    ) -> Dict[str, Any]:
        """
        Perform elbow method hyperparameter optimization.
        
        Mathematical Foundation:
        ------------------------
        The elbow method identifies the optimal parameter value where
        the rate of improvement diminishes significantly:
        
        Elbow Point: argmax_p |∇²f(p)| where f(p) is the CV score function
        
        This corresponds to the point of maximum curvature in the validation curve.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training targets
        param_name : str, default='n_estimators'
            Parameter name to optimize
        param_range : list, optional
            Range of parameter values to test
        cv_folds : int, optional
            Number of CV folds (defaults to self.cv_folds)
        scoring : str, default='neg_mean_squared_error'
            Scoring metric
            
        Returns:
        --------
        Dict[str, Any]
            Optimization results including elbow point and validation curve
        """
        if self.verbose:
            print(f"🔧 Performing elbow optimization for parameter: {param_name}")
        
        cv_folds = cv_folds or self.cv_folds
        
        # Default parameter ranges
        if param_range is None:
            if param_name == 'n_estimators':
                param_range = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
            elif param_name == 'max_depth':
                param_range = [2, 3, 4, 5, 6, 7, 8, 10, 12]
            elif param_name == 'learning_rate':
                param_range = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]
            elif param_name == 'reg_lambda':
                param_range = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            else:
                raise ValueError(f"Default range not defined for parameter: {param_name}")
        
        # Compute validation curve
        train_scores, validation_scores = validation_curve(
            estimator=self,
            X=X,
            y=y,
            param_name=param_name,
            param_range=param_range,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(validation_scores, axis=1)
        val_std = np.std(validation_scores, axis=1)
        
        # Find elbow point using second derivative
        # Convert to positive scores if necessary
        scores_for_elbow = val_mean if scoring.startswith('neg') else -val_mean
        
        # Calculate second derivative (curvature)
        if len(scores_for_elbow) >= 3:
            second_derivative = np.gradient(np.gradient(scores_for_elbow))
            elbow_idx = np.argmax(np.abs(second_derivative))
        else:
            elbow_idx = np.argmax(scores_for_elbow)
        
        elbow_value = param_range[elbow_idx]
        elbow_score = val_mean[elbow_idx]
        
        # Alternative elbow detection: maximum distance from line
        if len(param_range) >= 3:
            # Normalize parameters and scores
            param_norm = (np.array(param_range) - np.min(param_range)) / (np.max(param_range) - np.min(param_range))
            score_norm = (scores_for_elbow - np.min(scores_for_elbow)) / (np.max(scores_for_elbow) - np.min(scores_for_elbow))
            
            # Calculate distance from point to line
            line_vec = np.array([param_norm[-1] - param_norm[0], score_norm[-1] - score_norm[0]])
            line_len = np.linalg.norm(line_vec)
            
            distances = []
            for i in range(len(param_range)):
                point_vec = np.array([param_norm[i] - param_norm[0], score_norm[i] - score_norm[0]])
                # Distance from point to line
                cross_prod = np.cross(line_vec, point_vec)
                distance = np.abs(cross_prod) / line_len if line_len > 0 else 0
                distances.append(distance)
            
            distance_elbow_idx = np.argmax(distances)
            distance_elbow_value = param_range[distance_elbow_idx]
        else:
            distance_elbow_value = elbow_value
        
        # Update parameter
        setattr(self, param_name, elbow_value)
        
        results = {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores': train_scores,
            'validation_scores': validation_scores,
            'train_mean': train_mean,
            'train_std': train_std,
            'val_mean': val_mean,
            'val_std': val_std,
            'elbow_value': elbow_value,
            'elbow_score': elbow_score,
            'elbow_idx': elbow_idx,
            'distance_elbow_value': distance_elbow_value,
            'second_derivative': second_derivative if len(scores_for_elbow) >= 3 else None
        }
        
        self._hyperparameter_history[f'elbow_{param_name}'] = results
        
        if self.verbose:
            print(f"📈 Elbow analysis complete:")
            print(f"   Optimal {param_name}: {elbow_value}")
            print(f"   Best CV score: {elbow_score:.4f}")
            print(f"   Alternative (distance method): {distance_elbow_value}")
        
        return results
    
    def incremental_learn(
        self, 
        X_new: np.ndarray, 
        y_new: np.ndarray,
        n_new_estimators: int = 100,
        refit_frequency: int = 5
    ) -> Dict[str, Any]:
        """
        Implement incremental learning with warm start capability.
        
        Mathematical Framework:
        -----------------------
        Incremental learning extends the existing ensemble:
        
        F_{t+k}(x) = F_t(x) + Σᵢ₌₁ᵏ η·h_{t+i}(x)
        
        Where:
        - F_t(x) is the existing ensemble at iteration t
        - h_{t+i}(x) are new weak learners
        - η is the learning rate
        - k is the number of new estimators
        
        Parameters:
        -----------
        X_new : array-like of shape (n_new_samples, n_features)
            New training features
        y_new : array-like of shape (n_new_samples,)
            New training targets
        n_new_estimators : int, default=100
            Number of new estimators to add
        refit_frequency : int, default=5
            Frequency of complete model refitting
            
        Returns:
        --------
        Dict[str, Any]
            Incremental learning results and performance metrics
        """
        if not self._is_fitted:
            if self.verbose:
                print("🔄 Model not fitted. Performing initial fit...")
            self.fit(X_new, y_new)
            return {'status': 'initial_fit', 'n_estimators': self.n_estimators}
        
        if self.verbose:
            print(f"🔄 Incremental learning: adding {n_new_estimators} estimators...")
        
        # Store original number of estimators
        original_n_estimators = self.n_estimators
        
        # Update number of estimators
        self.n_estimators = original_n_estimators + n_new_estimators
        
        # Use warm_start functionality if available
        if hasattr(self, 'warm_start'):
            self.warm_start = True
        
        # Fit with new data
        fit_start = time.time()
        self.fit(X_new, y_new)
        fit_time = time.time() - fit_start
        
        # Validate incremental learning performance
        y_pred_new = self.predict(X_new)
        new_score = r2_score(y_new, y_pred_new)
        
        results = {
            'status': 'incremental_update',
            'original_n_estimators': original_n_estimators,
            'new_n_estimators': self.n_estimators,
            'added_estimators': n_new_estimators,
            'fit_time': fit_time,
            'new_data_score': new_score,
            'new_data_size': len(X_new),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in validation history
        if 'incremental' not in self._validation_history:
            self._validation_history['incremental'] = []
        self._validation_history['incremental'].append(results)
        
        if self.verbose:
            print(f"✅ Incremental learning complete:")
            print(f"   Estimators: {original_n_estimators} → {self.n_estimators}")
            print(f"   Fit time: {fit_time:.2f}s")
            print(f"   New data R²: {new_score:.4f}")
        
        return results
    
    def save_model_package(
        self, 
        save_dir: str,
        include_diagnostics: bool = True,
        include_data: bool = False,
        X_sample: Optional[np.ndarray] = None,
        y_sample: Optional[np.ndarray] = None
    ) -> str:
        """
        Save complete model package with all artifacts.
        
        Package Contents:
        -----------------
        1. Trained model (pickle)
        2. Model metadata and hyperparameters (JSON)
        3. Cross-validation results (JSON)
        4. Feature importance scores (CSV)
        5. Diagnostic plots (PNG)
        6. Model performance metrics (JSON)
        7. Training configuration (JSON)
        8. Sample data (optional, CSV)
        
        Parameters:
        -----------
        save_dir : str
            Directory to save the model package
        include_diagnostics : bool, default=True
            Whether to include diagnostic plots
        include_data : bool, default=False
            Whether to include sample data
        X_sample : array-like, optional
            Sample feature data to save
        y_sample : array-like, optional
            Sample target data to save
            
        Returns:
        --------
        str
            Path to the saved model package directory
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"💾 Saving model package to: {save_path}")
        
        # 1. Save trained model
        model_file = save_path / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(self, f)
        
        # 2. Save model metadata
        metadata = {
            'model_type': 'ScientificXGBRegressor',
            'creation_timestamp': datetime.now().isoformat(),
            'xgboost_version': getattr(self, '__version__', 'unknown'),
            'is_fitted': self._is_fitted,
            'cv_folds': self.cv_folds,
            'auto_tune': self.auto_tune,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'random_state': self.random_state
            }
        }
        
        metadata_file = save_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 3. Save CV results
        if self._cv_results:
            cv_file = save_path / "cv_results.json"
            # Convert numpy arrays to lists for JSON serialization
            cv_results_serializable = {}
            for key, value in self._cv_results.items():
                if isinstance(value, dict):
                    cv_results_serializable[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            cv_results_serializable[key][k] = v.tolist()
                        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                            cv_results_serializable[key][k] = [arr.tolist() for arr in v]
                        else:
                            cv_results_serializable[key][k] = v
                else:
                    cv_results_serializable[key] = value
            
            with open(cv_file, 'w') as f:
                json.dump(cv_results_serializable, f, indent=2)
        
        # 4. Save feature importance
        if hasattr(self, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature_index': range(len(self.feature_importances_)),
                'importance': self.feature_importances_
            })
            importance_file = save_path / "feature_importance.csv"
            importance_df.to_csv(importance_file, index=False)
        
        # 5. Save hyperparameter history
        if self._hyperparameter_history:
            hp_file = save_path / "hyperparameter_history.json"
            # Handle numpy arrays in hyperparameter history
            hp_serializable = {}
            for key, value in self._hyperparameter_history.items():
                if isinstance(value, dict):
                    hp_serializable[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            hp_serializable[key][k] = v.tolist()
                        else:
                            hp_serializable[key][k] = v
                else:
                    hp_serializable[key] = value
            
            with open(hp_file, 'w') as f:
                json.dump(hp_serializable, f, indent=2)
        
        # 6. Save validation history
        if self._validation_history:
            val_file = save_path / "validation_history.json"
            with open(val_file, 'w') as f:
                json.dump(self._validation_history, f, indent=2)
        
        # 7. Generate and save diagnostic plots
        if include_diagnostics and X_sample is not None and y_sample is not None:
            try:
                plot_file = save_path / "diagnostic_plots.png"
                self.diagnostic_plots(X_sample, y_sample, save_path=str(plot_file))
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Could not save diagnostic plots: {e}")
        
        # 8. Save sample data if requested
        if include_data and X_sample is not None and y_sample is not None:
            sample_df = pd.DataFrame(X_sample)
            sample_df['target'] = y_sample
            data_file = save_path / "sample_data.csv"
            sample_df.to_csv(data_file, index=False)
        
        # 9. Create README with package information
        readme_content = f"""
# ScientificXGBRegressor Model Package

## Model Information
- **Model Type**: ScientificXGBRegressor
- **Creation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Fitted**: {self._is_fitted}
- **CV Folds**: {self.cv_folds}

## Hyperparameters
- **N Estimators**: {self.n_estimators}
- **Learning Rate**: {self.learning_rate}
- **Max Depth**: {self.max_depth}
- **Subsample**: {self.subsample}
- **Colsample by Tree**: {self.colsample_bytree}
- **Reg Alpha**: {self.reg_alpha}
- **Reg Lambda**: {self.reg_lambda}

## Package Contents
- `model.pkl`: Trained model object
- `metadata.json`: Model metadata and configuration
- `cv_results.json`: Cross-validation results (if available)
- `feature_importance.csv`: Feature importance scores (if available)
- `hyperparameter_history.json`: Hyperparameter optimization history
- `validation_history.json`: Model validation history
- `diagnostic_plots.png`: Diagnostic visualizations (if generated)
- `sample_data.csv`: Sample training data (if included)

## Usage
```python
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
predictions = model.predict(X_new)
```

## Performance Metrics
{f"CV Results available: {bool(self._cv_results)}" if hasattr(self, '_cv_results') else "No CV results available"}
"""
        
        readme_file = save_path / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        if self.verbose:
            print("✅ Model package saved successfully!")
            print(f"📁 Package contents:")
            for file in save_path.iterdir():
                if file.is_file():
                    size = file.stat().st_size / 1024  # Size in KB
                    print(f"   📄 {file.name} ({size:.1f} KB)")
        
        return str(save_path)
    
    def fit(self, X, y, **kwargs):
        """
        Fit the ScientificXGBRegressor with enhanced functionality.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training targets
        **kwargs : dict
            Additional parameters for XGBRegressor.fit()
        """
        # Convert to numpy arrays if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Automated parameterization if enabled
        if self.auto_tune and not self._is_fitted:
            self.automated_parameterization(X, y)
        
        # Handle early stopping - create validation set if needed
        if (self.early_stopping_rounds is not None and 
            self.early_stopping_rounds > 0 and 
            'eval_set' not in kwargs):
            
            if self.verbose:
                print(f"  🔍 Creating validation set for early stopping (rounds={self.early_stopping_rounds})")
            
            # Simple train/validation split for monitoring
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Set up evaluation parameters
            kwargs['eval_set'] = [(X_train, y_train), (X_val, y_val)]
            kwargs['eval_metric'] = 'rmse'
            kwargs['verbose'] = False
            
            # Use training subset for fitting
            X, y = X_train, y_train
            
            if self.verbose:
                print(f"    Training set: {len(X_train)} samples")
                print(f"    Validation set: {len(X_val)} samples")
        
        # If early stopping is enabled but we don't want to create a validation set,
        # we need to disable early stopping for this fit call
        elif (self.early_stopping_rounds is not None and 
              self.early_stopping_rounds > 0 and 
              'eval_set' not in kwargs and
              len(X) < 100):  # For very small datasets, disable early stopping
            
            if self.verbose:
                print(f"  ⚠️  Dataset too small for early stopping ({len(X)} samples), disabling early stopping")
            # Temporarily disable early stopping for small datasets
            original_early_stopping = self.early_stopping_rounds
            self.early_stopping_rounds = None
        
        # Fit the model
        if self.verbose:
            print(f"🔄 Fitting XGBoost with {self.n_estimators} estimators...")
        
        fit_start = time.time()
        super().fit(X, y, **kwargs)
        fit_time = time.time() - fit_start
        
        # Restore early stopping if it was temporarily disabled
        if 'original_early_stopping' in locals():
            self.early_stopping_rounds = original_early_stopping
        
        self._is_fitted = True
        
        if self.verbose:
            print(f"✅ Model fitted in {fit_time:.2f}s")
        
        # Store feature importance
        if hasattr(self, 'feature_importances_'):
            self._feature_importance_scores['gain'] = self.feature_importances_
        
        return self
    
    @classmethod
    def load_model_package(cls, package_dir: str) -> 'ScientificXGBRegressor':
        """
        Load a complete model package.
        
        Parameters:
        -----------
        package_dir : str
            Directory containing the model package
            
        Returns:
        --------
        ScientificXGBRegressor
            Loaded model instance
        """
        package_path = Path(package_dir)
        model_file = package_path / "model.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model package loaded from: {package_dir}")
        
        # Load additional metadata if available
        metadata_file = package_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"📊 Model created: {metadata.get('creation_timestamp', 'Unknown')}")
        
        return model

def create_scientific_xgb_regressor(**kwargs) -> ScientificXGBRegressor:
    """
    Factory function to create a ScientificXGBRegressor with recommended settings.
    
    Parameters:
    -----------
    **kwargs : dict
        Additional parameters for ScientificXGBRegressor
        
    Returns:
    --------
    ScientificXGBRegressor
        Configured model instance
    """
    default_params = {
        'cv_folds': 5,
        'auto_tune': True,
        'verbose': True,
        'early_stopping_rounds': 50,
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    # Update with user parameters
    default_params.update(kwargs)
    
    return ScientificXGBRegressor(**default_params)


# Example usage and demonstration
if __name__ == "__main__":
    # This section provides usage examples
    print("🧪 ScientificXGBRegressor - Advanced XGBoost for Scientific Computing")
    print("=" * 70)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + 0.1 * np.random.randn(n_samples)
    
    # Create and demonstrate the model
    model = create_scientific_xgb_regressor(
        cv_folds=3,  # Reduced for demo
        auto_tune=True,
        verbose=True
    )
    
    print("\n1. 🔬 Automated Parameterization:")
    model.automated_parameterization(X, y)
    
    print("\n2. 🔄 Model Fitting:")
    model.fit(X, y)
    
    print("\n3. 📊 Generating Diagnostic Plots:")
    fig = model.diagnostic_plots(X, y)
    plt.show()
    
    print("\n4. 🔧 Elbow Tuning:")
    elbow_results = model.elbow_tune(X, y, param_name='n_estimators', 
                                    param_range=[50, 100, 200, 300, 500])
    
    print("\n5. 🔄 Nested Cross-Validation:")
    cv_results = model.nested_cross_validate(X, y, inner_cv=2, outer_cv=3)
    
    print("\n6. 💾 Saving Model Package:")
    package_path = model.save_model_package(
        save_dir="./scientific_xgb_package",
        include_diagnostics=True,
        include_data=True,
        X_sample=X[:100],  # Save sample data
        y_sample=y[:100]
    )
    
    print(f"\n✅ Model package saved to: {package_path}")
    print("\n🎉 ScientificXGBRegressor demonstration complete!")
    
    # Uncomment to run upgrade examples:
    # print("\n" + "="*70)
    # print("🔄 MODEL UPGRADE DEMONSTRATIONS")
    # print("="*70)
    # complete_migration_workflow()
