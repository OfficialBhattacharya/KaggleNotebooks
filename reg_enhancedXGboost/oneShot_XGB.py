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
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: Pandas not available due to compatibility issues")
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
import gc
from dataclasses import dataclass

# Scientific computing and ML libraries
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold, GridSearchCV, 
    RandomizedSearchCV, validation_curve, train_test_split
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

# GPU Detection and Support
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# CUDA availability check
try:
    import subprocess
    import os
    CUDA_AVAILABLE = False
    try:
        # Check if nvidia-smi is available
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            CUDA_AVAILABLE = True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass
except ImportError:
    CUDA_AVAILABLE = False


class GPUManager:
    """
    GPU Detection and Management for XGBoost
    """
    
    @staticmethod
    def detect_gpus() -> Dict[str, Any]:
        """
        Detect available GPUs and their capabilities.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing GPU information
        """
        gpu_info = {
            'available': False,
            'count': 0,
            'devices': [],
            'cuda_available': CUDA_AVAILABLE,
            'memory_info': [],
            'driver_version': None,
            'cuda_version': None
        }
        
        # Check CUDA availability first
        if not CUDA_AVAILABLE:
            return gpu_info
        
        # Try multiple methods to detect GPUs
        
        # Method 1: GPUtil (most reliable)
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info['available'] = True
                    gpu_info['count'] = len(gpus)
                    for i, gpu in enumerate(gpus):
                        device_info = {
                            'id': gpu.id,
                            'name': gpu.name,
                            'memory_total': gpu.memoryTotal,
                            'memory_free': gpu.memoryFree,
                            'memory_used': gpu.memoryUsed,
                            'utilization': gpu.load * 100,
                            'temperature': gpu.temperature
                        }
                        gpu_info['devices'].append(device_info)
                        gpu_info['memory_info'].append({
                            'total_mb': gpu.memoryTotal,
                            'free_mb': gpu.memoryFree,
                            'used_mb': gpu.memoryUsed
                        })
                    return gpu_info
            except Exception:
                pass
        
        # Method 2: PyTorch CUDA detection
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    gpu_info['available'] = True
                    gpu_info['count'] = torch.cuda.device_count()
                    gpu_info['cuda_version'] = torch.version.cuda
                    
                    for i in range(gpu_info['count']):
                        props = torch.cuda.get_device_properties(i)
                        device_info = {
                            'id': i,
                            'name': props.name,
                            'memory_total': props.total_memory // (1024**2),  # Convert to MB
                            'compute_capability': f"{props.major}.{props.minor}"
                        }
                        gpu_info['devices'].append(device_info)
                        
                        # Get memory info
                        torch.cuda.set_device(i)
                        memory_free = torch.cuda.memory_reserved(i) // (1024**2)
                        memory_total = props.total_memory // (1024**2)
                        gpu_info['memory_info'].append({
                            'total_mb': memory_total,
                            'free_mb': memory_free,
                            'used_mb': memory_total - memory_free
                        })
                    return gpu_info
            except Exception:
                pass
        
        # Method 3: TensorFlow GPU detection
        if TF_AVAILABLE:
            try:
                physical_devices = tf.config.list_physical_devices('GPU')
                if physical_devices:
                    gpu_info['available'] = True
                    gpu_info['count'] = len(physical_devices)
                    for i, device in enumerate(physical_devices):
                        device_info = {
                            'id': i,
                            'name': device.name,
                            'device_type': device.device_type
                        }
                        gpu_info['devices'].append(device_info)
                    return gpu_info
            except Exception:
                pass
        
        # Method 4: Direct nvidia-smi parsing
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    gpu_info['available'] = True
                    gpu_info['count'] = len(lines)
                    
                    for line in lines:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            device_info = {
                                'id': int(parts[0]),
                                'name': parts[1],
                                'memory_total': int(parts[2]),
                                'memory_free': int(parts[3]),
                                'memory_used': int(parts[4]),
                                'utilization': float(parts[5]) if parts[5] != '[Not Supported]' else 0.0,
                                'temperature': float(parts[6]) if parts[6] != '[Not Supported]' else 0.0
                            }
                            gpu_info['devices'].append(device_info)
                            gpu_info['memory_info'].append({
                                'total_mb': int(parts[2]),
                                'free_mb': int(parts[3]),
                                'used_mb': int(parts[4])
                            })
                    return gpu_info
        except Exception:
            pass
        
        return gpu_info
    
    @staticmethod
    def get_optimal_gpu_config(gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine optimal GPU configuration for XGBoost.
        
        Parameters:
        -----------
        gpu_info : Dict[str, Any]
            GPU information from detect_gpus()
            
        Returns:
        --------
        Dict[str, Any]
            Optimal GPU configuration
        """
        config = {
            'tree_method': 'auto',
            'gpu_id': None,
            'n_gpus': 0,
            'predictor': 'auto',
            'use_gpu': False
        }
        
        if not gpu_info['available'] or gpu_info['count'] == 0:
            return config
        
        # GPU is available, configure for GPU usage
        config['use_gpu'] = True
        config['tree_method'] = 'gpu_hist'  # GPU-accelerated histogram
        config['predictor'] = 'gpu_predictor'
        config['n_gpus'] = gpu_info['count']
        
        # Select best GPU based on available memory
        if gpu_info['memory_info']:
            best_gpu_idx = 0
            max_free_memory = 0
            
            for i, memory_info in enumerate(gpu_info['memory_info']):
                free_memory = memory_info.get('free_mb', 0)
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu_idx = i
            
            config['gpu_id'] = best_gpu_idx
        else:
            config['gpu_id'] = 0  # Default to first GPU
        
        # Use all GPUs if multiple are available and have sufficient memory
        min_memory_threshold = 2048  # 2GB minimum
        suitable_gpus = []
        
        for i, memory_info in enumerate(gpu_info['memory_info']):
            if memory_info.get('free_mb', 0) >= min_memory_threshold:
                suitable_gpus.append(i)
        
        if len(suitable_gpus) > 1:
            # Use multi-GPU setup
            config['gpu_id'] = ','.join(map(str, suitable_gpus))
            config['n_gpus'] = len(suitable_gpus)
        
        return config

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
        use_gpu: Optional[bool] = None,  # None = auto-detect, True = force GPU, False = force CPU
        **kwargs
    ):
        # Store scientific parameters separately (don't pass to parent)
        self.cv_folds = cv_folds
        self.auto_tune = auto_tune
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self._is_fitted = False
        self._cv_results = {}
        self._feature_importance_scores = {}
        self._validation_history = {}
        self._hyperparameter_history = {}
        self.performance_history = []  # Initialize performance history
        
        # GPU Detection and Configuration
        if self.verbose:
            print("🔍 Detecting GPU availability...")
        
        self._gpu_info = GPUManager.detect_gpus()
        self._gpu_config = GPUManager.get_optimal_gpu_config(self._gpu_info)
        
        # Determine GPU usage
        if use_gpu is None:
            # Auto-detect GPU usage
            use_gpu_final = self._gpu_config['use_gpu']
        else:
            # User specified GPU preference
            use_gpu_final = use_gpu and self._gpu_config['use_gpu']
        
        # Store GPU usage decision as instance attribute for sklearn compatibility
        self.use_gpu = use_gpu_final
        
        # Display GPU information
        if self.verbose:
            if self._gpu_info['available']:
                gpu_names = [device['name'] for device in self._gpu_info['devices']]
                print(f"🎮 Found {self._gpu_info['count']} GPU(s): {', '.join(gpu_names)}")
                
                if use_gpu_final:
                    if self._gpu_config['n_gpus'] > 1:
                        print(f"⚡ Using {self._gpu_config['n_gpus']} GPU(s) for training (IDs: {self._gpu_config['gpu_id']})")
                    else:
                        print(f"⚡ Using GPU {self._gpu_config['gpu_id']} for training")
                    print(f"🚀 Tree method: {self._gpu_config['tree_method']}")
                    print(f"🎯 Predictor: {self._gpu_config['predictor']}")
                else:
                    print("💻 Using CPU for training (GPU disabled by user)")
            else:
                print("💻 No GPUs detected, using CPU for training")
        
        # Prepare XGBoost parameters only (exclude scientific parameters)
        xgb_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'early_stopping_rounds': early_stopping_rounds,
            'eval_metric': 'rmse',  # Set eval_metric here to avoid deprecation warning
        }
        
        # Add GPU parameters if GPU is available and enabled
        if use_gpu_final:
            xgb_params.update({
                'tree_method': self._gpu_config['tree_method'],
                'predictor': self._gpu_config['predictor'],
                'gpu_id': self._gpu_config['gpu_id']
            })
        
        # Add any additional XGBoost parameters from kwargs
        # Filter out any GPU-related parameters that might conflict
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['tree_method', 'predictor', 'gpu_id', 'use_gpu']}
        xgb_params.update(filtered_kwargs)
        
        # Store GPU configuration for later use
        self._using_gpu = use_gpu_final
        self._gpu_params = {k: v for k, v in xgb_params.items() 
                           if k in ['tree_method', 'predictor', 'gpu_id']} if use_gpu_final else {}
        
        # Initialize XGBRegressor with only XGBoost parameters
        super().__init__(**xgb_params)
        
        if self.verbose:
            print(f"🧪 ScientificXGBRegressor initialized with {cv_folds}-fold CV")
            if auto_tune:
                print("🔧 Automatic hyperparameter tuning enabled")
            if use_gpu_final:
                print("⚡ GPU acceleration enabled")

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
        
        # Data validation and debugging
        if self.verbose:
            print(f"   📊 Data validation:")
            print(f"      X shape: {X.shape}, dtype: {X.dtype}")
            print(f"      y shape: {y.shape}, dtype: {y.dtype}")
            
            # Check for problematic values
            if hasattr(X, 'isnull'):
                nan_count = X.isnull().sum().sum()
                print(f"      NaN values in X: {nan_count}")
            else:
                nan_count = np.isnan(X).sum()
                print(f"      NaN values in X: {nan_count}")
            
            if hasattr(y, 'isnull'):
                y_nan_count = y.isnull().sum()
                print(f"      NaN values in y: {y_nan_count}")
            else:
                y_nan_count = np.isnan(y).sum()
                print(f"      NaN values in y: {y_nan_count}")
            
            # Check for infinite values
            if np.any(np.isinf(X)):
                print(f"      ⚠️ Infinite values detected in X")
            if np.any(np.isinf(y)):
                print(f"      ⚠️ Infinite values detected in y")
        
        char = self._calculate_data_characteristics(X, y)
        
        # Scientific parameter calculation with explicit type conversion
        params = {}
        
        # Learning rate: Adaptive based on sample size and complexity
        base_lr = min(0.3, 1.0 / np.sqrt(char['n_samples']))
        complexity_factor = min(2.0, char['dimensionality_ratio'] * 10)
        params['learning_rate'] = float(max(0.01, base_lr / complexity_factor))
        
        # Number of estimators: Based on learning rate and convergence theory
        # CRITICAL FIX: Ensure n_estimators is always an integer
        raw_n_estimators = min(2000, int(1000 / params['learning_rate']))
        params['n_estimators'] = int(raw_n_estimators)
        
        # Maximum depth: Information-theoretic approach
        base_depth = int(np.log2(char['n_samples'])) + 1
        noise_adjustment = -1 if char['signal_to_noise'] < 5 else 0
        params['max_depth'] = int(max(3, min(10, base_depth + noise_adjustment)))
        
        # Regularization: Based on dimensionality and overfitting risk
        reg_factor = char['dimensionality_ratio'] * char['feature_correlation_max']
        params['reg_lambda'] = float(min(10.0, max(0.1, reg_factor * 5)))
        params['reg_alpha'] = float(params['reg_lambda'] * 0.1)  # L1 regularization
        
        # Subsampling: Prevent overfitting in high-dimensional spaces
        params['subsample'] = float(max(0.5, min(0.9, 1.0 - char['dimensionality_ratio'] * 0.5)))
        params['colsample_bytree'] = float(max(0.5, min(0.9, 1.0 - char['dimensionality_ratio'] * 0.3)))
        
        # Feature bagging: Based on feature correlation
        if char['feature_correlation_max'] > 0.8:
            params['colsample_bylevel'] = float(0.7)
        
        # Minimum child weight: Based on sample size
        params['min_child_weight'] = int(max(1, int(char['n_samples'] / 1000)))
        
        # Gamma: Minimum loss reduction (conservative for small datasets)
        if char['n_samples'] < 1000:
            params['gamma'] = float(0.1)
        
        # GPU-specific optimizations
        if self._using_gpu and self._gpu_info['available']:
            if self.verbose:
                print("⚡ Applying GPU-specific optimizations...")
            
            # GPU memory optimization: adjust tree_method and max_depth for GPU efficiency
            available_memory = 0
            if self._gpu_info['memory_info']:
                available_memory = max(info['free_mb'] for info in self._gpu_info['memory_info'])
            
            data_size_mb = X.nbytes / (1024**2) if hasattr(X, 'nbytes') else 0
            
            # Optimize based on GPU memory and dataset size
            if available_memory > 0:
                memory_ratio = data_size_mb / available_memory
                
                if memory_ratio > 0.7:  # Dataset uses >70% of GPU memory
                    if self.verbose:
                        print(f"   📊 Large dataset ({data_size_mb:.1f}MB) relative to GPU memory ({available_memory:.1f}MB)")
                        print("   🔧 Adjusting parameters for memory efficiency...")
                    
                    # Reduce memory usage
                    params['max_depth'] = int(min(params['max_depth'], 6))  # Limit tree depth
                    params['subsample'] = float(min(params['subsample'], 0.8))  # Reduce sample size
                    params['colsample_bytree'] = float(min(params['colsample_bytree'], 0.8))
                    
                elif memory_ratio < 0.3:  # Dataset uses <30% of GPU memory
                    if self.verbose:
                        print(f"   📊 Small dataset ({data_size_mb:.1f}MB) relative to GPU memory ({available_memory:.1f}MB)")
                        print("   🚀 Increasing parameters for better GPU utilization...")
                    
                    # Can afford more complex models
                    params['max_depth'] = int(min(params['max_depth'] + 1, 10))
                    # CRITICAL FIX: Ensure n_estimators multiplication results in integer
                    increased_estimators = int(params['n_estimators'] * 1.2)
                    params['n_estimators'] = int(min(increased_estimators, 3000))
            
            # Multi-GPU optimizations
            if self._gpu_config['n_gpus'] > 1:
                if self.verbose:
                    print(f"   🎮 Optimizing for {self._gpu_config['n_gpus']} GPUs...")
                
                # Increase batch size effectively by allowing more complex models
                if char['n_samples'] > 5000:  # Only for larger datasets
                    # CRITICAL FIX: Ensure n_estimators multiplication results in integer
                    increased_estimators = int(params['n_estimators'] * 1.5)
                    params['n_estimators'] = int(min(increased_estimators, 4000))
                    params['learning_rate'] = float(params['learning_rate'] * 0.9)  # Slightly reduce LR for stability
            
            # GPU histogram method optimizations
            if char['n_features'] > 100:
                # For high-dimensional data, GPU histogram is very efficient
                if self.verbose:
                    print("   📊 High-dimensional data detected, optimizing for GPU histogram method...")
                params['max_bin'] = int(min(256, max(64, char['n_features'])))  # Optimize bin count
            
            # Store GPU-specific optimizations
            gpu_optimizations = {
                'memory_optimization': memory_ratio > 0.7 if available_memory > 0 else False,
                'multi_gpu_optimization': self._gpu_config['n_gpus'] > 1,
                'high_dim_optimization': char['n_features'] > 100,
                'available_memory_mb': available_memory,
                'data_size_mb': data_size_mb,
                'memory_ratio': memory_ratio if available_memory > 0 else 0.0
            }
            
            self._hyperparameter_history['gpu_optimizations'] = gpu_optimizations
        
        # CRITICAL: Final parameter validation and type conversion
        validated_params = {}
        integer_params = ['n_estimators', 'max_depth', 'min_child_weight', 'max_bin']
        float_params = ['learning_rate', 'reg_lambda', 'reg_alpha', 'subsample', 'colsample_bytree', 'colsample_bylevel', 'gamma']
        
        for key, value in params.items():
            if key in integer_params:
                validated_params[key] = int(value)
            elif key in float_params:
                validated_params[key] = float(value)
            else:
                validated_params[key] = value
        
        # Update model parameters with validated types
        for key, value in validated_params.items():
            setattr(self, key, value)
        
        if self.verbose:
            print("📊 Dataset Characteristics:")
            for key, value in char.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            
            print("🎯 Automated Parameters (with type validation):")
            for key, value in validated_params.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f} (type: {type(value).__name__})")
                else:
                    print(f"   {key}: {value} (type: {type(value).__name__})")
            
            if self._using_gpu:
                print("⚡ GPU-Enhanced Configuration Applied")
        
        self._hyperparameter_history['automated'] = {
            'parameters': validated_params,
            'characteristics': char,
            'gpu_enhanced': self._using_gpu,
            'timestamp': datetime.now().isoformat()
        }
        
        return validated_params
    
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
        
        # Suppress numpy divide warnings temporarily
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            warnings.filterwarnings("ignore", message="divide by zero encountered")
            
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
        
        # Suppress numpy warnings for mathematical calculations
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            warnings.filterwarnings("ignore", message="divide by zero encountered")
            
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
                # Normalize parameters and scores - handle division by zero
                param_min, param_max = np.min(param_range), np.max(param_range)
                score_min, score_max = np.min(scores_for_elbow), np.max(scores_for_elbow)
                
                # Check for constant values to avoid division by zero
                param_range_span = param_max - param_min
                score_range_span = score_max - score_min
                
                if param_range_span > 1e-10 and score_range_span > 1e-10:
                    param_norm = (np.array(param_range) - param_min) / param_range_span
                    score_norm = (scores_for_elbow - score_min) / score_range_span
                    
                    # Calculate distance from point to line
                    line_vec = np.array([param_norm[-1] - param_norm[0], score_norm[-1] - score_norm[0]])
                    line_len = np.linalg.norm(line_vec)
                    
                    distances = []
                    for i in range(len(param_range)):
                        point_vec = np.array([param_norm[i] - param_norm[0], score_norm[i] - score_norm[0]])
                        # Distance from point to line
                        if line_len > 1e-10:  # Avoid division by very small numbers
                            cross_prod = np.cross(line_vec, point_vec)
                            distance = np.abs(cross_prod) / line_len
                        else:
                            distance = 0.0
                        distances.append(distance)
                    
                    distance_elbow_idx = np.argmax(distances)
                    distance_elbow_value = param_range[distance_elbow_idx]
                else:
                    # If parameters or scores are constant, use curvature elbow
                    distance_elbow_value = elbow_value
            else:
                distance_elbow_value = elbow_value
        
        # Update parameter
        setattr(self, param_name, distance_elbow_value)
        
        results = {
            'param_name': param_name,
            'param_range': param_range,
            'train_scores': train_scores,
            'validation_scores': validation_scores,
            'train_mean': train_mean,
            'train_std': train_std,
            'val_mean': val_mean,
            'val_std': val_std,
            'elbow_value': distance_elbow_value,
            'elbow_score': elbow_score,
            'elbow_idx': elbow_idx,
            'distance_elbow_value': distance_elbow_value,
            'second_derivative': second_derivative if len(scores_for_elbow) >= 3 else None
        }
        
        self._hyperparameter_history[f'elbow_{param_name}'] = results
        
        if self.verbose:
            print(f"📈 Elbow analysis complete:")
            print(f"   Optimal {param_name}: {distance_elbow_value}")
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
        
        F_{t+k}(x) = F_t(x) + Σᵢ₌₁ᵏ η·h_{t+i}(x)
        
        Where:
        - F_t(x) is the existing ensemble at iteration t
        - h_{t+i}(x) are new weak learners
        - η is the learning rate
        - k is the number of new estimators
        
        Parameters:
        -----------
        X_new : array-like of shape (n_new_samples, n_features)
            New training features (already preprocessed)
        y_new : array-like of shape (n_new_samples,)
            New training targets (already preprocessed)
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
            # Data is already preprocessed, so fit directly
            self.fit(X_new, y_new)
            return {'status': 'initial_fit', 'n_estimators': self.n_estimators}
        
        if self.verbose:
            print(f"🔄 Incremental learning: adding {n_new_estimators} estimators...")
        
        # Store original number of estimators
        original_n_estimators = self.n_estimators
        
        # Update number of estimators (ensure it's an integer)
        self.n_estimators = int(original_n_estimators + n_new_estimators)
        
        # Use warm_start functionality if available
        if hasattr(self, 'warm_start'):
            self.warm_start = True
        
        # Fit with new data (already preprocessed)
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
            'final_features': X_new.shape[1],  # Track feature count
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
            print(f"   Features: {X_new.shape[1]} (preprocessed)")
        
        return results
    
    def _make_json_serializable(self, obj):
        """
        Convert numpy types and other non-serializable objects to JSON-serializable types.
        
        Parameters:
        -----------
        obj : Any
            Object to convert
            
        Returns:
        --------
        Any
            JSON-serializable version of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        else:
            return obj

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
            # Convert numpy arrays to lists for JSON serialization using helper
            cv_results_serializable = self._make_json_serializable(self._cv_results)
            
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
            # Use helper function to handle all numpy types
            hp_serializable = self._make_json_serializable(self._hyperparameter_history)
            
            with open(hp_file, 'w') as f:
                json.dump(hp_serializable, f, indent=2)
        
        # 6. Save validation history
        if self._validation_history:
            val_file = save_path / "validation_history.json"
            # Use helper function for validation history too
            val_serializable = self._make_json_serializable(self._validation_history)
            with open(val_file, 'w') as f:
                json.dump(val_serializable, f, indent=2)
        
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
    
    def _preprocess_data(self, X, y, operation_name="training", problematic_columns=None, good_columns=None):
        """
        Preprocess data to handle problematic values and ensure XGBoost compatibility.
        
        Parameters:
        -----------
        X : array-like
            Feature data
        y : array-like
            Target data
        operation_name : str
            Name of the operation for logging
        problematic_columns : list, optional
            Pre-identified problematic column indices to remove
        good_columns : list, optional
            Pre-identified good column indices to keep
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Cleaned X and y arrays
        """
        if self.verbose:
            print(f"🧹 Preprocessing data for {operation_name}...")
        
        # Convert to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        original_shape = X.shape
        
        # Apply consistent column removal if specified
        if good_columns is not None:
            if self.verbose:
                print(f"   🔧 Applying consistent column selection: keeping {len(good_columns)} of {X.shape[1]} columns")
            X = X[:, good_columns]
            if self.verbose:
                print(f"   📊 Data shape: {original_shape} → {X.shape}")
        elif problematic_columns is not None:
            if self.verbose:
                print(f"   🗑️ Removing pre-identified problematic columns: {problematic_columns}")
            good_columns = [i for i in range(X.shape[1]) if i not in problematic_columns]
            X = X[:, good_columns]
            if self.verbose:
                print(f"   📊 Data shape: {original_shape} → {X.shape}")
        else:
            # Original column detection logic (for backward compatibility)
            problematic_columns = []
            
            # Check each column for issues
            for i in range(X.shape[1]):
                col_data = X[:, i]
                
                # Check for all NaN
                if np.all(np.isnan(col_data)):
                    problematic_columns.append(i)
                    if self.verbose:
                        print(f"   ⚠️ Column {i}: All NaN values")
                    continue
                
                # Check for all infinite
                if np.all(np.isinf(col_data)):
                    problematic_columns.append(i)
                    if self.verbose:
                        print(f"   ⚠️ Column {i}: All infinite values")
                    continue
                
                # Check for constant values (no variance)
                if len(np.unique(col_data[~np.isnan(col_data)])) <= 1:
                    problematic_columns.append(i)
                    if self.verbose:
                        print(f"   ⚠️ Column {i}: Constant values (no variance)")
                    continue
            
            # Remove problematic columns
            if problematic_columns:
                if self.verbose:
                    print(f"   🗑️ Removing {len(problematic_columns)} problematic columns: {problematic_columns}")
                
                # Keep only good columns
                good_columns = [i for i in range(X.shape[1]) if i not in problematic_columns]
                X = X[:, good_columns]
                
                if self.verbose:
                    print(f"   📊 Data shape: {original_shape} → {X.shape}")
        
        # Handle remaining NaN values
        nan_count_X = np.isnan(X).sum()
        if nan_count_X > 0:
            if self.verbose:
                print(f"   🔧 Handling {nan_count_X} remaining NaN values in X")
            
            # Replace NaN with column means
            col_means = np.nanmean(X, axis=0)
            for i in range(X.shape[1]):
                col_mask = np.isnan(X[:, i])
                if np.any(col_mask):
                    X[col_mask, i] = col_means[i] if not np.isnan(col_means[i]) else 0.0
        
        # Handle infinite values
        inf_count_X = np.isinf(X).sum()
        if inf_count_X > 0:
            if self.verbose:
                print(f"   🔧 Handling {inf_count_X} infinite values in X")
            
            # Replace infinite with very large finite values
            X[np.isposinf(X)] = np.finfo(np.float64).max / 1e6
            X[np.isneginf(X)] = np.finfo(np.float64).min / 1e6
        
        # Handle target variable
        nan_count_y = np.isnan(y).sum()
        if nan_count_y > 0:
            if self.verbose:
                print(f"   🔧 Handling {nan_count_y} NaN values in y")
            
            # For regression, use mean imputation
            y_mean = np.nanmean(y)
            y[np.isnan(y)] = y_mean if not np.isnan(y_mean) else 0.0
        
        inf_count_y = np.isinf(y).sum()
        if inf_count_y > 0:
            if self.verbose:
                print(f"   🔧 Handling {inf_count_y} infinite values in y")
            
            y[np.isposinf(y)] = np.finfo(np.float64).max / 1e6
            y[np.isneginf(y)] = np.finfo(np.float64).min / 1e6
        
        # Final validation
        assert not np.any(np.isnan(X)), "NaN values still present in X after preprocessing"
        assert not np.any(np.isnan(y)), "NaN values still present in y after preprocessing"
        assert not np.any(np.isinf(X)), "Infinite values still present in X after preprocessing"
        assert not np.any(np.isinf(y)), "Infinite values still present in y after preprocessing"
        
        if self.verbose:
            print(f"   ✅ Data preprocessing complete for {operation_name}")
            print(f"      Final X shape: {X.shape}, dtype: {X.dtype}")
            print(f"      Final y shape: {y.shape}, dtype: {y.dtype}")
        
        return X, y

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
        # Preprocess data first
        X, y = self._preprocess_data(X, y, "initial training")
        
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
            
            # Set up evaluation parameters (don't set eval_metric here since it's in constructor)
            kwargs['eval_set'] = [(X_train, y_train), (X_val, y_val)]
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
        
        # Final parameter validation before fitting
        if self.verbose:
            print(f"🔄 Fitting XGBoost with validated parameters:")
            print(f"   n_estimators: {self.n_estimators} (type: {type(self.n_estimators).__name__})")
            print(f"   learning_rate: {self.learning_rate} (type: {type(self.learning_rate).__name__})")
            print(f"   max_depth: {self.max_depth} (type: {type(self.max_depth).__name__})")
        
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

    # ========================================
    # GPU MANAGEMENT METHODS
    # ========================================
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get comprehensive GPU information and current configuration.
        
        Returns:
        --------
        Dict[str, Any]
            Complete GPU information including devices, memory, and current config
        """
        return {
            'gpu_info': self._gpu_info,
            'gpu_config': self._gpu_config,
            'using_gpu': self._using_gpu,
            'gpu_params': self._gpu_params,
            'cuda_available': CUDA_AVAILABLE,
            'gputil_available': GPUTIL_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'tf_available': TF_AVAILABLE
        }
    
    def print_gpu_status(self) -> None:
        """
        Print detailed GPU status information.
        """
        print("🎮 GPU Status Report")
        print("=" * 50)
        
        gpu_info = self.get_gpu_info()
        
        print(f"CUDA Available: {gpu_info['cuda_available']}")
        print(f"GPUtil Available: {gpu_info['gputil_available']}")
        print(f"PyTorch Available: {gpu_info['torch_available']}")
        print(f"TensorFlow Available: {gpu_info['tf_available']}")
        print()
        
        if gpu_info['gpu_info']['available']:
            print(f"GPUs Detected: {gpu_info['gpu_info']['count']}")
            print()
            
            for i, device in enumerate(gpu_info['gpu_info']['devices']):
                print(f"GPU {device.get('id', i)}:")
                print(f"  Name: {device.get('name', 'Unknown')}")
                if 'memory_total' in device:
                    print(f"  Memory: {device['memory_total']:.0f} MB total")
                    if 'memory_free' in device:
                        print(f"  Free: {device['memory_free']:.0f} MB ({device['memory_free']/device['memory_total']*100:.1f}%)")
                if 'utilization' in device:
                    print(f"  Utilization: {device['utilization']:.1f}%")
                if 'temperature' in device:
                    print(f"  Temperature: {device['temperature']:.1f}°C")
                print()
            
            print(f"Current XGBoost Configuration:")
            print(f"  Using GPU: {gpu_info['using_gpu']}")
            if gpu_info['using_gpu']:
                for param, value in gpu_info['gpu_params'].items():
                    print(f"  {param}: {value}")
        else:
            print("No GPUs detected or available")
        
        print("=" * 50)
    
    def switch_to_gpu(self, gpu_id: Optional[Union[int, str]] = None) -> bool:
        """
        Switch the model to use GPU acceleration.
        
        Parameters:
        -----------
        gpu_id : int, str, optional
            Specific GPU ID to use. If None, uses the optimal GPU.
            Can be a single ID (0) or multiple IDs ("0,1,2")
            
        Returns:
        --------
        bool
            True if successfully switched to GPU, False otherwise
        """
        if not self._gpu_info['available']:
            if self.verbose:
                print("❌ No GPUs available for switching")
            return False
        
        if gpu_id is None:
            gpu_id = self._gpu_config['gpu_id']
        
        # Update GPU parameters
        gpu_params = {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': gpu_id
        }
        
        # Set parameters on the model
        try:
            self.set_params(**gpu_params)
            self._using_gpu = True
            self._gpu_params = gpu_params
            
            if self.verbose:
                print(f"✅ Switched to GPU acceleration (GPU ID: {gpu_id})")
            return True
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to switch to GPU: {e}")
            return False
    
    def switch_to_cpu(self) -> bool:
        """
        Switch the model to use CPU processing.
        
        Returns:
        --------
        bool
            True if successfully switched to CPU
        """
        cpu_params = {
            'tree_method': 'hist',  # CPU histogram method
            'predictor': 'cpu_predictor',
            'gpu_id': None
        }
        
        try:
            # Remove GPU parameters and set CPU parameters
            self.set_params(**cpu_params)
            self._using_gpu = False
            self._gpu_params = {}
            
            if self.verbose:
                print("✅ Switched to CPU processing")
            return True
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to switch to CPU: {e}")
            return False
    
    def optimize_gpu_usage(self, X: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize GPU usage based on dataset size and available GPU memory.
        
        Parameters:
        -----------
        X : array-like, optional
            Training data to analyze for optimal GPU configuration
            
        Returns:
        --------
        Dict[str, Any]
            Optimization results and recommendations
        """
        if not self._gpu_info['available']:
            return {'status': 'no_gpu', 'recommendation': 'Use CPU processing'}
        
        optimization_results = {
            'status': 'optimized',
            'original_config': self._gpu_params.copy(),
            'recommendations': [],
            'applied_changes': []
        }
        
        # Analyze dataset if provided
        if X is not None:
            data_size_mb = X.nbytes / (1024**2)
            n_samples, n_features = X.shape
            
            # Get available GPU memory
            max_gpu_memory = 0
            if self._gpu_info['memory_info']:
                max_gpu_memory = max(info['free_mb'] for info in self._gpu_info['memory_info'])
            
            optimization_results['dataset_analysis'] = {
                'data_size_mb': data_size_mb,
                'n_samples': n_samples,
                'n_features': n_features,
                'max_gpu_memory_mb': max_gpu_memory
            }
            
            # Memory-based recommendations
            if data_size_mb > max_gpu_memory * 0.7:  # Use 70% of GPU memory as threshold
                optimization_results['recommendations'].append(
                    f"Dataset ({data_size_mb:.1f} MB) may exceed available GPU memory ({max_gpu_memory:.1f} MB). "
                    "Consider using CPU or reducing batch size."
                )
            
            # Multi-GPU recommendations
            if self._gpu_info['count'] > 1 and n_samples > 10000:
                suitable_gpus = []
                for i, memory_info in enumerate(self._gpu_info['memory_info']):
                    if memory_info.get('free_mb', 0) >= 2048:  # 2GB minimum
                        suitable_gpus.append(i)
                
                if len(suitable_gpus) > 1:
                    new_gpu_id = ','.join(map(str, suitable_gpus))
                    if self.switch_to_gpu(new_gpu_id):
                        optimization_results['applied_changes'].append(
                            f"Switched to multi-GPU setup: {new_gpu_id}"
                        )
        
        return optimization_results
    
    def extract_optimal_hyperparameters(self) -> Dict[str, Any]:
        """
        Extract the current optimal hyperparameters from the fitted model.
        
        This method extracts all hyperparameters from the current model state,
        which can be used as a baseline for further optimization or for
        refitting on new data.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing all current hyperparameters
            
        Example:
        --------
        ```python
        # After incremental learning
        optimal_params = model.extract_optimal_hyperparameters()
        
        # Create new model with these parameters
        new_model = ScientificXGBRegressor(**optimal_params)
        ```
        """
        if not self._is_fitted:
            if self.verbose:
                print("⚠️ Model not fitted yet. Returning initialization parameters.")
        
        # Get all XGBoost parameters
        base_params = self.get_params()
        
        # Add scientific parameters
        scientific_params = {
            'cv_folds': self.cv_folds,
            'auto_tune': self.auto_tune,
            'verbose': self.verbose,
            'use_gpu': self.use_gpu,
        }
        
        # Combine all parameters
        optimal_params = {**base_params, **scientific_params}
        
        # Add performance metadata if available
        if self.performance_history:
            latest_performance = self.performance_history[-1]
            optimal_params['_performance_metadata'] = {
                'r2_score': latest_performance.get('r2', 'unknown'),
                'rmse': latest_performance.get('rmse', 'unknown'),
                'n_estimators': self.n_estimators,
                'extraction_timestamp': datetime.now().isoformat()
            }
        
        if self.verbose:
            print("📋 Extracted Optimal Hyperparameters:")
            print(f"   N Estimators: {optimal_params.get('n_estimators', 'unknown')}")
            print(f"   Learning Rate: {optimal_params.get('learning_rate', 'unknown')}")
            print(f"   Max Depth: {optimal_params.get('max_depth', 'unknown')}")
            print(f"   Regularization (L1/L2): {optimal_params.get('reg_alpha', 'unknown')}/{optimal_params.get('reg_lambda', 'unknown')}")
        
        return optimal_params
    
    def refit_on_full_data(
        self, 
        X_full: np.ndarray, 
        y_full: np.ndarray,
        use_current_params: bool = True,
        apply_auto_tuning: bool = False,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Refit the model on the complete dataset using current or optimized hyperparameters.
        
        This method allows you to take a model trained incrementally and refit it
        on the complete dataset, optionally applying additional hyperparameter
        optimization.
        
        Parameters:
        -----------
        X_full : array-like of shape (n_samples, n_features)
            Complete feature dataset
        y_full : array-like of shape (n_samples,)
            Complete target dataset
        use_current_params : bool, default=True
            Whether to use current hyperparameters as baseline
        apply_auto_tuning : bool, default=False
            Whether to apply automated hyperparameter tuning before fitting
        validation_split : float, default=0.2
            Fraction of data to use for validation during fitting
            
        Returns:
        --------
        Dict[str, Any]
            Refitting results including performance comparisons
            
        Example:
        --------
        ```python
        # After incremental learning on chunks
        refit_results = model.refit_on_full_data(
            X_full, y_full,
            use_current_params=True,
            apply_auto_tuning=True
        )
        ```
        """
        if self.verbose:
            print("🔄 Refitting model on complete dataset...")
            print(f"   Dataset size: {len(X_full):,} samples, {X_full.shape[1]} features")
        
        refit_start = time.time()
        results = {}
        
        # Convert to numpy arrays for consistent handling
        if hasattr(X_full, 'values'):
            X_full_array = X_full.values
        else:
            X_full_array = np.array(X_full)
            
        if hasattr(y_full, 'values'):
            y_full_array = y_full.values
        else:
            y_full_array = np.array(y_full)
        
        # Store original performance if model was fitted
        if self._is_fitted:
            # Quick evaluation on a sample to get baseline
            sample_size = min(5000, len(X_full_array))
            sample_idx = np.random.choice(len(X_full_array), sample_size, replace=False)
            X_sample, y_sample = X_full_array[sample_idx], y_full_array[sample_idx]
            
            y_pred_baseline = self.predict(X_sample)
            baseline_r2 = r2_score(y_sample, y_pred_baseline)
            baseline_rmse = np.sqrt(mean_squared_error(y_sample, y_pred_baseline))
            
            results['baseline_performance'] = {
                'r2': baseline_r2,
                'rmse': baseline_rmse,
                'n_estimators': self.n_estimators
            }
            
            if self.verbose:
                print(f"   📊 Baseline performance (sample): R²={baseline_r2:.4f}, RMSE={baseline_rmse:.4f}")
        
        # Extract current parameters if requested
        if use_current_params:
            current_params = self.extract_optimal_hyperparameters()
            # Remove metadata that's not a parameter
            current_params.pop('_performance_metadata', None)
            results['used_parameters'] = current_params
        
        # Apply automated hyperparameter tuning if requested
        if apply_auto_tuning:
            if self.verbose:
                print("   🔬 Applying automated hyperparameter tuning...")
            tuning_results = self.automated_parameterization(X_full_array, y_full_array)
            results['auto_tuning_results'] = tuning_results
        
        # Create train/validation split
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_full_array, y_full_array,
                test_size=validation_split,
                random_state=42
            )
            
            # Set up early stopping with validation set
            fit_kwargs = {
                'eval_set': [(X_train, y_train), (X_val, y_val)],
                'verbose': False
            }
            
            if self.verbose:
                print(f"   📊 Train/Val split: {len(X_train):,}/{len(X_val):,} samples")
        else:
            X_train, y_train = X_full_array, y_full_array
            fit_kwargs = {}
        
        # Refit the model
        if self.verbose:
            print(f"   🎯 Fitting model with {self.n_estimators} estimators...")
        
        # Reset fitted state for clean refit
        self._is_fitted = False
        
        # Fit on complete training data
        self.fit(X_train, y_train, **fit_kwargs)
        
        # Evaluate on full dataset
        y_pred_full = self.predict(X_full_array)
        full_r2 = r2_score(y_full_array, y_pred_full)
        full_rmse = np.sqrt(mean_squared_error(y_full_array, y_pred_full))
        
        results['final_performance'] = {
            'r2': full_r2,
            'rmse': full_rmse,
            'mae': mean_absolute_error(y_full_array, y_pred_full),
            'n_estimators': self.n_estimators,
            'training_samples': len(X_train)
        }
        
        # Calculate improvement if baseline exists
        if 'baseline_performance' in results:
            r2_improvement = full_r2 - results['baseline_performance']['r2']
            rmse_improvement = results['baseline_performance']['rmse'] - full_rmse
            
            results['performance_improvement'] = {
                'r2_improvement': r2_improvement,
                'rmse_improvement': rmse_improvement,
                'relative_r2_improvement': r2_improvement / max(abs(results['baseline_performance']['r2']), 1e-6),
                'relative_rmse_improvement': rmse_improvement / max(results['baseline_performance']['rmse'], 1e-6)
            }
        
        # Store timing
        refit_time = time.time() - refit_start
        results['refit_time'] = refit_time
        results['refit_timestamp'] = datetime.now().isoformat()
        
        # Update model history
        if not hasattr(self, '_refit_history'):
            self._refit_history = []
        self._refit_history.append(results)
        
        if self.verbose:
            print(f"   ✅ Refitting completed in {refit_time:.2f}s")
            print(f"   📈 Final performance: R²={full_r2:.4f}, RMSE={full_rmse:.4f}")
            
            if 'performance_improvement' in results:
                imp = results['performance_improvement']
                print(f"   📊 Improvement: R² {imp['r2_improvement']:+.4f} ({imp['relative_r2_improvement']:+.2%})")
                print(f"                  RMSE {imp['rmse_improvement']:+.4f} ({imp['relative_rmse_improvement']:+.2%})")
        
        return results
    
    def comprehensive_hyperparameter_optimization(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        optimization_budget: str = 'medium',  # 'fast', 'medium', 'thorough'
        use_optuna: bool = True,
        n_trials: Optional[int] = None,
        cv_folds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive hyperparameter optimization using multiple methods.
        
        This method combines multiple optimization techniques:
        1. Automated data-driven parameterization
        2. Elbow method for key parameters
        3. Optuna-based Bayesian optimization (if available)
        4. Nested cross-validation for unbiased evaluation
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Training targets
        optimization_budget : str, default='medium'
            Optimization intensity:
            - 'fast': Quick optimization for rapid iteration
            - 'medium': Balanced optimization (recommended)
            - 'thorough': Extensive optimization for best results
        use_optuna : bool, default=True
            Whether to use Optuna for Bayesian optimization
        n_trials : int, optional
            Number of Optuna trials (auto-set based on budget if None)
        cv_folds : int, optional
            Number of CV folds (auto-set based on budget if None)
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive optimization results with best parameters
            
        Example:
        --------
        ```python
        # Comprehensive optimization on full dataset
        optimization_results = model.comprehensive_hyperparameter_optimization(
            X_full, y_full,
            optimization_budget='medium',
            use_optuna=True
        )
        
        # The model is automatically updated with best parameters
        print(f"Best R²: {optimization_results['best_score']:.4f}")
        ```
        """
        if self.verbose:
            print("🔬 Starting Comprehensive Hyperparameter Optimization")
            print("=" * 60)
            print(f"   Budget: {optimization_budget}")
            print(f"   Dataset: {len(X):,} samples, {X.shape[1]} features")
        
        optimization_start = time.time()
        results = {
            'optimization_budget': optimization_budget,
            'dataset_size': len(X),
            'n_features': X.shape[1],
            'methods_used': [],
            'best_parameters': {},
            'optimization_history': {}
        }
        
        # Set budget-based defaults
        budget_configs = {
            'fast': {'n_trials': 20, 'cv_folds': 3, 'elbow_params': ['n_estimators']},
            'medium': {'n_trials': 50, 'cv_folds': 5, 'elbow_params': ['n_estimators', 'learning_rate']},
            'thorough': {'n_trials': 100, 'cv_folds': 8, 'elbow_params': ['n_estimators', 'learning_rate', 'max_depth']}
        }
        
        config = budget_configs.get(optimization_budget, budget_configs['medium'])
        
        if n_trials is None:
            n_trials = config['n_trials']
        if cv_folds is None:
            cv_folds = config['cv_folds']
        
        # Step 1: Automated data-driven parameterization
        if self.verbose:
            print("\n🎯 Step 1: Automated Data-Driven Parameterization")
        
        auto_params = self.automated_parameterization(X, y)
        results['methods_used'].append('automated_parameterization')
        results['optimization_history']['automated_params'] = auto_params
        
        # Store baseline performance
        baseline_pred = self.predict(X) if self._is_fitted else None
        if baseline_pred is not None:
            baseline_r2 = r2_score(y, baseline_pred)
            results['baseline_performance'] = {'r2': baseline_r2}
        
        # Step 2: Elbow method optimization for key parameters
        if self.verbose:
            print("\n📈 Step 2: Elbow Method Optimization")
        
        elbow_results = {}
        for param in config['elbow_params']:
            if self.verbose:
                print(f"   Optimizing {param}...")
            try:
                elbow_result = self.elbow_tune(X, y, param_name=param, cv_folds=cv_folds)
                elbow_results[param] = elbow_result
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Elbow optimization failed for {param}: {e}")
        
        if elbow_results:
            results['methods_used'].append('elbow_method')
            results['optimization_history']['elbow_results'] = elbow_results
        
        # Step 3: Optuna Bayesian optimization
        if use_optuna and OPTUNA_AVAILABLE:
            if self.verbose:
                print(f"\n🧠 Step 3: Bayesian Optimization ({n_trials} trials)")
            
            try:
                optuna_results = self._optuna_optimization(X, y, n_trials, cv_folds)
                results['methods_used'].append('optuna_bayesian')
                results['optimization_history']['optuna_results'] = optuna_results
                
                # Update model with best Optuna parameters
                if 'best_params' in optuna_results:
                    for param, value in optuna_results['best_params'].items():
                        setattr(self, param, value)
                
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Optuna optimization failed: {e}")
        elif use_optuna and not OPTUNA_AVAILABLE:
            if self.verbose:
                print("   ⚠️ Optuna not available, skipping Bayesian optimization")
        
        # Step 4: Nested cross-validation for final evaluation
        if self.verbose:
            print(f"\n🔄 Step 4: Nested Cross-Validation ({cv_folds} folds)")
        
        try:
            nested_cv_results = self.nested_cross_validate(
                X, y, 
                inner_cv=max(3, cv_folds-2), 
                outer_cv=cv_folds,
                scoring='neg_mean_squared_error'
            )
            results['methods_used'].append('nested_cross_validation')
            results['optimization_history']['nested_cv'] = nested_cv_results
            
            # Extract best performance
            best_r2 = nested_cv_results['mean_scores']['r2']
            best_rmse = nested_cv_results['mean_scores']['rmse']
            
            results['best_score'] = best_r2
            results['best_rmse'] = best_rmse
            results['best_parameters'] = nested_cv_results['consensus_params']
            
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Nested CV failed: {e}")
            
            # Fallback: simple CV evaluation
            scores = cross_val_score(self, X, y, cv=cv_folds, scoring='r2')
            results['best_score'] = np.mean(scores)
            results['best_parameters'] = self.get_params()
        
        # Step 5: Final model update with best parameters
        if results['best_parameters']:
            if self.verbose:
                print("\n🎯 Step 5: Updating Model with Best Parameters")
            
            # Apply best parameters
            for param, value in results['best_parameters'].items():
                if hasattr(self, param):
                    setattr(self, param, value)
            
            # Refit with best parameters
            self._is_fitted = False
            self.fit(X, y)
        
        # Calculate total optimization time
        optimization_time = time.time() - optimization_start
        results['optimization_time'] = optimization_time
        results['timestamp'] = datetime.now().isoformat()
        
        # Store in model history
        if not hasattr(self, '_optimization_history'):
            self._optimization_history = []
        self._optimization_history.append(results)
        
        # Final summary
        if self.verbose:
            print("\n🎉 Comprehensive Optimization Complete!")
            print("=" * 60)
            print(f"   Methods used: {', '.join(results['methods_used'])}")
            print(f"   Optimization time: {optimization_time:.2f}s")
            print(f"   Best R² score: {results.get('best_score', 'N/A'):.4f}")
            if 'best_rmse' in results:
                print(f"   Best RMSE: {results['best_rmse']:.4f}")
            print(f"   Final estimators: {self.n_estimators}")
        
        return results
    
    def _optuna_optimization(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_trials: int, 
        cv_folds: int
    ) -> Dict[str, Any]:
        """
        Internal method for Optuna-based Bayesian optimization.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training targets
        n_trials : int
            Number of optimization trials
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        Dict[str, Any]
            Optuna optimization results
        """
        import optuna
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
            
            # Create temporary model with trial parameters
            temp_model = XGBRegressor(
                random_state=self.random_state,
                **params
            )
            
            # Cross-validation evaluation
            scores = cross_val_score(temp_model, X, y, cv=cv_folds, scoring='r2')
            return np.mean(scores)
        
        # Create and run study
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return {
            'best_score': study.best_value,
            'best_params': study.best_params,
            'n_trials': n_trials,
            'study_summary': {
                'n_trials': len(study.trials),
                'best_trial': study.best_trial.number,
                'optimization_direction': 'maximize'
            }
        }

def create_scientific_xgb_regressor(**kwargs) -> ScientificXGBRegressor:
    """
    Factory function to create a ScientificXGBRegressor with recommended settings.
    
    This function provides intelligent defaults and automatically detects and 
    configures GPU acceleration when available.
    
    Parameters:
    -----------
    use_gpu : bool, optional
        GPU usage preference:
        - None (default): Auto-detect and use GPU if available
        - True: Force GPU usage (will fail if no GPU available)
        - False: Force CPU usage
    **kwargs : dict
        Additional parameters for ScientificXGBRegressor
        
    Returns:
    --------
    ScientificXGBRegressor
        Configured model instance with optimal GPU/CPU settings
        
    Example:
    --------
    ```python
    # Auto-detect GPU (recommended)
    model = create_scientific_xgb_regressor()
    
    # Force GPU usage
    model = create_scientific_xgb_regressor(use_gpu=True)
    
    # Force CPU usage
    model = create_scientific_xgb_regressor(use_gpu=False)
    
    # Custom configuration with GPU auto-detection
    model = create_scientific_xgb_regressor(
        n_estimators=2000,
        learning_rate=0.05,
        cv_folds=10
    )
    ```
    """
    default_params = {
        'cv_folds': 5,
        'auto_tune': True,
        'verbose': True,
        'early_stopping_rounds': 50,
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'random_state': 42,
        'use_gpu': None  # Auto-detect GPU by default
    }
    
    # Update with user parameters
    default_params.update(kwargs)
    
    # Quick GPU check for informative messaging
    if default_params.get('verbose', True):
        gpu_info = GPUManager.detect_gpus()
        if gpu_info['available']:
            print(f"🎮 GPU acceleration available: {gpu_info['count']} GPU(s) detected")
        else:
            print("💻 GPU acceleration not available, using CPU")
    
    return ScientificXGBRegressor(**default_params)


"""
Incremental Learning Pipeline for Large Datasets
================================================

This script provides a comprehensive pipeline for training ScientificXGBRegressor
on large datasets using incremental learning. It chunks the dataset and progressively
improves the model by training on each chunk sequentially.

Features:
---------
- Configurable dataset chunking
- Progressive model improvement tracking
- Memory-efficient processing for large datasets
- Comprehensive performance monitoring
- Cross-validation integration
- GPU/CPU optimization
- Model checkpoint saving
- Detailed progress visualization

Mathematical Foundation:
------------------------
Incremental learning implements the following update rule:

F_t(x) = F_{t-1}(x) + Σᵢ₌₁ᵏ η·h_i(x)

Where:
- F_t(x) is the model after chunk t
- h_i(x) are new weak learners from chunk t
- η is the learning rate
- k is the number of new estimators per chunk

"""

@dataclass
class IncrementalConfig:
    """
    Configuration class for incremental learning pipeline.
    
    Attributes:
    -----------
    n_chunks : int
        Number of chunks to split the dataset into
    chunk_size : Optional[int]
        Specific chunk size (overrides n_chunks if provided)
    inner_cv_folds : int
        Number of inner cross-validation folds
    outer_cv_folds : int
        Number of outer cross-validation folds
    n_estimators_per_chunk : int
        Number of new estimators to add per chunk
    validation_split : float
        Fraction of data to use for validation
    enable_early_stopping : bool
        Whether to use early stopping
    save_checkpoints : bool
        Whether to save model checkpoints after each chunk
    checkpoint_dir : str
        Directory to save checkpoints
    memory_optimization : bool
        Whether to apply memory optimization techniques
    verbose : bool
        Whether to print detailed progress information
    use_gpu : Optional[bool]
        GPU usage preference (None=auto, True=force, False=disable)
    plot_progress : bool
        Whether to generate progress plots
    """
    n_chunks: int = 10
    chunk_size: Optional[int] = None
    inner_cv_folds: int = 3
    outer_cv_folds: int = 5
    n_estimators_per_chunk: int = 100
    validation_split: float = 0.2
    enable_early_stopping: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "./incremental_checkpoints"
    memory_optimization: bool = True
    verbose: bool = True
    use_gpu: Optional[bool] = None
    plot_progress: bool = True


class IncrementalLearningPipeline:
    """
    Comprehensive pipeline for incremental learning on large datasets.
    
    This class manages the entire incremental learning process, including
    data chunking, progressive training, performance monitoring, and
    checkpoint management.
    
    Mathematical Framework:
    -----------------------
    The pipeline implements incremental ensemble learning:
    
    For each chunk C_t:
    1. Load chunk: (X_t, y_t) = C_t
    2. Update model: M_t = M_{t-1} + ΔM_t
    3. Evaluate: R_t = evaluate(M_t, V)
    4. Save checkpoint: save(M_t) if R_t > R_{t-1}
    
    Where ΔM_t represents the incremental model update from chunk t.
    """
    
    def __init__(self, config: IncrementalConfig):
        """
        Initialize the incremental learning pipeline.
        
        Parameters:
        -----------
        config : IncrementalConfig
            Configuration object with all pipeline settings
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler() if config.memory_optimization else None
        self.training_history = []
        self.performance_history = []
        self.chunk_info = []
        self.validation_data = None
        
        # Create checkpoint directory
        if config.save_checkpoints:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracking
        self.start_time = None
        self.chunk_times = []
        
        if config.verbose:
            print("🚀 Incremental Learning Pipeline Initialized")
            print(f"   📊 Configuration: {config.n_chunks} chunks, {config.n_estimators_per_chunk} estimators/chunk")
            print(f"   🔄 Cross-validation: {config.inner_cv_folds} inner × {config.outer_cv_folds} outer folds")
            print(f"   💾 Checkpoints: {'Enabled' if config.save_checkpoints else 'Disabled'}")
    
    def prepare_data_chunks(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        shuffle: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare data chunks for incremental learning.
        
        This method efficiently chunks large datasets while preserving
        statistical properties and enabling memory-efficient processing.
        All chunks are fully preprocessed and ready for model training.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target vector
        shuffle : bool, default=True
            Whether to shuffle data before chunking
            
        Returns:
        --------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (X_chunk, y_chunk) tuples, all preprocessed and ready for training
        """
        if self.config.verbose:
            print(f"📦 Preparing data chunks from {len(X)} samples...")
        
        # Convert to numpy arrays for consistent handling
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        X = np.array(X)
        y = np.array(y)
        
        # Identify and remove problematic columns across entire dataset
        if self.config.verbose:
            print("🔍 Analyzing entire dataset for problematic columns...")
        
        original_shape = X.shape
        problematic_columns = []
        
        # Check each column across the entire dataset
        for i in range(X.shape[1]):
            col_data = X[:, i]
            
            # Check for all NaN
            if np.all(np.isnan(col_data)):
                problematic_columns.append(i)
                if self.config.verbose:
                    print(f"   ⚠️ Column {i}: All NaN values")
                continue
            
            # Check for all infinite
            if np.all(np.isinf(col_data)):
                problematic_columns.append(i)
                if self.config.verbose:
                    print(f"   ⚠️ Column {i}: All infinite values")
                continue
            
            # Check for constant values (no variance)
            unique_vals = np.unique(col_data[~np.isnan(col_data) & ~np.isinf(col_data)])
            if len(unique_vals) <= 1:
                problematic_columns.append(i)
                if self.config.verbose:
                    print(f"   ⚠️ Column {i}: Constant values (no variance)")
                continue
        
        # Store information for tracking and validation
        self.problematic_columns = problematic_columns
        self.original_n_features = X.shape[1]
        
        # Remove problematic columns from entire dataset
        if problematic_columns:
            if self.config.verbose:
                print(f"   🗑️ Removing {len(problematic_columns)} problematic columns globally: {problematic_columns}")
            
            # Keep only good columns
            good_columns = [i for i in range(X.shape[1]) if i not in problematic_columns]
            X = X[:, good_columns]
            
            if self.config.verbose:
                print(f"   📊 Data shape: {original_shape} → {X.shape}")
        
        # Store final feature count for validation
        self.final_n_features = X.shape[1]
        
        # Additional preprocessing: handle remaining NaN and infinite values
        if self.config.verbose:
            print("🧹 Final data cleaning...")
        
        # Handle remaining NaN values
        nan_count_X = np.isnan(X).sum()
        if nan_count_X > 0:
            if self.config.verbose:
                print(f"   🔧 Handling {nan_count_X} remaining NaN values in X")
            
            # Replace NaN with column means
            col_means = np.nanmean(X, axis=0)
            for i in range(X.shape[1]):
                col_mask = np.isnan(X[:, i])
                if np.any(col_mask):
                    X[col_mask, i] = col_means[i] if not np.isnan(col_means[i]) else 0.0
        
        # Handle infinite values
        inf_count_X = np.isinf(X).sum()
        if inf_count_X > 0:
            if self.config.verbose:
                print(f"   🔧 Handling {inf_count_X} infinite values in X")
            
            # Replace infinite with very large finite values
            X[np.isposinf(X)] = np.finfo(np.float64).max / 1e6
            X[np.isneginf(X)] = np.finfo(np.float64).min / 1e6
        
        # Handle target variable
        nan_count_y = np.isnan(y).sum()
        if nan_count_y > 0:
            if self.config.verbose:
                print(f"   🔧 Handling {nan_count_y} NaN values in y")
            
            # For regression, use mean imputation
            y_mean = np.nanmean(y)
            y[np.isnan(y)] = y_mean if not np.isnan(y_mean) else 0.0
        
        inf_count_y = np.isinf(y).sum()
        if inf_count_y > 0:
            if self.config.verbose:
                print(f"   🔧 Handling {inf_count_y} infinite values in y")
            
            y[np.isposinf(y)] = np.finfo(np.float64).max / 1e6
            y[np.isneginf(y)] = np.finfo(np.float64).min / 1e6
        
        # Final validation
        assert not np.any(np.isnan(X)), "NaN values still present in X after preprocessing"
        assert not np.any(np.isnan(y)), "NaN values still present in y after preprocessing"
        assert not np.any(np.isinf(X)), "Infinite values still present in X after preprocessing"
        assert not np.any(np.isinf(y)), "Infinite values still present in y after preprocessing"
        
        if self.config.verbose:
            print(f"   ✅ Data preprocessing complete")
            print(f"      Final shape: {X.shape}, dtype: {X.dtype}")
        
        # Shuffle data if requested
        if shuffle:
            if self.config.verbose:
                print("   🔀 Shuffling data...")
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
        
        # Calculate chunk size
        if self.config.chunk_size is not None:
            chunk_size = self.config.chunk_size
            n_chunks = len(X) // chunk_size + (1 if len(X) % chunk_size > 0 else 0)
        else:
            n_chunks = self.config.n_chunks
            chunk_size = len(X) // n_chunks
        
        if self.config.verbose:
            print(f"   📏 Chunk configuration:")
            print(f"      Number of chunks: {n_chunks}")
            print(f"      Samples per chunk: ~{chunk_size:,}")
            print(f"      Features per chunk: {self.final_n_features}")
        
        # Create chunks from preprocessed data
        chunks = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(X))
            
            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]
            
            chunk_info = {
                'chunk_id': i,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size': len(X_chunk),
                'feature_shape': X_chunk.shape,
                'target_stats': {
                    'mean': np.mean(y_chunk),
                    'std': np.std(y_chunk),
                    'min': np.min(y_chunk),
                    'max': np.max(y_chunk)
                },
                'n_features': X_chunk.shape[1]
            }
            
            chunks.append((X_chunk, y_chunk))
            self.chunk_info.append(chunk_info)
            
            if self.config.verbose and i < 3:  # Show details for first 3 chunks
                print(f"      Chunk {i}: {len(X_chunk):,} samples, {X_chunk.shape[1]} features, target μ={chunk_info['target_stats']['mean']:.4f}")
        
        if self.config.verbose and n_chunks > 3:
            print(f"      ... and {n_chunks - 3} more chunks")
        
        # Validate all chunks have same number of features
        feature_counts = [chunk[0].shape[1] for chunk in chunks]
        if len(set(feature_counts)) > 1:
            raise ValueError(f"Inconsistent feature counts across chunks: {set(feature_counts)}")
        
        if self.config.verbose:
            print(f"   ✅ All chunks validated with consistent {self.final_n_features} features")
        
        # Memory optimization
        if self.config.memory_optimization:
            del X, y  # Free original data memory
            gc.collect()
        
        return chunks
    
    def create_validation_set(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create training and validation sets from the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target vector
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, X_val, y_train, y_val
        """
        if self.config.validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config.validation_split,
                random_state=42,
                stratify=None  # Regression, no stratification
            )
            
            # Store validation data for consistent evaluation
            self.validation_data = (X_val, y_val)
            
            if self.config.verbose:
                print(f"   📊 Validation set: {len(X_val):,} samples ({self.config.validation_split:.1%})")
            
            return X_train, X_val, y_train, y_val
        else:
            return X, None, y, None
    
    def initialize_model(self, X_first_chunk: np.ndarray, y_first_chunk: np.ndarray) -> None:
        """
        Initialize the ScientificXGBRegressor with the first chunk.
        
        Parameters:
        -----------
        X_first_chunk : array-like
            Features from the first data chunk (already preprocessed)
        y_first_chunk : array-like
            Targets from the first data chunk (already preprocessed)
        """
        if self.config.verbose:
            print("🧪 Initializing ScientificXGBRegressor...")
        
        # Create validation set from first chunk
        # Note: X_first_chunk is already preprocessed in prepare_data_chunks
        X_train, X_val, y_train, y_val = self.create_validation_set(X_first_chunk, y_first_chunk)
        
        # Initialize model with configuration
        model_params = {
            'cv_folds': self.config.outer_cv_folds,
            'auto_tune': True,
            'verbose': self.config.verbose,
            'n_estimators': self.config.n_estimators_per_chunk,
            'early_stopping_rounds': 50 if self.config.enable_early_stopping else None,
            'use_gpu': self.config.use_gpu,
            'random_state': 42
        }
        
        self.model = create_scientific_xgb_regressor(**model_params)
        
        # Apply GPU optimization
        if self.model._gpu_info['available'] and self.config.use_gpu != False:
            gpu_optimization = self.model.optimize_gpu_usage(X_train)
            if self.config.verbose:
                print(f"   ⚡ GPU optimization: {gpu_optimization['status']}")
        
        # Fit initial model
        if self.config.verbose:
            print("   🎯 Training initial model on first chunk...")
        
        fit_start = time.time()
        
        # Add evaluation set for early stopping if validation data available
        fit_kwargs = {}
        if X_val is not None and self.config.enable_early_stopping:
            fit_kwargs['eval_set'] = [(X_train, y_train), (X_val, y_val)]
            fit_kwargs['verbose'] = False
        
        # Fit the model on preprocessed data
        self.model.fit(X_train, y_train, **fit_kwargs)
        
        fit_time = time.time() - fit_start
        
        # Evaluate initial performance
        initial_performance = self.evaluate_model(X_first_chunk, y_first_chunk, chunk_id=0)
        initial_performance['fit_time'] = fit_time
        initial_performance['n_estimators'] = self.model.n_estimators
        initial_performance['is_initial'] = True
        initial_performance['n_features'] = X_first_chunk.shape[1]
        
        self.performance_history.append(initial_performance)
        
        if self.config.verbose:
            print(f"   ✅ Initial model trained in {fit_time:.2f}s")
            print(f"      R² score: {initial_performance['r2']:.4f}")
            print(f"      RMSE: {initial_performance['rmse']:.4f}")
            print(f"      Features: {X_first_chunk.shape[1]} (preprocessed)")
    
    def evaluate_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        chunk_id: int = -1
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Parameters:
        -----------
        X : array-like
            Features for evaluation
        y : array-like
            True targets
        chunk_id : int
            ID of the current chunk
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive evaluation metrics
        """
        y_pred = self.model.predict(X)
        
        evaluation = {
            'chunk_id': chunk_id,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X),
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'model_n_estimators': self.model.n_estimators,
        }
        
        # Add explained variance and additional metrics
        y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        evaluation['explained_variance'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Residual analysis
        residuals = y - y_pred
        evaluation['residual_std'] = np.std(residuals)
        evaluation['residual_mean'] = np.mean(residuals)
        evaluation['residual_skewness'] = float(np.mean(((residuals - np.mean(residuals)) / np.std(residuals)) ** 3))
        
        # Validation set evaluation if available
        if self.validation_data is not None:
            X_val, y_val = self.validation_data
            y_val_pred = self.model.predict(X_val)
            evaluation['val_r2'] = r2_score(y_val, y_val_pred)
            evaluation['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_val_pred))
            evaluation['val_mae'] = mean_absolute_error(y_val, y_val_pred)
        
        return evaluation
    
    def save_checkpoint(self, chunk_id: int) -> str:
        """
        Save model checkpoint after processing a chunk.
        
        Parameters:
        -----------
        chunk_id : int
            ID of the completed chunk
            
        Returns:
        --------
        str
            Path to saved checkpoint
        """
        if not self.config.save_checkpoints:
            return ""
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_file = checkpoint_dir / f"model_checkpoint_chunk_{chunk_id:03d}.pkl"
        
        # Save model
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            'chunk_id': chunk_id,
            'timestamp': datetime.now().isoformat(),
            'n_estimators': self.model.n_estimators,
            'performance_history': self.performance_history,
            'chunk_info': self.chunk_info[:chunk_id + 1],
            'config': self.config.__dict__
        }
        
        metadata_file = checkpoint_dir / f"metadata_chunk_{chunk_id:03d}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        if self.config.verbose:
            file_size = checkpoint_file.stat().st_size / (1024 * 1024)  # MB
            print(f"      💾 Checkpoint saved: {checkpoint_file.name} ({file_size:.1f} MB)")
        
        return str(checkpoint_file)
    
    def process_chunk(self, chunk_id: int, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Process a single data chunk with incremental learning.
        
        Parameters:
        -----------
        chunk_id : int
            ID of the current chunk
        X_chunk : array-like
            Features for the current chunk (already preprocessed)
        y_chunk : array-like
            Targets for the current chunk (already preprocessed)
            
        Returns:
        --------
        Dict[str, Any]
            Processing results and performance metrics
        """
        if self.config.verbose:
            print(f"🔄 Processing chunk {chunk_id + 1}/{len(self.chunk_info)}")
            print(f"   📊 Chunk size: {len(X_chunk):,} samples")
        
        chunk_start = time.time()
        
        # Use incremental learning for non-initial chunks
        if chunk_id == 0:
            # First chunk - initialize model
            self.initialize_model(X_chunk, y_chunk)
            processing_results = {
                'method': 'initial_fit',
                'chunk_id': chunk_id,
                'n_estimators_added': self.model.n_estimators
            }
        else:
            # Subsequent chunks - incremental learning
            if self.config.verbose:
                print(f"   🔄 Incremental learning: adding {self.config.n_estimators_per_chunk} estimators...")
            
            # Apply incremental learning on preprocessed data
            # No need to pass good_columns since chunks are already preprocessed
            incremental_results = self.model.incremental_learn(
                X_chunk, y_chunk,
                n_new_estimators=self.config.n_estimators_per_chunk
            )
            
            processing_results = {
                'method': 'incremental_learn',
                'chunk_id': chunk_id,
                'incremental_results': incremental_results,
                'n_estimators_added': self.config.n_estimators_per_chunk
            }
        
        # Evaluate performance on current chunk
        chunk_performance = self.evaluate_model(X_chunk, y_chunk, chunk_id)
        
        # Add timing information
        chunk_time = time.time() - chunk_start
        chunk_performance['processing_time'] = chunk_time
        chunk_performance['processing_method'] = processing_results['method']
        
        # Store results
        self.performance_history.append(chunk_performance)
        self.chunk_times.append(chunk_time)
        
        # Save checkpoint
        checkpoint_path = self.save_checkpoint(chunk_id)
        processing_results['checkpoint_path'] = checkpoint_path
        
        # Progress reporting
        if self.config.verbose:
            print(f"   ✅ Chunk {chunk_id + 1} completed in {chunk_time:.2f}s")
            print(f"      R² score: {chunk_performance['r2']:.4f}")
            print(f"      RMSE: {chunk_performance['rmse']:.4f}")
            print(f"      Total estimators: {self.model.n_estimators}")
            print(f"      Features: {X_chunk.shape[1]} (preprocessed)")
            
            if 'val_r2' in chunk_performance:
                print(f"      Validation R²: {chunk_performance['val_r2']:.4f}")
        
        # Memory cleanup
        if self.config.memory_optimization:
            del X_chunk, y_chunk
            gc.collect()
        
        return processing_results
    
    def run_incremental_training(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Run the complete incremental training pipeline.
        
        This is the main method that orchestrates the entire incremental
        learning process from data preparation to final model evaluation.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Complete feature dataset
        y : array-like of shape (n_samples,)
            Complete target dataset
            
        Returns:
        --------
        Dict[str, Any]
            Complete training results and performance history
        """
        if self.config.verbose:
            print("🚀 Starting Incremental Learning Pipeline")
            print("=" * 60)
        
        self.start_time = time.time()
        
        # Step 1: Prepare data chunks
        chunks = self.prepare_data_chunks(X, y)
        
        if self.config.verbose:
            print(f"\n📦 Data preparation complete: {len(chunks)} chunks ready")
        
        # Step 2: Process each chunk
        processing_results = []
        
        for chunk_id, (X_chunk, y_chunk) in enumerate(chunks):
            try:
                result = self.process_chunk(chunk_id, X_chunk, y_chunk)
                processing_results.append(result)
                
                # Progress update
                progress = (chunk_id + 1) / len(chunks) * 100
                if self.config.verbose:
                    print(f"   📈 Progress: {progress:.1f}% ({chunk_id + 1}/{len(chunks)} chunks)")
                    
            except Exception as e:
                error_msg = f"Error processing chunk {chunk_id}: {str(e)}"
                if self.config.verbose:
                    print(f"   ❌ {error_msg}")
                
                processing_results.append({
                    'chunk_id': chunk_id,
                    'error': error_msg,
                    'method': 'failed'
                })
                
                # Continue with next chunk
                continue
        
        # Step 3: Final evaluation and summary
        total_time = time.time() - self.start_time
        
        final_results = {
            'pipeline_config': self.config.__dict__,
            'total_chunks_processed': len([r for r in processing_results if 'error' not in r]),
            'failed_chunks': len([r for r in processing_results if 'error' in r]),
            'total_training_time': total_time,
            'average_chunk_time': np.mean(self.chunk_times) if self.chunk_times else 0,
            'processing_results': processing_results,
            'performance_history': self.performance_history,
            'chunk_info': self.chunk_info,
            'final_model_estimators': self.model.n_estimators if self.model else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate performance improvements
        if len(self.performance_history) >= 2:
            initial_r2 = self.performance_history[0]['r2']
            final_r2 = self.performance_history[-1]['r2']
            r2_improvement = final_r2 - initial_r2
            
            initial_rmse = self.performance_history[0]['rmse']
            final_rmse = self.performance_history[-1]['rmse']
            rmse_improvement = initial_rmse - final_rmse
            
            final_results['performance_improvement'] = {
                'r2_improvement': r2_improvement,
                'rmse_improvement': rmse_improvement,
                'relative_r2_improvement': r2_improvement / max(abs(initial_r2), 1e-6),
                'relative_rmse_improvement': rmse_improvement / max(initial_rmse, 1e-6)
            }
        
        # Generate progress plots if requested
        if self.config.plot_progress:
            self.plot_training_progress()
        
        # Final summary
        if self.config.verbose:
            print("\n🎉 Incremental Learning Pipeline Complete!")
            print("=" * 60)
            print(f"📊 Summary:")
            print(f"   Total chunks: {len(chunks)}")
            print(f"   Successful: {final_results['total_chunks_processed']}")
            print(f"   Failed: {final_results['failed_chunks']}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Final estimators: {final_results['final_model_estimators']}")
            
            if 'performance_improvement' in final_results:
                perf = final_results['performance_improvement']
                print(f"   R² improvement: {perf['r2_improvement']:+.4f} ({perf['relative_r2_improvement']:+.2%})")
                print(f"   RMSE improvement: {perf['rmse_improvement']:+.4f} ({perf['relative_rmse_improvement']:+.2%})")
        
        return final_results
    
    def plot_training_progress(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Generate comprehensive training progress visualization.
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size for the plots
            
        Returns:
        --------
        matplotlib.Figure
            Figure containing all progress plots
        """
        if not self.performance_history:
            print("⚠️ No performance history available for plotting")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Incremental Learning Progress Dashboard', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        chunk_ids = [p['chunk_id'] for p in self.performance_history]
        r2_scores = [p['r2'] for p in self.performance_history]
        rmse_scores = [p['rmse'] for p in self.performance_history]
        mae_scores = [p['mae'] for p in self.performance_history]
        n_estimators = [p['model_n_estimators'] for p in self.performance_history]
        
        # Plot 1: R² Score Progress
        axes[0, 0].plot(chunk_ids, r2_scores, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Chunk ID')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('R² Score Progression')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(bottom=0)
        
        # Add trend line
        if len(chunk_ids) > 2:
            z = np.polyfit(chunk_ids, r2_scores, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(chunk_ids, p(chunk_ids), "r--", alpha=0.8, label=f'Trend: {z[0]:+.4f}/chunk')
            axes[0, 0].legend()
        
        # Plot 2: RMSE Progress
        axes[0, 1].plot(chunk_ids, rmse_scores, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Chunk ID')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('RMSE Progression')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Model Complexity (Number of Estimators)
        axes[0, 2].plot(chunk_ids, n_estimators, 'g-o', linewidth=2, markersize=6)
        axes[0, 2].set_xlabel('Chunk ID')
        axes[0, 2].set_ylabel('Number of Estimators')
        axes[0, 2].set_title('Model Complexity Growth')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Multiple Metrics Comparison
        axes[1, 0].plot(chunk_ids, r2_scores, 'b-', label='R² Score', linewidth=2)
        if self.validation_data is not None:
            val_r2_scores = [p.get('val_r2', 0) for p in self.performance_history if 'val_r2' in p]
            if val_r2_scores:
                val_chunk_ids = [p['chunk_id'] for p in self.performance_history if 'val_r2' in p]
                axes[1, 0].plot(val_chunk_ids, val_r2_scores, 'b--', label='Validation R²', linewidth=2)
        
        axes[1, 0].set_xlabel('Chunk ID')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Training vs Validation Performance')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 5: Processing Time per Chunk
        if self.chunk_times:
            axes[1, 1].bar(range(len(self.chunk_times)), self.chunk_times, alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('Chunk ID')
            axes[1, 1].set_ylabel('Processing Time (seconds)')
            axes[1, 1].set_title('Processing Time per Chunk')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add average line
            avg_time = np.mean(self.chunk_times)
            axes[1, 1].axhline(y=avg_time, color='red', linestyle='--', 
                             label=f'Average: {avg_time:.2f}s')
            axes[1, 1].legend()
        
        # Plot 6: Residual Analysis (Latest Chunk)
        if len(self.performance_history) > 0:
            latest_perf = self.performance_history[-1]
            residual_std = latest_perf.get('residual_std', 0)
            residual_mean = latest_perf.get('residual_mean', 0)
            
            # Create a simple residual distribution plot
            # Since we don't have actual residuals, we'll show summary statistics
            metrics = ['RMSE', 'MAE', 'Residual Std']
            values = [latest_perf['rmse'], latest_perf['mae'], residual_std]
            
            bars = axes[1, 2].bar(metrics, values, alpha=0.7, color=['red', 'orange', 'yellow'])
            axes[1, 2].set_ylabel('Error Magnitude')
            axes[1, 2].set_title('Final Model Error Metrics')
            axes[1, 2].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot if checkpoints are enabled
        if self.config.save_checkpoints:
            plot_path = Path(self.config.checkpoint_dir) / "training_progress.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            if self.config.verbose:
                print(f"📊 Progress plots saved to: {plot_path}")
        
        return fig
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Find the checkpoint with the best validation performance.
        
        Returns:
        --------
        Optional[str]
            Path to the best checkpoint file, or None if no checkpoints exist
        """
        if not self.config.save_checkpoints or not self.performance_history:
            return None
        
        # Find best performance based on validation R² if available, otherwise training R²
        best_idx = 0
        best_score = -np.inf
        
        for i, perf in enumerate(self.performance_history):
            score = perf.get('val_r2', perf.get('r2', -np.inf))
            if score > best_score:
                best_score = score
                best_idx = i
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        best_checkpoint = checkpoint_dir / f"model_checkpoint_chunk_{best_idx:03d}.pkl"
        
        if best_checkpoint.exists():
            return str(best_checkpoint)
        return None


def create_incremental_pipeline(
    n_chunks: int = 10,
    chunk_size: Optional[int] = None,
    inner_cv_folds: int = 3,
    outer_cv_folds: int = 5,
    n_estimators_per_chunk: int = 100,
    use_gpu: Optional[bool] = None,
    **kwargs
) -> IncrementalLearningPipeline:
    """
    Factory function to create an incremental learning pipeline with common configurations.
    
    Parameters:
    -----------
    n_chunks : int, default=10
        Number of chunks to split the dataset into
    chunk_size : Optional[int], default=None
        Specific chunk size (overrides n_chunks if provided)
    inner_cv_folds : int, default=3
        Number of inner cross-validation folds
    outer_cv_folds : int, default=5
        Number of outer cross-validation folds
    n_estimators_per_chunk : int, default=100
        Number of new estimators to add per chunk
    use_gpu : Optional[bool], default=None
        GPU usage preference (None=auto, True=force, False=disable)
    **kwargs : dict
        Additional configuration parameters for IncrementalConfig
        
    Returns:
    --------
    IncrementalLearningPipeline
        Configured pipeline ready for incremental learning
        
    Example:
    --------
    ```python
    # Create pipeline for 20 chunks with GPU acceleration
    pipeline = create_incremental_pipeline(
        n_chunks=20,
        n_estimators_per_chunk=150,
        use_gpu=True,
        verbose=True
    )
    
    # Run incremental training
    results = pipeline.run_incremental_training(X, y)
    ```
    """
    config_params = {
        'n_chunks': n_chunks,
        'chunk_size': chunk_size,
        'inner_cv_folds': inner_cv_folds,
        'outer_cv_folds': outer_cv_folds,
        'n_estimators_per_chunk': n_estimators_per_chunk,
        'use_gpu': use_gpu,
    }
    
    # Add any additional parameters
    config_params.update(kwargs)
    
    config = IncrementalConfig(**config_params)
    return IncrementalLearningPipeline(config)


def extract_model_from_pipeline(pipeline: IncrementalLearningPipeline) -> ScientificXGBRegressor:
    """
    Extract the trained ScientificXGBRegressor from an incremental learning pipeline.
    
    This is a convenience function to get the final trained model after
    running incremental learning, ready for further optimization or evaluation.
    
    Parameters:
    -----------
    pipeline : IncrementalLearningPipeline
        The pipeline object after running incremental training
        
    Returns:
    --------
    ScientificXGBRegressor
        The trained model from the pipeline
        
    Example:
    --------
    ```python
    # After running incremental learning
    pipeline = create_incremental_pipeline(...)
    results = pipeline.run_incremental_training(X, y)
    
    # Extract the final model
    final_model = extract_model_from_pipeline(pipeline)
    
    # Now use the model for further optimization
    refit_results = final_model.refit_on_full_data(X_full, y_full)
    ```
    """
    if pipeline.model is None:
        raise ValueError("Pipeline has no trained model. Please run incremental training first.")
    
    if not pipeline.model._is_fitted:
        raise ValueError("Pipeline model is not fitted. Please run incremental training first.")
    
    # Ensure performance history is available
    if hasattr(pipeline, 'performance_history') and pipeline.performance_history:
        pipeline.model.performance_history = pipeline.performance_history
    
    return pipeline.model


"""
# ============================================================================
# COMPREHENSIVE USAGE GUIDE: Starting from Existing Parameters
# ============================================================================

This section provides detailed examples for different scenarios of using 
ScientificXGBRegressor with existing parameters and improving them systematically.

## 🎯 Method 1: Direct Initialization with Existing Parameters

```python
# Your existing parameters dictionary
existing_params = {
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 0.1,
    # ... other XGBoost parameters
}

# Create model with existing parameters + scientific features
model = create_scientific_xgb_regressor(
    **existing_params,
    cv_folds=5,
    auto_tune=False,  # Start with existing params, don't auto-tune yet
    verbose=True,
    use_gpu=None  # Auto-detect
)

# Train with existing parameters first
model.fit(X_train, y_train)
print(f"Baseline with existing params - R²: {model.score(X_test, y_test):.4f}")
```

## 🔬 Method 2: Incremental Parameter Improvement

```python
# Step 1: Start with existing parameters
model = create_scientific_xgb_regressor(**existing_params, auto_tune=False)
model.fit(X_train, y_train)

# Step 2: Apply elbow optimization for key parameters
elbow_results = {}
key_params = ['n_estimators', 'learning_rate', 'max_depth']

for param in key_params:
    print(f"🔧 Optimizing {param}...")
    result = model.elbow_tune(X_train, y_train, param_name=param)
    elbow_results[param] = result
    print(f"   Best {param}: {result['elbow_value']}")

# Step 3: Apply automated data-driven improvements
print("🧪 Applying automated parameterization...")
auto_params = model.automated_parameterization(X_train, y_train)

# Step 4: Refit with improved parameters
model.fit(X_train, y_train)
print(f"After incremental improvement - R²: {model.score(X_test, y_test):.4f}")
```

## 🚀 Method 3: Comprehensive Optimization Starting from Existing Params

```python
# Initialize with existing parameters
model = create_scientific_xgb_regressor(**existing_params, auto_tune=False)

# Apply comprehensive optimization starting from existing baseline
optimization_results = model.comprehensive_hyperparameter_optimization(
    X_train, y_train,
    optimization_budget='medium',  # or 'fast' / 'thorough'
    use_optuna=True,
    n_trials=50
)

print("🎉 Comprehensive optimization complete!")
print(f"Best R² score: {optimization_results['best_score']:.4f}")
print(f"Best parameters: {optimization_results['best_parameters']}")
```

## 🔄 Method 4: Migration and Enhancement Workflow

```python
# Create a standard XGBRegressor with existing params first
base_model = XGBRegressor(**existing_params)
base_model.fit(X_train, y_train)

# Upgrade to ScientificXGBRegressor
scientific_model = ScientificXGBRegressor.from_xgb_regressor(
    base_model,
    auto_tune=True,
    preserve_training=True
)

# Apply migration and retraining with enhancements
migration_results = scientific_model.migrate_and_retrain(
    X_train, y_train,
    apply_scientific_tuning=True,
    refit_strategy='enhanced'
)

print(f"Performance improvement: {migration_results['performance_improvement']}")
```

## 📈 Method 5: Incremental Learning from Existing Model

```python
# Start with existing parameters
model = create_scientific_xgb_regressor(**existing_params)

# Create incremental pipeline with existing model parameters
pipeline = create_incremental_pipeline(
    n_chunks=10,
    n_estimators_per_chunk=100,  # Add estimators incrementally
    use_gpu=None,
    verbose=True
)

# Override the default model creation with your parameters
pipeline.config.use_gpu = model.use_gpu
pipeline.config.verbose = True

# Run incremental training
results = pipeline.run_incremental_training(X, y)

# Extract the improved model
final_model = extract_model_from_pipeline(pipeline)
print(f"Final model estimators: {final_model.n_estimators}")
```

## 🎯 Method 6: Parameter Extraction and Improvement Loop

```python
def iterative_improvement(X, y, initial_params, max_iterations=3):
    '''
    Iteratively improve model parameters starting from initial params.
    '''
    current_params = initial_params.copy()
    best_score = -np.inf
    improvement_history = []
    
    for iteration in range(max_iterations):
        print(f"\\n🔄 Iteration {iteration + 1}/{max_iterations}")
        
        # Create model with current parameters
        model = create_scientific_xgb_regressor(**current_params, auto_tune=False)
        
        # Apply optimization
        optimization_results = model.comprehensive_hyperparameter_optimization(
            X, y,
            optimization_budget='fast',
            n_trials=20
        )
        
        current_score = optimization_results['best_score']
        
        if current_score > best_score:
            best_score = current_score
            # Extract improved parameters for next iteration
            current_params = model.extract_optimal_hyperparameters()
            # Remove metadata
            current_params.pop('_performance_metadata', None)
            
            improvement_history.append({
                'iteration': iteration + 1,
                'score': current_score,
                'parameters': current_params.copy()
            })
            
            print(f"✅ Improvement found: R² = {current_score:.4f}")
        else:
            print(f"⏹️ No improvement: R² = {current_score:.4f}")
            break
    
    return model, improvement_history

# Run iterative improvement
final_model, history = iterative_improvement(X_train, y_train, existing_params)
```

## 🔧 Method 7: Warm Start from Saved Model

```python
# If you have a saved model pickle
try:
    # Load existing model
    with open('existing_model.pkl', 'rb') as f:
        existing_model = pickle.load(f)
    
    # Upgrade and improve
    scientific_model = ScientificXGBRegressor.from_xgb_regressor(
        existing_model,
        preserve_training=True
    )
    
    # Continue training with more estimators
    incremental_results = scientific_model.incremental_learn(
        X_new, y_new,
        n_new_estimators=500
    )
    
except FileNotFoundError:
    print("No existing model found, starting fresh...")
```

## 📊 Parameter Tracking and Comparison

```python
def compare_parameter_sets(X, y, param_sets, set_names):
    '''
    Compare multiple parameter sets and return the best one.
    '''
    results = {}
    
    for name, params in zip(set_names, param_sets):
        print(f"\\n🧪 Testing parameter set: {name}")
        
        model = create_scientific_xgb_regressor(**params, auto_tune=False)
        
        # Quick cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        results[name] = {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'params': params
        }
        
        print(f"   R² Score: {results[name]['mean_score']:.4f} ± {results[name]['std_score']:.4f}")
    
    # Find best parameter set
    best_name = max(results.keys(), key=lambda k: results[k]['mean_score'])
    print(f"\\n🏆 Best parameter set: {best_name}")
    
    return results, best_name

# Example usage
param_sets = [
    existing_params,  # Your original params
    {**existing_params, 'n_estimators': 1500},  # More estimators
    {**existing_params, 'learning_rate': 0.05, 'n_estimators': 2000}  # Lower LR, more trees
]
set_names = ['Original', 'More Estimators', 'Lower LR + More Trees']

comparison_results, best_set = compare_parameter_sets(X_train, y_train, param_sets, set_names)
```

## 🎯 Recommended Workflow for Most Use Cases

```python
# 1. Start with existing parameters
model = create_scientific_xgb_regressor(**existing_params, auto_tune=False)
baseline_score = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
print(f"Baseline R²: {baseline_score:.4f}")

# 2. Apply automated improvements
model.automated_parameterization(X, y)
improved_score = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
print(f"After auto-tuning R²: {improved_score:.4f}")

# 3. If significant data, use comprehensive optimization
if len(X) > 10000:
    optimization_results = model.comprehensive_hyperparameter_optimization(
        X, y, optimization_budget='medium'
    )
    final_score = optimization_results['best_score']
    print(f"After comprehensive optimization R²: {final_score:.4f}")

# 4. Extract final optimized parameters for future use
final_params = model.extract_optimal_hyperparameters()
print("🎯 Final optimized parameters ready for reuse!")
```

## 🏗️ Advanced Use Cases

### A. Ensemble of Different Parameter Sets

```python
def create_parameter_ensemble(X, y, base_params, n_variants=5):
    '''
    Create ensemble from multiple parameter variations.
    '''
    models = []
    param_variants = []
    
    for i in range(n_variants):
        # Create parameter variations
        variant_params = base_params.copy()
        variant_params['n_estimators'] = base_params['n_estimators'] + i * 100
        variant_params['learning_rate'] = base_params['learning_rate'] * (0.8 + i * 0.1)
        variant_params['max_depth'] = max(3, base_params['max_depth'] + i - 2)
        
        # Train model with variant
        model = create_scientific_xgb_regressor(**variant_params, auto_tune=False)
        model.fit(X, y)
        
        models.append(model)
        param_variants.append(variant_params)
    
    return models, param_variants

# Create ensemble
ensemble_models, ensemble_params = create_parameter_ensemble(X_train, y_train, existing_params)

# Make ensemble predictions
def ensemble_predict(models, X):
    predictions = np.array([model.predict(X) for model in models])
    return np.mean(predictions, axis=0)

ensemble_pred = ensemble_predict(ensemble_models, X_test)
ensemble_score = r2_score(y_test, ensemble_pred)
print(f"Ensemble R² Score: {ensemble_score:.4f}")
```

### B. Parameter Evolution with Genetic Algorithm Concept

```python
def evolve_parameters(X, y, base_params, generations=5, population_size=10):
    '''
    Evolve parameters using genetic algorithm-inspired approach.
    '''
    population = []
    
    # Initialize population with variations of base parameters
    for _ in range(population_size):
        variant = base_params.copy()
        # Add random mutations
        variant['n_estimators'] = max(50, base_params['n_estimators'] + np.random.randint(-200, 200))
        variant['learning_rate'] = max(0.01, base_params['learning_rate'] * np.random.uniform(0.5, 1.5))
        variant['max_depth'] = max(2, base_params['max_depth'] + np.random.randint(-2, 3))
        variant['subsample'] = max(0.5, min(1.0, base_params['subsample'] + np.random.uniform(-0.2, 0.2)))
        
        population.append(variant)
    
    best_params = None
    best_score = -np.inf
    
    for generation in range(generations):
        print(f"🧬 Generation {generation + 1}/{generations}")
        
        # Evaluate population
        scores = []
        for params in population:
            model = create_scientific_xgb_regressor(**params, auto_tune=False)
            score = cross_val_score(model, X, y, cv=3, scoring='r2').mean()
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        # Select top performers for next generation
        top_indices = np.argsort(scores)[-population_size//2:]
        survivors = [population[i] for i in top_indices]
        
        # Create new generation
        new_population = survivors.copy()
        while len(new_population) < population_size:
            # Crossover: combine two random survivors
            parent1, parent2 = np.random.choice(survivors, 2, replace=False)
            child = {}
            for key in parent1.keys():
                child[key] = parent1[key] if np.random.random() < 0.5 else parent2[key]
            new_population.append(child)
        
        population = new_population
        print(f"   Best score this generation: {max(scores):.4f}")
    
    return best_params, best_score

# Evolve parameters
evolved_params, evolved_score = evolve_parameters(X_train, y_train, existing_params)
print(f"\\nEvolved parameters achieved R²: {evolved_score:.4f}")
```

### C. Adaptive Learning Rate Schedule

```python
def adaptive_learning_schedule(X, y, base_params, stages=3):
    '''
    Train with decreasing learning rates in stages.
    '''
    current_params = base_params.copy()
    model = create_scientific_xgb_regressor(**current_params, auto_tune=False)
    
    learning_rates = [current_params['learning_rate'] * (0.5 ** i) for i in range(stages)]
    estimators_per_stage = current_params['n_estimators'] // stages
    
    for stage, lr in enumerate(learning_rates):
        print(f"📚 Training stage {stage + 1}/{stages} with LR: {lr:.4f}")
        
        # Update learning rate and estimators for this stage
        stage_params = current_params.copy()
        stage_params['learning_rate'] = lr
        stage_params['n_estimators'] = estimators_per_stage
        
        if stage == 0:
            # First stage: fresh training
            model = create_scientific_xgb_regressor(**stage_params, auto_tune=False)
            model.fit(X, y)
        else:
            # Subsequent stages: incremental learning
            incremental_results = model.incremental_learn(X, y, n_new_estimators=estimators_per_stage)
            # Update learning rate for next incremental learning
            model.learning_rate = lr
        
        # Evaluate progress
        score = model.score(X, y)
        print(f"   Stage {stage + 1} R² Score: {score:.4f}")
    
    return model

# Apply adaptive learning schedule
adaptive_model = adaptive_learning_schedule(X_train, y_train, existing_params)
```

## 💡 Quick Start Templates

### Template 1: Basic Parameter Improvement
```python
# Load your existing parameters
existing_params = {...}  # Your parameter dictionary

# Quick improvement workflow
model = create_scientific_xgb_regressor(**existing_params, auto_tune=False)
model.automated_parameterization(X, y)
model.fit(X, y)
improved_params = model.extract_optimal_hyperparameters()
```

### Template 2: Full Optimization Pipeline
```python
# Complete optimization starting from existing parameters
model = create_scientific_xgb_regressor(**existing_params, auto_tune=False)
optimization_results = model.comprehensive_hyperparameter_optimization(X, y)
final_model = model
final_params = model.extract_optimal_hyperparameters()
```

### Template 3: Large Dataset Incremental Approach
```python
# For large datasets
pipeline = create_incremental_pipeline(n_chunks=10, n_estimators_per_chunk=100)
results = pipeline.run_incremental_training(X, y)
final_model = extract_model_from_pipeline(pipeline)
refit_results = final_model.refit_on_full_data(X, y, apply_auto_tuning=True)
```

This comprehensive guide covers all major scenarios for working with existing 
parameters and systematically improving them using the ScientificXGBRegressor framework.
"""