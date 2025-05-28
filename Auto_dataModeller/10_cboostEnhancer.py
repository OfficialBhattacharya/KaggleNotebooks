import pandas as pd
import numpy as np
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            accuracy_score, f1_score, roc_auc_score, log_loss,
                            precision_score, recall_score, classification_report)
from catboost import CatBoostRegressor, CatBoostClassifier
import os
import warnings
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')

sns.set_palette("husl")

class ModelLogger:
    """Enhanced model logging with detailed tracking and visualization capabilities."""
    
    def __init__(self, log_dir: str = "model_logs"):
        """
        Initialize the model logger.
        
        Parameters:
        -----------
        log_dir : str
            Directory to save logs and visualizations
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.log_df = pd.DataFrame(columns=[
            'timestamp', 'model_name', 'transformations', 'feature_group',
            'cv_scheme', 'train_metrics', 'test_metrics', 'hyperparameters',
            'training_time', 'task_type', 'optimization_method'
        ])
        
        self.best_models = {}
        self.optimization_history = []
    
    def add_model(self, model_name: str, transformations: str, feature_group: List[str],
                 cv_scheme: str, train_metrics: Dict, test_metrics: Dict,
                 hyperparameters: Dict = None, training_time: float = None,
                 task_type: str = None, optimization_method: str = None):
        """Add a model to the log with comprehensive tracking."""
        
        new_row = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'transformations': transformations,
            'feature_group': json.dumps(feature_group) if isinstance(feature_group, list) else feature_group,
            'cv_scheme': cv_scheme,
            'train_metrics': json.dumps(train_metrics),
            'test_metrics': json.dumps(test_metrics),
            'hyperparameters': json.dumps(hyperparameters) if hyperparameters else None,
            'training_time': training_time,
            'task_type': task_type,
            'optimization_method': optimization_method
        }
        
        self.log_df = pd.concat([self.log_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Track best models by primary metric
        primary_metric = self._get_primary_metric(test_metrics, task_type)
        if primary_metric:
            if model_name not in self.best_models or self._is_better_score(
                test_metrics[primary_metric], 
                json.loads(self.best_models[model_name]['test_metrics'])[primary_metric],
                task_type
            ):
                self.best_models[model_name] = new_row
    
    def _get_primary_metric(self, metrics: Dict, task_type: str) -> str:
        """Get the primary metric for comparison based on task type."""
        if task_type == 'regression':
            if 'rmse' in metrics:
                return 'rmse'
            elif 'mse' in metrics:
                return 'mse'
            elif 'mae' in metrics:
                return 'mae'
        else:  # classification
            if 'roc_auc' in metrics:
                return 'roc_auc'
            elif 'f1' in metrics:
                return 'f1'
            elif 'accuracy' in metrics:
                return 'accuracy'
        return list(metrics.keys())[0] if metrics else None
    
    def _is_better_score(self, new_score: float, old_score: float, task_type: str) -> bool:
        """Determine if new score is better than old score."""
        if task_type == 'regression':
            return new_score < old_score  # Lower is better for regression metrics
        else:
            return new_score > old_score  # Higher is better for classification metrics
    
    def save_log(self, filename: str = None):
        """Save the model log to CSV with timestamp."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_log_{timestamp}.csv"
        
        filepath = self.log_dir / filename
        self.log_df.to_csv(filepath, index=False)
        print(f"📊 Model log saved to {filepath}")
        
        # Save best models summary
        if self.best_models:
            best_df = pd.DataFrame(list(self.best_models.values()))
            best_filepath = self.log_dir / f"best_models_{timestamp}.csv"
            best_df.to_csv(best_filepath, index=False)
            print(f"🏆 Best models summary saved to {best_filepath}")
    
    def plot_model_comparison(self, metric: str = None, task_type: str = 'regression'):
        """Create visualization comparing model performances."""
        if self.log_df.empty:
            print("No models to compare yet.")
            return
        
        # Parse metrics from JSON strings
        test_metrics_list = []
        model_names = []
        
        for idx, row in self.log_df.iterrows():
            try:
                metrics = json.loads(row['test_metrics'])
                test_metrics_list.append(metrics)
                model_names.append(row['model_name'])
            except:
                continue
        
        if not test_metrics_list:
            print("No valid metrics found for comparison.")
            return
        
        # Determine metric to plot
        if metric is None:
            metric = self._get_primary_metric(test_metrics_list[0], task_type)
        
        if metric not in test_metrics_list[0]:
            print(f"Metric '{metric}' not found in model results.")
            return
        
        # Extract metric values
        metric_values = [metrics[metric] for metrics in test_metrics_list]
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(model_names)), metric_values, alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel(metric.upper())
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(metric_values),
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = self.log_dir / f"model_comparison_{metric}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📈 Model comparison plot saved to {plot_path}")

def check_gpu_availability():
    """
    Check GPU availability and return configuration information.
    Enhanced for CatBoost GPU support detection.
    """
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'gpu_memory': [],
        'total_memory_gb': 0,
        'free_memory_gb': 0,
        'catboost_gpu_capable': False,
        'recommended_task_type': 'CPU'
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['catboost_gpu_capable'] = True
            gpu_info['recommended_task_type'] = 'GPU'
            
            total_memory = 0
            free_memory = 0
            
            print(f"\n🎮 GPU Detection Results:")
            print(f"   CUDA Available: {gpu_info['cuda_available']}")
            print(f"   Number of GPUs: {gpu_info['gpu_count']}")
            
            for i in range(gpu_info['gpu_count']):
                name = torch.cuda.get_device_name(i)
                gpu_info['gpu_names'].append(name)
                
                # Get memory information
                torch.cuda.set_device(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                memory_free = memory_total - memory_reserved
                
                gpu_info['gpu_memory'].append({
                    'total_gb': memory_total,
                    'free_gb': memory_free
                })
                
                total_memory += memory_total
                free_memory += memory_free
                
                print(f"   GPU {i}: {name}")
                print(f"      Memory: {memory_total:.1f} GB total, {memory_free:.1f} GB free")
            
            gpu_info['total_memory_gb'] = total_memory
            gpu_info['free_memory_gb'] = free_memory
            
            print(f"   🚀 CatBoost GPU Support: Available")
            print(f"   💡 Note: CatBoost uses single GPU optimization")
            
            logger.info(f"CUDA available with {gpu_info['gpu_count']} GPU(s) for CatBoost")
        else:
            print(f"\n💻 GPU Detection Results:")
            print(f"   CUDA Available: False")
            print(f"   Using CPU for computation")
            logger.warning("CUDA not available - using CPU")
    except ImportError:
        print(f"\n💻 GPU Detection Results:")
        print(f"   PyTorch not available - cannot check CUDA status")
        print(f"   Using CPU for computation")
        logger.warning("PyTorch not available - cannot check CUDA status")
    
    return gpu_info

def load_data(train_path: str, test_path: str, target_col: str, feature_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load training and test data with enhanced error handling and validation.
    
    Parameters:
    -----------
    train_path : str
        Path to training data CSV file
    test_path : str
        Path to test data CSV file
    target_col : str
        Name of the target column
    feature_cols : List[str], optional
        List of feature column names. If None, uses all columns except target
    
    Returns:
    --------
    X_train, y_train, X_test, y_test : tuple
        Training and test features and targets
    """
    try:
        print(f"📂 Loading data from {train_path} and {test_path}")
        
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        
        print(f"   Training data shape: {train.shape}")
        print(f"   Test data shape: {test.shape}")
        
        # Validate target column exists
        if target_col not in train.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data")
        if target_col not in test.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data")
        
        # Select features
        if feature_cols:
            missing_cols = set(feature_cols) - set(train.columns)
            if missing_cols:
                raise ValueError(f"Feature columns not found in training data: {missing_cols}")
            X_train = train[feature_cols].copy()
            X_test = test[feature_cols].copy()
        else:
            X_train = train.drop(columns=[target_col]).copy()
            X_test = test.drop(columns=[target_col]).copy()
        
        y_train = train[target_col].copy()
        y_test = test[target_col].copy()
        
        print(f"   Features shape: {X_train.shape}")
        print(f"   Target distribution (train): {y_train.describe()}")
        
        # Check for missing values
        train_missing = X_train.isnull().sum().sum()
        test_missing = X_test.isnull().sum().sum()
        if train_missing > 0 or test_missing > 0:
            print(f"   ⚠️  Missing values found - Train: {train_missing}, Test: {test_missing}")
        
        # Detect categorical features for CatBoost
        cat_features = []
        for col in X_train.columns:
            if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                cat_features.append(col)
        
        if cat_features:
            print(f"   🏷️  Categorical features detected: {len(cat_features)} columns")
            print(f"      {cat_features[:5]}{'...' if len(cat_features) > 5 else ''}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def load_model(model_path: str):
    """Load a pre-trained model with error handling."""
    try:
        print(f"📦 Loading model from {model_path}")
        model = joblib.load(model_path)
        print(f"   Model type: {type(model).__name__}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def detect_task(y: pd.Series) -> str:
    """
    Automatically detect the machine learning task type based on target variable.
    
    Parameters:
    -----------
    y : pd.Series
        Target variable
    
    Returns:
    --------
    task_type : str
        One of 'binary_classification', 'multiclass_classification', or 'regression'
    """
    unique_vals = y.unique()
    n_unique = len(unique_vals)
    
    # Check if all values are integers
    is_integer = y.dtype in ['int64', 'int32'] or y.apply(lambda x: float(x).is_integer()).all()
    
    if n_unique == 2:
        task_type = 'binary_classification'
    elif n_unique <= 20 and is_integer:
        task_type = 'multiclass_classification'
    else:
        task_type = 'regression'
    
    print(f"🎯 Task Detection:")
    print(f"   Unique values: {n_unique}")
    print(f"   Task type: {task_type}")
    print(f"   Target range: [{y.min():.3f}, {y.max():.3f}]")
    
    return task_type

def detect_categorical_features(X: pd.DataFrame) -> List[int]:
    """
    Detect categorical features for CatBoost.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    
    Returns:
    --------
    cat_features : List[int]
        List of categorical feature indices
    """
    cat_features = []
    for i, col in enumerate(X.columns):
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            cat_features.append(i)
        elif X[col].nunique() <= 10 and X[col].dtype in ['int64', 'int32']:
            # Consider low-cardinality integer features as categorical
            cat_features.append(i)
    
    return cat_features

def evaluate_model(model, X: pd.DataFrame, y: pd.Series, task: str) -> Dict[str, float]:
    """
    Comprehensive model evaluation for both regression and classification tasks.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X : pd.DataFrame
        Features
    y : pd.Series
        True target values
    task : str
        Task type ('regression', 'binary_classification', 'multiclass_classification')
    
    Returns:
    --------
    metrics : Dict[str, float]
        Dictionary of evaluation metrics
    """
    try:
        pred = model.predict(X)
        metrics = {}
        
        if task in ['binary_classification', 'multiclass_classification']:
            metrics['accuracy'] = accuracy_score(y, pred)
            
            if task == 'binary_classification':
                metrics['precision'] = precision_score(y, pred, zero_division=0)
                metrics['recall'] = recall_score(y, pred, zero_division=0)
                metrics['f1'] = f1_score(y, pred, zero_division=0)
            else:
                metrics['precision'] = precision_score(y, pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y, pred, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(y, pred, average='weighted', zero_division=0)
            
            # Add probability-based metrics if available
            try:
                proba = model.predict_proba(X)
                if task == 'binary_classification':
                    metrics['roc_auc'] = roc_auc_score(y, proba[:, 1])
                    metrics['log_loss'] = log_loss(y, proba)
                else:
                    metrics['roc_auc'] = roc_auc_score(y, proba, multi_class='ovr', average='weighted')
                    metrics['log_loss'] = log_loss(y, proba)
            except (AttributeError, ValueError) as e:
                logger.warning(f"Could not calculate probability-based metrics: {str(e)}")
        
        else:  # regression
            metrics['mse'] = mean_squared_error(y, pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y, pred)
            metrics['r2'] = r2_score(y, pred)
            
            # Add RMSLE for positive targets
            if (y > 0).all() and (pred > 0).all():
                metrics['rmsle'] = np.sqrt(mean_squared_error(np.log1p(y), np.log1p(pred)))
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return {}

def elbow_tune_n_estimators(X_train: pd.DataFrame, y_train: pd.Series, 
                           X_val: pd.DataFrame, y_val: pd.Series, 
                           task: str, base_params: Dict, cat_features: List[int] = None,
                           min_estimators: int = 50, max_estimators: int = 1000, 
                           step: int = 50, use_gpu: bool = False, 
                           early_stopping_rounds: int = 10) -> Tuple[int, Any]:
    """
    Use elbow method to find optimal number of estimators for CatBoost.
    
    Parameters:
    -----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training data
    X_val, y_val : pd.DataFrame, pd.Series
        Validation data
    task : str
        Task type
    base_params : Dict
        Base parameters for the model
    cat_features : List[int], optional
        List of categorical feature indices
    min_estimators, max_estimators : int
        Range for number of estimators
    step : int
        Step size for the search
    use_gpu : bool
        Whether to use GPU acceleration
    early_stopping_rounds : int
        Early stopping patience
    
    Returns:
    --------
    optimal_estimators : int
        Optimal number of estimators
    best_model : model
        Best model found during the search
    """
    print(f"🔍 Elbow Method: Tuning n_estimators from {min_estimators} to {max_estimators}")
    
    if use_gpu:
        base_params = base_params.copy()
        base_params['task_type'] = 'GPU'
        base_params['devices'] = '0'
    
    metrics = []
    best_score = float('inf') if task == 'regression' else float('-inf')
    best_model = None
    
    n_estimators_range = range(min_estimators, max_estimators + 1, step)
    
    for i, n_estimators in enumerate(n_estimators_range):
        print(f"   Testing n_estimators={n_estimators} ({i+1}/{len(n_estimators_range)})")
        
        try:
            if task == 'regression':
                model = CatBoostRegressor(iterations=n_estimators, **base_params)
            else:
                model = CatBoostClassifier(iterations=n_estimators, **base_params)
            
            # CatBoost specific training with early stopping
            model.fit(X_train, y_train, 
                     eval_set=(X_val, y_val),
                     cat_features=cat_features,
                     early_stopping_rounds=early_stopping_rounds, 
                     verbose=False)
            
            # Get validation metric
            if task == 'regression':
                score = mean_squared_error(y_val, model.predict(X_val))
                is_better = score < best_score
            else:
                try:
                    proba = model.predict_proba(X_val)
                    score = log_loss(y_val, proba)
                    is_better = score < best_score
                except:
                    score = accuracy_score(y_val, model.predict(X_val))
                    is_better = score > best_score
            
            metrics.append((n_estimators, score))
            
            if is_better:
                best_score = score
                best_model = model
                
        except Exception as e:
            logger.warning(f"Error with n_estimators={n_estimators}: {str(e)}")
            continue
    
    if not metrics:
        raise ValueError("No valid models found during elbow tuning")
    
    # Find elbow point
    n_est_values = [m[0] for m in metrics]
    scores = [m[1] for m in metrics]
    elbow_index = find_elbow_index(scores)
    optimal_estimators = n_est_values[elbow_index]
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(n_est_values, scores, 'b-o', markersize=4, linewidth=2)
    plt.xlabel('Number of Estimators (Iterations)')
    plt.ylabel('Validation Score' if task == 'regression' else 'Validation Loss')
    plt.title(f'CatBoost Elbow Method for n_estimators Optimization\nTask: {task}')
    plt.axvline(x=optimal_estimators, color='r', linestyle='--', linewidth=2, 
                label=f'Optimal: {optimal_estimators}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = f'elbow_n_estimators_catboost_{task}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   📈 Elbow plot saved to {plot_path}")
    print(f"   🎯 Optimal n_estimators: {optimal_estimators}")
    print(f"   📊 Best validation score: {best_score:.6f}")
    
    return optimal_estimators, best_model

def find_elbow_index(values: List[float]) -> int:
    """
    Find elbow point using the method of maximum curvature.
    
    Parameters:
    -----------
    values : List[float]
        List of metric values
    
    Returns:
    --------
    elbow_index : int
        Index of the elbow point
    """
    if len(values) < 3:
        return 0
    
    n_points = len(values)
    all_coords = np.vstack((range(n_points), values)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    
    return np.argmax(dist_to_line)

def optuna_objective(trial, X: pd.DataFrame, y: pd.Series, task: str, 
                    base_params: Dict, cat_features: List[int] = None,
                    use_gpu: bool = False, cv_folds: int = 3) -> float:
    """
    Optuna objective function for CatBoost hyperparameter optimization.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    X, y : pd.DataFrame, pd.Series
        Training data
    task : str
        Task type
    base_params : Dict
        Base parameters for the model
    cat_features : List[int], optional
        List of categorical feature indices
    use_gpu : bool
        Whether to use GPU acceleration
    cv_folds : int
        Number of cross-validation folds
    
    Returns:
    --------
    score : float
        Cross-validation score to optimize
    """
    # Suggest CatBoost-specific hyperparameters
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'rsm': trial.suggest_float('rsm', 0.5, 1.0),  # Random subspace method
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
    }
    
    # CatBoost specific parameters
    if len(cat_features) > 0 if cat_features else False:
        params['one_hot_max_size'] = trial.suggest_int('one_hot_max_size', 2, 25)
    
    if use_gpu:
        params['task_type'] = 'GPU'
        params['devices'] = '0'
        # GPU-specific optimizations
        params['gpu_ram_part'] = trial.suggest_float('gpu_ram_part', 0.5, 0.95)
    
    params.update(base_params)
    
    # Cross-validation
    if task in ['binary_classification', 'multiclass_classification']:
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        try:
            if task == 'regression':
                model = CatBoostRegressor(**params, iterations=500, random_state=42)
            else:
                model = CatBoostClassifier(**params, iterations=500, random_state=42)
            
            model.fit(X_train_fold, y_train_fold,
                     eval_set=(X_val_fold, y_val_fold),
                     cat_features=cat_features,
                     early_stopping_rounds=20,
                     verbose=False)
            
            # Calculate score based on task
            if task == 'regression':
                pred = model.predict(X_val_fold)
                score = mean_squared_error(y_val_fold, pred)
            else:
                try:
                    proba = model.predict_proba(X_val_fold)
                    score = log_loss(y_val_fold, proba)
                except:
                    pred = model.predict(X_val_fold)
                    score = 1 - accuracy_score(y_val_fold, pred)  # Convert to loss
            
            scores.append(score)
            
        except Exception as e:
            logger.warning(f"Error in fold {fold}: {str(e)}")
            # Return a poor score to indicate failure
            return float('inf') if task == 'regression' else 1.0
    
    return np.mean(scores)

def run_optuna_optimization(X_train: pd.DataFrame, y_train: pd.Series, task: str,
                           base_params: Dict, cat_features: List[int] = None,
                           use_gpu: bool = False, n_trials: int = 100, cv_folds: int = 3,
                           timeout: int = 3600) -> Tuple[Dict, optuna.Study]:
    """
    Run Optuna hyperparameter optimization for CatBoost.
    
    Parameters:
    -----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training data
    task : str
        Task type
    base_params : Dict
        Base parameters for the model
    cat_features : List[int], optional
        List of categorical feature indices
    use_gpu : bool
        Whether to use GPU acceleration
    n_trials : int
        Number of optimization trials
    cv_folds : int
        Number of cross-validation folds
    timeout : int
        Timeout in seconds
    
    Returns:
    --------
    best_params : Dict
        Best hyperparameters found
    study : optuna.Study
        Optuna study object with optimization history
    """
    print(f"🔬 Starting CatBoost Optuna optimization with {n_trials} trials")
    print(f"   Task: {task}")
    print(f"   CV folds: {cv_folds}")
    print(f"   GPU acceleration: {use_gpu}")
    print(f"   Categorical features: {len(cat_features) if cat_features else 0}")
    print(f"   Timeout: {timeout} seconds")
    
    # Create study
    direction = 'minimize'  # We minimize loss for both regression and classification
    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Define objective with fixed parameters
    objective = lambda trial: optuna_objective(
        trial, X_train, y_train, task, base_params, cat_features, use_gpu, cv_folds
    )
    
    # Run optimization
    start_time = time.time()
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    except KeyboardInterrupt:
        print("   ⚠️  Optimization interrupted by user")
    
    optimization_time = time.time() - start_time
    
    print(f"   ✅ Optimization completed in {optimization_time:.2f} seconds")
    print(f"   🏆 Best score: {study.best_value:.6f}")
    print(f"   📊 Completed trials: {len(study.trials)}")
    
    # Get best parameters
    best_params = study.best_params.copy()
    best_params.update(base_params)
    
    return best_params, study

def create_optimization_plots(study: optuna.Study, save_dir: str = "optimization_plots"):
    """Create visualization plots for Optuna optimization results."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    try:
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(save_path / "catboost_optimization_history.png")
        
        # Parameter importance
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(save_path / "catboost_param_importances.png")
        
        # Parallel coordinate plot
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(save_path / "catboost_parallel_coordinate.png")
        
        print(f"📊 CatBoost optimization plots saved to {save_path}")
        
    except Exception as e:
        logger.warning(f"Could not create optimization plots: {str(e)}")

def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, 
                     X_test: pd.DataFrame, y_test: pd.Series,
                     task: str, best_params: Dict, cat_features: List[int] = None,
                     model_name: str = "optimized_catboost") -> Tuple[Any, Dict, Dict]:
    """
    Train the final CatBoost model with optimized hyperparameters.
    
    Parameters:
    -----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training data
    X_test, y_test : pd.DataFrame, pd.Series
        Test data
    task : str
        Task type
    best_params : Dict
        Optimized hyperparameters
    cat_features : List[int], optional
        List of categorical feature indices
    model_name : str
        Name for the model
    
    Returns:
    --------
    model : trained model
        Final trained model
    train_metrics : Dict
        Training metrics
    test_metrics : Dict
        Test metrics
    """
    print(f"🎯 Training final CatBoost model: {model_name}")
    
    start_time = time.time()
    
    # Create and train model
    if task == 'regression':
        model = CatBoostRegressor(**best_params, random_state=42)
    else:
        model = CatBoostClassifier(**best_params, random_state=42)
    
    # Train with early stopping
    model.fit(X_train, y_train,
             eval_set=(X_test, y_test),
             cat_features=cat_features,
             early_stopping_rounds=20,
             verbose=False)
    
    training_time = time.time() - start_time
    
    # Evaluate model
    train_metrics = evaluate_model(model, X_train, y_train, task)
    test_metrics = evaluate_model(model, X_test, y_test, task)
    
    print(f"   ⏱️  Training time: {training_time:.2f} seconds")
    print(f"   📊 Training metrics: {train_metrics}")
    print(f"   📊 Test metrics: {test_metrics}")
    
    # Save model
    model_path = f"{model_name}_{task}.joblib"
    joblib.dump(model, model_path)
    print(f"   💾 Model saved to {model_path}")
    
    return model, train_metrics, test_metrics

def create_feature_importance_plot(model, feature_names: List[str], 
                                 model_name: str = "model", top_n: int = 20):
    """Create feature importance visualization for CatBoost."""
    try:
        # Get feature importance
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        top_features = feature_importance_df.head(top_n)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name} (CatBoost)')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        plot_path = f"feature_importance_catboost_{model_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 CatBoost feature importance plot saved to {plot_path}")
        
        return feature_importance_df
        
    except Exception as e:
        logger.warning(f"Could not create feature importance plot: {str(e)}")
        return None

def main():
    """
    Main function to run the enhanced CatBoost pipeline.
    """
    print("🚀 Enhanced CatBoost Pipeline Starting...")
    print("=" * 60)
    
    # Configuration
    config = {
        'train_path': 'train.csv',
        'test_path': 'test.csv',
        'model_path': 'baseline_model.joblib',
        'target_col': 'target',
        'feature_cols': None,  # Use all columns except target if None
        'transformations': "Standard Scaling + Feature Engineering",
        'cv_scheme': "5-Fold Cross Validation",
        
        # Optimization settings
        'use_elbow_method': True,
        'use_optuna': True,
        'optuna_trials': 100,
        'optuna_timeout': 3600,  # 1 hour
        'cv_folds': 5,
        
        # Performance settings
        'use_gpu': True,
        
        # Output settings
        'save_models': True,
        'create_plots': True,
        'log_results': True
    }
    
    try:
        # Check GPU availability
        gpu_info = check_gpu_availability()
        config['use_gpu'] = gpu_info['catboost_gpu_capable'] and config['use_gpu']
        
        # Load data
        X_train, y_train, X_test, y_test = load_data(
            config['train_path'], config['test_path'], 
            config['target_col'], config['feature_cols']
        )
        
        feature_names = list(X_train.columns)
        
        # Detect categorical features
        cat_features = detect_categorical_features(X_train)
        
        # Detect task type
        task = detect_task(y_train)
        
        # Initialize model logger
        logger_instance = ModelLogger()
        
        # Load and evaluate baseline model
        print("\n" + "="*60)
        print("📊 BASELINE MODEL EVALUATION")
        print("="*60)
        
        try:
            baseline_model = load_model(config['model_path'])
            baseline_train_metrics = evaluate_model(baseline_model, X_train, y_train, task)
            baseline_test_metrics = evaluate_model(baseline_model, X_test, y_test, task)
            
            logger_instance.add_model(
                model_name='baseline_model',
                transformations=config['transformations'],
                feature_group=feature_names,
                cv_scheme='None',
                train_metrics=baseline_train_metrics,
                test_metrics=baseline_test_metrics,
                task_type=task,
                optimization_method='None'
            )
            
            print("✅ Baseline model evaluation completed")
            
        except Exception as e:
            print(f"⚠️  Could not load baseline model: {str(e)}")
            baseline_model = None
        
        # Base parameters for CatBoost
        base_params = {
            'random_state': 42,
            'verbose': False,
            'allow_writing_files': False
        }
        
        if config['use_gpu']:
            base_params['task_type'] = gpu_info['recommended_task_type']
            if gpu_info['cuda_available']:
                base_params['devices'] = '0'
        
        # Step 1: Elbow method for n_estimators
        if config['use_elbow_method']:
            print("\n" + "="*60)
            print("📈 ELBOW METHOD OPTIMIZATION")
            print("="*60)
            
            # Split training data for elbow method
            X_train_elbow, X_val_elbow, y_train_elbow, y_val_elbow = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42,
                stratify=y_train if task != 'regression' else None
            )
            
            optimal_estimators, elbow_model = elbow_tune_n_estimators(
                X_train_elbow, y_train_elbow, X_val_elbow, y_val_elbow,
                task, base_params, cat_features, use_gpu=config['use_gpu']
            )
            
            base_params['iterations'] = optimal_estimators
            
            # Evaluate elbow model
            elbow_train_metrics = evaluate_model(elbow_model, X_train, y_train, task)
            elbow_test_metrics = evaluate_model(elbow_model, X_test, y_test, task)
            
            logger_instance.add_model(
                model_name='elbow_optimized',
                transformations=config['transformations'],
                feature_group=feature_names,
                cv_scheme='Train-Validation Split',
                train_metrics=elbow_train_metrics,
                test_metrics=elbow_test_metrics,
                hyperparameters={'iterations': optimal_estimators},
                task_type=task,
                optimization_method='Elbow Method'
            )
            
            print("✅ Elbow method optimization completed")
        
        # Step 2: Optuna hyperparameter optimization
        if config['use_optuna']:
            print("\n" + "="*60)
            print("🔬 OPTUNA HYPERPARAMETER OPTIMIZATION")
            print("="*60)
            
            best_params, study = run_optuna_optimization(
                X_train, y_train, task, base_params, cat_features,
                use_gpu=config['use_gpu'],
                n_trials=config['optuna_trials'],
                cv_folds=config['cv_folds'],
                timeout=config['optuna_timeout']
            )
            
            # Create optimization plots
            if config['create_plots']:
                create_optimization_plots(study)
            
            print("✅ Optuna optimization completed")
        else:
            best_params = base_params.copy()
            study = None
        
        # Step 3: Train final optimized model
        print("\n" + "="*60)
        print("🎯 FINAL MODEL TRAINING")
        print("="*60)
        
        final_model, final_train_metrics, final_test_metrics = train_final_model(
            X_train, y_train, X_test, y_test, task, best_params, cat_features, "final_optimized_catboost"
        )
        
        # Log final model
        logger_instance.add_model(
            model_name='final_optimized',
            transformations=config['transformations'],
            feature_group=feature_names,
            cv_scheme=config['cv_scheme'],
            train_metrics=final_train_metrics,
            test_metrics=final_test_metrics,
            hyperparameters=best_params,
            task_type=task,
            optimization_method='Elbow + Optuna'
        )
        
        # Create feature importance plot
        if config['create_plots']:
            create_feature_importance_plot(final_model, feature_names, "final_optimized")
        
        # Step 4: Results summary and comparison
        print("\n" + "="*60)
        print("📊 RESULTS SUMMARY")
        print("="*60)
        
        # Save logs
        if config['log_results']:
            logger_instance.save_log()
            logger_instance.plot_model_comparison(task_type=task)
        
        # Print summary
        print("\n🏆 Model Performance Summary:")
        if baseline_model:
            print(f"   Baseline Model: {baseline_test_metrics}")
        if config['use_elbow_method']:
            print(f"   Elbow Optimized: {elbow_test_metrics}")
        print(f"   Final Optimized: {final_test_metrics}")
        
        print("\n✅ Enhanced CatBoost pipeline completed successfully!")
        
        return {
            'final_model': final_model,
            'best_params': best_params,
            'study': study,
            'logger': logger_instance,
            'task_type': task,
            'cat_features': cat_features
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    print("🌟 CatBoost Enhanced Pipeline")
    print("=" * 60)
    print("This script provides comprehensive CatBoost model optimization including:")
    print("• GPU acceleration support (CUDA)")
    print("• Automatic categorical feature detection and handling")
    print("• Elbow method for iterations optimization")
    print("• Optuna hyperparameter optimization")
    print("• Comprehensive model evaluation and comparison")
    print("• Feature importance analysis")
    print("• Model logging and visualization")
    print("=" * 60)
    
    # Run the pipeline
    try:
        results = main()
        
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Task Type: {results['task_type']}")
        print(f"Best Parameters: {results['best_params']}")
        print(f"Categorical Features: {len(results['cat_features'])} detected")
        print("Check the generated plots and logs for detailed analysis.")
        
        # Display key CatBoost-specific optimizations
        if 'task_type' in results['best_params']:
            print(f"\n🎮 GPU Configuration: {results['best_params']['task_type']}")
        if 'depth' in results['best_params']:
            print(f"🌳 Optimized depth: {results['best_params']['depth']}")
        if 'learning_rate' in results['best_params']:
            print(f"📈 Optimized learning_rate: {results['best_params']['learning_rate']:.4f}")
        if 'l2_leaf_reg' in results['best_params']:
            print(f"🔧 Optimized l2_leaf_reg: {results['best_params']['l2_leaf_reg']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        print("Please check your data paths and ensure all dependencies are installed.")
        print("\nRequired packages:")
        print("• catboost")
        print("• optuna")
        print("• scikit-learn")
        print("• pandas")
        print("• numpy")
        print("• matplotlib")
        print("• seaborn")
        print("\nFor GPU support:")
        print("• torch (for CUDA detection)")
        
        raise

"""
Example Usage:
--------------

# Basic usage with default configuration
python 10_cboostEnhancer.py

# Or import and use programmatically:
from cboostEnhancer import main, check_gpu_availability, load_data

# Check GPU availability
gpu_info = check_gpu_availability()
print(f"GPU available: {gpu_info['catboost_gpu_capable']}")

# Load your data
X_train, y_train, X_test, y_test = load_data('train.csv', 'test.csv', 'target')

# Run optimization pipeline
results = main()

# Access optimized model
optimized_model = results['final_model']
best_hyperparameters = results['best_params']
categorical_features = results['cat_features']

# Make predictions
predictions = optimized_model.predict(X_test)

Key Features:
-------------
1. **GPU Acceleration**: Automatic detection and configuration for CUDA
2. **Categorical Feature Handling**: Automatic detection and proper handling
3. **Elbow Method**: Optimal iterations selection using validation curves
4. **Optuna Optimization**: Advanced hyperparameter tuning with pruning
5. **CatBoost-Specific Parameters**: 
   - depth, learning_rate, l2_leaf_reg
   - subsample, rsm, border_count
   - bagging_temperature, random_strength
   - one_hot_max_size, min_data_in_leaf
6. **Comprehensive Evaluation**: Multiple metrics for regression/classification
7. **Model Logging**: Track and compare different optimization strategies
8. **Visualization**: Feature importance, optimization history, model comparison
9. **Error Handling**: Robust error handling with fallback options

Hyperparameters Optimized:
--------------------------
• learning_rate: Learning rate for boosting
• depth: Maximum tree depth
• l2_leaf_reg: L2 regularization coefficient
• subsample: Fraction of samples to use for each tree
• rsm: Random subspace method (fraction of features)
• border_count: Number of splits for numerical features
• bagging_temperature: Controls intensity of Bayesian bagging
• random_strength: Amount of randomness for scoring splits
• min_data_in_leaf: Minimum number of training samples in a leaf
• one_hot_max_size: Maximum size for one-hot encoding

CatBoost Advantages:
--------------------
• Excellent categorical feature handling without preprocessing
• Robust to hyperparameter choices
• Built-in overfitting detection
• No need for extensive data preprocessing
• Strong performance on structured data
• Handles missing values automatically

GPU Optimizations:
------------------
• Automatic CUDA detection and configuration
• Memory-optimized parameters for GPU training
• Single GPU optimization (CatBoost doesn't support multi-GPU)
• Fallback to optimized CPU settings when GPU unavailable
• GPU-specific memory management (gpu_ram_part)
"""
