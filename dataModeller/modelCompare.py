import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, KBinsDiscretizer
from sklearn.base import clone
from scipy import stats
from typing import Tuple, Optional, Union, Any
import warnings
import logging
from copy import deepcopy
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """
    Check GPU availability and log information about available devices.
    Enhanced to support multi-GPU detection and configuration.
    """
    gpu_info = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'gpu_memory': [],
        'gpu_utilization': [],
        'total_memory_gb': 0,
        'free_memory_gb': 0,
        'multi_gpu_capable': False
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['multi_gpu_capable'] = gpu_info['gpu_count'] > 1
            
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
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_free = memory_total - memory_reserved
                
                gpu_info['gpu_memory'].append({
                    'total_gb': memory_total,
                    'reserved_gb': memory_reserved,
                    'allocated_gb': memory_allocated,
                    'free_gb': memory_free
                })
                
                total_memory += memory_total
                free_memory += memory_free
                
                print(f"   GPU {i}: {name}")
                print(f"      Memory: {memory_total:.1f} GB total, {memory_free:.1f} GB free")
                
                # Try to get utilization if possible
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info['gpu_utilization'].append(utilization.gpu)
                    print(f"      Utilization: {utilization.gpu}%")
                except:
                    gpu_info['gpu_utilization'].append(None)
                    print(f"      Utilization: N/A")
            
            gpu_info['total_memory_gb'] = total_memory
            gpu_info['free_memory_gb'] = free_memory
            
            if gpu_info['multi_gpu_capable']:
                print(f"   🚀 Multi-GPU Setup: {gpu_info['gpu_count']} GPUs with {total_memory:.1f} GB total memory")
            
            logger.info(f"CUDA available: {gpu_info['cuda_available']}")
            logger.info(f"Number of GPUs: {gpu_info['gpu_count']}")
            for i, name in enumerate(gpu_info['gpu_names']):
                logger.info(f"GPU {i}: {name} ({gpu_info['gpu_memory'][i]['total_gb']:.1f} GB)")
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

# Set style for better plots
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('default')
        
sns.set_palette("husl")

def calculate_rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Logarithmic Error.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    rmsle : float
        Root Mean Squared Logarithmic Error
    """
    # Ensure positive values for log calculation
    y_true_pos = np.maximum(y_true, 1e-8)
    y_pred_pos = np.maximum(y_pred, 1e-8)
    
    return np.sqrt(np.mean((np.log1p(y_true_pos) - np.log1p(y_pred_pos)) ** 2))

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all metrics
    """
    metrics = {
        'RMSLE': calculate_rmsle(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics

def create_diagnostic_plots(fold_results, cv_predictions, y_train_original, X_test, test_predictions, 
                          model_name, dataset_name):
    """
    Create comprehensive diagnostic plots for model analysis.
        
        Parameters:
        -----------
    fold_results : list
        Results from each CV fold
    cv_predictions : array-like
        Cross-validation predictions
    y_train_original : array-like
        Original training targets
    X_test : array-like
        Test features
    test_predictions : array-like
        Test predictions
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset
            
        Returns:
        --------
    fig : matplotlib.Figure
            Figure object containing all diagnostic plots
        """
    logger.info("Creating diagnostic plots")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Model Diagnostic Analysis: {model_name} on {dataset_name}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Cross-Validation Metrics Across Folds
    metrics_df = pd.DataFrame([fold['oof_metrics'] for fold in fold_results])
    metrics_df.index = [f'Fold {i+1}' for i in range(len(fold_results))]
    
    metrics_to_plot = ['RMSLE', 'RMSE', 'R2']
    for i, metric in enumerate(metrics_to_plot):
        if i < 3:
            axes[0, i].bar(metrics_df.index, metrics_df[metric], alpha=0.7)
            axes[0, i].set_title(f'{metric} Across CV Folds')
            axes[0, i].set_ylabel(metric)
            axes[0, i].tick_params(axis='x', rotation=45)
            axes[0, i].grid(True, alpha=0.3)
            
            # Add mean line
            mean_val = metrics_df[metric].mean()
            axes[0, i].axhline(y=mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.4f}')
            axes[0, i].legend()
    
    # 2. Residual Analysis (using CV predictions)
    residuals = y_train_original - cv_predictions
    axes[1, 0].scatter(cv_predictions, residuals, alpha=0.6, s=20)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Analysis (CV Predictions)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3. Actual vs Predicted (CV)
    axes[1, 1].scatter(y_train_original, cv_predictions, alpha=0.6, s=20)
    min_val = min(y_train_original.min(), cv_predictions.min())
    max_val = max(y_train_original.max(), cv_predictions.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Predicted Values')
    axes[1, 1].set_title('Actual vs Predicted (CV)')
    r2_cv = r2_score(y_train_original, cv_predictions)
    axes[1, 1].text(0.05, 0.95, f'R² = {r2_cv:.4f}', transform=axes[1, 1].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 4. Residual Distribution
    axes[1, 2].hist(residuals, bins=30, alpha=0.7, density=True, color='skyblue')
    axes[1, 2].set_xlabel('Residuals')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Residual Distribution')
    
    # Overlay normal distribution
    mu, sigma = stats.norm.fit(residuals)
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1, 2].plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'r-', lw=2, label='Normal fit')
    axes[1, 2].legend()
    
    # 5. Feature Importance (if available)
    if hasattr(fold_results[0]['model'], 'feature_importances_'):
        # Average feature importance across folds
        importance_matrix = np.array([fold['model'].feature_importances_ for fold in fold_results])
        avg_importance = np.mean(importance_matrix, axis=0)
        std_importance = np.std(importance_matrix, axis=0)
        
        feature_names = [f'Feature_{i}' for i in range(len(avg_importance))]
        
        # Get top 10 features
        top_indices = np.argsort(avg_importance)[-10:]
        
        axes[2, 0].barh(range(len(top_indices)), avg_importance[top_indices], 
                       xerr=std_importance[top_indices], alpha=0.7)
        axes[2, 0].set_yticks(range(len(top_indices)))
        axes[2, 0].set_yticklabels([feature_names[i] for i in top_indices])
        axes[2, 0].set_xlabel('Feature Importance')
        axes[2, 0].set_title('Top 10 Feature Importance (Avg ± Std)')
    else:
        axes[2, 0].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                       ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Feature Importance')
    
    # 6. Cross-Validation Stability
    cv_scores_by_metric = {metric: [fold['oof_metrics'][metric] for fold in fold_results] 
                          for metric in ['RMSLE', 'RMSE', 'R2']}
    
    box_data = [cv_scores_by_metric['RMSLE'], cv_scores_by_metric['RMSE']]
    axes[2, 1].boxplot(box_data, labels=['RMSLE', 'RMSE'])
    axes[2, 1].set_ylabel('Score')
    axes[2, 1].set_title('Cross-Validation Stability')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 7. Prediction Intervals (using fold predictions std)
    if len(fold_results) > 1:
        # Calculate prediction uncertainty from fold variations
        fold_test_preds = np.array([fold['test_predictions'] for fold in fold_results])
        pred_mean = np.mean(fold_test_preds, axis=0)
        pred_std = np.std(fold_test_preds, axis=0)
        
        # Plot first 100 predictions with uncertainty
        n_plot = min(100, len(pred_mean))
        x_range = range(n_plot)
        
        axes[2, 2].plot(x_range, pred_mean[:n_plot], 'b-', label='Mean Prediction')
        axes[2, 2].fill_between(x_range, 
                               (pred_mean - 1.96 * pred_std)[:n_plot],
                               (pred_mean + 1.96 * pred_std)[:n_plot],
                               alpha=0.3, label='95% Prediction Interval')
        axes[2, 2].set_xlabel('Sample Index')
        axes[2, 2].set_ylabel('Predicted Value')
        axes[2, 2].set_title('Prediction Intervals (First 100 samples)')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
    else:
        axes[2, 2].text(0.5, 0.5, 'Prediction intervals\nrequire multiple folds', 
                       ha='center', va='center', transform=axes[2, 2].transAxes)
        axes[2, 2].set_title('Prediction Intervals')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

def fitPlotAndPredict(X_train, y_train, X_test, model, datasetName, xTransform, yTransform, modelName, cvScheme, time_column=None):
    """
    Fit model using cross-validation, generate diagnostic plots, and make predictions.
    Enhanced with detailed progress reporting and GPU monitoring.
    
    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix
    y_train : array-like of shape (n_samples,)
        Training target vector
    X_test : array-like of shape (n_test_samples, n_features)
        Test feature matrix
    model : sklearn estimator
        Machine learning model to fit
    datasetName : str
        Name of the dataset
    xTransform : str
        Transformation applied to features (for logging purposes)
    yTransform : str
        Transformation applied to target (for logging purposes)
    modelName : str
        Name of the model
    cvScheme : str
        Cross-validation scheme (e.g., '5-fold CV')
    time_column : str, optional
        Name of the time/duration column in X_train for stratified cross-validation. 
        If provided, will use StratifiedKFold based on time bins instead of regular KFold.
        Example: time_column='Duration'
    
    Returns:
    --------
    results : pd.DataFrame
        Single row dataframe with evaluation metrics
    predictions : np.ndarray
        Predictions on X_test using averaged CV models
    """
    print(f"\n{'='*80}")
    print(f"🚀 STARTING MODEL TRAINING: {modelName}")
    print(f"{'='*80}")
    print(f"📊 Dataset: {datasetName}")
    print(f"🔄 Transforms: X={xTransform}, y={yTransform}")
    print(f"📈 CV Scheme: {cvScheme}")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    logger.info(f"Starting fitPlotAndPredict for {modelName} on {datasetName}")
    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"Transforms - X: {xTransform}, y: {yTransform}")
    logger.info(f"CV Scheme: {cvScheme}")
    
    # Check GPU availability with enhanced reporting
    print(f"\n🔍 Checking GPU Configuration...")
    gpu_info = check_gpu_availability()
    
    # Parse CV scheme
    if 'fold' in cvScheme.lower():
        n_folds = int(cvScheme.split('-')[0])
    else:
        n_folds = 5
        logger.warning(f"Could not parse CV scheme '{cvScheme}', using 5 folds")
    
    print(f"\n📋 Training Configuration:")
    print(f"   Cross-Validation: {n_folds}-fold")
    print(f"   Training samples: {X_train.shape[0]:,}")
    print(f"   Features: {X_train.shape[1]:,}")
    print(f"   Test samples: {X_test.shape[0]:,}")
    
    logger.info(f"Using {n_folds}-fold cross-validation")
    
    # Extract time column data before converting to numpy (if needed for stratified CV)
    time_data = None
    if time_column is not None:
        if hasattr(X_train, 'columns') and time_column in X_train.columns:
            time_data = X_train[time_column].values
            print(f"   Stratification: Using '{time_column}' column for time-based stratification")
            logger.info(f"Extracted time column '{time_column}' for stratified CV")
            logger.info(f"Time data stats - mean: {time_data.mean():.6f}, std: {time_data.std():.6f}")
        else:
            print(f"   ⚠️  Time column '{time_column}' not found, using regular KFold")
            logger.warning(f"Time column '{time_column}' not found in X_train. Using regular KFold instead.")
            time_column = None
    else:
        print(f"   Stratification: Regular KFold (no time column specified)")
    
    # Convert to numpy arrays if they are pandas objects
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    
    # Store original data for metrics calculation
    y_train_original = y_train.copy()
    
    # Set up cross-validation based on duration bins if provided
    if time_column is not None and time_data is not None:
        logger.info("Using StratifiedKFold with time bins")
        # Create duration bins using KBinsDiscretizer
        bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        duration_bins = bins.fit_transform(time_data.reshape(-1, 1)).astype(int).flatten()
        logger.info(f"Duration bins distribution: {np.bincount(duration_bins)}")
        
        # Use StratifiedKFold with duration bins
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_splits = list(kf.split(X_train, duration_bins))
    else:
        logger.info("Using regular KFold cross-validation")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_splits = list(kf.split(X_train))
    
    # Initialize storage for results
    fold_results = []
    oof_predictions = np.zeros(len(y_train))
    test_predictions_folds = []
    
    print(f"\n🔄 Starting Cross-Validation Training...")
    logger.info("Starting cross-validation...")
    
    # Track overall progress
    fold_times = []
    fold_scores = []
    
    # Perform cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        fold_start_time = time.time()
        
        print(f"\n{'─'*60}")
        print(f"📊 FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'─'*60}")
        print(f"⏰ Fold Start: {datetime.now().strftime('%H:%M:%S')}")
        
        logger.info(f"Processing fold {fold_idx + 1}/{n_folds}")
        
        # Split data for this fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        print(f"📈 Data Split:")
        print(f"   Train: {X_fold_train.shape[0]:,} samples")
        print(f"   Validation: {X_fold_val.shape[0]:,} samples")
        
        # Clone and fit model
        fold_model = clone(model)
        
        logger.info(f"Fitting model for fold {fold_idx + 1}")
        logger.info(f"X_fold_train shape: {X_fold_train.shape}, y_fold_train shape: {y_fold_train.shape}")
        logger.info(f"X_fold_train stats - mean: {X_fold_train.mean():.6f}, std: {X_fold_train.std():.6f}")
        logger.info(f"y_fold_train stats - mean: {y_fold_train.mean():.6f}, std: {y_fold_train.std():.6f}")
        
        # Handle different model types, especially XGBoost with early stopping
        model_type = type(fold_model).__name__
        print(f"🤖 Model Type: {model_type}")
        logger.info(f"Model type: {model_type}")
        
        # Monitor GPU utilization during training
        def monitor_gpu_utilization():
            if gpu_info['cuda_available']:
                try:
                    import torch
                    gpu_usage = []
                    for i in range(gpu_info['gpu_count']):
                        torch.cuda.set_device(i)
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                        gpu_usage.append(f"GPU{i}: {memory_allocated:.1f}GB/{memory_reserved:.1f}GB")
                    return " | ".join(gpu_usage)
                except:
                    return "GPU monitoring unavailable"
            return "CPU mode"
        
        print(f"💾 GPU Status: {monitor_gpu_utilization()}")
        
        training_start = time.time()
        
        if 'XGB' in model_type and hasattr(fold_model, 'early_stopping_rounds') and fold_model.early_stopping_rounds is not None:
            # XGBoost model with early stopping - provide validation set
            print(f"⚡ Training XGBoost with early stopping...")
            logger.info("XGBoost model detected with early stopping - providing validation set")
            
            # Check and log GPU usage for XGBoost
            gpu_detected = False
            if hasattr(fold_model, 'device'):
                device_info = fold_model.device
                if 'cuda' in str(device_info):
                    print(f"   🎮 GPU Device: {device_info}")
                    if gpu_info['multi_gpu_capable']:
                        print(f"   🚀 Multi-GPU Training: {gpu_info['gpu_count']} GPUs available")
                    logger.info(f"✓ XGBoost using GPU acceleration (device='{device_info}')")
                    gpu_detected = True
                else:
                    print(f"   💻 CPU Device: {device_info}")
            
            if hasattr(fold_model, 'tree_method'):
                print(f"   🌳 Tree Method: {fold_model.tree_method}")
                if fold_model.tree_method == 'gpu_hist':
                    logger.info("✓ XGBoost using GPU acceleration (tree_method='gpu_hist')")
                    gpu_detected = True
                elif fold_model.tree_method == 'hist' and hasattr(fold_model, 'device') and 'cuda' in str(fold_model.device):
                    logger.info("✓ XGBoost using GPU acceleration (tree_method='hist' + device='cuda')")
                    gpu_detected = True
                else:
                    logger.info(f"⚠️  XGBoost tree_method: {fold_model.tree_method}")
            
            if not gpu_detected and gpu_info['cuda_available']:
                print(f"   ⚠️  GPU available but not being used by XGBoost")
                logger.warning("⚠️  GPU available but XGBoost may not be using it")
                logger.warning("   Consider setting device='cuda' and tree_method='hist' for GPU acceleration")
            elif not gpu_info['cuda_available']:
                print(f"   ℹ️  No GPU available - using CPU")
                logger.info("ℹ️  No GPU available - using CPU")
            
            # Custom callback to show training progress
            try:
                from xgboost.callback import TrainingCallback
                
                class ProgressCallback(TrainingCallback):
                    def __init__(self, fold_idx, n_folds):
                        self.fold_idx = fold_idx
                        self.n_folds = n_folds
                        self.last_print_time = time.time()
                        self.print_interval = 10  # Print every 10 seconds
                    
                    def after_iteration(self, model, epoch, evals_log):
                        current_time = time.time()
                        if current_time - self.last_print_time >= self.print_interval:
                            iteration = epoch
                            # Handle different evaluation result formats
                            train_score = "N/A"
                            val_score = "N/A"
                            
                            if evals_log:
                                # Try to get training score
                                if 'validation_0' in evals_log and 'rmse' in evals_log['validation_0']:
                                    train_score = f"{evals_log['validation_0']['rmse'][-1]:8.4f}"
                                # Try to get validation score
                                if 'validation_1' in evals_log and 'rmse' in evals_log['validation_1']:
                                    val_score = f"{evals_log['validation_1']['rmse'][-1]:8.4f}"
                            
                            print(f"   📊 Iteration {iteration:4d} | Train RMSE: {train_score} | Val RMSE: {val_score} | GPU: {monitor_gpu_utilization()}")
                            self.last_print_time = current_time
                        
                        return False  # Continue training
                
                progress_callback = ProgressCallback(fold_idx, n_folds)
                callbacks_list = [progress_callback]
                use_verbose = False
                
            except (ImportError, Exception) as e:
                # Fallback for older XGBoost versions or callback issues
                print(f"   📊 Using standard XGBoost progress reporting (custom callback not available: {type(e).__name__})")
                callbacks_list = None
                use_verbose = 100  # Show progress every 100 iterations
            
            try:
                fold_model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)],
                    callbacks=callbacks_list,
                    verbose=use_verbose
                )
            except Exception as e:
                # Ultimate fallback - train without evaluation set
                print(f"   ⚠️  Callback training failed ({type(e).__name__}), using simple training...")
                fold_model.fit(X_fold_train, y_fold_train, verbose=use_verbose)
            
        elif 'CatBoost' in model_type and hasattr(fold_model, 'early_stopping_rounds') and fold_model.early_stopping_rounds is not None:
            # CatBoost model with early stopping - provide validation set
            print(f"⚡ Training CatBoost with early stopping...")
            logger.info("CatBoost model detected with early stopping - providing validation set")
            
            # Check GPU usage for CatBoost
            if hasattr(fold_model, 'task_type') and fold_model.task_type == 'GPU':
                print(f"   🎮 GPU Training: task_type='GPU'")
                if hasattr(fold_model, 'devices'):
                    print(f"   🚀 GPU Devices: {fold_model.devices}")
                logger.info("✓ CatBoost using GPU acceleration (task_type='GPU')")
            elif gpu_info['cuda_available']:
                print(f"   ⚠️  GPU available but not being used by CatBoost")
                logger.warning("⚠️  GPU available but CatBoost may not be using it")
                logger.warning("   Consider setting task_type='GPU' for GPU acceleration")
            else:
                print(f"   ℹ️  No GPU available - using CPU")
                logger.info("ℹ️  No GPU available - using CPU")
            
            fold_model.fit(
                X_fold_train, y_fold_train,
                eval_set=(X_fold_val, y_fold_val),
                verbose=100  # Print every 100 iterations
            )
        else:
            # Standard model fitting for other models
            print(f"⚡ Training {model_type}...")
            logger.info("Standard model fitting")
            fold_model.fit(X_fold_train, y_fold_train)
        
        training_time = time.time() - training_start
        print(f"⏱️  Training completed in {training_time:.1f} seconds")
        
        # Check if model learned anything meaningful
        if hasattr(fold_model, 'coef_'):
            logger.info(f"Model coefficients: {fold_model.coef_}")
            logger.info(f"Model intercept: {fold_model.intercept_}")
            logger.info(f"Coefficient magnitudes: {np.abs(fold_model.coef_).max():.6f}")
        
        # Make predictions on validation set
        print(f"🔮 Making predictions...")
        pred_start = time.time()
        y_val_pred = fold_model.predict(X_fold_val)
        pred_time = time.time() - pred_start
        
        logger.info(f"Validation predictions stats - mean: {y_val_pred.mean():.6f}, std: {y_val_pred.std():.6f}")
        logger.info(f"Validation predictions range: [{y_val_pred.min():.6f}, {y_val_pred.max():.6f}]")
        
        # Store out-of-fold predictions
        oof_predictions[val_idx] = y_val_pred
        
        # Calculate metrics for this fold
        y_val_true = y_train_original[val_idx]
        oof_metrics = calculate_metrics(y_val_true, y_val_pred)
        
        # Make predictions on test set
        test_pred = fold_model.predict(X_test)
        test_predictions_folds.append(test_pred)
        
        # Calculate train metrics
        y_train_pred = fold_model.predict(X_fold_train)
        y_train_true = y_train_original[train_idx]
        train_metrics = calculate_metrics(y_train_true, y_train_pred)
        
        # Store fold results
        fold_result = {
            'fold': fold_idx + 1,
            'model': fold_model,
            'oof_metrics': oof_metrics,
            'train_metrics': train_metrics,
            'test_predictions': test_pred,
            'train_indices': train_idx,
            'val_indices': val_idx,
            'training_time': training_time,
            'prediction_time': pred_time
        }
        fold_results.append(fold_result)
        
        fold_time = time.time() - fold_start_time
        fold_times.append(fold_time)
        fold_scores.append(oof_metrics['RMSE'])
        
        # Display fold results
        print(f"📊 Fold {fold_idx + 1} Results:")
        print(f"   🎯 Validation RMSE: {oof_metrics['RMSE']:.6f}")
        print(f"   🎯 Validation RMSLE: {oof_metrics['RMSLE']:.6f}")
        print(f"   🎯 Validation R²: {oof_metrics['R2']:.6f}")
        print(f"   🏃 Train RMSE: {train_metrics['RMSE']:.6f}")
        print(f"   ⏱️  Total Fold Time: {fold_time:.1f}s (Train: {training_time:.1f}s, Pred: {pred_time:.1f}s)")
        print(f"   💾 GPU Status: {monitor_gpu_utilization()}")
        
        # Show progress summary
        avg_time = np.mean(fold_times)
        remaining_folds = n_folds - (fold_idx + 1)
        eta = remaining_folds * avg_time
        
        if remaining_folds > 0:
            print(f"   📈 Progress: {fold_idx + 1}/{n_folds} folds completed")
            print(f"   ⏰ ETA: {eta/60:.1f} minutes remaining")
            print(f"   📊 Average RMSE so far: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
        
        logger.info(f"Fold {fold_idx + 1} - OOF RMSLE: {oof_metrics['RMSLE']:.4f}, "
                   f"RMSE: {oof_metrics['RMSE']:.4f}, R²: {oof_metrics['R2']:.4f}")
    
    total_training_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ CROSS-VALIDATION COMPLETED!")
    print(f"{'='*80}")
    print(f"⏰ Total Training Time: {total_training_time/60:.1f} minutes")
    print(f"⚡ Average Fold Time: {np.mean(fold_times):.1f} seconds")
    
    # Calculate average test predictions
    test_predictions = np.mean(test_predictions_folds, axis=0)
    
    print(f"\n🔮 Final Prediction Averaging:")
    logger.info("Final prediction averaging:")
    logger.info(f"Number of fold predictions: {len(test_predictions_folds)}")
    for i, fold_preds in enumerate(test_predictions_folds):
        logger.info(f"Fold {i+1} predictions stats - mean: {fold_preds.mean():.6f}, std: {fold_preds.std():.6f}")
    
    print(f"   📊 Averaged {len(test_predictions_folds)} fold predictions")
    print(f"   📈 Final predictions - mean: {test_predictions.mean():.6f}, std: {test_predictions.std():.6f}")
    print(f"   📉 Final predictions range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")
    
    logger.info(f"Final averaged predictions stats - mean: {test_predictions.mean():.6f}, std: {test_predictions.std():.6f}")
    logger.info(f"Final averaged predictions range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")
    
    # Check for constant predictions (potential issue)
    prediction_std = test_predictions.std()
    if prediction_std < 1e-10:
        print(f"⚠️  WARNING: ALL PREDICTIONS ARE IDENTICAL!")
        print(f"   This usually indicates a serious issue with the model or data.")
        logger.warning("⚠️  ALL PREDICTIONS ARE IDENTICAL!")
        logger.warning("This usually indicates one of the following issues:")
        logger.warning("1. Features have no relationship with the target variable")
        logger.warning("2. All features are constant or nearly constant")
        logger.warning("3. Data preprocessing removed all signal")
        logger.warning("4. Model is predicting the mean of the target")
        logger.warning("Suggestion: Check your data with validate_input_data() function")
    elif prediction_std < 0.01:
        print(f"⚠️  WARNING: Very low prediction variance (std={prediction_std:.6f})")
        print(f"   Model may not be learning meaningful patterns from the data")
        logger.warning(f"⚠️  Very low prediction variance (std={prediction_std:.6f})")
        logger.warning("Model may not be learning meaningful patterns from the data")
    
    # Calculate overall CV metrics
    cv_metrics = calculate_metrics(y_train_original, oof_predictions)
    
    # Calculate average train metrics across folds
    avg_train_metrics = {}
    for metric in ['RMSLE', 'RMSE', 'MSE', 'MAE', 'R2']:
        avg_train_metrics[metric] = np.mean([fold['train_metrics'][metric] for fold in fold_results])
    
    print(f"\n📊 FINAL CROSS-VALIDATION METRICS:")
    print(f"   🎯 CV RMSE: {cv_metrics['RMSE']:.6f}")
    print(f"   🎯 CV RMSLE: {cv_metrics['RMSLE']:.6f}")
    print(f"   🎯 CV R²: {cv_metrics['R2']:.6f}")
    print(f"   🎯 CV MAE: {cv_metrics['MAE']:.6f}")
    print(f"   🏃 Avg Train RMSE: {avg_train_metrics['RMSE']:.6f}")
    print(f"   📊 RMSE Std Dev: {np.std(fold_scores):.6f}")
    
    logger.info("Cross-validation completed!")
    logger.info(f"Average CV metrics - RMSLE: {cv_metrics['RMSLE']:.4f}, "
               f"RMSE: {cv_metrics['RMSE']:.4f}, R²: {cv_metrics['R2']:.4f}")
    
    # Create diagnostic plots
    print(f"\n🎨 Creating diagnostic plots...")
    logger.info("Creating diagnostic plots...")
    fig = create_diagnostic_plots(fold_results, oof_predictions, y_train_original, 
                                X_test, test_predictions, modelName, datasetName)
    plt.show()
    
    # Create results dataframe
    results = pd.DataFrame({
        'Dataset': [datasetName],
        'xTransform': [xTransform],
        'yTransform': [yTransform],
        'modelName': [modelName],
        'cvScheme': [cvScheme],
        'Train_RMSLE': [avg_train_metrics['RMSLE']],
        'Train_RMSE': [avg_train_metrics['RMSE']],
        'Train_MSE': [avg_train_metrics['MSE']],
        'Train_MAE': [avg_train_metrics['MAE']],
        'Train_R2': [avg_train_metrics['R2']],
        'CV_RMSLE': [cv_metrics['RMSLE']],
        'CV_RMSE': [cv_metrics['RMSE']],
        'CV_MSE': [cv_metrics['MSE']],
        'CV_MAE': [cv_metrics['MAE']],
        'CV_R2': [cv_metrics['R2']],
        'Total_Time_Minutes': [total_training_time/60],
        'Avg_Fold_Time_Seconds': [np.mean(fold_times)]
    })
    
    print(f"\n📋 RESULTS SUMMARY:")
    print(results[['modelName', 'CV_RMSE', 'CV_RMSLE', 'CV_R2', 'Total_Time_Minutes']].to_string(index=False))
    
    logger.info("Results summary:")
    logger.info(f"\n{results.to_string(index=False)}")
    
    print(f"\n🎉 Training completed successfully!")
    print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    return results, test_predictions

# Example usage
"""
# Simple example usage:

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Example parameters
model = LinearRegression(fit_intercept=True)
datasetName = 'Sample_Dataset'
xTransform = 'none'
yTransform = 'none'
modelName = 'LinearRegression_Example'
cvScheme = '5-fold CV'

# Generate sample data with Duration column
np.random.seed(42)
n_samples, n_features = 1000, 5
feature_data = np.random.randn(n_samples, n_features)
duration_data = np.random.exponential(scale=2.0, size=n_samples) + 1.0  # Simulated duration data

# Create DataFrame with Duration column
X_train = pd.DataFrame(feature_data, columns=[f'Feature_{i}' for i in range(n_features)])
X_train['Duration'] = duration_data

# Create target with some relationship to features
y_train = 2*X_train['Feature_0'] + 1.5*X_train['Feature_1'] + np.random.randn(n_samples)*0.1 + 3

# Create test data (without Duration column for prediction)
X_test = pd.DataFrame(np.random.randn(200, n_features), columns=[f'Feature_{i}' for i in range(n_features)])

# Option 1: Regular cross-validation (without time column)
results, predictions = fitPlotAndPredict(
    X_train.drop('Duration', axis=1), y_train, X_test, model, datasetName, 
    xTransform, yTransform, modelName, cvScheme
)

# Option 2: Stratified cross-validation with Duration column
results_stratified, predictions_stratified = fitPlotAndPredict(
    X_train, y_train, X_test, model, datasetName, 
    xTransform, yTransform, modelName, cvScheme, 
    time_column='Duration'
)

print("Regular CV Results:")
print(results)
print(f"Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
print(f"Sample predictions: {predictions[:5]}")

print("\nStratified CV Results:")
print(results_stratified)
print(f"Predictions range: [{predictions_stratified.min():.6f}, {predictions_stratified.max():.6f}]")
print(f"Sample predictions: {predictions_stratified[:5]}")
"""

def validate_input_data(X_train, y_train, X_test):
    """
    Validate input data to identify potential issues.
    """
    print("="*60)
    print("VALIDATING INPUT DATA")
    print("="*60)
    
    # Check data shapes and types
    print(f"X_train shape: {X_train.shape}, type: {type(X_train)}")
    print(f"y_train shape: {y_train.shape}, type: {type(y_train)}")
    print(f"X_test shape: {X_test.shape}, type: {type(X_test)}")
    
    # Convert to numpy if needed
    if hasattr(X_train, 'values'):
        X_train_np = X_train.values
    else:
        X_train_np = np.array(X_train)
    
    if hasattr(y_train, 'values'):
        y_train_np = y_train.values
    else:
        y_train_np = np.array(y_train)
    
    # Check for data issues
    print(f"\nData statistics:")
    print(f"X_train - mean: {X_train_np.mean():.6f}, std: {X_train_np.std():.6f}")
    print(f"X_train - min: {X_train_np.min():.6f}, max: {X_train_np.max():.6f}")
    print(f"y_train - mean: {y_train_np.mean():.6f}, std: {y_train_np.std():.6f}")
    print(f"y_train - min: {y_train_np.min():.6f}, max: {y_train_np.max():.6f}")
    
    # Check for constant features
    feature_stds = X_train_np.std(axis=0)
    constant_features = np.sum(feature_stds < 1e-10)
    print(f"\nFeature analysis:")
    print(f"Feature standard deviations: {feature_stds}")
    print(f"Number of constant features: {constant_features}")
    
    # Check for NaN or infinite values
    print(f"\nData quality:")
    print(f"X_train NaN count: {np.isnan(X_train_np).sum()}")
    print(f"y_train NaN count: {np.isnan(y_train_np).sum()}")
    print(f"X_train infinite count: {np.isinf(X_train_np).sum()}")
    print(f"y_train infinite count: {np.isinf(y_train_np).sum()}")
    
    # Simple correlation check
    if X_train_np.shape[1] <= 10:  # Only for small number of features
        correlations = []
        for i in range(X_train_np.shape[1]):
            corr = np.corrcoef(X_train_np[:, i], y_train_np)[0, 1]
            correlations.append(corr)
        print(f"\nFeature-target correlations: {correlations}")
        max_corr = np.max(np.abs(correlations))
        print(f"Maximum absolute correlation: {max_corr:.6f}")
        
        if max_corr < 0.01:
            print("⚠️  WARNING: Very low correlations between features and target!")
            print("   This might explain why all predictions are similar.")
    
    print("="*60)

def create_gpu_optimized_model(model_type='xgboost', **kwargs):
    """
    Create GPU-optimized models with proper multi-GPU configuration.
    
    Parameters:
    -----------
    model_type : str
        Type of model to create ('xgboost' or 'catboost')
    **kwargs : dict
        Additional parameters for the model
    
    Returns:
    --------
    model : sklearn estimator
        Configured model with GPU optimization if available
    """
    gpu_info = check_gpu_availability()
    
    if model_type.lower() == 'xgboost':
        try:
            from xgboost import XGBRegressor
            
            # Enhanced XGBoost parameters optimized for multi-GPU
            default_params = {
                'max_depth': 9,
                'colsample_bytree': 0.7,
                'subsample': 0.9,
                'n_estimators': 3000,
                'learning_rate': 0.009,
                'gamma': 0.01,
                'max_delta_step': 2,
                'eval_metric': 'rmse',
                'enable_categorical': False,
                'random_state': 42,
                'early_stopping_rounds': 100,
                'tree_method': 'hist',
                'device': 'cpu'  # Default to CPU, will be overridden if GPU available
            }
            
            # Configure GPU settings if available
            if gpu_info['cuda_available']:
                if gpu_info['multi_gpu_capable']:
                    # Multi-GPU configuration
                    print(f"\n⚡ Configuring XGBoost for Multi-GPU Training:")
                    print(f"   Available GPUs: {gpu_info['gpu_count']}")
                    print(f"   Total GPU Memory: {gpu_info['total_memory_gb']:.1f} GB")
                    
                    # For XGBoost 2.0+, use device='cuda' for multi-GPU
                    # XGBoost automatically utilizes all available GPUs
                    default_params.update({
                        'device': 'cuda',  # XGBoost 2.0+ automatically uses all GPUs
                        'tree_method': 'hist',  # GPU histogram method
                        'max_bin': 256,  # Optimize for GPU memory
                        'grow_policy': 'lossguide',  # Better for GPU
                        'sampling_method': 'gradient_based'  # GPU-optimized sampling
                    })
                    
                    # Adjust parameters for multi-GPU efficiency
                    if gpu_info['gpu_count'] >= 2:
                        # Increase batch size effectively by allowing more complex models
                        default_params['n_estimators'] = min(default_params['n_estimators'] * 1.2, 4000)
                        default_params['max_depth'] = min(default_params['max_depth'] + 1, 12)
                        # Slightly reduce learning rate for stability with more complex models
                        default_params['learning_rate'] = default_params['learning_rate'] * 0.95
                    
                    print(f"   Configuration: device='cuda' (auto multi-GPU)")
                    print(f"   Tree Method: {default_params['tree_method']}")
                    print(f"   Max Depth: {default_params['max_depth']}")
                    print(f"   N Estimators: {default_params['n_estimators']}")
                    
                else:
                    # Single GPU configuration
                    print(f"\n⚡ Configuring XGBoost for Single GPU Training:")
                    print(f"   GPU: {gpu_info['gpu_names'][0]}")
                    print(f"   Memory: {gpu_info['gpu_memory'][0]['total_gb']:.1f} GB")
                    
                    default_params.update({
                        'device': 'cuda:0',  # Specific GPU device
                        'tree_method': 'hist',
                        'max_bin': 256,
                        'grow_policy': 'lossguide'
                    })
                    
                    print(f"   Configuration: device='cuda:0'")
                    print(f"   Tree Method: {default_params['tree_method']}")
            else:
                print(f"\n💻 Configuring XGBoost for CPU Training:")
                print(f"   No GPU available, using optimized CPU settings")
                default_params.update({
                    'device': 'cpu',
                    'tree_method': 'hist',  # CPU histogram method
                    'n_jobs': -1  # Use all CPU cores
                })
            
            # Update with user parameters (user parameters override defaults)
            default_params.update(kwargs)
            
            model = XGBRegressor(**default_params)
            
            if gpu_info['cuda_available']:
                if gpu_info['multi_gpu_capable']:
                    logger.info(f"✓ Created XGBoost model with multi-GPU acceleration ({gpu_info['gpu_count']} GPUs)")
                    print(f"   ✅ Multi-GPU XGBoost model created successfully")
                else:
                    logger.info("✓ Created XGBoost model with single GPU acceleration")
                    print(f"   ✅ Single GPU XGBoost model created successfully")
            else:
                logger.info("ℹ️  Created XGBoost model for CPU (no GPU available)")
                print(f"   ✅ CPU XGBoost model created successfully")
                
            return model
            
        except ImportError:
            logger.error("XGBoost not available. Please install with: pip install xgboost")
            print("❌ XGBoost not available. Please install with: pip install xgboost")
            return None
    
    elif model_type.lower() == 'catboost':
        try:
            from catboost import CatBoostRegressor
            
            # Enhanced CatBoost parameters optimized for GPU
            default_params = {
                'iterations': 3500,
                'learning_rate': 0.02,
                'depth': 12,
                'loss_function': 'RMSE',
                'l2_leaf_reg': 3,
                'random_seed': 42,
                'eval_metric': 'RMSE',
                'early_stopping_rounds': 200,
                'verbose': 0,
                'task_type': 'CPU'  # Default to CPU
            }
            
            # Configure GPU settings if available
            if gpu_info['cuda_available']:
                print(f"\n⚡ Configuring CatBoost for GPU Training:")
                print(f"   Available GPUs: {gpu_info['gpu_count']}")
                
                default_params.update({
                    'task_type': 'GPU',
                    'devices': '0' if gpu_info['gpu_count'] == 1 else f"0:{gpu_info['gpu_count']-1}",
                    'gpu_ram_part': 0.8  # Use 80% of GPU memory
                })
                
                if gpu_info['multi_gpu_capable']:
                    print(f"   Configuration: Multi-GPU (devices 0-{gpu_info['gpu_count']-1})")
                    # Adjust parameters for multi-GPU
                    default_params['iterations'] = min(default_params['iterations'] * 1.2, 5000)
                else:
                    print(f"   Configuration: Single GPU (device 0)")
                
                print(f"   Task Type: {default_params['task_type']}")
                print(f"   Devices: {default_params['devices']}")
            else:
                print(f"\n💻 Configuring CatBoost for CPU Training:")
                default_params.update({
                    'task_type': 'CPU',
                    'thread_count': -1  # Use all CPU cores
                })
            
            # Update with user parameters
            default_params.update(kwargs)
            
            model = CatBoostRegressor(**default_params)
            
            if gpu_info['cuda_available']:
                logger.info("✓ Created CatBoost model with GPU acceleration")
                print(f"   ✅ GPU CatBoost model created successfully")
            else:
                logger.info("ℹ️  Created CatBoost model for CPU (no GPU available)")
                print(f"   ✅ CPU CatBoost model created successfully")
                
            return model
            
        except ImportError:
            logger.error("CatBoost not available. Please install with: pip install catboost")
            print("❌ CatBoost not available. Please install with: pip install catboost")
            return None
    
    else:
        logger.error(f"Unsupported model type: {model_type}. Supported types: 'xgboost', 'catboost'")
        print(f"❌ Unsupported model type: {model_type}. Supported types: 'xgboost', 'catboost'")
        return None