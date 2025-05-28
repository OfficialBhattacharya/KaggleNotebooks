import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

def calculate_metrics(y_true, y_pred, task_type='auto'):
    """
    Calculate comprehensive evaluation metrics for both regression and classification.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    task_type : str, optional
        Type of task: 'regression', 'classification', or 'auto' to detect automatically
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all metrics appropriate for the task
    """
    # Auto-detect task type if not specified
    if task_type == 'auto':
        # Check if y_true contains only 0s and 1s (binary classification)
        unique_values = np.unique(y_true)
        if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
            task_type = 'classification'
        else:
            task_type = 'regression'
    
    if task_type == 'regression':
        metrics = {
            'RMSLE': calculate_rmsle(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MSE': mean_squared_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    elif task_type == 'classification':
        # For classification, we need to handle probability predictions vs class predictions
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
            
            # Convert predictions to binary if they are probabilities
            if np.all((y_pred >= 0) & (y_pred <= 1)) and not np.all(np.isin(y_pred, [0, 1])):
                # Predictions are probabilities
                y_pred_binary = (y_pred > 0.5).astype(int)
                
                metrics = {
                    'Accuracy': accuracy_score(y_true, y_pred_binary),
                    'Precision': precision_score(y_true, y_pred_binary, zero_division=0),
                    'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
                    'F1': f1_score(y_true, y_pred_binary, zero_division=0),
                    'AUC_ROC': roc_auc_score(y_true, y_pred),
                    'LogLoss': log_loss(y_true, y_pred)
                }
            else:
                # Predictions are already binary classes
                y_pred_binary = y_pred.astype(int)
                
                metrics = {
                    'Accuracy': accuracy_score(y_true, y_pred_binary),
                    'Precision': precision_score(y_true, y_pred_binary, zero_division=0),
                    'Recall': recall_score(y_true, y_pred_binary, zero_division=0),
                    'F1': f1_score(y_true, y_pred_binary, zero_division=0),
                    'AUC_ROC': None,  # Cannot calculate AUC without probabilities
                    'LogLoss': None   # Cannot calculate LogLoss without probabilities
                }
                
        except ImportError:
            # Fallback if sklearn classification metrics are not available
            logger.warning("Classification metrics not available, using basic accuracy")
            y_pred_binary = (y_pred > 0.5).astype(int) if np.all((y_pred >= 0) & (y_pred <= 1)) else y_pred.astype(int)
            accuracy = np.mean(y_true == y_pred_binary)
            metrics = {
                'Accuracy': accuracy,
                'Precision': None,
                'Recall': None,
                'F1': None,
                'AUC_ROC': None,
                'LogLoss': None
            }
    else:
        raise ValueError(f"Unsupported task_type: {task_type}. Use 'regression' or 'classification'.")
    
    return metrics

def create_diagnostic_plots(fold_results, cv_predictions, y_train_original, X_test, test_predictions, 
                          model_name, dataset_name, feature_names=None):
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
    feature_names : list, optional
        List of feature names. If None, will use generic names like 'Feature_0', 'Feature_1', etc.
            
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
    
    # Determine which metrics to plot based on what's available
    available_metrics = list(metrics_df.columns)
    
    # Choose metrics to plot based on task type
    if 'RMSE' in available_metrics:
        # Regression metrics
        metrics_to_plot = ['RMSLE', 'RMSE', 'R2'] if all(m in available_metrics for m in ['RMSLE', 'RMSE', 'R2']) else available_metrics[:3]
    else:
        # Classification metrics
        preferred_clf_metrics = ['Accuracy', 'F1', 'AUC_ROC']
        metrics_to_plot = [m for m in preferred_clf_metrics if m in available_metrics]
        if len(metrics_to_plot) < 3:
            metrics_to_plot.extend([m for m in available_metrics if m not in metrics_to_plot][:3-len(metrics_to_plot)])
    
    for i, metric in enumerate(metrics_to_plot[:3]):  # Only plot first 3 metrics
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
        
        # Use provided feature names or create generic ones
        if feature_names is not None and len(feature_names) == len(avg_importance):
            feature_name_list = feature_names
        else:
            feature_name_list = [f'Feature_{i}' for i in range(len(avg_importance))]
            if feature_names is not None:
                logger.warning(f"Feature names length ({len(feature_names)}) doesn't match importance length ({len(avg_importance)}). Using generic names.")
        
        # Get top 10 features
        top_indices = np.argsort(avg_importance)[-10:]
        
        axes[2, 0].barh(range(len(top_indices)), avg_importance[top_indices], 
                       xerr=std_importance[top_indices], alpha=0.7)
        axes[2, 0].set_yticks(range(len(top_indices)))
        axes[2, 0].set_yticklabels([feature_name_list[i] for i in top_indices])
        axes[2, 0].set_xlabel('Feature Importance')
        axes[2, 0].set_title('Top 10 Feature Importance (Avg ± Std)')
    else:
        axes[2, 0].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                       ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Feature Importance')
    
    # 6. Cross-Validation Stability
    # Use the same metrics that were plotted above for consistency
    if len(metrics_to_plot) >= 2:
        cv_scores_by_metric = {metric: [fold['oof_metrics'][metric] for fold in fold_results] 
                              for metric in metrics_to_plot[:2]}  # Use first 2 metrics
        
        box_data = [cv_scores_by_metric[metric] for metric in list(cv_scores_by_metric.keys())]
        axes[2, 1].boxplot(box_data, labels=list(cv_scores_by_metric.keys()))
        axes[2, 1].set_ylabel('Score')
        axes[2, 1].set_title('Cross-Validation Stability')
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, 'Insufficient metrics\nfor stability analysis', 
                       ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('Cross-Validation Stability')
    
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

def fitPlotAndPredict(X_train, y_train, X_test, model, datasetName, xTransform, yTransform, modelName, cvScheme, time_column=None, X_train_full=None, y_train_full=None):
    """
    Fit model using cross-validation, generate diagnostic plots, and make predictions.
    Enhanced with detailed progress reporting and GPU monitoring.
    
    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix for cross-validation (can be a subset)
    y_train : array-like of shape (n_samples,)
        Training target vector for cross-validation (can be a subset)
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
    X_train_full : array-like of shape (n_full_samples, n_features), optional
        Full training feature matrix for final predictions. If None, uses X_train.
    y_train_full : array-like of shape (n_full_samples,), optional
        Full training target vector for final predictions. If None, uses y_train.
    
    Returns:
    --------
    results : pd.DataFrame
        Single row dataframe with evaluation metrics
    predictions : np.ndarray
        Predictions on X_test using final model trained on CV subset
    train_predictions : pd.DataFrame
        DataFrame with 'actual' and 'predicted' columns containing predictions on full training set
    """
    print(f"\n{'='*80}")
    print(f"🚀 STARTING MODEL TRAINING: {modelName}")
    print(f"{'='*80}")
    print(f"📊 Dataset: {datasetName}")
    print(f"🔄 Transforms: X={xTransform}, y={yTransform}")
    print(f"📈 CV Scheme: {cvScheme}")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Determine if we're using separate datasets for CV and final predictions
    use_full_dataset = X_train_full is not None and y_train_full is not None
    
    if use_full_dataset:
        print(f"📊 Two-stage prediction:")
        print(f"   CV Dataset: {X_train.shape[0]:,} samples")
        print(f"   Full Dataset for final predictions: {X_train_full.shape[0]:,} samples")
        logger.info(f"Two-stage prediction - CV on {X_train.shape[0]} samples, final predictions on {X_train_full.shape[0]} samples")
    else:
        print(f"📊 Single dataset training: {X_train.shape[0]:,} samples")
        logger.info(f"Single dataset training on {X_train.shape[0]} samples")
    
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
    print(f"   CV Training samples: {X_train.shape[0]:,}")
    print(f"   Features: {X_train.shape[1]:,}")
    print(f"   Test samples: {X_test.shape[0]:,}")
    if use_full_dataset:
        print(f"   Full Dataset samples: {X_train_full.shape[0]:,}")
    
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
    
    # Extract feature names before converting to numpy arrays
    feature_names = None
    if hasattr(X_train, 'columns'):
        feature_names = list(X_train.columns)
        print(f"   📋 Extracted {len(feature_names)} feature names for diagnostic plots")
        logger.info(f"Extracted feature names: {feature_names[:10]}{'...' if len(feature_names) > 10 else ''}")
    else:
        print(f"   📋 No column names available - will use generic feature names in plots")
        logger.info("No column names available in X_train - using generic feature names")
    
    # Convert to numpy arrays if they are pandas objects
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    
    # Handle full dataset conversion
    if use_full_dataset:
        if hasattr(X_train_full, 'values'):
            X_train_full = X_train_full.values
        if hasattr(y_train_full, 'values'):
            y_train_full = y_train_full.values
    
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
            """Monitor GPU utilization and memory usage"""
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                
                gpu_status = []
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    used_gb = mem_info.used / 1024**3
                    total_gb = mem_info.total / 1024**3
                    
                    # Get utilization
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_util = util.gpu
                    except:
                        gpu_util = 0
                    
                    gpu_status.append(f"GPU{i}: {used_gb:.1f}GB/{total_gb:.1f}GB ({gpu_util}%)")
                
                return " | ".join(gpu_status)
                
            except Exception as e:
                # Fallback for when pynvml is not available
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_count = torch.cuda.device_count()
                        gpu_status = []
                        for i in range(gpu_count):
                            allocated = torch.cuda.memory_allocated(i) / 1024**3
                            cached = torch.cuda.memory_reserved(i) / 1024**3
                            gpu_status.append(f"GPU{i}: {allocated:.1f}GB/{cached:.1f}GB")
                        return " | ".join(gpu_status)
                    else:
                        return "No GPU"
                except:
                    return "GPU status unavailable"
        
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
                    def __init__(self, fold_idx, n_folds, eval_metric='rmse'):
                        self.fold_idx = fold_idx
                        self.n_folds = n_folds
                        self.eval_metric = eval_metric
                        self.last_print_time = time.time()
                        self.print_interval = 10  # Print every 10 seconds
                        self.best_score = float('inf') if 'rmse' in eval_metric or 'mae' in eval_metric or 'logloss' in eval_metric else float('-inf')
                        self.rounds_without_improvement = 0
                        self.max_rounds_without_improvement = 2000  # Allow up to 2000 rounds without improvement
                        self.max_iterations = 2000  # Maximum total iterations
                        self.is_higher_better = 'auc' in eval_metric or 'accuracy' in eval_metric  # Metrics where higher is better
                    
                    def after_iteration(self, model, epoch, evals_log):
                        current_time = time.time()
                        iteration = epoch
                        
                        # Check for improvement - dynamically find the metric
                        current_score = None
                        metric_found = False
                        
                        if evals_log and 'validation_1' in evals_log:
                            # Try to find the evaluation metric in the logs
                            for metric_name in evals_log['validation_1'].keys():
                                if metric_name in self.eval_metric or self.eval_metric in metric_name:
                                    current_score = evals_log['validation_1'][metric_name][-1]
                                    metric_found = True
                                    break
                            
                            # If exact match not found, use the first available metric
                            if not metric_found and evals_log['validation_1']:
                                metric_name = list(evals_log['validation_1'].keys())[0]
                                current_score = evals_log['validation_1'][metric_name][-1]
                                metric_found = True
                        
                        # Update best score and improvement tracking
                        if current_score is not None:
                            if self.is_higher_better:
                                if current_score > self.best_score:
                                    self.best_score = current_score
                                    self.rounds_without_improvement = 0
                                else:
                                    self.rounds_without_improvement += 1
                            else:
                                if current_score < self.best_score:
                                    self.best_score = current_score
                                    self.rounds_without_improvement = 0
                                else:
                                    self.rounds_without_improvement += 1
                        
                        # Print progress every 10 seconds
                        if current_time - self.last_print_time >= self.print_interval:
                            # Handle different evaluation result formats
                            train_score = "N/A"
                            val_score = "N/A"
                            
                            if evals_log:
                                # Try to get training score
                                if 'validation_0' in evals_log and evals_log['validation_0']:
                                    first_metric = list(evals_log['validation_0'].keys())[0]
                                    train_score = f"{evals_log['validation_0'][first_metric][-1]:8.4f}"
                                
                                # Try to get validation score
                                if 'validation_1' in evals_log and evals_log['validation_1']:
                                    first_metric = list(evals_log['validation_1'].keys())[0]
                                    val_score = f"{evals_log['validation_1'][first_metric][-1]:8.4f}"
                                    metric_display_name = first_metric.upper()
                                else:
                                    metric_display_name = self.eval_metric.upper()
                            else:
                                metric_display_name = self.eval_metric.upper()
                            
                            print(f"   📊 Iteration {iteration:4d} | Train {metric_display_name}: {train_score} | Val {metric_display_name}: {val_score} | GPU: {monitor_gpu_utilization()}")
                            self.last_print_time = current_time
                        
                        # Custom early stopping logic: stop only if we've reached max iterations
                        # OR if we've gone 2000 rounds without improvement (which is very generous)
                        if iteration >= self.max_iterations:
                            print(f"   🛑 Stopping at maximum iterations: {self.max_iterations}")
                            return True  # Stop training
                        elif self.rounds_without_improvement >= self.max_rounds_without_improvement:
                            print(f"   🛑 Stopping after {self.max_rounds_without_improvement} rounds without improvement")
                            return True  # Stop training
                        
                        return False  # Continue training
                
                progress_callback = ProgressCallback(fold_idx, n_folds, fold_model.eval_metric)
                callbacks_list = [progress_callback]
                use_verbose = False
                
            except (ImportError, Exception) as e:
                # Fallback for older XGBoost versions or callback issues
                print(f"   📊 Using standard XGBoost progress reporting (custom callback not available: {type(e).__name__})")
                callbacks_list = None
                use_verbose = 100  # Show progress every 100 iterations
            
            try:
                # Temporarily disable built-in early stopping since we're using custom logic
                original_early_stopping = fold_model.early_stopping_rounds
                fold_model.early_stopping_rounds = None
                
                fold_model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)],
                    callbacks=callbacks_list,
                    verbose=use_verbose
                )
                
                # Restore original early stopping setting
                fold_model.early_stopping_rounds = original_early_stopping
                
            except Exception as e:
                # Ultimate fallback - train without evaluation set
                print(f"   ⚠️  Callback training failed ({type(e).__name__}), using simple training...")
                fold_model.fit(X_fold_train, y_fold_train, verbose=use_verbose)
            
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
    
    cv_training_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ CROSS-VALIDATION COMPLETED!")
    print(f"{'='*80}")
    print(f"⏰ CV Training Time: {cv_training_time/60:.1f} minutes")
    print(f"⚡ Average Fold Time: {np.mean(fold_times):.1f} seconds")
    
    # Calculate overall CV metrics using out-of-fold predictions
    cv_metrics = calculate_metrics(y_train_original, oof_predictions)
    
    # Calculate average train metrics across folds
    avg_train_metrics = {}
    for metric in ['RMSLE', 'RMSE', 'MSE', 'MAE', 'R2']:
        avg_train_metrics[metric] = np.mean([fold['train_metrics'][metric] for fold in fold_results])
    
    print(f"\n📊 CROSS-VALIDATION METRICS:")
    print(f"   🎯 CV RMSE: {cv_metrics['RMSE']:.6f}")
    print(f"   🎯 CV RMSLE: {cv_metrics['RMSLE']:.6f}")
    print(f"   🎯 CV R²: {cv_metrics['R2']:.6f}")
    print(f"   🎯 CV MAE: {cv_metrics['MAE']:.6f}")
    print(f"   🏃 Avg Train RMSE: {avg_train_metrics['RMSE']:.6f}")
    print(f"   📊 RMSE Std Dev: {np.std(fold_scores):.6f}")
    
    logger.info("Cross-validation completed!")
    logger.info(f"Average CV metrics - RMSLE: {cv_metrics['RMSLE']:.4f}, "
               f"RMSE: {cv_metrics['RMSE']:.4f}, R²: {cv_metrics['R2']:.4f}")
    
    # ========================================================================
    # TRAIN FINAL MODEL ON CV SUBSET AND PREDICT ON FULL DATASET
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"🎯 TRAINING FINAL MODEL ON CV SUBSET")
    print(f"{'='*80}")
    
    final_model_start = time.time()
    
    # Clone the original model for final training
    final_model = clone(model)
    model_type = type(final_model).__name__
    
    print(f"🤖 Final Model Type: {model_type}")
    print(f"📊 Training on CV subset: {X_train.shape[0]:,} samples with {X_train.shape[1]:,} features")
    if use_full_dataset:
        print(f"   🎯 Will predict on full dataset: {X_train_full.shape[0]:,} samples")
    print(f"💾 GPU Status: {monitor_gpu_utilization()}")
    
    logger.info("Training final model on CV subset")
    logger.info(f"Final model type: {model_type}")
    logger.info(f"CV subset shape: {X_train.shape}")
    if use_full_dataset:
        logger.info(f"Full dataset size for predictions: {X_train_full.shape[0]}")
    
    # Handle XGBoost with early stopping for final model
    if 'XGB' in model_type and hasattr(final_model, 'early_stopping_rounds') and final_model.early_stopping_rounds is not None:
        print(f"⚡ Training final XGBoost model...")
        logger.info("Final XGBoost model training with early stopping disabled for full training")
        
        # For final model, we disable early stopping to use all CV data
        final_model_params = final_model.get_params()
        final_model_params['early_stopping_rounds'] = None
        final_model.set_params(**final_model_params)
        
        print(f"   ℹ️  Early stopping disabled for final model training")
        logger.info("Early stopping disabled for final model to use entire CV subset")
        
        # Check GPU configuration
        if hasattr(final_model, 'device') and 'cuda' in str(final_model.device):
            print(f"   🎮 GPU Device: {final_model.device}")
            if gpu_info['multi_gpu_capable']:
                print(f"   🚀 Multi-GPU Training: {gpu_info['gpu_count']} GPUs available")
        
        if hasattr(final_model, 'tree_method'):
            print(f"   🌳 Tree Method: {final_model.tree_method}")
        
        # Train final model on CV subset
        final_model.fit(X_train, y_train, verbose=100)
        
    else:
        # Standard model fitting for other models
        print(f"⚡ Training final {model_type} model...")
        logger.info("Final standard model training")
        final_model.fit(X_train, y_train)
    
    final_training_time = time.time() - final_model_start
    total_time = time.time() - start_time
    
    print(f"⏱️  Final model training completed in {final_training_time:.1f} seconds")
    print(f"⏰ Total time (CV + Final): {total_time/60:.1f} minutes")
    
    # Make predictions using final model
    print(f"\n🔮 Making final predictions...")
    
    # Determine which dataset to use for final train predictions
    if use_full_dataset:
        X_final = X_train_full
        y_final = y_train_full
        print(f"   📈 Using final model (trained on CV subset) to predict on full dataset")
        logger.info("Using final model trained on CV subset to predict on full dataset")
    else:
        X_final = X_train
        y_final = y_train
        print(f"   📈 Using final model to predict on CV subset")
        logger.info("Using final model to predict on CV subset")
    
    # Predictions on final dataset (full or CV subset)
    final_train_predictions = final_model.predict(X_final)
    final_train_metrics = calculate_metrics(y_final, final_train_predictions)
    
    # Predictions on test set
    final_test_predictions = final_model.predict(X_test)
    
    logger.info(f"Final train predictions stats - mean: {final_train_predictions.mean():.6f}, std: {final_train_predictions.std():.6f}")
    logger.info(f"Final test predictions stats - mean: {final_test_predictions.mean():.6f}, std: {final_test_predictions.std():.6f}")
    
    print(f"   📈 Final train predictions - mean: {final_train_predictions.mean():.6f}, std: {final_train_predictions.std():.6f}")
    print(f"   📈 Final test predictions - mean: {final_test_predictions.mean():.6f}, std: {final_test_predictions.std():.6f}")
    
    # Check for constant predictions (potential issue)
    prediction_std = final_test_predictions.std()
    if prediction_std < 1e-10:
        print(f"⚠️  WARNING: ALL FINAL PREDICTIONS ARE IDENTICAL!")
        print(f"   This usually indicates a serious issue with the model or data.")
        logger.warning("⚠️  ALL FINAL PREDICTIONS ARE IDENTICAL!")
        logger.warning("This usually indicates one of the following issues:")
        logger.warning("1. Features have no relationship with the target variable")
        logger.warning("2. All features are constant or nearly constant")
        logger.warning("3. Data preprocessing removed all signal")
        logger.warning("4. Model is predicting the mean of the target")
        logger.warning("Suggestion: Check your data with validate_input_data() function")
    elif prediction_std < 0.01:
        print(f"⚠️  WARNING: Very low final prediction variance (std={prediction_std:.6f})")
        print(f"   Model may not be learning meaningful patterns from the data")
        logger.warning(f"⚠️  Very low final prediction variance (std={prediction_std:.6f})")
        logger.warning("Model may not be learning meaningful patterns from the data")
    
    print(f"\n📊 FINAL MODEL METRICS (trained on CV subset, predicting on {'full dataset' if use_full_dataset else 'CV subset'}):")
    print(f"   🎯 Final Train RMSE: {final_train_metrics['RMSE']:.6f}")
    print(f"   🎯 Final Train RMSLE: {final_train_metrics['RMSLE']:.6f}")
    print(f"   🎯 Final Train R²: {final_train_metrics['R2']:.6f}")
    print(f"   🎯 Final Train MAE: {final_train_metrics['MAE']:.6f}")
    
    logger.info(f"Final model metrics - RMSLE: {final_train_metrics['RMSLE']:.4f}, "
               f"RMSE: {final_train_metrics['RMSE']:.4f}, R²: {final_train_metrics['R2']:.4f}")
    
    # Create diagnostic plots using out-of-fold predictions for CV analysis
    print(f"\n🎨 Creating diagnostic plots...")
    logger.info("Creating diagnostic plots...")
    fig = create_diagnostic_plots(fold_results, oof_predictions, y_train_original, 
                                X_test, final_test_predictions, modelName, datasetName, feature_names)
    plt.show()
    
    # Create train predictions dataframe using final model predictions
    print(f"\n📊 Creating train predictions dataframe (final model predictions on {'full dataset' if use_full_dataset else 'CV subset'})...")
    train_predictions_df = pd.DataFrame({
        'actual': y_final,
        'predicted': final_train_predictions
    })
    
    logger.info(f"Train predictions dataframe created with shape: {train_predictions_df.shape}")
    logger.info(f"Train predictions stats - actual mean: {train_predictions_df['actual'].mean():.6f}, predicted mean: {train_predictions_df['predicted'].mean():.6f}")
    
    # Create results dataframe with dynamic metrics
    results_dict = {
        'Dataset': [datasetName],
        'xTransform': [xTransform],
        'yTransform': [yTransform],
        'modelName': [modelName],
        'cvScheme': [cvScheme],
        'Total_Time_Minutes': [total_time/60],
        'CV_Time_Minutes': [cv_training_time/60],
        'Final_Training_Time_Seconds': [final_training_time],
        'Avg_Fold_Time_Seconds': [np.mean(fold_times)],
        'CV_Samples': [X_train.shape[0]],
        'Final_Prediction_Samples': [X_final.shape[0]]
    }
    
    # Add train metrics with prefix
    for metric_name, metric_value in final_train_metrics.items():
        results_dict[f'Train_{metric_name}'] = [metric_value]
    
    # Add CV metrics with prefix
    for metric_name, metric_value in cv_metrics.items():
        results_dict[f'CV_{metric_name}'] = [metric_value]
    
    results = pd.DataFrame(results_dict)
    
    # Determine which metrics to show in summary based on task type
    summary_columns = ['modelName', 'CV_Samples', 'Final_Prediction_Samples', 'Total_Time_Minutes']
    
    # Add task-specific metrics to summary
    if 'CV_RMSE' in results.columns:
        # Regression task
        summary_columns.extend(['CV_RMSE', 'CV_RMSLE', 'CV_R2', 'Train_RMSE', 'Train_R2'])
    elif 'CV_Accuracy' in results.columns:
        # Classification task
        summary_columns.extend(['CV_Accuracy', 'CV_F1', 'Train_Accuracy'])
        if 'CV_AUC_ROC' in results.columns:
            summary_columns.append('CV_AUC_ROC')
    
    # Filter summary columns to only include those that exist
    summary_columns = [col for col in summary_columns if col in results.columns]
    
    print(f"\n📋 RESULTS SUMMARY:")
    print(results[summary_columns].to_string(index=False))
    
    logger.info("Results summary:")
    logger.info(f"\n{results.to_string(index=False)}")
    
    print(f"\n🎉 Training completed successfully!")
    print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if use_full_dataset:
        print(f"📊 Model trained on CV subset: {X_train.shape[0]:,} samples")
        print(f"📊 Final predictions made on full dataset: {X_final.shape[0]:,} samples")
        print(f"🔮 Test predictions generated using final model (trained on CV subset)")
        print(f"📈 Train predictions generated using final model (on full dataset)")
    else:
        print(f"📊 Model trained and predictions made on: {X_final.shape[0]:,} samples")
        print(f"🔮 Test predictions generated using final model")
        print(f"📈 Train predictions generated using final model")
    print(f"{'='*80}")
    
    return results, final_test_predictions, train_predictions_df


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

def create_gpu_optimized_model(model_type='xgboost', task_type='regression', monotonic_constraints=None, interaction_constraints=None, **kwargs):
    """
    Create GPU-optimized XGBoost models with proper multi-GPU configuration.
    
    Parameters:
    -----------
    model_type : str
        Type of model to create (only 'xgboost' supported)
    task_type : str
        Type of task: 'regression' or 'classification'
    monotonic_constraints : dict or list, optional
        Monotonic constraints for features. Can be:
        - dict: {feature_index: constraint} where constraint is 1 (increasing), -1 (decreasing), or 0 (no constraint)
        - list: [constraint1, constraint2, ...] for each feature in order
        Example: {0: 1, 2: -1} means feature 0 should be monotonically increasing, feature 2 decreasing
    interaction_constraints : list of lists, optional
        Feature interaction constraints. Each inner list contains feature indices that can interact.
        Features not in any list cannot interact with any other features.
        Example: [[0, 1], [2, 3, 4]] means features 0&1 can interact, features 2&3&4 can interact,
        but no interactions between these groups or with other features.
    **kwargs : dict
        Additional parameters for the model
    
    Returns:
    --------
    model : sklearn estimator
        Configured XGBoost model with GPU optimization if available
    """
    gpu_info = check_gpu_availability()
    
    if model_type.lower() == 'xgboost':
        try:
            if task_type.lower() == 'regression':
                from xgboost import XGBRegressor
                ModelClass = XGBRegressor
                default_objective = 'reg:squarederror'
                default_eval_metric = 'rmse'
            elif task_type.lower() == 'classification':
                from xgboost import XGBClassifier
                ModelClass = XGBClassifier
                default_objective = 'binary:logistic'
                default_eval_metric = 'logloss'
            else:
                raise ValueError(f"Unsupported task_type: {task_type}. Use 'regression' or 'classification'.")
            
            # Enhanced XGBoost parameters optimized for multi-GPU
            default_params = {
                'max_depth': 9,
                'colsample_bytree': 0.7,
                'subsample': 0.9,
                'n_estimators': 2000,  # Set max iterations to 2000
                'learning_rate': 0.009,
                'gamma': 0.01,
                'max_delta_step': 2,
                'objective': default_objective,
                'eval_metric': default_eval_metric,
                'enable_categorical': False,
                'random_state': 42,
                'early_stopping_rounds': None,  # Disable early stopping initially
                'tree_method': 'hist',
                'device': 'cpu'  # Default to CPU, will be overridden if GPU available
            }
            
            # Add monotonic constraints if provided
            if monotonic_constraints is not None:
                if isinstance(monotonic_constraints, dict):
                    # Convert dict to list format expected by XGBoost
                    # Assume we need to know the total number of features
                    max_feature_idx = max(monotonic_constraints.keys()) if monotonic_constraints else 0
                    monotonic_list = [0] * (max_feature_idx + 1)
                    for feature_idx, constraint in monotonic_constraints.items():
                        monotonic_list[feature_idx] = constraint
                    default_params['monotone_constraints'] = monotonic_list
                elif isinstance(monotonic_constraints, (list, tuple)):
                    default_params['monotone_constraints'] = list(monotonic_constraints)
                else:
                    raise ValueError("monotonic_constraints must be a dict or list")
                
                print(f"\n📊 Monotonic Constraints Applied: {default_params['monotone_constraints']}")
                logger.info(f"Monotonic constraints: {default_params['monotone_constraints']}")
            
            # Add interaction constraints if provided
            if interaction_constraints is not None:
                if not isinstance(interaction_constraints, list):
                    raise ValueError("interaction_constraints must be a list of lists")
                default_params['interaction_constraints'] = interaction_constraints
                print(f"\n🔗 Interaction Constraints Applied: {interaction_constraints}")
                logger.info(f"Interaction constraints: {interaction_constraints}")
            
            # Configure GPU settings if available
            if gpu_info['cuda_available']:
                if gpu_info['multi_gpu_capable']:
                    # Multi-GPU configuration
                    print(f"\n⚡ Configuring XGBoost for Multi-GPU Training:")
                    print(f"   Task Type: {task_type.title()}")
                    print(f"   Available GPUs: {gpu_info['gpu_count']}")
                    print(f"   Total GPU Memory: {gpu_info['total_memory_gb']:.1f} GB")
                    
                    # For XGBoost 2.0+, use device='cuda' for multi-GPU
                    # XGBoost automatically utilizes all available GPUs
                    default_params.update({
                        'device': 'cuda',  # XGBoost 2.0+ automatically uses all GPUs
                        'tree_method': 'hist',  # GPU histogram method
                        'max_bin': 256,  # Optimize for GPU memory
                        'grow_policy': 'lossguide',  # Better for GPU
                        'sampling_method': 'gradient_based',  # GPU-optimized sampling
                        'n_jobs': gpu_info['gpu_count'],  # Explicitly set number of parallel jobs
                        'early_stopping_rounds': 2000  # Continue until 2000 iterations or 2000 rounds without improvement
                    })
                    
                    # Adjust parameters for multi-GPU efficiency
                    if gpu_info['gpu_count'] >= 2:
                        # Increase batch size effectively by allowing more complex models
                        default_params['n_estimators'] = 2000  # Keep at 2000 max
                        default_params['max_depth'] = min(default_params['max_depth'] + 1, 12)
                        # Slightly reduce learning rate for stability with more complex models
                        default_params['learning_rate'] = default_params['learning_rate'] * 0.95
                    
                    print(f"   Configuration: device='cuda' (auto multi-GPU)")
                    print(f"   Tree Method: {default_params['tree_method']}")
                    print(f"   Max Depth: {default_params['max_depth']}")
                    print(f"   N Estimators: {default_params['n_estimators']}")
                    print(f"   Early Stopping: {default_params['early_stopping_rounds']} rounds")
                    print(f"   Parallel Jobs: {default_params['n_jobs']}")
                    
                else:
                    # Single GPU configuration
                    print(f"\n⚡ Configuring XGBoost for Single GPU Training:")
                    print(f"   Task Type: {task_type.title()}")
                    print(f"   GPU: {gpu_info['gpu_names'][0]}")
                    print(f"   Memory: {gpu_info['gpu_memory'][0]['total_gb']:.1f} GB")
                    
                    default_params.update({
                        'device': 'cuda:0',  # Specific GPU device
                        'tree_method': 'hist',
                        'max_bin': 256,
                        'grow_policy': 'lossguide',
                        'early_stopping_rounds': 2000  # Continue until 2000 iterations or 2000 rounds without improvement
                    })
                    
                    print(f"   Configuration: device='cuda:0'")
                    print(f"   Tree Method: {default_params['tree_method']}")
                    print(f"   Early Stopping: {default_params['early_stopping_rounds']} rounds")
            else:
                print(f"\n💻 Configuring XGBoost for CPU Training:")
                print(f"   Task Type: {task_type.title()}")
                print(f"   No GPU available, using optimized CPU settings")
                default_params.update({
                    'device': 'cpu',
                    'tree_method': 'hist',  # CPU histogram method
                    'n_jobs': -1  # Use all CPU cores
                })
            
            # Update with user parameters (user parameters override defaults)
            default_params.update(kwargs)
            
            model = ModelClass(**default_params)
            
            constraint_info = ""
            if monotonic_constraints is not None:
                constraint_info += f" with monotonic constraints"
            if interaction_constraints is not None:
                constraint_info += f" with interaction constraints"
            
            if gpu_info['cuda_available']:
                if gpu_info['multi_gpu_capable']:
                    logger.info(f"✓ Created XGBoost {task_type} model with multi-GPU acceleration ({gpu_info['gpu_count']} GPUs){constraint_info}")
                    print(f"   ✅ Multi-GPU XGBoost {task_type} model created successfully{constraint_info}")
                else:
                    logger.info(f"✓ Created XGBoost {task_type} model with single GPU acceleration{constraint_info}")
                    print(f"   ✅ Single GPU XGBoost {task_type} model created successfully{constraint_info}")
            else:
                logger.info(f"ℹ️  Created XGBoost {task_type} model for CPU (no GPU available){constraint_info}")
                print(f"   ✅ CPU XGBoost {task_type} model created successfully{constraint_info}")
                
            return model
            
        except ImportError:
            logger.error("XGBoost not available. Please install with: pip install xgboost")
            print("❌ XGBoost not available. Please install with: pip install xgboost")
            return None
    
    else:
        logger.error(f"Unsupported model type: {model_type}. Only 'xgboost' is supported.")
        print(f"❌ Unsupported model type: {model_type}. Only 'xgboost' is supported.")
        return None


# ============================================================================
# EXAMPLE USAGE FOR ALL 4 XGBOOST MODEL TYPES
# ============================================================================

"""
# Example usage of the modelCompare.py script for all 4 XGBoost model types
# Uncomment and modify the sections below to use in your own projects

# ============================================================================
# CASE 1: BASIC REGRESSION WITH XGBOOST
# ============================================================================

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your regression dataset
# Replace this with your actual data loading
# data = pd.read_csv('your_regression_dataset.csv')
# X = data.drop('target_column', axis=1)  # Features
# y = data['target_column']  # Continuous target variable

# Example with synthetic data
np.random.seed(42)
n_samples, n_features = 1000, 5
X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                columns=[f'feature_{i}' for i in range(n_features)])
y = pd.Series(2 * X['feature_0'] + 1.5 * X['feature_1'] - X['feature_2'] + np.random.randn(n_samples) * 0.1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create basic XGBoost regression model
model_basic_regression = create_gpu_optimized_model(
    model_type='xgboost',
    task_type='regression'
)

# Train and evaluate the model
results_basic, predictions_basic, train_preds_basic = fitPlotAndPredict(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    model=model_basic_regression,
    datasetName="Basic_Regression_Example",
    xTransform="StandardScaler",
    yTransform="None",
    modelName="XGBoost_Basic_Regression",
    cvScheme="5-fold CV"
)

print("Basic Regression Results:")
print(results_basic)

# ============================================================================
# CASE 2: REGRESSION WITH MONOTONIC CONSTRAINTS AND FEATURE INTERACTIONS
# ============================================================================

# For this example, let's assume:
# - feature_0 should have a monotonically increasing relationship with target
# - feature_1 should have a monotonically decreasing relationship with target
# - feature_2 has no monotonic constraint
# - Features 0 and 1 can interact with each other
# - Features 2, 3, and 4 can interact among themselves
# - No interactions between the two groups

# Define monotonic constraints
monotonic_constraints = {
    0: 1,   # feature_0: monotonically increasing
    1: -1,  # feature_1: monotonically decreasing
    2: 0,   # feature_2: no constraint
    3: 0,   # feature_3: no constraint
    4: 0    # feature_4: no constraint
}

# Define interaction constraints
interaction_constraints = [
    [0, 1],      # Features 0 and 1 can interact
    [2, 3, 4]    # Features 2, 3, and 4 can interact among themselves
]

# Create XGBoost regression model with constraints
model_constrained_regression = create_gpu_optimized_model(
    model_type='xgboost',
    task_type='regression',
    monotonic_constraints=monotonic_constraints,
    interaction_constraints=interaction_constraints,
    learning_rate=0.01,  # Slightly higher learning rate for constrained model
    max_depth=8          # Slightly lower depth due to constraints
)

# Train and evaluate the constrained model
results_constrained, predictions_constrained, train_preds_constrained = fitPlotAndPredict(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    model=model_constrained_regression,
    datasetName="Constrained_Regression_Example",
    xTransform="StandardScaler",
    yTransform="None",
    modelName="XGBoost_Constrained_Regression",
    cvScheme="5-fold CV"
)

print("Constrained Regression Results:")
print(results_constrained)

# ============================================================================
# CASE 3: BASIC BINARY CLASSIFICATION WITH XGBOOST
# ============================================================================

# Create binary classification dataset
# Replace this with your actual binary classification data loading
# data_clf = pd.read_csv('your_classification_dataset.csv')
# X_clf = data_clf.drop('target_column', axis=1)  # Features
# y_clf = data_clf['target_column']  # Binary target (0/1)

# Example with synthetic binary classification data
np.random.seed(42)
n_samples_clf, n_features_clf = 1000, 5
X_clf = pd.DataFrame(np.random.randn(n_samples_clf, n_features_clf), 
                    columns=[f'feature_{i}' for i in range(n_features_clf)])
# Create binary target with some relationship to features
prob = 1 / (1 + np.exp(-(2 * X_clf['feature_0'] + 1.5 * X_clf['feature_1'] - X_clf['feature_2'])))
y_clf = pd.Series(np.random.binomial(1, prob, n_samples_clf))

# Split the classification data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Create basic XGBoost classification model
model_basic_classification = create_gpu_optimized_model(
    model_type='xgboost',
    task_type='classification'
)

# Note: For classification, you might want to modify the fitPlotAndPredict function
# to handle classification metrics (accuracy, precision, recall, F1, AUC-ROC)
# The current function is optimized for regression metrics

# Train and evaluate the classification model
results_clf_basic, predictions_clf_basic, train_preds_clf_basic = fitPlotAndPredict(
    X_train=X_train_clf,
    y_train=y_train_clf,
    X_test=X_test_clf,
    model=model_basic_classification,
    datasetName="Basic_Classification_Example",
    xTransform="StandardScaler",
    yTransform="None",
    modelName="XGBoost_Basic_Classification",
    cvScheme="5-fold CV"
)

print("Basic Classification Results:")
print(results_clf_basic)

# ============================================================================
# CASE 4: BINARY CLASSIFICATION WITH MONOTONIC CONSTRAINTS AND INTERACTIONS
# ============================================================================

# For classification with constraints, let's assume:
# - feature_0 should monotonically increase the probability of class 1
# - feature_1 should monotonically decrease the probability of class 1
# - Features 0 and 1 can interact
# - Features 2, 3, and 4 can interact among themselves

# Define monotonic constraints for classification
monotonic_constraints_clf = {
    0: 1,   # feature_0: increasing probability of class 1
    1: -1,  # feature_1: decreasing probability of class 1
    2: 0,   # feature_2: no constraint
    3: 0,   # feature_3: no constraint
    4: 0    # feature_4: no constraint
}

# Define interaction constraints for classification
interaction_constraints_clf = [
    [0, 1],      # Features 0 and 1 can interact
    [2, 3, 4]    # Features 2, 3, and 4 can interact among themselves
]

# Create XGBoost classification model with constraints
model_constrained_classification = create_gpu_optimized_model(
    model_type='xgboost',
    task_type='classification',
    monotonic_constraints=monotonic_constraints_clf,
    interaction_constraints=interaction_constraints_clf,
    learning_rate=0.01,  # Slightly higher learning rate for constrained model
    max_depth=8          # Slightly lower depth due to constraints
)

# Train and evaluate the constrained classification model
results_clf_constrained, predictions_clf_constrained, train_preds_clf_constrained = fitPlotAndPredict(
    X_train=X_train_clf,
    y_train=y_train_clf,
    X_test=X_test_clf,
    model=model_constrained_classification,
    datasetName="Constrained_Classification_Example",
    xTransform="StandardScaler",
    yTransform="None",
    modelName="XGBoost_Constrained_Classification",
    cvScheme="5-fold CV"
)

print("Constrained Classification Results:")
print(results_clf_constrained)

# ============================================================================
# ADVANCED USAGE EXAMPLES
# ============================================================================

# Example with time-based stratified cross-validation (useful for time series data)
# Assuming you have a 'duration' or 'time' column in your dataset

# results_time_based, predictions_time_based, train_preds_time_based = fitPlotAndPredict(
#     X_train=X_train,
#     y_train=y_train,
#     X_test=X_test,
#     model=model_basic_regression,
#     datasetName="Time_Based_CV_Example",
#     xTransform="StandardScaler",
#     yTransform="None",
#     modelName="XGBoost_Time_Based_CV",
#     cvScheme="5-fold CV",
#     time_column='duration'  # Specify the time column for stratified CV
# )

# Example with two-stage prediction (CV on subset, final predictions on full dataset)
# This is useful when you have a large dataset and want to do CV on a subset for speed

# # Create a subset for CV
# X_train_subset = X_train.sample(n=500, random_state=42)
# y_train_subset = y_train.loc[X_train_subset.index]

# results_two_stage, predictions_two_stage, train_preds_two_stage = fitPlotAndPredict(
#     X_train=X_train_subset,      # CV subset
#     y_train=y_train_subset,      # CV subset
#     X_test=X_test,
#     model=model_basic_regression,
#     datasetName="Two_Stage_Example",
#     xTransform="StandardScaler",
#     yTransform="None",
#     modelName="XGBoost_Two_Stage",
#     cvScheme="5-fold CV",
#     X_train_full=X_train,       # Full dataset for final predictions
#     y_train_full=y_train        # Full dataset for final predictions
# )

# ============================================================================
# CUSTOM PARAMETER EXAMPLES
# ============================================================================

# Example with custom XGBoost parameters
custom_model = create_gpu_optimized_model(
    model_type='xgboost',
    task_type='regression',
    # Custom parameters that override defaults
    n_estimators=1500,
    max_depth=12,
    learning_rate=0.005,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    min_child_weight=3
)

# Example with different evaluation metrics
custom_model_mae = create_gpu_optimized_model(
    model_type='xgboost',
    task_type='regression',
    eval_metric='mae',  # Use MAE instead of RMSE
    objective='reg:absoluteerror'  # Use absolute error objective
)

# ============================================================================
# TIPS FOR USING MONOTONIC CONSTRAINTS AND INTERACTIONS
# ============================================================================

# Monotonic Constraints Tips:
# - Use 1 for features that should monotonically increase the target
# - Use -1 for features that should monotonically decrease the target  
# - Use 0 for no constraint (default)
# - Constraints can help with model interpretability and prevent overfitting
# - Be careful with constraints - they can hurt performance if incorrectly specified

# Interaction Constraints Tips:
# - Each list in interaction_constraints represents a group of features that can interact
# - Features not in any group cannot interact with any other features
# - This can help prevent overfitting and improve interpretability
# - Useful when you have domain knowledge about which features should interact
# - Can significantly reduce model complexity

# Example of more complex interaction constraints:
# interaction_constraints_complex = [
#     [0, 1, 2],    # Group 1: Features 0, 1, 2 can interact among themselves
#     [3, 4],       # Group 2: Features 3, 4 can interact with each other
#     [5],          # Group 3: Feature 5 can only have main effects (no interactions)
#     [6, 7, 8, 9]  # Group 4: Features 6, 7, 8, 9 can interact among themselves
# ]

# ============================================================================
# VALIDATION AND DEBUGGING
# ============================================================================

# Always validate your input data before training
# validate_input_data(X_train, y_train, X_test)

# Check GPU availability
# gpu_info = check_gpu_availability()
# print(f"GPU Available: {gpu_info['cuda_available']}")
# print(f"Number of GPUs: {gpu_info['gpu_count']}")

# Monitor training progress and GPU utilization during training
# The fitPlotAndPredict function automatically provides detailed progress reporting

"""