import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress specific model warnings
os.environ['LIGHTGBM_SILENT'] = '1'
os.environ['XGBOOST_SILENT'] = '1'
os.environ['CATBOOST_SILENT'] = '1'

# Suppress matplotlib warnings
plt.set_loglevel('error')

# Suppress pandas warnings
pd.options.mode.chained_assignment = None

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Try to import optional libraries
try:
    from lightgbm import LGBMClassifier, log_evaluation, early_stopping
    import lightgbm as lgb
    # Additional LightGBM warning suppression
    import logging
    logging.basicConfig(level=logging.ERROR)  # Set logging level to ERROR
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

try:
    from xgboost import XGBClassifier
    # Additional XGBoost warning suppression
    import xgboost as xgb
    xgb.set_config(verbosity=0)  # Disable XGBoost verbosity
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

try:
    from catboost import CatBoostClassifier
    # Additional CatBoost warning suppression
    from catboost import CatBoostError
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available")

# GPU Support Detection
def detect_gpu_support():
    """Detect available GPUs"""
    gpu_info = {'cuda_available': False, 'gpu_count': 0}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['gpu_count'] = torch.cuda.device_count()
    except ImportError:
        pass
    return gpu_info

GPU_INFO = detect_gpu_support()

@dataclass
class Config:
    """Simplified configuration class"""
    # Data paths
    train_paths: List[str] = field(default_factory=list)  # List of training data paths
    test_path: str = ""
    target_col: str = "Fertilizer Name"
    id_col: str = "id"
    
    # Training settings
    n_folds: int = 5
    seed: int = 42
    frac: float = 1.0  # Fraction of training data to use
    
    # GPU settings
    use_gpu: bool = True
    
    # Output settings
    output_dir: str = "ensemble_outputs"
    top_k_predictions: int = 3
    
    def __post_init__(self):
        if not self.train_paths or not self.test_path:
            raise ValueError("train_paths and test_path must be specified")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.use_gpu and not GPU_INFO['cuda_available']:
            print("⚠️  GPU requested but CUDA not available. Using CPU.")
            self.use_gpu = False

def mean_average_precision_at_k(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int = 3) -> float:
    """Calculate Mean Average Precision at K"""
    if len(y_true) == 0:
        return 0.0
    
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:][:, ::-1]
    
    def average_precision_at_k(y_true_sample: int, y_pred_sample: np.ndarray, k: int) -> float:
        y_pred_sample = y_pred_sample[:k]
        if len(y_pred_sample) == 0:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        
        for i, pred in enumerate(y_pred_sample):
            if pred == y_true_sample and pred not in y_pred_sample[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score
    
    ap_scores = []
    for i in range(len(y_true)):
        ap = average_precision_at_k(y_true[i], top_k_preds[i], k)
        ap_scores.append(ap)
    
    return np.mean(ap_scores)

def load_and_preprocess_data(config: Config) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, LabelEncoder, List[str]]:
    """Load and preprocess data"""
    print(f"\n🔄 Loading data...")
    
    # Load and combine training data
    train_dfs = []
    for train_path in config.train_paths:
        print(f"📥 Loading training data from: {train_path}")
        train_df = pd.read_csv(train_path)
        train_dfs.append(train_df)
    
    # Combine all training data
    train = pd.concat(train_dfs, ignore_index=True)
    print(f"📊 Combined train shape: {train.shape}")
    
    # Load test data
    test = pd.read_csv(config.test_path)
    print(f"🧪 Test shape: {test.shape}")
    
    if config.id_col in train.columns:
        train = train.set_index(config.id_col)
    if config.id_col in test.columns:
        test = test.set_index(config.id_col)
    
    # Sample data if needed
    if config.frac < 1:
        print(f"🎲 Sampling {config.frac*100:.1f}% of training data...")
        train = train.sample(frac=config.frac, random_state=config.seed)
        print(f"📊 Sampled train shape: {train.shape}")
    
    # Handle categorical columns
    cat_cols = train.select_dtypes(include="object").columns.tolist()
    if config.target_col in cat_cols:
        cat_cols.remove(config.target_col)
    
    # Store categorical columns for later use
    categorical_features = cat_cols.copy()
    
    # For RandomForest, we need to encode categorical features
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    
    # Encode target
    label_encoder = LabelEncoder()
    train[config.target_col] = label_encoder.fit_transform(train[config.target_col])
    
    # Split features and target
    X = train.drop(config.target_col, axis=1)
    y = train[config.target_col]
    X_test = test
    
    print(f"🎯 Classes: {len(label_encoder.classes_)}")
    print(f"📝 Features: {len(X.columns)}")
    print(f"🔤 Categorical features: {categorical_features}")
    
    return X, y, X_test, label_encoder, categorical_features

def get_model_configurations(config: Config, categorical_features: List[str]) -> Dict[str, Dict]:
    """Get model configurations with GPU support"""
    configs = {}
    
    # Determine GPU settings
    if config.use_gpu and GPU_INFO['cuda_available']:
        lgb_device = "gpu"
        xgb_device = "cuda"
        catboost_task_type = "GPU"
        print("🎮 GPU acceleration enabled for compatible models")
    else:
        lgb_device = "cpu"
        xgb_device = "cpu"
        catboost_task_type = "CPU"
        print("💻 Using CPU for all models")
    
    # LightGBM GBDT
    if LIGHTGBM_AVAILABLE:
        lgbm_params = {
            "boosting_type": "gbdt",
            "device": lgb_device,
            "colsample_bytree": 0.4366677273946288,
            "learning_rate": 0.016164161953515117,
            "max_depth": 12,
            "min_child_samples": 67,
            "n_estimators": 10000,
            "n_jobs": -1,
            "num_leaves": 243,
            "random_state": config.seed,
            "reg_alpha": 6.38288560443373,
            "reg_lambda": 9.392999314379155,
            "subsample": 0.7989164499431718,
            "verbose": -1
        }
        
        configs["LightGBM_GBDT"] = {
            "model": LGBMClassifier(**lgbm_params),
            "fit_args": {
                "eval_metric": "multi_logloss",
                "callbacks": [early_stopping(100, verbose=False)],
                "verbose": False
            }
        }
        
        # LightGBM GOSS
        lgbm_goss_params = {
            "boosting_type": "goss",
            "device": lgb_device,
            "colsample_bytree": 0.32751831793031183,
            "learning_rate": 0.006700715059604966,
            "max_depth": 12,
            "min_child_samples": 84,
            "n_estimators": 10000,
            "n_jobs": -1,
            "num_leaves": 229,
            "random_state": config.seed,
            "reg_alpha": 6.879977008084246,
            "reg_lambda": 4.739518466581721,
            "subsample": 0.5411572049978781,
            "verbose": -1
        }
        
        configs["LightGBM_GOSS"] = {
            "model": LGBMClassifier(**lgbm_goss_params),
            "fit_args": {
                "eval_metric": "multi_logloss",
                "callbacks": [early_stopping(100, verbose=False)],
                "verbose": False
            }
        }
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        configs["XGBoost"] = {
            "model": XGBClassifier(
                device=xgb_device,
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                random_state=config.seed,
                verbosity=0,
                enable_categorical=True,
                silent=True  # Additional silence control
            ),
            "fit_args": {
                "early_stopping_rounds": 100,
                "verbose": False
            }
        }
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        configs["CatBoost"] = {
            "model": CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                task_type=catboost_task_type,
                random_seed=config.seed,
                cat_features=categorical_features,
                verbose=False  # Only set verbose here
            ),
            "fit_args": {
                "early_stopping_rounds": 100,
                "verbose": False
            }
        }
    
    # Random Forest (always available)
    configs["RandomForest"] = {
        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=config.seed,
            n_jobs=-1
        ),
        "fit_args": {}
    }
    
    return configs

class SimpleModelTrainer:
    """Simplified model trainer with sequential fold processing"""
    
    def __init__(self, model: Any, model_name: str, config: Config):
        self.model = model
        self.model_name = model_name
        self.config = config
    
    def train_single_fold(self, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray, 
                         X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, 
                         fit_args: Dict) -> Tuple[np.ndarray, np.ndarray, float]:
        """Train a single fold"""
        print(f"    📊 Fold {fold_idx + 1}: Training on {len(train_idx)} samples, validating on {len(val_idx)} samples")
        
        # Prepare data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Clone model for this fold
        fold_model = clone(self.model)
        
        # Train model
        try:
            if fit_args and ('eval_set' in fit_args or 'callbacks' in fit_args or 'early_stopping_rounds' in fit_args):
                # Models with early stopping
                if hasattr(fold_model, 'fit') and 'LightGBM' in str(type(fold_model)):
                    fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_args)
                elif hasattr(fold_model, 'fit') and 'XGB' in str(type(fold_model)):
                    fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_args)
                elif hasattr(fold_model, 'fit') and 'CatBoost' in str(type(fold_model)):
                    fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_args)
                else:
                    fold_model.fit(X_train, y_train)
            else:
                fold_model.fit(X_train, y_train)
        except Exception as e:
            print(f"    ⚠️  Training error: {str(e)}")
            # Fallback to basic training
            fold_model.fit(X_train, y_train)
        
        # Get predictions
        val_pred_probs = fold_model.predict_proba(X_val)
        test_pred_probs = fold_model.predict_proba(X_test)
        
        # Calculate MAP@K
        map_score = mean_average_precision_at_k(y_val.values, val_pred_probs, self.config.top_k_predictions)
        
        print(f"    ✅ Fold {fold_idx + 1} MAP@{self.config.top_k_predictions}: {map_score:.6f}")
        
        return val_pred_probs, test_pred_probs, map_score
    
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, 
                   fit_args: Dict = None) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Train model with cross-validation"""
        print(f"\n🔥 Training {self.model_name}")
        print(f"{'='*50}")
        
        if fit_args is None:
            fit_args = {}
        
        n_classes = y.nunique()
        scores = []
        
        # Initialize prediction arrays
        oof_pred_probs = np.zeros((X.shape[0], n_classes))
        test_pred_probs = np.zeros((X_test.shape[0], n_classes))
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=self.config.n_folds, random_state=self.config.seed, shuffle=True)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            val_pred, test_pred, map_score = self.train_single_fold(
                fold_idx, train_idx, val_idx, X, y, X_test, fit_args
            )
            
            # Store predictions
            oof_pred_probs[val_idx] = val_pred
            test_pred_probs += test_pred / self.config.n_folds
            scores.append(map_score)
        
        # Calculate overall score
        overall_map = mean_average_precision_at_k(y.values, oof_pred_probs, self.config.top_k_predictions)
        
        print(f"🎯 {self.model_name} Results:")
        print(f"   Overall MAP@{self.config.top_k_predictions}: {overall_map:.6f}")
        print(f"   Average MAP@{self.config.top_k_predictions}: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
        
        return oof_pred_probs, test_pred_probs, scores

def create_ensemble_and_predict(oof_predictions: Dict[str, np.ndarray], 
                               test_predictions: Dict[str, np.ndarray],
                               y: pd.Series, X_test: pd.DataFrame, 
                               label_encoder: LabelEncoder, config: Config) -> pd.DataFrame:
    """Create ensemble predictions"""
    print(f"\n🎯 Creating ensemble predictions...")
    
    # Combine predictions (simple average)
    oof_ensemble = np.mean(list(oof_predictions.values()), axis=0)
    test_ensemble = np.mean(list(test_predictions.values()), axis=0)
    
    # Calculate ensemble score
    ensemble_score = mean_average_precision_at_k(y.values, oof_ensemble, config.top_k_predictions)
    print(f"🏆 Ensemble MAP@{config.top_k_predictions}: {ensemble_score:.6f}")
    
    # Generate final predictions
    top_k_indices = np.argsort(test_ensemble, axis=1)[:, -config.top_k_predictions:][:, ::-1]
    
    final_predictions = []
    for indices in top_k_indices:
        prediction_names = label_encoder.inverse_transform(indices)
        final_predictions.append(" ".join(prediction_names))
    
    # Create submission
    test_ids = X_test.index.tolist() if hasattr(X_test.index, 'tolist') else list(range(len(X_test)))
    submission = pd.DataFrame({
        config.id_col: test_ids,
        config.target_col: final_predictions
    })
    
    return submission

def visualize_results(scores: Dict[str, List[float]], config: Config) -> None:
    """Create visualization of results"""
    print(f"\n📈 Creating results visualization...")
    
    scores_df = pd.DataFrame(scores)
    mean_scores = scores_df.mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Mean scores
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(mean_scores)), mean_scores.values, alpha=0.7)
    plt.title(f'Mean MAP@{config.top_k_predictions} Scores')
    plt.xticks(range(len(mean_scores)), mean_scores.index, rotation=45)
    plt.ylabel(f'MAP@{config.top_k_predictions}')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Subplot 2: Score distribution
    plt.subplot(2, 2, 2)
    scores_df[mean_scores.index].boxplot()
    plt.title('Score Distribution')
    plt.xticks(rotation=45)
    plt.ylabel(f'MAP@{config.top_k_predictions}')
    
    # Subplot 3: Standard deviation
    plt.subplot(2, 2, 3)
    score_stds = scores_df.std().sort_values()
    plt.bar(range(len(score_stds)), score_stds.values, alpha=0.7, color='orange')
    plt.title('Model Stability (Lower is Better)')
    plt.xticks(range(len(score_stds)), score_stds.index, rotation=45)
    plt.ylabel('Standard Deviation')
    
    # Subplot 4: Summary table
    plt.subplot(2, 2, 4)
    plt.axis('tight')
    plt.axis('off')
    
    summary_data = []
    for model in mean_scores.index:
        summary_data.append([
            model,
            f"{mean_scores[model]:.4f}",
            f"{scores_df[model].std():.4f}",
            f"{scores_df[model].min():.4f}",
            f"{scores_df[model].max():.4f}"
        ])
    
    table = plt.table(
        cellText=summary_data,
        colLabels=['Model', 'Mean', 'Std', 'Min', 'Max'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title('Performance Summary')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(config.output_dir) / 'results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Results saved to: {plot_path}")
    
    plt.show()

def main(config: Config) -> None:
    """Main execution function"""
    print("\n" + "="*60)
    print("🚀 SIMPLIFIED MULTI-CLASS ENSEMBLE FRAMEWORK")
    print("="*60)
    print(f"📂 Output directory: {config.output_dir}")
    print(f"🎲 Random seed: {config.seed}")
    print(f"🔄 Cross-validation folds: {config.n_folds}")
    print(f"📊 Top-K predictions: {config.top_k_predictions}")
    
    if config.use_gpu and GPU_INFO['cuda_available']:
        print(f"🎮 GPU acceleration: ENABLED ({GPU_INFO['gpu_count']} GPU(s))")
    else:
        print(f"💻 GPU acceleration: DISABLED")
    print("="*60)
    
    try:
        # Load and preprocess data
        X, y, X_test, label_encoder, categorical_features = load_and_preprocess_data(config)
        
        # Get model configurations
        model_configs = get_model_configurations(config, categorical_features)
        print(f"\n🤖 Available models: {list(model_configs.keys())}")
        
        # Train models
        oof_predictions = {}
        test_predictions = {}
        scores = {}
        
        for model_name, model_config in model_configs.items():
            try:
                trainer = SimpleModelTrainer(
                    model_config["model"], 
                    model_name, 
                    config
                )
                
                oof_pred, test_pred, model_scores = trainer.fit_predict(
                    X, y, X_test, model_config["fit_args"]
                )
                
                oof_predictions[model_name] = oof_pred
                test_predictions[model_name] = test_pred
                scores[model_name] = model_scores
                
                print(f"✅ {model_name} completed successfully!")
                
            except Exception as e:
                print(f"❌ Failed to train {model_name}: {str(e)}")
                continue
        
        if len(oof_predictions) == 0:
            print("❌ No models trained successfully!")
            return
        
        # Create ensemble predictions
        submission = create_ensemble_and_predict(
            oof_predictions, test_predictions, y, X_test, label_encoder, config
        )
        
        # Save results
        submission_path = Path(config.output_dir) / 'submission.csv'
        submission.to_csv(submission_path, index=False)
        submission.to_csv('submission.csv', index=False)  # Also save to working directory
        print(f"💾 Submission saved to: {submission_path}")
        
        # Visualize results
        if len(scores) > 0:
            visualize_results(scores, config)
        
        # Print final summary
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📊 FINAL MODEL PERFORMANCE:")
        for model_name, model_scores in scores.items():
            mean_score = np.mean(model_scores)
            std_score = np.std(model_scores)
            print(f"🏆 {model_name:15s}: {mean_score:.6f} ± {std_score:.6f}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    config = Config(
        train_paths=[
            "/kaggle/input/playground-series-s5e6/train.csv",
            "/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv"
        ],
        test_path="/kaggle/input/playground-series-s5e6/test.csv",
        target_col="Fertilizer Name",
        id_col="id",
        n_folds=5,
        seed=42,
        frac=0.1,  # Use 10% for testing
        use_gpu=True,
        output_dir="ensemble_outputs"
    )
    
    print("🚀 Simplified Multi-Class Classification Ensemble Framework")
    print("Please set the correct data paths in the Config object before running!")
    print(f"Current config: {config}")
    # Uncomment the following line when paths are set
    main(config) 