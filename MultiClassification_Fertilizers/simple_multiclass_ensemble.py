#!/usr/bin/env python3
"""
Simple Multi-Class Classification Ensemble Framework
Focuses on reliability and ease of use over complex features.
"""

import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Try to import gradient boosting libraries
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")


@dataclass
class SimpleConfig:
    """Simple configuration class"""
    train_path: str = ""
    test_path: str = ""
    target_col: str = "target"
    id_col: str = "id"
    n_folds: int = 5
    seed: int = 42
    top_k_predictions: int = 3
    output_dir: str = "simple_ensemble_outputs"
    use_gpu: bool = False  # Simplified - no complex GPU handling
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def mean_average_precision_at_k(y_true: np.ndarray, y_pred_probs: np.ndarray, k: int = 3) -> float:
    """Calculate Mean Average Precision at K (MAP@K)"""
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


class SimpleModelTrainer:
    """Simple, reliable model trainer"""
    
    def __init__(self, model, model_name: str, config: SimpleConfig):
        self.model = model
        self.model_name = model_name
        self.config = config
        
    def train_cv(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Simple cross-validation training"""
        
        print(f"\n{'='*60}")
        print(f"🚀 Training {self.model_name}")
        print(f"{'='*60}")
        
        n_classes = y.nunique()
        scores = []
        
        oof_pred_probs = np.zeros((X.shape[0], n_classes))
        test_pred_probs = np.zeros((X_test.shape[0], n_classes))
        
        skf = StratifiedKFold(n_splits=self.config.n_folds, random_state=self.config.seed, shuffle=True)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n📊 Fold {fold_idx + 1}/{self.config.n_folds}")
            
            # Prepare fold data
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y[train_idx].copy(), y[val_idx].copy()
            
            # Clone model for this fold
            fold_model = clone(self.model)
            
            # Train model based on type
            try:
                if 'LightGBM' in str(type(fold_model)):
                    # Simple LightGBM training
                    print("   🚂 Training LightGBM...")
                    # Use simple fit without complex callbacks
                    fold_model.fit(X_train, y_train)
                
                elif 'XGB' in str(type(fold_model)):
                    # Simple XGBoost training
                    print("   🚂 Training XGBoost...")
                    # Use simple fit without eval_set to avoid complications
                    fold_model.fit(X_train, y_train)
                
                else:
                    # Simple sklearn model training
                    print(f"   🚂 Training {self.model_name}...")
                    fold_model.fit(X_train, y_train)
                
                # Make predictions
                print("   🔮 Making predictions...")
                y_pred_probs = fold_model.predict_proba(X_val)
                test_pred_probs_fold = fold_model.predict_proba(X_test)
                
                # Store predictions
                oof_pred_probs[val_idx] = y_pred_probs
                test_pred_probs += test_pred_probs_fold / self.config.n_folds
                
                # Calculate metrics
                map_score = mean_average_precision_at_k(y_val.values, y_pred_probs, self.config.top_k_predictions)
                accuracy = accuracy_score(y_val, fold_model.predict(X_val))
                
                scores.append(map_score)
                
                print(f"   ✅ MAP@{self.config.top_k_predictions}: {map_score:.6f} | Accuracy: {accuracy:.6f}")
                
            except Exception as e:
                print(f"   ❌ Error in fold {fold_idx + 1}: {str(e)}")
                raise
            
            finally:
                # Cleanup
                del fold_model
                gc.collect()
        
        # Final results
        overall_map = mean_average_precision_at_k(y.values, oof_pred_probs, self.config.top_k_predictions)
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\n🎯 Final Results:")
        print(f"   📊 Overall MAP@{self.config.top_k_predictions}: {overall_map:.6f}")
        print(f"   📈 CV MAP@{self.config.top_k_predictions}: {avg_score:.6f} ± {std_score:.6f}")
        
        return oof_pred_probs, test_pred_probs, scores


def load_and_preprocess_data(config: SimpleConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """Load and preprocess data simply"""
    
    print("🔄 Loading and preprocessing data...")
    
    # Load data
    train = pd.read_csv(config.train_path)
    test = pd.read_csv(config.test_path)
    
    print(f"📊 Train shape: {train.shape}")
    print(f"📊 Test shape: {test.shape}")
    
    # Handle ID column
    if config.id_col in train.columns:
        train = train.set_index(config.id_col)
    if config.id_col in test.columns:
        test = test.set_index(config.id_col)
    
    # Handle categorical columns - convert to category type
    categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
    if config.target_col in categorical_cols:
        categorical_cols.remove(config.target_col)
    
    print(f"🏷️  Categorical columns: {categorical_cols}")
    
    for col in categorical_cols:
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')
    
    # Encode target
    print("🎯 Encoding target variable...")
    label_encoder = LabelEncoder()
    train[config.target_col] = label_encoder.fit_transform(train[config.target_col])
    
    print(f"📈 Classes: {len(label_encoder.classes_)} -> {list(label_encoder.classes_)}")
    
    # Split features and target
    X = train.drop(config.target_col, axis=1)
    y = train[config.target_col]
    X_test = test
    
    return X, y, X_test, label_encoder


def get_simple_models(config: SimpleConfig) -> Dict:
    """Get simple, reliable model configurations"""
    
    models = {}
    
    # LightGBM - simple configuration
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=500,  # Reduced to avoid long training
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=config.seed,
            device='gpu' if config.use_gpu else 'cpu',
            verbose=-1
        )
    
    # XGBoost - simple configuration
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,  # Reduced to avoid long training
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.seed,
            device='cuda' if config.use_gpu else 'cpu',
            verbosity=0
        )
    
    # Random Forest - always available
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=config.seed,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    return models


def create_ensemble(oof_predictions: Dict, test_predictions: Dict, y: pd.Series, config: SimpleConfig) -> Tuple[np.ndarray, LogisticRegression]:
    """Create simple ensemble"""
    
    print(f"\n{'='*60}")
    print("🎯 Creating Ensemble")
    print(f"{'='*60}")
    
    # Combine predictions
    X_ensemble = np.concatenate(list(oof_predictions.values()), axis=1)
    X_test_ensemble = np.concatenate(list(test_predictions.values()), axis=1)
    
    print(f"📊 Ensemble features: {X_ensemble.shape[1]} (from {len(oof_predictions)} models)")
    
    # Scale features
    scaler = StandardScaler()
    X_ensemble_scaled = scaler.fit_transform(X_ensemble)
    X_test_ensemble_scaled = scaler.transform(X_test_ensemble)
    
    # Train ensemble
    ensemble_model = LogisticRegression(
        random_state=config.seed,
        max_iter=1000,
        class_weight='balanced'
    )
    
    ensemble_model.fit(X_ensemble_scaled, y)
    
    # Make final predictions
    final_test_probs = ensemble_model.predict_proba(X_test_ensemble_scaled)
    
    print("✅ Ensemble created successfully!")
    
    return final_test_probs, ensemble_model


def generate_submission(test_probs: np.ndarray, X_test: pd.DataFrame, label_encoder: LabelEncoder, config: SimpleConfig) -> pd.DataFrame:
    """Generate submission file"""
    
    print("\n🔮 Generating final predictions...")
    
    # Get top-k predictions
    top_k_indices = np.argsort(test_probs, axis=1)[:, -config.top_k_predictions:][:, ::-1]
    
    predictions = []
    for indices in top_k_indices:
        pred_names = label_encoder.inverse_transform(indices)
        predictions.append(" ".join(pred_names))
    
    # Create submission dataframe
    test_ids = X_test.index.tolist() if hasattr(X_test.index, 'tolist') else list(range(len(X_test)))
    
    submission = pd.DataFrame({
        config.id_col: test_ids,
        config.target_col: predictions
    })
    
    print(f"✅ Generated {len(predictions)} predictions")
    return submission


def main():
    """Main execution function"""
    
    print("🚀 SIMPLE MULTI-CLASS ENSEMBLE FRAMEWORK")
    print("="*60)
    
    # Configuration - UPDATE THESE PATHS
    config = SimpleConfig(
        train_path="/kaggle/input/playground-series-s5e6/train.csv",  # UPDATE THIS
        test_path="/kaggle/input/playground-series-s5e6/test.csv",    # UPDATE THIS
        target_col="Fertilizer Name",
        id_col="id",
        n_folds=5,
        seed=42,
        top_k_predictions=3,
        use_gpu=True  # GPU enabled - simple single GPU usage
    )
    
    # Check GPU availability
    if config.use_gpu:
        print("🎮 GPU acceleration: ENABLED")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"   ✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
            else:
                print("   ⚠️  CUDA not available - will fall back to CPU")
        except ImportError:
            print("   ⚠️  PyTorch not installed - GPU detection limited")
    else:
        print("💻 GPU acceleration: DISABLED")
    
    try:
        # Load data
        X, y, X_test, label_encoder = load_and_preprocess_data(config)
        
        # Get models
        models = get_simple_models(config)
        print(f"\n📋 Available models: {list(models.keys())}")
        
        # Train models
        oof_predictions = {}
        test_predictions = {}
        all_scores = {}
        
        for model_name, model in models.items():
            print(f"\n🔄 Processing {model_name}...")
            
            trainer = SimpleModelTrainer(model, model_name, config)
            oof_pred, test_pred, scores = trainer.train_cv(X, y, X_test)
            
            oof_predictions[model_name] = oof_pred
            test_predictions[model_name] = test_pred
            all_scores[model_name] = scores
        
        # Create ensemble
        if len(oof_predictions) > 1:
            final_test_probs, ensemble_model = create_ensemble(oof_predictions, test_predictions, y, config)
        else:
            # Use single model if only one available
            final_test_probs = list(test_predictions.values())[0]
        
        # Generate submission
        submission = generate_submission(final_test_probs, X_test, label_encoder, config)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = Path(config.output_dir) / f"submission_{timestamp}.csv"
        submission.to_csv(submission_path, index=False)
        
        # Save scores
        scores_df = pd.DataFrame(all_scores)
        scores_path = Path(config.output_dir) / f"cv_scores_{timestamp}.csv"
        scores_df.to_csv(scores_path, index=False)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📄 Submission saved: {submission_path}")
        print(f"📊 CV scores saved: {scores_path}")
        
        # Print final summary
        print(f"\n📈 FINAL RESULTS:")
        for model_name, scores in all_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"   {model_name:15s}: {mean_score:.6f} ± {std_score:.6f}")
        
    except FileNotFoundError:
        print("❌ ERROR: Data files not found!")
        print("Please update the train_path and test_path in the config above.")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main() 