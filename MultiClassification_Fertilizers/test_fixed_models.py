#!/usr/bin/env python3
"""
Test script to verify that the fixed LightGBM and XGBoost configurations work properly.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Import the models
try:
    from lightgbm import LGBMClassifier, early_stopping
    LIGHTGBM_AVAILABLE = True
    print("✅ LightGBM available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("❌ LightGBM not available")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("❌ XGBoost not available")

def test_lightgbm_fixed_config():
    """Test the fixed LightGBM configuration"""
    if not LIGHTGBM_AVAILABLE:
        print("Skipping LightGBM test - not available")
        return
    
    print("\n" + "="*50)
    print("🧪 TESTING FIXED LIGHTGBM CONFIGURATION")
    print("="*50)
    
    # Create dummy multiclass data
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_classes=7, 
        n_informative=8,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training data: {X_train.shape}")
    print(f"🎯 Classes: {len(np.unique(y))}")
    
    # Test GBDT configuration
    print("\n🔧 Testing LightGBM GBDT with fixed configuration...")
    model_gbdt = LGBMClassifier(
        boosting_type="gbdt",
        device="cpu",  # Use CPU for testing
        colsample_bytree=0.4366677273946288,
        learning_rate=0.016164161953515117,
        max_depth=12,
        min_child_samples=67,
        n_estimators=100,  # Reduced for testing
        n_jobs=-1,
        num_leaves=243,
        random_state=42,
        reg_alpha=6.38288560443373,
        reg_lambda=9.392999314379155,
        subsample=0.7989164499431718,
        verbose=-1
    )
    
    # Fixed fit_args - using both eval_metric and callbacks
    fit_args_gbdt = {
        "eval_metric": "multi_logloss",
        "callbacks": [
            early_stopping(stopping_rounds=20, verbose=False)
        ]
    }
    
    try:
        model_gbdt.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            **fit_args_gbdt
        )
        
        # Make predictions
        pred_proba = model_gbdt.predict_proba(X_val)
        pred_classes = model_gbdt.predict(X_val)
        accuracy = accuracy_score(y_val, pred_classes)
        
        print(f"✅ LightGBM GBDT training successful!")
        print(f"   📈 Validation accuracy: {accuracy:.4f}")
        print(f"   🔮 Prediction shape: {pred_proba.shape}")
        
    except Exception as e:
        print(f"❌ LightGBM GBDT failed: {str(e)}")
        return False
    
    # Test GOSS configuration
    print("\n🔧 Testing LightGBM GOSS with fixed configuration...")
    model_goss = LGBMClassifier(
        boosting_type="goss",
        device="cpu",  # Use CPU for testing
        colsample_bytree=0.32751831793031183,
        learning_rate=0.006700715059604966,
        max_depth=12,
        min_child_samples=84,
        n_estimators=100,  # Reduced for testing
        n_jobs=-1,
        num_leaves=229,
        random_state=42,
        reg_alpha=6.879977008084246,
        reg_lambda=4.739518466581721,
        subsample=0.5411572049978781,
        verbose=-1
    )
    
    # Fixed fit_args - using both eval_metric and callbacks
    fit_args_goss = {
        "eval_metric": "multi_logloss",
        "callbacks": [
            early_stopping(stopping_rounds=20, verbose=False)
        ]
    }
    
    try:
        model_goss.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            **fit_args_goss
        )
        
        # Make predictions
        pred_proba = model_goss.predict_proba(X_val)
        pred_classes = model_goss.predict(X_val)
        accuracy = accuracy_score(y_val, pred_classes)
        
        print(f"✅ LightGBM GOSS training successful!")
        print(f"   📈 Validation accuracy: {accuracy:.4f}")
        print(f"   🔮 Prediction shape: {pred_proba.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ LightGBM GOSS failed: {str(e)}")
        return False


def test_xgboost_fixed_config():
    """Test the fixed XGBoost configuration"""
    if not XGBOOST_AVAILABLE:
        print("Skipping XGBoost test - not available")
        return
    
    print("\n" + "="*50)
    print("🧪 TESTING FIXED XGBOOST CONFIGURATION")
    print("="*50)
    
    # Create dummy multiclass data
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_classes=7, 
        n_informative=8,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training data: {X_train.shape}")
    print(f"🎯 Classes: {len(np.unique(y))}")
    
    print("\n🔧 Testing XGBoost with fixed configuration...")
    model_xgb = XGBClassifier(
        device="cpu",  # Use CPU for testing
        max_depth=12,
        colsample_bytree=0.467,
        subsample=0.86,
        n_estimators=100,  # Reduced for testing
        learning_rate=0.03,
        gamma=0.26,
        max_delta_step=4,
        reg_alpha=2.7,
        reg_lambda=1.4,
        objective='multi:softprob',
        random_state=13,
        enable_categorical=True,
        verbosity=0
    )
    
    # Fixed fit_args - eval_metric moved here
    fit_args_xgb = {
        "early_stopping_rounds": 20,  # Reduced for testing
        "verbose": False,
        "eval_metric": "mlogloss"
    }
    
    try:
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            **fit_args_xgb
        )
        
        # Make predictions
        pred_proba = model_xgb.predict_proba(X_val)
        pred_classes = model_xgb.predict(X_val)
        accuracy = accuracy_score(y_val, pred_classes)
        
        print(f"✅ XGBoost training successful!")
        print(f"   📈 Validation accuracy: {accuracy:.4f}")
        print(f"   🔮 Prediction shape: {pred_proba.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ XGBoost failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("🚀 TESTING FIXED MODEL CONFIGURATIONS")
    print("="*60)
    
    lightgbm_success = test_lightgbm_fixed_config()
    xgboost_success = test_xgboost_fixed_config()
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    if LIGHTGBM_AVAILABLE:
        status = "✅ PASSED" if lightgbm_success else "❌ FAILED"
        print(f"LightGBM Configuration: {status}")
    else:
        print("LightGBM Configuration: ⚠️  SKIPPED (not available)")
    
    if XGBOOST_AVAILABLE:
        status = "✅ PASSED" if xgboost_success else "❌ FAILED"
        print(f"XGBoost Configuration:  {status}")
    else:
        print("XGBoost Configuration:  ⚠️  SKIPPED (not available)")
    
    overall_success = (lightgbm_success if LIGHTGBM_AVAILABLE else True) and (xgboost_success if XGBOOST_AVAILABLE else True)
    
    if overall_success:
        print("\n🎉 ALL CONFIGURATIONS FIXED SUCCESSFULLY!")
        print("   The early stopping errors have been resolved.")
    else:
        print("\n❌ SOME CONFIGURATIONS STILL HAVE ISSUES")
        print("   Please check the error messages above.")
    
    return overall_success


if __name__ == "__main__":
    main() 