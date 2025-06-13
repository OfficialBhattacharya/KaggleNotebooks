#!/usr/bin/env python3
"""
Quick test script to verify model configuration fixes
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import the fixed configurations
from enhanced_multiclass_ensemble import get_model_configurations, Config

def test_model_configs():
    """Test that all model configurations work without errors"""
    
    print("🧪 Testing model configurations...")
    
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Create features (mix of numerical and categorical)
    X = pd.DataFrame({
        'numeric_1': np.random.randn(n_samples),
        'numeric_2': np.random.randn(n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples)
    })
    
    # Convert categorical columns to category dtype
    X['categorical_1'] = X['categorical_1'].astype('category')
    X['categorical_2'] = X['categorical_2'].astype('category')
    
    # Create target variable
    y = np.random.choice([0, 1, 2], n_samples)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   📊 Test data: {X_train.shape[0]} train, {X_val.shape[0]} validation")
    print(f"   🎯 Classes: {len(np.unique(y))}")
    print(f"   📝 Features: {len(X.columns)} ({X.select_dtypes(include=['category']).shape[1]} categorical)")
    
    # Create test config
    config = Config(
        train_path="dummy.csv",
        test_path="dummy.csv",
        use_gpu=False,  # Use CPU for testing
        parallel_folds=False
    )
    
    # Get model configurations
    model_configs = get_model_configurations(config)
    
    print(f"\n🚀 Testing {len(model_configs)} model configurations...")
    
    successful_models = []
    failed_models = []
    
    for model_name, model_config in model_configs.items():
        try:
            print(f"\n   🔧 Testing {model_name}...")
            
            model = model_config["model"]
            fit_args = model_config["fit_args"]
            
            # Clone the model
            from sklearn.base import clone
            test_model = clone(model)
            
            # Test fitting
            if fit_args:
                # Handle different model types
                if ('LightGBM' in str(type(test_model)) or 
                    'XGB' in str(type(test_model)) or 
                    'CatBoost' in str(type(test_model))):
                    test_model.fit(X_train, y_train, **fit_args, eval_set=[(X_val, y_val)])
                else:
                    # Filter out eval_set specific parameters
                    filtered_fit_args = {k: v for k, v in fit_args.items() 
                                       if k not in ['early_stopping_rounds', 'eval_set']}
                    
                    # Handle RandomForest categorical data
                    if 'RandomForest' in str(type(test_model)):
                        # Convert categorical columns for RandomForest
                        X_train_processed = X_train.copy()
                        X_val_processed = X_val.copy()
                        
                        categorical_columns = X_train.select_dtypes(include=['category', 'object']).columns
                        if len(categorical_columns) > 0:
                            from sklearn.preprocessing import LabelEncoder
                            for col in categorical_columns:
                                le = LabelEncoder()
                                combined_data = pd.concat([X_train[col], X_val[col]], ignore_index=True)
                                le.fit(combined_data.astype(str))
                                X_train_processed[col] = le.transform(X_train[col].astype(str))
                                X_val_processed[col] = le.transform(X_val[col].astype(str))
                        
                        test_model.fit(X_train_processed, y_train, **filtered_fit_args)
                        # Test predictions
                        _ = test_model.predict_proba(X_val_processed)
                    else:
                        test_model.fit(X_train, y_train, **filtered_fit_args)
                        # Test predictions
                        _ = test_model.predict_proba(X_val)
            else:
                test_model.fit(X_train, y_train)
                # Test predictions
                _ = test_model.predict_proba(X_val)
            
            print(f"   ✅ {model_name} - SUCCESS")
            successful_models.append(model_name)
            
        except Exception as e:
            print(f"   ❌ {model_name} - FAILED: {str(e)}")
            failed_models.append((model_name, str(e)))
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Successful: {len(successful_models)}")
    print(f"   ❌ Failed: {len(failed_models)}")
    
    if successful_models:
        print(f"\n🎉 Working models:")
        for model in successful_models:
            print(f"   - {model}")
    
    if failed_models:
        print(f"\n💥 Failed models:")
        for model, error in failed_models:
            print(f"   - {model}: {error}")
    
    return len(failed_models) == 0

if __name__ == "__main__":
    success = test_model_configs()
    if success:
        print(f"\n🎉 All model configurations are working correctly!")
    else:
        print(f"\n⚠️  Some model configurations need further fixes.") 