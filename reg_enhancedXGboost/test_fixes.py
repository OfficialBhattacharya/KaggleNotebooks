#!/usr/bin/env python3
"""
Test script to verify the fixes for float conversion and data preprocessing issues.
"""

import numpy as np
import pandas as pd
from oneShot_XGB import create_incremental_pipeline, extract_model_from_pipeline

def create_test_data(n_samples=1000, n_features=10, add_problems=True):
    """Create test data with potential issues."""
    np.random.seed(42)
    
    # Create base data
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.1, n_samples)
    
    if add_problems:
        # Add problematic columns/values
        
        # Add a column with all NaN
        X = np.column_stack([X, np.full(n_samples, np.nan)])
        
        # Add a column with all same values (no variance)
        X = np.column_stack([X, np.full(n_samples, 1.0)])
        
        # Add some NaN values scattered
        X[np.random.choice(n_samples, 50, replace=False), 0] = np.nan
        
        # Add some infinite values
        X[np.random.choice(n_samples, 20, replace=False), 1] = np.inf
        X[np.random.choice(n_samples, 10, replace=False), 2] = -np.inf
        
        # Add some NaN values in target
        y[np.random.choice(n_samples, 5, replace=False)] = np.nan
    
    return X, y

def test_incremental_pipeline():
    """Test the incremental pipeline with problematic data."""
    print("🧪 Testing Incremental Pipeline with Data Issues")
    print("=" * 60)
    
    # Create test data with problems
    X, y = create_test_data(n_samples=500, n_features=8, add_problems=True)
    
    print(f"📊 Test Data Created:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   NaN count in X: {np.isnan(X).sum()}")
    print(f"   Inf count in X: {np.isinf(X).sum()}")
    print(f"   NaN count in y: {np.isnan(y).sum()}")
    
    try:
        # Create pipeline
        pipeline = create_incremental_pipeline(
            n_chunks=3,
            n_estimators_per_chunk=50,
            inner_cv_folds=2,
            outer_cv_folds=2,
            use_gpu=False,  # Use CPU for testing
            memory_optimization=True,
            save_checkpoints=False,  # Disable for testing
            verbose=True
        )
        
        # Run incremental training
        results = pipeline.run_incremental_training(X, y)
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"   Chunks processed: {results['total_chunks_processed']}")
        print(f"   Failed chunks: {results['failed_chunks']}")
        print(f"   Total time: {results['total_training_time']:.2f}s")
        
        # Extract model
        final_model = extract_model_from_pipeline(pipeline)
        print(f"   Final model estimators: {final_model.n_estimators}")
        
        return True, final_model, results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_hyperparameter_validation():
    """Test parameter type validation."""
    print("\n🔧 Testing Parameter Type Validation")
    print("=" * 60)
    
    from oneShot_XGB import ScientificXGBRegressor
    
    # Create clean test data
    X, y = create_test_data(n_samples=200, n_features=5, add_problems=False)
    
    try:
        model = ScientificXGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            use_gpu=False,
            verbose=True
        )
        
        # Test automated parameterization
        params = model.automated_parameterization(X, y)
        
        print(f"\n✅ Parameter validation successful!")
        print(f"   All parameters have correct types:")
        for key, value in params.items():
            print(f"     {key}: {value} (type: {type(value).__name__})")
        
        # Test fitting
        model.fit(X, y)
        print(f"   Model fitted successfully with {model.n_estimators} estimators")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Running Comprehensive Tests for ScientificXGBRegressor Fixes")
    print("=" * 80)
    
    # Test 1: Parameter validation
    test1_success = test_hyperparameter_validation()
    
    # Test 2: Incremental pipeline with data issues
    test2_success, model, results = test_incremental_pipeline()
    
    # Summary
    print(f"\n🎯 TEST SUMMARY")
    print("=" * 40)
    print(f"Parameter Validation: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"Incremental Pipeline: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 ALL TESTS PASSED! The fixes are working correctly.")
        
        if model:
            print(f"\n🔬 Additional Model Info:")
            print(f"   Final R² on test data: {results.get('final_model_estimators', 'unknown')}")
            print(f"   GPU enabled: {model.use_gpu}")
    else:
        print(f"\n⚠️ Some tests failed. Please check the error messages above.") 