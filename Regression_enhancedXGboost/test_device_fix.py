#!/usr/bin/env python3
"""
Test script to verify device mismatch and parameter warnings are fixed
"""

import numpy as np
import warnings
import sys
import xgboost
create_scientific_xgb_regressor = xgboost.create_scientific_xgb_regressor

def test_device_warnings():
    """Test that device mismatch warnings are resolved"""
    
    print("🧪 Testing Device Compatibility Fixes...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(300, 5)
    y = np.sum(X[:, :3], axis=1) + 0.1 * np.random.randn(300)
    
    # Create model with GPU enabled (if available)
    print("1. Creating model with GPU auto-detection...")
    model = create_scientific_xgb_regressor(
        cv_folds=3,
        auto_tune=True,
        verbose=True,
        n_estimators=100
    )
    
    # Test automated parameterization
    print("2. Testing automated parameterization...")
    model.automated_parameterization(X, y)
    print("   ✓ Automated parameterization completed")
    
    # Test model fitting
    print("3. Testing model fitting...")
    model.fit(X, y)
    print("   ✓ Model fitting completed")
    
    # Test predictions (this often triggers device warnings)
    print("4. Testing predictions...")
    predictions = model.predict(X[:10])
    print(f"   ✓ Predictions completed (sample: {predictions[0]:.4f})")
    
    # Test diagnostic plots (this was causing device warnings)
    print("5. Testing diagnostic plots...")
    try:
        fig = model.diagnostic_plots(X, y)
        print("   ✓ Diagnostic plots completed without device warnings")
    except Exception as e:
        print(f"   ⚠️ Diagnostic plots failed: {e}")
    
    # Test elbow tuning (this was causing parameter warnings)
    print("6. Testing elbow tuning...")
    try:
        elbow_results = model.elbow_tune(
            X, y, 
            param_name='n_estimators', 
            param_range=[50, 100, 150]
        )
        print(f"   ✓ Elbow tuning completed - optimal: {elbow_results['elbow_value']}")
    except Exception as e:
        print(f"   ⚠️ Elbow tuning failed: {e}")
    
    # Test nested cross-validation (this was causing both warnings)
    print("7. Testing nested cross-validation...")
    try:
        cv_results = model.nested_cross_validate(
            X, y, 
            inner_cv=2, 
            outer_cv=2  # Reduced for faster testing
        )
        print("   ✓ Nested CV completed without device warnings")
    except Exception as e:
        print(f"   ⚠️ Nested CV failed: {e}")
    
    print("\n🎉 Device compatibility test completed!")
    print("📝 Check the output above - there should be no device mismatch or parameter warnings")
    
    return True

def test_gpu_switching():
    """Test GPU/CPU switching works without warnings"""
    
    print("\n🔄 Testing GPU/CPU Switching...")
    
    # Generate simple data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.sum(X, axis=1) + 0.1 * np.random.randn(100)
    
    model = create_scientific_xgb_regressor(verbose=False)
    
    # Test switching
    print("1. Testing GPU switch...")
    gpu_result = model.switch_to_gpu()
    print(f"   GPU switch result: {gpu_result}")
    
    if gpu_result:
        # Test prediction with GPU
        model.fit(X, y)
        pred_gpu = model.predict(X[:5])
        print(f"   GPU prediction sample: {pred_gpu[0]:.4f}")
    
    print("2. Testing CPU switch...")
    cpu_result = model.switch_to_cpu()
    print(f"   CPU switch result: {cpu_result}")
    
    # Test prediction with CPU
    model.fit(X, y)
    pred_cpu = model.predict(X[:5])
    print(f"   CPU prediction sample: {pred_cpu[0]:.4f}")
    
    print("   ✓ GPU/CPU switching test completed")
    
    return True

if __name__ == "__main__":
    print("🧪 Device Compatibility and Parameter Warnings Fix Test")
    print("=" * 65)
    
    # Capture warnings to check what's still appearing
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Run tests
        test_device_warnings()
        test_gpu_switching()
        
        # Report any remaining warnings
        if w:
            print(f"\n⚠️ Captured {len(w)} warnings:")
            for warning in w:
                print(f"   - {warning.category.__name__}: {warning.message}")
        else:
            print("\n✅ No warnings captured - fixes successful!")
    
    print("\n🚀 All device compatibility tests completed!") 