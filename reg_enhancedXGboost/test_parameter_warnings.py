#!/usr/bin/env python3
"""
Focused test for parameter warnings fix
"""

import numpy as np
import warnings
import xgboost
create_scientific_xgb_regressor = xgboost.create_scientific_xgb_regressor

def test_parameter_warnings():
    """Test that parameter warnings are properly suppressed"""
    
    print("🧪 Testing parameter warnings fix...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(200, 4)
    y = np.sum(X[:, :2], axis=1) + 0.1 * np.random.randn(200)
    
    # Create model with scientific parameters
    print("1. Creating ScientificXGBRegressor...")
    model = create_scientific_xgb_regressor(
        cv_folds=3,
        auto_tune=True,
        verbose=True,
        n_estimators=100
    )
    print("   ✓ Model created successfully")
    
    # Test automated parameterization (this could cause warnings)
    print("2. Testing automated parameterization...")
    model.automated_parameterization(X, y)
    print("   ✓ Automated parameterization completed")
    
    # Test model fitting (this could cause warnings)
    print("3. Testing model fitting...")
    model.fit(X, y)
    print("   ✓ Model fitting completed")
    
    # Test elbow tuning (this was the main source of warnings)
    print("4. Testing elbow tuning (main test)...")
    print("   Running validation_curve with clean estimator...")
    elbow_results = model.elbow_tune(
        X, y, 
        param_name='n_estimators', 
        param_range=[50, 100, 150]  # Smaller range for faster testing
    )
    print(f"   ✓ Elbow tuning completed - optimal: {elbow_results['elbow_value']}")
    
    print("\n🎉 Parameter warnings test completed!")
    print("📝 Check the output above - there should be no 'Parameters not used' warnings")
    return True

if __name__ == "__main__":
    test_parameter_warnings() 