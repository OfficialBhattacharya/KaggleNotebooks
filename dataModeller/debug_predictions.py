"""
Debug script to identify why all predictions are the same.
Run this script with your actual data to diagnose the issue.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from dataModeller.modelCompare import validate_input_data, test_fitPlotAndPredict_with_meaningful_data

def debug_your_data(X_train, y_train, X_test):
    """
    Debug function to identify issues with your specific data.
    """
    print("🔍 DEBUGGING YOUR DATA")
    print("="*60)
    
    # Step 1: Validate your input data
    print("Step 1: Validating your input data...")
    validate_input_data(X_train, y_train, X_test)
    
    # Step 2: Test with a simple linear relationship
    print("\nStep 2: Testing with a simple model...")
    
    # Convert to numpy arrays
    if hasattr(X_train, 'values'):
        X_np = X_train.values
    else:
        X_np = np.array(X_train)
    
    if hasattr(y_train, 'values'):
        y_np = y_train.values
    else:
        y_np = np.array(y_train)
    
    if hasattr(X_test, 'values'):
        X_test_np = X_test.values
    else:
        X_test_np = np.array(X_test)
    
    # Fit a simple linear regression
    model = LinearRegression()
    model.fit(X_np, y_np)
    
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")
    print(f"Max coefficient magnitude: {np.abs(model.coef_).max():.6f}")
    
    # Make predictions
    y_pred = model.predict(X_test_np)
    print(f"Raw predictions range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    print(f"Raw predictions std: {y_pred.std():.6f}")
    print(f"Sample raw predictions: {y_pred[:5]}")
    
    # Check if predictions are all the same
    if y_pred.std() < 1e-10:
        print("\n❌ ISSUE CONFIRMED: All predictions are identical!")
        print("\nPossible causes:")
        print("1. Your features have no predictive power for the target")
        print("2. Features are all constant or nearly constant")
        print("3. Target variable has no relationship with features")
        
        # Check feature correlations
        correlations = []
        for i in range(X_np.shape[1]):
            corr = np.corrcoef(X_np[:, i], y_np)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        if correlations:
            max_corr = np.max(np.abs(correlations))
            print(f"\nMaximum feature-target correlation: {max_corr:.6f}")
            if max_corr < 0.01:
                print("🔍 ROOT CAUSE: Features have virtually no correlation with target!")
                print("   Your X_train data doesn't contain information to predict y_train")
        
    else:
        print("\n✅ Predictions are diverse - the issue might be elsewhere")
    
    # Step 3: Test with known good data
    print("\n" + "="*60)
    print("Step 3: Testing with known good data for comparison...")
    test_fitPlotAndPredict_with_meaningful_data()

def quick_fix_suggestion():
    """
    Provide quick fix suggestions.
    """
    print("\n" + "="*60)
    print("🛠️  QUICK FIX SUGGESTIONS")
    print("="*60)
    print("1. Check your data generation/loading process")
    print("2. Ensure X_train features are related to y_train target")
    print("3. Verify that features are not all constant")
    print("4. Consider feature engineering or selection")
    print("5. Try a different model (Random Forest, XGBoost)")
    print("6. Check for data leakage or preprocessing issues")
    print("\nExample of creating meaningful test data:")
    print("""
# Create data with clear relationships
X_train = np.random.randn(1000, 5)
# Make y depend on X features
y_train = 2*X_train[:, 0] + 1.5*X_train[:, 1] + np.random.randn(1000)*0.1 + 3
y_train = np.maximum(y_train, 0.1)  # Ensure positive for log transform
X_test = np.random.randn(200, 5)
""")

if __name__ == "__main__":
    print("This is a debug script. Import and use the functions with your data:")
    print("from debug_predictions import debug_your_data")
    print("debug_your_data(X_train, y_train, X_test)")
    quick_fix_suggestion() 