"""
Test script using the exact usage pattern requested by the user.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from dataModeller.getModelReadyData import getModelReadyData
from dataModeller.modelCompare import fitPlotAndPredict

def create_test_dataset():
    """Create a test dataset with Calories column."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic features for calorie prediction
    data = {
        'Duration': np.random.uniform(10, 120, n_samples),
        'Heart_Rate': np.random.uniform(60, 180, n_samples),
        'Body_Temp': np.random.uniform(36.5, 39.0, n_samples),
        'Age': np.random.randint(18, 65, n_samples),
        'Weight': np.random.uniform(50, 120, n_samples),
        'Height': np.random.uniform(150, 200, n_samples),
        'Gender': np.random.choice([0, 1], n_samples)
    }
    
    # Create realistic calorie calculation
    calories = (
        data['Duration'] * 0.8 +
        (data['Heart_Rate'] - 60) * 0.1 +
        data['Weight'] * 0.3 +
        data['Age'] * 0.05 +
        data['Gender'] * 20 +
        np.random.normal(0, 10, n_samples)
    )
    
    # Ensure calories are positive and realistic
    calories = np.clip(calories, 10, 500)
    data['Calories'] = calories
    
    return pd.DataFrame(data)

def main():
    """Test the exact usage pattern."""
    print("="*60)
    print("TESTING EXACT USAGE PATTERN")
    print("="*60)
    
    # Create test datasets
    train_dataset0 = create_test_dataset()
    test_dataset0 = create_test_dataset()  # In practice, this would be different
    
    print(f"Train dataset shape: {train_dataset0.shape}")
    print(f"Test dataset shape: {test_dataset0.shape}")
    print(f"Calories range: [{train_dataset0['Calories'].min():.2f}, {train_dataset0['Calories'].max():.2f}]")
    
    # EXACT USAGE PATTERN AS REQUESTED
    target = 'Calories'
    datasetName = 'Raw'
    xTransform = 'none'
    yTransform = 'none'
    X_train, y_train, X_test = getModelReadyData(train_dataset0, test_dataset0, xTransform, yTransform, target) 
    
    modelName = 'OLSWithIntercept_Ver0'
    cvScheme = '5-fold CV'
    
    model = LinearRegression(fit_intercept=True)
    results, predictions = fitPlotAndPredict(X_train, y_train, X_test, model, datasetName, xTransform, yTransform, modelName, cvScheme)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(results.to_string(index=False))
    
    print(f"\nPredictions Summary:")
    print(f"Shape: {predictions.shape}")
    print(f"Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"Mean: {predictions.mean():.6f}")
    print(f"Std: {predictions.std():.6f}")
    print(f"Sample predictions: {predictions[:5]}")
    
    # Check if predictions are diverse
    unique_predictions = len(np.unique(np.round(predictions, 6)))
    print(f"Number of unique predictions: {unique_predictions}")
    
    if unique_predictions > 1:
        print("✅ SUCCESS: Function works with exact usage pattern!")
    else:
        print("❌ WARNING: All predictions are identical!")
    
    return results, predictions

if __name__ == "__main__":
    results, predictions = main()
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60) 