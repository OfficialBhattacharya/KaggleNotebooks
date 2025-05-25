"""
Complete example script following the exact usage pattern requested.
This demonstrates the full workflow with getModelReadyData and fitPlotAndPredict.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from dataModeller.getModelReadyData import getModelReadyData
from dataModeller.modelCompare import fitPlotAndPredict

# Initialize dictionaries to store results
ModelParamsDictionary = {}
ModelResultsDictionary = {}
ModelPredictionssDictionary = {}

def create_sample_calories_dataset(n_samples=1000):
    """
    Create a sample dataset that mimics a calories prediction problem.
    """
    np.random.seed(42)
    
    # Create realistic features for calorie prediction
    data = {
        'Duration': np.random.uniform(10, 120, n_samples),  # Exercise duration in minutes
        'Heart_Rate': np.random.uniform(60, 180, n_samples),  # Heart rate
        'Body_Temp': np.random.uniform(36.5, 39.0, n_samples),  # Body temperature
        'Age': np.random.randint(18, 65, n_samples),  # Age
        'Weight': np.random.uniform(50, 120, n_samples),  # Weight in kg
        'Height': np.random.uniform(150, 200, n_samples),  # Height in cm
        'Gender': np.random.choice([0, 1], n_samples)  # 0=Female, 1=Male
    }
    
    # Create realistic calorie calculation
    # Calories burned = f(duration, heart_rate, weight, age, gender) + noise
    calories = (
        data['Duration'] * 0.8 +  # Duration effect
        (data['Heart_Rate'] - 60) * 0.1 +  # Heart rate effect
        data['Weight'] * 0.3 +  # Weight effect
        data['Age'] * 0.05 +  # Age effect
        data['Gender'] * 20 +  # Gender effect
        np.random.normal(0, 10, n_samples)  # Noise
    )
    
    # Ensure calories are positive and realistic (between 10 and 500)
    calories = np.clip(calories, 10, 500)
    data['Calories'] = calories
    
    return pd.DataFrame(data)

def main():
    """
    Main function demonstrating the complete workflow.
    """
    print("="*80)
    print("COMPLETE EXAMPLE: CALORIES PREDICTION WITH FITPLOTANDPREDICT")
    print("="*80)
    
    # Step 1: Create sample dataset
    print("Step 1: Creating sample Calories dataset...")
    train_dataset0 = create_sample_calories_dataset(n_samples=1000)
    
    print(f"Dataset created with shape: {train_dataset0.shape}")
    print(f"Columns: {list(train_dataset0.columns)}")
    print(f"Calories range: [{train_dataset0['Calories'].min():.2f}, {train_dataset0['Calories'].max():.2f}]")
    print(f"Sample data:")
    print(train_dataset0.head())
    
    # Step 2: Set parameters exactly as requested
    print("\nStep 2: Setting parameters...")
    target = 'Calories'
    datasetName = 'Raw'
    xTransform = 'none'
    yTransform = 'none'
    modelName = 'OLSWithIntercept_Ver0'
    cvScheme = '5-fold CV'
    
    print(f"Target: {target}")
    print(f"Dataset Name: {datasetName}")
    print(f"X Transform: {xTransform}")
    print(f"Y Transform: {yTransform}")
    print(f"Model Name: {modelName}")
    print(f"CV Scheme: {cvScheme}")
    
    # Step 3: Prepare data using getModelReadyData
    print("\nStep 3: Preparing data with getModelReadyData...")
    X_train, y_train, X_test = getModelReadyData(
        train_dataset0, train_dataset0, xTransform, yTransform, target
    )
    
    print(f"✓ X_train shape: {X_train.shape}")
    print(f"✓ y_train shape: {y_train.shape}")
    print(f"✓ X_test shape: {X_test.shape}")
    print(f"✓ y_train range after transformation: [{y_train.min():.6f}, {y_train.max():.6f}]")
    
    # Step 4: Create model
    print("\nStep 4: Creating LinearRegression model...")
    model = LinearRegression(fit_intercept=True)
    print(f"✓ Model created: {model}")
    
    # Step 5: Run fitPlotAndPredict
    print("\nStep 5: Running fitPlotAndPredict...")
    print("="*60)
    
    results, predictions = fitPlotAndPredict(
        X_train, y_train, X_test, model, datasetName, 
        xTransform, yTransform, modelName, cvScheme
    )
    
    print("="*60)
    print("✓ fitPlotAndPredict completed successfully!")
    
    # Step 6: Store results in dictionaries as requested
    print("\nStep 6: Storing results in dictionaries...")
    ModelParamsDictionary[modelName] = model
    ModelResultsDictionary[modelName] = results
    ModelPredictionssDictionary[modelName] = predictions
    
    print(f"✓ Model stored in ModelParamsDictionary['{modelName}']")
    print(f"✓ Results stored in ModelResultsDictionary['{modelName}']")
    print(f"✓ Predictions stored in ModelPredictionssDictionary['{modelName}']")
    
    # Step 7: Display final results
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print("\nModel Results:")
    print(results.to_string(index=False))
    
    print(f"\nPredictions Summary:")
    print(f"Shape: {predictions.shape}")
    print(f"Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"Mean: {predictions.mean():.6f}")
    print(f"Std: {predictions.std():.6f}")
    print(f"Sample predictions: {predictions[:5]}")
    
    # Verify predictions are properly clipped
    clipped_low = np.sum(predictions <= 0)
    clipped_high = np.sum(predictions >= 314)
    print(f"\nClipping verification:")
    print(f"Predictions at lower bound (0): {clipped_low}")
    print(f"Predictions at upper bound (314): {clipped_high}")
    
    # Check if predictions are diverse
    unique_predictions = len(np.unique(np.round(predictions, 6)))
    print(f"Number of unique predictions: {unique_predictions}")
    
    if unique_predictions > 1:
        print("✅ SUCCESS: Predictions are diverse and properly transformed!")
    else:
        print("❌ WARNING: All predictions are identical - check your data!")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return ModelParamsDictionary, ModelResultsDictionary, ModelPredictionssDictionary

if __name__ == "__main__":
    # Run the complete example
    model_params, model_results, model_predictions = main()
    
    # Additional verification
    print("\nDictionary contents verification:")
    print(f"ModelParamsDictionary keys: {list(model_params.keys())}")
    print(f"ModelResultsDictionary keys: {list(model_results.keys())}")
    print(f"ModelPredictionssDictionary keys: {list(model_predictions.keys())}")
    
    # Show how to access the stored results
    modelName = 'OLSWithIntercept_Ver0'
    print(f"\nAccessing stored results for '{modelName}':")
    print(f"Model type: {type(model_params[modelName])}")
    print(f"Results shape: {model_results[modelName].shape}")
    print(f"Predictions shape: {model_predictions[modelName].shape}") 