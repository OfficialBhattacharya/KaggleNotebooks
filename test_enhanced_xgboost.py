# Test script for enhanced multi-GPU XGBoost functionality
import sys
import os
sys.path.append('dataModeller')

from modelCompare import create_gpu_optimized_model, fitPlotAndPredict, check_gpu_availability
import numpy as np
import pandas as pd

def test_enhanced_xgboost():
    """
    Test the enhanced XGBoost functionality with multi-GPU support and detailed progress reporting.
    """
    print("🧪 Testing Enhanced XGBoost Multi-GPU Functionality")
    print("="*60)
    
    # Check GPU availability first
    gpu_info = check_gpu_availability()
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    np.random.seed(42)
    n_samples, n_features = 5000, 10
    
    # Create feature data
    X_train = np.random.randn(n_samples, n_features)
    
    # Create target with some relationship to features + noise
    y_train = (2*X_train[:, 0] + 1.5*X_train[:, 1] - 0.8*X_train[:, 2] + 
               0.5*X_train[:, 3] + np.random.randn(n_samples)*0.5 + 10)
    
    # Create test data
    X_test = np.random.randn(1000, n_features)
    
    # Add Duration column for stratified CV
    duration_data = np.random.exponential(scale=2.0, size=n_samples) + 1.0
    X_train_df = pd.DataFrame(X_train, columns=[f'Feature_{i}' for i in range(n_features)])
    X_train_df['Duration'] = duration_data
    
    X_test_df = pd.DataFrame(X_test, columns=[f'Feature_{i}' for i in range(n_features)])
    
    print(f"   Training samples: {n_samples:,}")
    print(f"   Features: {n_features}")
    print(f"   Test samples: {X_test.shape[0]:,}")
    
    # Test parameters
    target = 'TestTarget'
    datasetName = 'Enhanced_Test_Dataset'
    xTransform = 'none'
    yTransform = 'none'
    modelName = 'XGBRegressor_Enhanced_MultiGPU'
    cvScheme = '3-fold CV'  # Use 3 folds for faster testing
    
    print(f"\n🤖 Creating enhanced XGBoost model...")
    
    # Create GPU-optimized XGBoost model
    model_xgb = create_gpu_optimized_model('xgboost', 
                                       max_depth=6,  # Reduced for faster testing
                                       colsample_bytree=0.8, 
                                       subsample=0.9,
                                       n_estimators=500,  # Reduced for faster testing
                                       learning_rate=0.1,  # Increased for faster convergence
                                       gamma=0.01,
                                       max_delta_step=2, 
                                       eval_metric='rmse',
                                       enable_categorical=False, 
                                       random_state=42,
                                       early_stopping_rounds=50)  # Reduced for faster testing
    
    if model_xgb is None:
        print("❌ Failed to create XGBoost model")
        return
    
    print(f"\n🚀 Starting enhanced training with detailed progress reporting...")
    
    # Run enhanced fitPlotAndPredict with Duration-based stratification
    results, predictions = fitPlotAndPredict(
        X_train_df, y_train, X_test_df, model_xgb, datasetName, 
        xTransform, yTransform, modelName, cvScheme,
        time_column='Duration'  # Use Duration column for stratified CV
    )
    
    print(f"\n✅ Training completed successfully!")
    print(f"\n📊 Final Results:")
    print(f"   CV RMSE: {results['CV_RMSE'].iloc[0]:.6f}")
    print(f"   CV RMSLE: {results['CV_RMSLE'].iloc[0]:.6f}")
    print(f"   CV R²: {results['CV_R2'].iloc[0]:.6f}")
    print(f"   Training Time: {results['Total_Time_Minutes'].iloc[0]:.2f} minutes")
    print(f"   Avg Fold Time: {results['Avg_Fold_Time_Seconds'].iloc[0]:.1f} seconds")
    
    print(f"\n🔮 Prediction Summary:")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"   Predictions mean: {predictions.mean():.6f}")
    print(f"   Predictions std: {predictions.std():.6f}")
    
    # Test GPU utilization monitoring
    if gpu_info['cuda_available']:
        print(f"\n🎮 GPU Configuration Summary:")
        print(f"   GPUs Used: {gpu_info['gpu_count']}")
        print(f"   Multi-GPU: {'Yes' if gpu_info['multi_gpu_capable'] else 'No'}")
        print(f"   Total GPU Memory: {gpu_info['total_memory_gb']:.1f} GB")
        for i, gpu_name in enumerate(gpu_info['gpu_names']):
            memory_info = gpu_info['gpu_memory'][i]
            print(f"   GPU {i}: {gpu_name} ({memory_info['total_gb']:.1f} GB)")
    
    return results, predictions

if __name__ == "__main__":
    try:
        results, predictions = test_enhanced_xgboost()
        print(f"\n🎉 Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 