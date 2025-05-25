# Quick test to verify the XGBoost callback fix
import sys
import os
sys.path.append('dataModeller')

def test_xgboost_callback_fix():
    """
    Quick test to verify the XGBoost callback fix works.
    """
    print("🧪 Testing XGBoost Callback Fix")
    print("="*40)
    
    try:
        from modelCompare import create_gpu_optimized_model, fitPlotAndPredict
        import numpy as np
        import pandas as pd
        
        print("✅ Imports successful")
        
        # Create minimal test data
        np.random.seed(42)
        n_samples = 100
        X_train = np.random.randn(n_samples, 5)
        y_train = np.random.randn(n_samples)
        X_test = np.random.randn(20, 5)
        
        # Convert to DataFrame
        X_train_df = pd.DataFrame(X_train, columns=[f'Feature_{i}' for i in range(5)])
        X_test_df = pd.DataFrame(X_test, columns=[f'Feature_{i}' for i in range(5)])
        
        print("✅ Test data created")
        
        # Create model with minimal parameters for quick test
        model_xgb = create_gpu_optimized_model('xgboost', 
                                           n_estimators=10,  # Very small for quick test
                                           max_depth=3,
                                           learning_rate=0.3,
                                           random_state=42,
                                           early_stopping_rounds=5)
        
        if model_xgb is None:
            print("❌ Model creation failed")
            return False
        
        print("✅ Model created successfully")
        
        # Test the training with callback
        print("\n🚀 Testing training with callback fix...")
        
        results, predictions = fitPlotAndPredict(
            X_train_df, y_train, X_test_df, model_xgb, 
            'TestDataset', 'none', 'none', 'TestModel', '2-fold CV'
        )
        
        print("✅ Training completed successfully!")
        print(f"📊 Results shape: {results.shape}")
        print(f"🔮 Predictions shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_xgboost_callback_fix()
    if success:
        print("\n🎉 All tests passed! The callback fix is working correctly.")
    else:
        print("\n💥 Test failed. Please check the error messages above.") 