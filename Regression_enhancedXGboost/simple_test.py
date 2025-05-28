import numpy as np
# import pandas as pd  # Temporarily disabled due to numpy compatibility issue

print("Testing ScientificXGBRegressor fixes...")

try:
    from oneShot_XGB import ScientificXGBRegressor, create_incremental_pipeline
    print("✅ Import successful")
    
    # Test data with potential issues
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X.sum(axis=1) + np.random.randn(100) * 0.1
    
    # Add some problematic values
    X[0, 0] = np.nan
    X[1, 1] = np.inf
    y[2] = np.nan
    
    print(f"Test data: X{X.shape}, y{y.shape}")
    print(f"NaN in X: {np.isnan(X).sum()}, Inf in X: {np.isinf(X).sum()}")
    print(f"NaN in y: {np.isnan(y).sum()}")
    
    # Test model creation and fitting
    model = ScientificXGBRegressor(
        n_estimators=50,
        use_gpu=False,
        verbose=True
    )
    
    print("Testing fit...")
    model.fit(X, y)
    print("✅ Model fit successful")
    
    # Test prediction
    pred = model.predict(X)
    print(f"✅ Prediction successful, shape: {pred.shape}")
    
    print("🎉 All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 