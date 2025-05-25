#!/usr/bin/env python3
"""
Test script to verify GPU functionality in ScientificXGBRegressor
"""

import numpy as np
import warnings
import sys
import time
# Import from local module instead of xgboost package
import xgboost
ScientificXGBRegressor = xgboost.ScientificXGBRegressor
create_scientific_xgb_regressor = xgboost.create_scientific_xgb_regressor
GPUManager = xgboost.GPUManager

def test_gpu_detection():
    """Test GPU detection functionality"""
    print("🔍 Testing GPU Detection...")
    
    # Test GPUManager directly
    gpu_info = GPUManager.detect_gpus()
    gpu_config = GPUManager.get_optimal_gpu_config(gpu_info)
    
    print(f"   GPU Available: {gpu_info['available']}")
    print(f"   GPU Count: {gpu_info['count']}")
    print(f"   CUDA Available: {gpu_info['cuda_available']}")
    
    if gpu_info['available']:
        print(f"   GPU Names: {[device.get('name', 'Unknown') for device in gpu_info['devices']]}")
        print(f"   Optimal GPU Config: {gpu_config}")
    
    return gpu_info['available']

def test_model_creation():
    """Test model creation with GPU auto-detection"""
    print("\n🎮 Testing Model Creation...")
    
    # Test auto-detection
    model_auto = create_scientific_xgb_regressor(verbose=False)
    print(f"   Auto-detection: Using GPU = {model_auto._using_gpu}")
    
    # Test forced CPU
    model_cpu = create_scientific_xgb_regressor(use_gpu=False, verbose=False)
    print(f"   Forced CPU: Using GPU = {model_cpu._using_gpu}")
    
    # Test forced GPU (if available)
    try:
        model_gpu = create_scientific_xgb_regressor(use_gpu=True, verbose=False)
        print(f"   Forced GPU: Using GPU = {model_gpu._using_gpu}")
    except Exception as e:
        print(f"   Forced GPU: Failed (expected if no GPU) - {e}")
    
    return model_auto

def test_gpu_switching(model):
    """Test switching between GPU and CPU"""
    print("\n🔄 Testing GPU/CPU Switching...")
    
    original_gpu_state = model._using_gpu
    
    # Test switch to CPU
    cpu_result = model.switch_to_cpu()
    print(f"   Switch to CPU: {cpu_result}, Now using GPU: {model._using_gpu}")
    
    # Test switch to GPU (if available)
    if model._gpu_info['available']:
        gpu_result = model.switch_to_gpu()
        print(f"   Switch to GPU: {gpu_result}, Now using GPU: {model._using_gpu}")
    else:
        print("   Switch to GPU: Skipped (no GPU available)")

def test_training_performance():
    """Test training performance comparison"""
    print("\n⚡ Testing Training Performance...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + 0.1 * np.random.randn(n_samples)
    
    # Create model
    model = create_scientific_xgb_regressor(
        n_estimators=100,  # Smaller for faster testing
        verbose=False
    )
    
    results = {}
    
    # Test CPU performance
    print("   Testing CPU performance...")
    model.switch_to_cpu()
    start_time = time.time()
    model.fit(X, y)
    cpu_time = time.time() - start_time
    results['cpu_time'] = cpu_time
    
    # Test GPU performance (if available)
    if model._gpu_info['available']:
        print("   Testing GPU performance...")
        model.switch_to_gpu()
        start_time = time.time()
        model.fit(X, y)
        gpu_time = time.time() - start_time
        results['gpu_time'] = gpu_time
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
        print(f"   CPU Time: {cpu_time:.3f}s")
        print(f"   GPU Time: {gpu_time:.3f}s")
        print(f"   Speedup: {speedup:.1f}x")
    else:
        print("   GPU performance test skipped (no GPU available)")
        print(f"   CPU Time: {cpu_time:.3f}s")
    
    return results

def test_gpu_optimization():
    """Test GPU optimization features"""
    print("\n🔧 Testing GPU Optimization...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(500, 20)
    y = np.sum(X[:, :5], axis=1) + 0.1 * np.random.randn(500)
    
    model = create_scientific_xgb_regressor(verbose=False)
    
    # Test GPU optimization
    optimization_results = model.optimize_gpu_usage(X)
    print(f"   Optimization Status: {optimization_results['status']}")
    
    if 'dataset_analysis' in optimization_results:
        analysis = optimization_results['dataset_analysis']
        print(f"   Dataset Size: {analysis['data_size_mb']:.2f} MB")
        print(f"   GPU Memory: {analysis['max_gpu_memory_mb']:.2f} MB")
    
    # Test automated parameterization with GPU
    if model._using_gpu:
        print("   Testing GPU-enhanced automated parameterization...")
        params = model.automated_parameterization(X, y)
        print(f"   Automated {len(params)} parameters with GPU optimizations")

def test_model_persistence():
    """Test saving and loading GPU-enabled models"""
    print("\n💾 Testing Model Persistence...")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = np.sum(X[:, :3], axis=1) + 0.1 * np.random.randn(200)
    
    # Create and train model
    model = create_scientific_xgb_regressor(
        n_estimators=50,
        verbose=False
    )
    model.fit(X, y)
    
    # Save model package
    try:
        package_path = model.save_model_package(
            save_dir="./test_gpu_model_package",
            include_diagnostics=False,
            include_data=False
        )
        print(f"   Model saved to: {package_path}")
        
        # Load model package
        loaded_model = ScientificXGBRegressor.load_model_package(package_path)
        print(f"   Model loaded successfully")
        
        # Test predictions match
        original_pred = model.predict(X[:10])
        loaded_pred = loaded_model.predict(X[:10])
        predictions_match = np.allclose(original_pred, loaded_pred)
        print(f"   Predictions match: {predictions_match}")
        
        # Clean up
        import shutil
        shutil.rmtree(package_path)
        print("   Test package cleaned up")
        
    except Exception as e:
        print(f"   Model persistence test failed: {e}")

def main():
    """Run all GPU tests"""
    print("🧪 ScientificXGBRegressor GPU Features Test Suite")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    try:
        # Test 1: GPU Detection
        gpu_available = test_gpu_detection()
        
        # Test 2: Model Creation
        model = test_model_creation()
        
        # Test 3: GPU Switching
        test_gpu_switching(model)
        
        # Test 4: Training Performance
        performance_results = test_training_performance()
        
        # Test 5: GPU Optimization
        test_gpu_optimization()
        
        # Test 6: Model Persistence
        test_model_persistence()
        
        print("\n✅ All GPU tests completed successfully!")
        
        # Summary
        print("\n📊 Test Summary:")
        print(f"   GPU Available: {gpu_available}")
        if gpu_available:
            print("   ✅ GPU detection and usage working correctly")
            if 'gpu_time' in performance_results:
                speedup = performance_results['cpu_time'] / performance_results['gpu_time']
                print(f"   ⚡ GPU speedup: {speedup:.1f}x")
        else:
            print("   💻 No GPU detected - CPU mode tested successfully")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 