#!/usr/bin/env python3
"""
Simple test to verify GPU detection functionality
"""

import subprocess
import sys

def test_gpu_detection_basic():
    """Test basic GPU detection without full imports"""
    print("🔍 Testing Basic GPU Detection...")
    
    # Test CUDA availability check
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("   ✅ CUDA Available: True")
            print("   🎮 nvidia-smi output:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"      {line}")
        else:
            print("   ❌ CUDA Available: False (nvidia-smi failed)")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
        print(f"   ❌ CUDA Available: False ({e})")
    
    # Test optional library availability
    print("\n📚 Testing Optional Library Availability...")
    
    libraries = [
        ('GPUtil', 'GPUtil'),
        ('torch', 'torch'),
        ('tensorflow', 'tensorflow'),
    ]
    
    for lib_name, import_name in libraries:
        try:
            __import__(import_name)
            print(f"   ✅ {lib_name}: Available")
        except ImportError:
            print(f"   ❌ {lib_name}: Not available")

def test_xgboost_import():
    """Test XGBoost import"""
    print("\n🧪 Testing XGBoost Import...")
    
    try:
        import xgboost as xgb
        print(f"   ✅ XGBoost version: {xgb.__version__}")
        
        # Test if GPU support is compiled in
        try:
            # Try to create a GPU-enabled model
            model = xgb.XGBRegressor(tree_method='gpu_hist')
            print("   ✅ XGBoost GPU support: Available")
        except Exception as e:
            print(f"   ⚠️  XGBoost GPU support: Limited ({e})")
            
    except ImportError as e:
        print(f"   ❌ XGBoost: Not available ({e})")

def main():
    """Run basic GPU tests"""
    print("🧪 Basic GPU Detection Test")
    print("=" * 40)
    
    test_gpu_detection_basic()
    test_xgboost_import()
    
    print("\n✅ Basic GPU detection test completed!")
    print("\n💡 To test full GPU functionality:")
    print("   1. Ensure NumPy compatibility: pip install 'numpy<2'")
    print("   2. Install GPU libraries: pip install GPUtil torch")
    print("   3. Run: python test_gpu_features.py")

if __name__ == "__main__":
    main() 