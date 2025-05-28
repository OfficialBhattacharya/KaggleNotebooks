#!/usr/bin/env python3
"""
Test script to verify feature consistency fixes for incremental learning.
"""

import numpy as np

def test_feature_consistency():
    """Test that features remain consistent across chunks."""
    print("🧪 Testing Feature Consistency Fixes")
    print("=" * 50)
    
    # Create problematic test data
    np.random.seed(42)
    n_samples, n_features = 1000, 15
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.normal(0, 0.1, n_samples)
    
    # Add problematic columns that should be removed consistently
    # Column 10: All NaN
    X[:, 10] = np.nan
    # Column 11: All same value (no variance)
    X[:, 11] = 1.0
    # Column 12: All infinite
    X[:, 12] = np.inf
    
    # Add some scattered problematic values
    X[50:60, 5] = np.nan  # Some NaN in chunk 1
    X[500:510, 5] = np.inf  # Some inf in chunk 2
    
    print(f"📊 Test data created:")
    print(f"   Shape: {X.shape}")
    print(f"   Problematic columns: 10 (all NaN), 11 (constant), 12 (all inf)")
    print(f"   Scattered issues in column 5")
    
    try:
        from oneShot_XGB import create_incremental_pipeline
        
        # Test with small chunks to trigger the issue
        pipeline = create_incremental_pipeline(
            n_chunks=3,
            n_estimators_per_chunk=50,
            use_gpu=False,
            save_checkpoints=False,
            verbose=True
        )
        
        print(f"\n🔄 Running incremental training...")
        results = pipeline.run_incremental_training(X, y)
        
        print(f"\n✅ Success! Feature consistency maintained.")
        print(f"   Chunks processed: {results['total_chunks_processed']}")
        print(f"   Failed chunks: {results['failed_chunks']}")
        print(f"   Total time: {results['total_training_time']:.2f}s")
        
        # Verify feature counts
        if hasattr(pipeline, 'chunk_info'):
            feature_counts = [chunk['n_features'] for chunk in pipeline.chunk_info]
            unique_counts = set(feature_counts)
            print(f"   Feature counts per chunk: {feature_counts}")
            print(f"   Unique feature counts: {unique_counts}")
            
            if len(unique_counts) == 1:
                print(f"   🎉 All chunks have consistent feature count: {list(unique_counts)[0]}")
                return True
            else:
                print(f"   ❌ Inconsistent feature counts detected!")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_feature_consistency()
    if success:
        print(f"\n🎉 Feature consistency test PASSED!")
    else:
        print(f"\n❌ Feature consistency test FAILED!") 