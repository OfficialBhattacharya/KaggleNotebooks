"""
Example Usage Script for Enhanced Multi-Class Classification Ensemble Framework
===============================================================================

This script demonstrates how to use the enhanced ensemble framework for 
multiclass classification problems.
"""

from enhanced_multiclass_ensemble import Config, main
import os

def run_ensemble_example():
    """
    Example function showing how to configure and run the ensemble framework
    """
    
    print("Enhanced Multi-Class Classification Ensemble Framework")
    print("=" * 60)
    
    # Configure the ensemble framework
    config = Config(
        # Data paths - UPDATE THESE TO YOUR ACTUAL PATHS
        train_path="train.csv",                    # Path to training data
        test_path="test.csv",                      # Path to test data  
        original_path="",                          # Optional: additional training data
        sample_sub_path="",                        # Optional: sample submission format
        
        # Column specifications
        target_col="Fertilizer Name",              # Target column name
        id_col="id",                               # ID column name
        
        # Cross-validation settings
        n_folds=5,                                 # Number of CV folds
        seed=42,                                   # Random seed for reproducibility
        shuffle=True,                              # Shuffle data in CV
        
        # Prediction settings
        top_k_predictions=3,                       # Number of top predictions (MAP@3)
        
        # GPU settings (optimized for T4 x 2 setup)
        use_gpu=True,                              # Enable GPU acceleration
        gpu_devices=[0, 1],                        # Use both T4 GPUs
        parallel_folds=True,                       # Parallel fold processing
        max_gpu_memory_fraction=0.8,               # GPU memory usage limit
        
        # Ensemble settings
        use_class_weights=True,                    # Use balanced class weights
        scale_features_for_lr=True,                # Scale features for logistic regression
        
        # Model training settings
        early_stopping_rounds=100,                 # Early stopping for tree models
        verbose_eval=1000,                         # Logging frequency
        
        # Output settings
        output_dir="ensemble_outputs",             # Output directory
        save_oof_predictions=True,                 # Save out-of-fold predictions
        save_feature_importance=True,              # Save feature importance
        
        # AutoGluon settings (if available)
        autogluon_time_limit=3600,                 # Time limit in seconds (1 hour)
        autogluon_presets="good_quality"           # AutoGluon preset
    )
    
    # Validate that data files exist
    print("\nValidating data files...")
    if not os.path.exists(config.train_path):
        print(f"ERROR: Training file not found: {config.train_path}")
        print("Please update the train_path in the config to point to your training data")
        return
    
    if not os.path.exists(config.test_path):
        print(f"ERROR: Test file not found: {config.test_path}")
        print("Please update the test_path in the config to point to your test data")
        return
    
    print("✓ Data files found")
    
    # Print configuration summary
    print(f"\nConfiguration Summary:")
    print(f"Training data: {config.train_path}")
    print(f"Test data: {config.test_path}")
    print(f"Target column: {config.target_col}")
    print(f"ID column: {config.id_col}")
    print(f"Cross-validation folds: {config.n_folds}")
    print(f"Top-K predictions: {config.top_k_predictions}")
    print(f"Output directory: {config.output_dir}")
    
    # Check GPU status
    print(f"\nChecking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"🎮 {gpu_count} CUDA GPU(s) detected:")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory}GB)")
            
            print(f"✅ GPU acceleration enabled for config.gpu_devices: {config.gpu_devices}")
            if config.parallel_folds and len(config.gpu_devices) >= 2:
                print(f"⚡ Parallel fold processing enabled across {len(config.gpu_devices)} GPUs")
                print(f"🚀 Expected significant speed improvement!")
        else:
            print("❌ CUDA not available - will run on CPU")
            config.use_gpu = False
            config.parallel_folds = False
    except ImportError:
        print("❌ PyTorch not installed - will run on CPU")
        config.use_gpu = False
        config.parallel_folds = False
    
    # Check available models
    print(f"\nChecking available ML libraries...")
    try:
        import lightgbm
        gpu_support = " (GPU-accelerated)" if config.use_gpu else " (CPU-only)"
        print(f"✓ LightGBM available{gpu_support}")
    except ImportError:
        print("✗ LightGBM not available (install with: pip install lightgbm)")
    
    try:
        import xgboost
        gpu_support = " (GPU-accelerated)" if config.use_gpu else " (CPU-only)"
        print(f"✓ XGBoost available{gpu_support}")
    except ImportError:
        print("✗ XGBoost not available (install with: pip install xgboost)")
    
    try:
        import catboost
        gpu_support = " (GPU-accelerated)" if config.use_gpu else " (CPU-only)"
        print(f"✓ CatBoost available{gpu_support}")
    except ImportError:
        print("✗ CatBoost not available (install with: pip install catboost)")
    
    try:
        import autogluon
        print("✓ AutoGluon available")
    except ImportError:
        print("✗ AutoGluon not available (install with: pip install autogluon)")
    
    print("✓ Scikit-learn models (RandomForest, LogisticRegression) always available")
    
    # Run the ensemble
    print(f"\nStarting ensemble training...")
    print("This may take a while depending on your data size and available models.")
    
    try:
        main(config)
        print(f"\n🎉 Ensemble training completed successfully!")
        print(f"📁 Check the '{config.output_dir}' directory for results:")
        print(f"   - submission_ensemble_*.csv: Final predictions")
        print(f"   - ensemble_log_*.log: Detailed training log")
        print(f"   - cv_scores.csv: Cross-validation scores")
        print(f"   - experiment_summary.json: Summary statistics")
        
    except Exception as e:
        print(f"\n❌ Error during ensemble training: {str(e)}")
        print("Please check the log files for more details.")


if __name__ == "__main__":
    run_ensemble_example() 