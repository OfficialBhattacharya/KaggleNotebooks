#!/usr/bin/env python3
"""
Clean Runner for Enhanced Multi-Class Classification Ensemble Framework
========================================================================

This script provides a clean, minimal console output experience while running
the ensemble framework with GPU acceleration.

All warnings and verbose training output are suppressed for a clean experience.
"""

import os
import sys
import warnings
from pathlib import Path

# Comprehensive warning suppression BEFORE importing any ML libraries
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CFLAGS'] = '-w'
os.environ['CPPFLAGS'] = '-w'

# Suppress stdout/stderr during imports to prevent compilation warnings
import contextlib
import io

print("🚀 Starting Enhanced Multi-Class Ensemble Framework...")
print("📥 Loading libraries (suppressing compilation warnings)...")

# Import with suppressed output
with contextlib.redirect_stderr(io.StringIO()):
    from enhanced_multiclass_ensemble import Config, main

def run_clean_ensemble():
    """
    Run the ensemble framework with clean, minimal output
    """
    
    print("✅ Libraries loaded successfully!")
    print("")
    
    # Configure the ensemble framework for T4 x 2 GPU setup
    config = Config(
        # Data paths - UPDATE THESE TO YOUR ACTUAL PATHS
        train_path="train.csv",                    # Your training data
        test_path="test.csv",                      # Your test data  
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
        verbose_eval=1000,                         # Logging frequency (suppressed)
        
        # Output settings
        output_dir="ensemble_outputs",             # Output directory
        save_oof_predictions=True,                 # Save out-of-fold predictions
        save_feature_importance=True,              # Save feature importance
        
        # AutoGluon settings (if available)
        autogluon_time_limit=3600,                 # Time limit in seconds (1 hour)
        autogluon_presets="good_quality"           # AutoGluon preset
    )
    
    # Validate that data files exist
    if not os.path.exists(config.train_path):
        print(f"❌ Error: Training file not found: {config.train_path}")
        print("   Please update the train_path in the config to point to your training data")
        return False
    
    if not os.path.exists(config.test_path):
        print(f"❌ Error: Test file not found: {config.test_path}")
        print("   Please update the test_path in the config to point to your test data")
        return False
    
    print("📊 Data files validated successfully!")
    
    # Check GPU status
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"🎮 {gpu_count} CUDA GPU(s) detected")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory}GB)")
            
            if config.parallel_folds and len(config.gpu_devices) >= 2:
                print(f"⚡ Parallel processing enabled - expect 3-5x speedup!")
        else:
            print("💻 Using CPU (no CUDA GPUs available)")
            config.use_gpu = False
            config.parallel_folds = False
    except ImportError:
        print("💻 Using CPU (PyTorch not installed)")
        config.use_gpu = False
        config.parallel_folds = False
    
    print("")
    print("🚀 Starting ensemble training...")
    print("   (All compilation warnings and verbose output suppressed)")
    print("   Check the log file for detailed progress")
    print("")
    
    try:
        # Run the ensemble with suppressed output
        main(config)
        
        print("")
        print("🎉 Ensemble training completed successfully!")
        print("")
        print("📁 Results saved to:")
        print(f"   📊 Submission: {config.output_dir}/submission_ensemble_*.csv")
        print(f"   📋 Detailed log: {config.output_dir}/ensemble_log_*.log")
        print(f"   📈 CV scores: {config.output_dir}/cv_scores.csv")
        print("")
        return True
        
    except Exception as e:
        print(f"❌ Error during ensemble training: {str(e)}")
        print("   Check the log file for detailed error information")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("🧠 ENHANCED MULTI-CLASS CLASSIFICATION ENSEMBLE FRAMEWORK")
    print("   Clean Mode - Suppressed Warnings & Minimal Output")
    print("=" * 80)
    print("")
    
    success = run_clean_ensemble()
    
    if success:
        print("✅ All done! Your ensemble model is ready.")
    else:
        print("❌ Training failed. Please check the configuration and try again.")
    
    print("")
    print("=" * 80) 