#!/usr/bin/env python3
"""
Ultra Clean Enhanced Multi-Class Classification Ensemble Runner
- Maximum compilation warning suppression
- Detailed real-time progress tracking
- Professional console output with emojis
- Complete silence during compilation
"""

import os
import sys
import warnings
import subprocess
from pathlib import Path

def setup_ultra_clean_environment():
    """Set up ultra-clean environment with maximum warning suppression"""
    print("🔧 Setting up ultra-clean environment...")
    
    # Suppress all Python warnings
    warnings.filterwarnings("ignore")
    
    # Environment variables for complete silence
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['AUTOGLUON_LITE'] = '1'
    
    # Compilation warning suppression
    os.environ['CFLAGS'] = '-w -O0 -Wno-unused-function -Wno-unused-variable'
    os.environ['CPPFLAGS'] = '-w -O0 -Wno-unused-function -Wno-unused-variable'
    os.environ['CC'] = 'gcc -w -O0'
    os.environ['CXX'] = 'g++ -w -O0'
    os.environ['LDFLAGS'] = '-w'
    
    # Library-specific suppressions
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['NUMBA_DISABLE_JIT'] = '1'
    os.environ['NUMBA_WARNINGS'] = '0'
    os.environ['SKLEARN_SILENCE_WARNINGS'] = '1'
    
    # CUDA suppressions
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # T4 x 2 setup
    
    print("✅ Environment configured for ultra-clean output")

def install_packages_silently():
    """Install required packages with complete output suppression"""
    print("📦 Installing/checking required packages...")
    print("⏳ This may take a few moments for first-time setup...")
    
    packages_to_check = [
        'lightgbm',
        'xgboost', 
        'catboost',
        'autogluon',
        'torch',
        'scikit-learn',
        'pandas',
        'numpy'
    ]
    
    # Redirect all output to null during package installation
    devnull = open(os.devnull, 'w')
    
    for i, package in enumerate(packages_to_check, 1):
        try:
            print(f"    📋 Checking {package} ({i}/{len(packages_to_check)})...")
            
            # Try importing to check if installed
            if package == 'torch':
                import torch
            elif package == 'lightgbm':
                import lightgbm
            elif package == 'xgboost':
                import xgboost
            elif package == 'catboost':
                import catboost
            elif package == 'autogluon':
                import autogluon
            elif package == 'scikit-learn':
                import sklearn
            elif package == 'pandas':
                import pandas
            elif package == 'numpy':
                import numpy
                
            print(f"    ✅ {package} is available")
            
        except ImportError:
            print(f"    📥 Installing {package}...")
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package, '--quiet', '--no-warn-script-location'
                ], stdout=devnull, stderr=devnull, check=True)
                print(f"    ✅ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"    ⚠️  Warning: Could not install {package}")
    
    devnull.close()
    print("✅ Package verification completed")

def run_ensemble():
    """Run the enhanced ensemble with ultra-clean output"""
    print("\n🚀 Starting Enhanced Multi-Class Classification Ensemble...")
    
    # Import the enhanced ensemble (this is where most compilation happens)
    print("⚙️  Importing ensemble framework (may show some compilation messages)...")
    print("💡 Note: Any compilation warnings below will be suppressed in actual training")
    print("-" * 80)
    
    # Suppress stderr during import to hide compilation warnings
    import contextlib
    import io
    
    with contextlib.redirect_stderr(io.StringIO()):
        from enhanced_multiclass_ensemble import main, Config
    
    print("-" * 80)
    print("✅ Framework imported successfully!")
    
    # Configuration
    config = Config(
        # Update these paths to match your data
        train_path="train.csv",  # Update this path
        test_path="test.csv",    # Update this path
        original_path="",        # Optional: add if you have original data
        sample_sub_path="",      # Optional: add if you have sample submission
        
        # Model configuration
        target_col="Fertilizer Name",
        id_col="id",
        n_folds=5,
        seed=42,
        top_k_predictions=3,
        
        # GPU acceleration settings (optimized for T4 x 2)
        use_gpu=True,
        gpu_devices=[0, 1],
        parallel_folds=True,
        
        # Output settings
        output_dir="ensemble_outputs"
    )
    
    print("\n📋 Configuration Summary:")
    print(f"   📂 Output directory: {config.output_dir}")
    print(f"   🎲 Random seed: {config.seed}")
    print(f"   🔄 Cross-validation folds: {config.n_folds}")
    print(f"   📊 Top-K predictions: {config.top_k_predictions}")
    print(f"   🎮 GPU acceleration: {'ENABLED' if config.use_gpu else 'DISABLED'}")
    if config.use_gpu:
        print(f"   🔥 GPU devices: {config.gpu_devices}")
        print(f"   ⚡ Parallel folds: {'ENABLED' if config.parallel_folds else 'DISABLED'}")
    
    # Check if data files exist
    data_files_exist = True
    if not Path(config.train_path).exists():
        print(f"❌ Training data not found: {config.train_path}")
        data_files_exist = False
    if not Path(config.test_path).exists():
        print(f"❌ Test data not found: {config.test_path}")
        data_files_exist = False
    
    if not data_files_exist:
        print("\n⚠️  DATA FILES NOT FOUND!")
        print("Please update the paths in the configuration section of this script.")
        print("\nExample configuration:")
        print('    train_path="MultiClassification_Fertilizers/train.csv"')
        print('    test_path="MultiClassification_Fertilizers/test.csv"')
        return
    
    print(f"\n✅ All data files found - starting ensemble training!")
    
    try:
        # Run the main ensemble
        main(config)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Step 1: Setup ultra-clean environment
        setup_ultra_clean_environment()
        
        # Step 2: Install/check packages silently
        install_packages_silently()
        
        # Step 3: Run the ensemble
        run_ensemble()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        raise 