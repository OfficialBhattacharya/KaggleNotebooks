"""
Performance-Based Ensemble Strategy Demo

This script demonstrates how to use the performance-based ensemble strategy,
which automatically calculates optimal weights based on cross-validation performance.

Key Features:
- Automatic weight calculation from CV scores
- No manual weight specification required
- Higher performing models get higher weights
- Uses softmax transformation for smooth weight distribution
"""

from XGBoost_Modeller import *

def create_performance_based_config():
    """Create a configuration using performance-based ensemble strategy."""
    
    # Sample dataset configurations (adjust paths as needed)
    train_datasets = [
        DatasetConfig(
            file_path="train.parquet",
            feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"],
            id_columns=["timestamp"],
            dataset_name="Train Features",
            is_required=True
        )
    ]
    
    test_datasets = [
        DatasetConfig(
            file_path="test.parquet",
            feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
            id_columns=["ID"],
            dataset_name="Test Features",
            is_required=True
        )
    ]
    
    # Performance-based ensemble configuration
    config = ModelConfig(
        TRAIN_DATASETS=train_datasets,
        TEST_DATASETS=test_datasets,
        
        # Key setting: Performance-based ensemble strategy
        ENSEMBLE_STRATEGY="performance_based",
        
        # No need to specify weights - they are calculated automatically!
        # The system will:
        # 1. Train all 6 models (3 XGBoost + 3 LightGBM with different data percentages)
        # 2. Calculate cross-validation scores for each model
        # 3. Convert scores to weights using softmax transformation
        # 4. Higher performing models automatically get higher weights
        
        # Standard configuration
        TARGET_COLUMN="label",
        ID_COLUMN="ID",
        TIMESTAMP_COLUMN="timestamp",
        SELECTED_FEATURES=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
        
        # Model parameters (example)
        XGB_PARAMS={
            "tree_method": "gpu_hist",
            "learning_rate": 0.02,
            "max_depth": 20,
            "n_estimators": 1000,
            "random_state": 42,
            "verbosity": 0
        },
        LGBM_PARAMS={
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
            "num_leaves": 37,
            "n_estimators": 800,
            "random_state": 42,
            "verbose": -1
        },
        
        # Output files
        SUBMISSION_FILENAME="performance_based_submission.csv",
        RESULTS_FILENAME="performance_based_results.csv",
        
        # Other settings
        N_FOLDS=5,
        RANDOM_STATE=42,
        USE_MULTI_GPU=True,
        REDUCE_MEMORY_USAGE=True
    )
    
    return config

def demonstrate_weight_calculation():
    """Demonstrate how performance-based weights are calculated."""
    
    print("🎯 PERFORMANCE-BASED WEIGHT CALCULATION EXAMPLE")
    print("="*80)
    
    # Example CV scores from 6 models
    example_scores = {
        "XGB_Full_Dataset_100%": 0.6543,
        "XGB_Recent_Data_75%": 0.6512, 
        "XGB_Recent_Data_50%": 0.6489,
        "LGB_Full_Dataset_100%": 0.6521,
        "LGB_Recent_Data_75%": 0.6498,
        "LGB_Recent_Data_50%": 0.6467
    }
    
    print("📊 Example CV Scores:")
    for model, score in example_scores.items():
        print(f"   {model}: {score:.6f}")
    
    # Calculate performance-based weights
    scores_array = np.array(list(example_scores.values()))
    
    # Softmax transformation (same as used in the pipeline)
    epsilon = 1e-8
    exp_scores = np.exp((scores_array - scores_array.max()) / 0.1)  # Temperature = 0.1
    weights = exp_scores / (exp_scores.sum() + epsilon)
    
    print("\n⚖️  Calculated Performance-Based Weights:")
    for i, (model, score) in enumerate(example_scores.items()):
        print(f"   {model}: {weights[i]:.4f} (score: {score:.6f})")
    
    print(f"\n📈 Weight Statistics:")
    print(f"   Best model weight: {weights.max():.4f}")
    print(f"   Worst model weight: {weights.min():.4f}")
    print(f"   Weight range: {weights.max() - weights.min():.4f}")
    print(f"   Total weight: {weights.sum():.6f}")
    
    # Show how weights relate to performance
    best_idx = np.argmax(scores_array)
    worst_idx = np.argmin(scores_array)
    best_model = list(example_scores.keys())[best_idx]
    worst_model = list(example_scores.keys())[worst_idx]
    
    print(f"\n🏆 Best performing model: {best_model}")
    print(f"   Score: {scores_array[best_idx]:.6f}")
    print(f"   Weight: {weights[best_idx]:.4f}")
    
    print(f"\n📉 Worst performing model: {worst_model}")
    print(f"   Score: {scores_array[worst_idx]:.6f}")
    print(f"   Weight: {weights[worst_idx]:.4f}")
    
    weight_ratio = weights[best_idx] / weights[worst_idx]
    print(f"\n📊 Best-to-worst weight ratio: {weight_ratio:.2f}x")

def compare_ensemble_strategies():
    """Compare different ensemble strategies."""
    
    print("\n🔍 ENSEMBLE STRATEGY COMPARISON")
    print("="*80)
    
    strategies = [
        {
            "name": "Equal Weights (Baseline)",
            "description": "All 6 models get equal weight (1/6 each)",
            "pros": ["Simple", "No configuration needed"],
            "cons": ["Ignores model performance differences"]
        },
        {
            "name": "Learner-Level Weights",
            "description": "Manually set weights for XGBoost vs LightGBM ensembles",
            "pros": ["Simple algorithm-level control", "Easy to understand"],
            "cons": ["Still equal weights within each algorithm", "Manual tuning needed"]
        },
        {
            "name": "Individual Model Weights",
            "description": "Manually specify weight for each of the 6 models",
            "pros": ["Maximum control", "Can encode domain knowledge"],
            "cons": ["Requires manual tuning", "6 parameters to optimize"]
        },
        {
            "name": "Performance-Based Weights (New!)",
            "description": "Automatically calculate weights from CV performance",
            "pros": ["Automatic optimization", "Data-driven", "No manual tuning"],
            "cons": ["May overfit to CV split", "Less interpretable"]
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   📝 {strategy['description']}")
        print(f"   ✅ Pros: {', '.join(strategy['pros'])}")
        print(f"   ⚠️  Cons: {', '.join(strategy['cons'])}")

def run_performance_based_example():
    """Run the performance-based ensemble example."""
    
    print("🚀 RUNNING PERFORMANCE-BASED ENSEMBLE")
    print("="*80)
    
    # Create configuration
    config = create_performance_based_config()
    
    print("Configuration created:")
    print(f"   Strategy: {config.ENSEMBLE_STRATEGY}")
    print(f"   Datasets: {len(config.TRAIN_DATASETS)} train, {len(config.TEST_DATASETS)} test")
    print(f"   Features: {len(config.SELECTED_FEATURES)}")
    
    # Validate configuration
    is_valid, error_msg = config.validate_weights()
    if is_valid:
        print("   ✅ Configuration validation passed")
    else:
        print(f"   ❌ Configuration validation failed: {error_msg}")
        return
    
    print("\nTo run the full pipeline:")
    print("   pipeline = XGBoostLightGBMPipeline(config)")
    print("   results = pipeline.run_pipeline()")
    
    print("\nExpected output:")
    print("   🎯 Performance-based ensemble: weights calculated from CV scores")
    print("   📊 Individual model scores will be displayed")
    print("   ⚖️  Performance-based weights will be calculated and shown")
    print("   📈 Improvement over simple average will be reported")

def main():
    """Main demo function."""
    
    print("🎯 PERFORMANCE-BASED ENSEMBLE DEMO")
    print("="*80)
    print("This demo shows how to use the new performance-based ensemble strategy")
    print("that automatically calculates optimal weights from cross-validation scores.\n")
    
    # Demonstrate weight calculation
    demonstrate_weight_calculation()
    
    # Compare strategies
    compare_ensemble_strategies()
    
    # Show usage example
    run_performance_based_example()
    
    print("\n💡 KEY BENEFITS:")
    print("="*50)
    print("✅ Zero manual configuration - weights calculated automatically")
    print("✅ Data-driven approach - better models get higher weights")
    print("✅ Robust ensemble - leverages model diversity optimally")
    print("✅ Easy to use - just set ENSEMBLE_STRATEGY='performance_based'")
    
    print("\n🔧 USAGE:")
    print("="*50)
    print("config = ModelConfig(")
    print("    ENSEMBLE_STRATEGY='performance_based',")
    print("    # ... other configuration ...")
    print(")")
    print("pipeline = XGBoostLightGBMPipeline(config)")
    print("results = pipeline.run_pipeline()")

if __name__ == "__main__":
    main() 