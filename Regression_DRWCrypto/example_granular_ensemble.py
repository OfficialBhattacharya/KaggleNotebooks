"""
Comprehensive Example: Granular Ensemble Control

This example demonstrates the extended ensemble functionality that supports both:
1. Learner-level weights (XGBoost vs LightGBM ensembles)
2. Individual model weights (granular control over all 6 models)

The system trains 6 individual models:
- XGB_Full_Dataset_100%
- XGB_Recent_Data_75% 
- XGB_Recent_Data_50%
- LGB_Full_Dataset_100%
- LGB_Recent_Data_75%
- LGB_Recent_Data_50%

You can now control the ensemble at either the learner level or individual model level.
"""

from XGBoost_Modeller import *

def example_learner_level_ensemble():
    """Example: Learner-level ensemble with custom weights"""
    
    print("="*80)
    print("EXAMPLE 1: LEARNER-LEVEL ENSEMBLE")
    print("="*80)
    
    # Configuration for learner-level ensemble
    config = ModelConfig(
        # ... dataset configurations would go here ...
        TRAIN_DATASETS=[
            DatasetConfig(
                file_path="train.parquet",
                feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"],
                id_columns=["timestamp"],
                dataset_name="Train Features",
                is_required=True
            )
        ],
        TEST_DATASETS=[
            DatasetConfig(
                file_path="test.parquet", 
                feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
                id_columns=["ID"],
                dataset_name="Test Features",
                is_required=True
            )
        ],
        
        # Ensemble strategy: learner-level
        ENSEMBLE_STRATEGY="learner_level",
        CUSTOM_ENSEMBLE_WEIGHTS=[0.7, 0.3],  # 70% XGBoost, 30% LightGBM
        
        # Other settings...
        TARGET_COLUMN="label",
        ID_COLUMN="ID", 
        TIMESTAMP_COLUMN="timestamp",
        SELECTED_FEATURES=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
        
        SUBMISSION_FILENAME="learner_level_ensemble_submission.csv",
        RESULTS_FILENAME="learner_level_ensemble_results.csv"
    )
    
    print("Configuration:")
    print(f"  Strategy: {config.ENSEMBLE_STRATEGY}")
    print(f"  Learner weights: {config.CUSTOM_ENSEMBLE_WEIGHTS}")
    print("  This will:")
    print("    1. Train 3 XGBoost models (100%, 75%, 50% data)")
    print("    2. Train 3 LightGBM models (100%, 75%, 50% data)")
    print("    3. Create XGBoost ensemble (equal weight avg of 3 XGB models)")
    print("    4. Create LightGBM ensemble (equal weight avg of 3 LGB models)")
    print("    5. Final ensemble: 70% XGB ensemble + 30% LGB ensemble")
    
    return config

def example_individual_model_ensemble():
    """Example: Individual model ensemble with granular weights"""
    
    print("="*80)
    print("EXAMPLE 2: INDIVIDUAL MODEL ENSEMBLE")
    print("="*80)
    
    # Configuration for individual model ensemble
    config = ModelConfig(
        # ... dataset configurations would go here ...
        TRAIN_DATASETS=[
            DatasetConfig(
                file_path="train.parquet",
                feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"],
                id_columns=["timestamp"],
                dataset_name="Train Features",
                is_required=True
            )
        ],
        TEST_DATASETS=[
            DatasetConfig(
                file_path="test.parquet",
                feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
                id_columns=["ID"],
                dataset_name="Test Features", 
                is_required=True
            )
        ],
        
        # Ensemble strategy: individual models
        ENSEMBLE_STRATEGY="individual_models",
        INDIVIDUAL_MODEL_WEIGHTS={
            # XGBoost models - prefer full dataset, but include recent data models
            "XGB_Full_Dataset_100%": 0.30,      # Highest weight to full dataset XGB
            "XGB_Recent_Data_75%": 0.20,        # Medium weight to 75% XGB
            "XGB_Recent_Data_50%": 0.10,        # Lower weight to 50% XGB
            
            # LightGBM models - also prefer full dataset
            "LGB_Full_Dataset_100%": 0.25,      # High weight to full dataset LGB
            "LGB_Recent_Data_75%": 0.10,        # Lower weight to 75% LGB 
            "LGB_Recent_Data_50%": 0.05,        # Lowest weight to 50% LGB
        },
        
        # Other settings...
        TARGET_COLUMN="label",
        ID_COLUMN="ID",
        TIMESTAMP_COLUMN="timestamp", 
        SELECTED_FEATURES=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
        
        SUBMISSION_FILENAME="individual_model_ensemble_submission.csv",
        RESULTS_FILENAME="individual_model_ensemble_results.csv"
    )
    
    print("Configuration:")
    print(f"  Strategy: {config.ENSEMBLE_STRATEGY}")
    print("  Individual model weights:")
    for model, weight in config.INDIVIDUAL_MODEL_WEIGHTS.items():
        print(f"    {model}: {weight:.2f}")
    print(f"  Total weight: {sum(config.INDIVIDUAL_MODEL_WEIGHTS.values()):.2f}")
    print("  This will:")
    print("    1. Train all 6 individual models")
    print("    2. Create final ensemble using specified individual weights")
    print("    3. No intermediate learner ensembles")
    
    return config

def example_balanced_individual_ensemble():
    """Example: Balanced individual model weights"""
    
    print("="*80)
    print("EXAMPLE 3: BALANCED INDIVIDUAL MODEL ENSEMBLE")
    print("="*80)
    
    config = ModelConfig(
        # ... dataset configurations ...
        TRAIN_DATASETS=[
            DatasetConfig(
                file_path="train.parquet",
                feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"],
                id_columns=["timestamp"],
                dataset_name="Train Features",
                is_required=True
            )
        ],
        TEST_DATASETS=[
            DatasetConfig(
                file_path="test.parquet",
                feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
                id_columns=["ID"],
                dataset_name="Test Features",
                is_required=True
            )
        ],
        
        ENSEMBLE_STRATEGY="individual_models",
        INDIVIDUAL_MODEL_WEIGHTS={
            # Equal weights for all models
            "XGB_Full_Dataset_100%": 1/6,
            "XGB_Recent_Data_75%": 1/6,
            "XGB_Recent_Data_50%": 1/6,
            "LGB_Full_Dataset_100%": 1/6,
            "LGB_Recent_Data_75%": 1/6,
            "LGB_Recent_Data_50%": 1/6,
        },
        
        TARGET_COLUMN="label",
        ID_COLUMN="ID",
        TIMESTAMP_COLUMN="timestamp",
        SELECTED_FEATURES=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
        
        SUBMISSION_FILENAME="balanced_individual_ensemble_submission.csv",
        RESULTS_FILENAME="balanced_individual_ensemble_results.csv"
    )
    
    print("Configuration:")
    print(f"  Strategy: {config.ENSEMBLE_STRATEGY}")
    print("  All models have equal weights (1/6 ≈ 0.167 each)")
    print("  This gives equal importance to all trained models")
    
    return config

def example_recent_data_focused_ensemble():
    """Example: Focus on recent data models"""
    
    print("="*80)
    print("EXAMPLE 4: RECENT DATA FOCUSED ENSEMBLE")
    print("="*80)
    
    config = ModelConfig(
        # ... dataset configurations ...
        TRAIN_DATASETS=[
            DatasetConfig(
                file_path="train.parquet",
                feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"],
                id_columns=["timestamp"],
                dataset_name="Train Features",
                is_required=True
            )
        ],
        TEST_DATASETS=[
            DatasetConfig(
                file_path="test.parquet",
                feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
                id_columns=["ID"],
                dataset_name="Test Features",
                is_required=True
            )
        ],
        
        ENSEMBLE_STRATEGY="individual_models",
        INDIVIDUAL_MODEL_WEIGHTS={
            # Focus on recent data models
            "XGB_Full_Dataset_100%": 0.10,      # Lower weight to full dataset
            "XGB_Recent_Data_75%": 0.30,        # High weight to 75% recent
            "XGB_Recent_Data_50%": 0.25,        # High weight to 50% recent
            "LGB_Full_Dataset_100%": 0.05,      # Lower weight to full dataset
            "LGB_Recent_Data_75%": 0.20,        # Medium weight to 75% recent
            "LGB_Recent_Data_50%": 0.10,        # Medium weight to 50% recent
        },
        
        TARGET_COLUMN="label",
        ID_COLUMN="ID",
        TIMESTAMP_COLUMN="timestamp",
        SELECTED_FEATURES=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
        
        SUBMISSION_FILENAME="recent_focused_ensemble_submission.csv", 
        RESULTS_FILENAME="recent_focused_ensemble_results.csv"
    )
    
    print("Configuration:")
    print(f"  Strategy: {config.ENSEMBLE_STRATEGY}")
    print("  Individual model weights (focused on recent data):")
    for model, weight in config.INDIVIDUAL_MODEL_WEIGHTS.items():
        print(f"    {model}: {weight:.2f}")
    print("  This prioritizes models trained on more recent data")
    
    return config

def example_performance_based_ensemble():
    """Example: Performance-based ensemble with automatic weight calculation"""
    
    print("="*80)
    print("EXAMPLE 5: PERFORMANCE-BASED ENSEMBLE")
    print("="*80)
    
    config = ModelConfig(
        # ... dataset configurations ...
        TRAIN_DATASETS=[
            DatasetConfig(
                file_path="train.parquet",
                feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", "label"],
                id_columns=["timestamp"],
                dataset_name="Train Features",
                is_required=True
            )
        ],
        TEST_DATASETS=[
            DatasetConfig(
                file_path="test.parquet",
                feature_columns=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
                id_columns=["ID"],
                dataset_name="Test Features",
                is_required=True
            )
        ],
        
        # Ensemble strategy: performance-based (automatic weight calculation)
        ENSEMBLE_STRATEGY="performance_based",
        # No need to specify weights - they are calculated automatically from CV scores
        
        TARGET_COLUMN="label",
        ID_COLUMN="ID",
        TIMESTAMP_COLUMN="timestamp",
        SELECTED_FEATURES=["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"],
        
        SUBMISSION_FILENAME="performance_based_ensemble_submission.csv",
        RESULTS_FILENAME="performance_based_ensemble_results.csv"
    )
    
    print("Configuration:")
    print(f"  Strategy: {config.ENSEMBLE_STRATEGY}")
    print("  Automatic weight calculation based on cross-validation scores")
    print("  This will:")
    print("    1. Train all 6 individual models")
    print("    2. Calculate CV scores for each model")
    print("    3. Use softmax transformation to convert scores to weights")
    print("    4. Higher performing models get higher weights automatically")
    print("    5. No manual weight specification required")
    
    return config

def run_ensemble_comparison():
    """Run comparison of different ensemble strategies"""
    
    print("🚀 ENSEMBLE STRATEGY COMPARISON")
    print("This example shows how to configure different ensemble strategies")
    print("Choose the strategy that best fits your use case:\n")
    
    # Show different configurations
    configs = [
        ("Learner-Level (70% XGB, 30% LGB)", example_learner_level_ensemble()),
        ("Individual Models (Full Dataset Preferred)", example_individual_model_ensemble()),
        ("Individual Models (Equal Weights)", example_balanced_individual_ensemble()),
        ("Individual Models (Recent Data Focused)", example_recent_data_focused_ensemble()),
        ("Performance-Based (Automatic Weights)", example_performance_based_ensemble())
    ]
    
    for name, config in configs:
        print(f"\n📋 {name}:")
        print(f"   Strategy: {config.ENSEMBLE_STRATEGY}")
        
        if config.ENSEMBLE_STRATEGY == "learner_level":
            if config.CUSTOM_ENSEMBLE_WEIGHTS:
                print(f"   Learner weights: XGB={config.CUSTOM_ENSEMBLE_WEIGHTS[0]:.1f}, LGB={config.CUSTOM_ENSEMBLE_WEIGHTS[1]:.1f}")
            else:
                print("   Learner weights: Equal (0.5, 0.5)")
        elif config.ENSEMBLE_STRATEGY == "performance_based":
            print("   Weights: Automatically calculated from CV performance")
            print("   Higher performing models receive higher weights")
        else:
            total_xgb_weight = sum(weight for model, weight in config.INDIVIDUAL_MODEL_WEIGHTS.items() if 'XGB' in model)
            total_lgb_weight = sum(weight for model, weight in config.INDIVIDUAL_MODEL_WEIGHTS.items() if 'LGB' in model)
            print(f"   Effective learner weights: XGB={total_xgb_weight:.2f}, LGB={total_lgb_weight:.2f}")
        
        print("   " + "="*50)

def main():
    """Main function to demonstrate granular ensemble control"""
    
    print("🎯 GRANULAR ENSEMBLE CONTROL EXAMPLES")
    print("="*80)
    print("This script demonstrates the extended ensemble functionality")
    print("that provides granular control over model weights.\n")
    
    # Show comparison of strategies
    run_ensemble_comparison()
    
    print("\n🔧 IMPLEMENTATION NOTES:")
    print("="*50)
    print("1. ENSEMBLE_STRATEGY controls the weighting approach:")
    print("   - 'learner_level': Weight XGBoost vs LightGBM ensembles")
    print("   - 'individual_models': Weight each of the 6 models individually")
    print("   - 'performance_based': Automatic weights from CV scores")
    print()
    print("2. For learner_level strategy:")
    print("   - Use CUSTOM_ENSEMBLE_WEIGHTS=[xgb_weight, lgb_weight]")
    print("   - Each learner ensemble uses equal weights for its models")
    print()
    print("3. For individual_models strategy:")
    print("   - Use INDIVIDUAL_MODEL_WEIGHTS dictionary")
    print("   - Must specify weights for all 6 models")
    print("   - Weights are automatically normalized if they don't sum to 1.0")
    print()
    print("4. For performance_based strategy:")
    print("   - No manual weight specification required")
    print("   - Weights calculated using softmax transformation of CV scores")
    print("   - Better performing models automatically get higher weights")
    print("   - Temperature parameter (0.1) controls weight smoothness")
    print()
    print("5. Validation:")
    print("   - Configuration is validated at pipeline initialization")
    print("   - Missing weights or invalid strategies raise errors")
    print("   - Negative weights are not allowed")
    print()
    print("6. Results:")
    print("   - Results summary shows individual model weights")
    print("   - Final ensemble type is reported in results")
    print("   - Weight normalization is logged if applied")
    print("   - Performance-based shows improvement over simple average")

if __name__ == "__main__":
    main() 