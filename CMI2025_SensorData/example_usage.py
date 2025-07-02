"""
Example usage of the Enhanced XGBoost Baseline
Demonstrates different ways to use the baseline solution.
"""

import logging
from enhanced_xgboost_baseline import EnhancedSensorBaseline
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Basic usage example - run complete pipeline."""
    logger.info("=== Basic Usage Example ===")
    
    # Initialize baseline
    baseline = EnhancedSensorBaseline()
    
    # Run complete pipeline (loads data, processes, trains, predicts)
    baseline.run_complete_pipeline("basic_submission.csv")
    
    logger.info("Basic pipeline completed!")

def example_custom_processing():
    """Custom processing example - step by step."""
    logger.info("=== Custom Processing Example ===")
    
    # Initialize baseline
    baseline = EnhancedSensorBaseline()
    
    # Step 1: Process data
    X_train, y_train, X_test, sequence_ids = baseline.process_pipeline()
    logger.info(f"Training data: {X_train.shape}")
    logger.info(f"Test data: {X_test.shape}")
    
    # Step 2: Optimize hyperparameters with more trials
    best_params = baseline.optimize_hyperparameters(X_train, y_train, n_trials=100)
    logger.info(f"Best parameters: {best_params}")
    
    # Step 3: Train model with custom feature selection
    baseline.train_model(X_train, y_train, use_feature_selection=True)
    
    # Step 4: Make predictions
    predictions = baseline.predict(X_test)
    logger.info(f"Made predictions for {len(predictions)} sequences")
    
    # Step 5: Create submission
    baseline.create_submission(predictions, sequence_ids, "custom_submission.csv")
    
    logger.info("Custom processing completed!")

def example_without_feature_selection():
    """Example without feature selection for faster processing."""
    logger.info("=== No Feature Selection Example ===")
    
    # Initialize baseline
    baseline = EnhancedSensorBaseline()
    
    # Process data
    X_train, y_train, X_test, sequence_ids = baseline.process_pipeline()
    
    # Train without feature selection (faster)
    baseline.train_model(X_train, y_train, use_feature_selection=False)
    
    # Make predictions
    predictions = baseline.predict(X_test)
    
    # Create submission
    baseline.create_submission(predictions, sequence_ids, "no_feature_selection_submission.csv")
    
    logger.info("No feature selection pipeline completed!")

def example_fast_optimization():
    """Example with fast optimization (fewer trials)."""
    logger.info("=== Fast Optimization Example ===")
    
    # Initialize baseline
    baseline = EnhancedSensorBaseline()
    
    # Process data
    X_train, y_train, X_test, sequence_ids = baseline.process_pipeline()
    
    # Fast optimization (fewer trials)
    best_params = baseline.optimize_hyperparameters(X_train, y_train, n_trials=20)
    
    # Train model
    baseline.train_model(X_train, y_train, use_feature_selection=True)
    
    # Make predictions
    predictions = baseline.predict(X_test)
    
    # Create submission
    baseline.create_submission(predictions, sequence_ids, "fast_optimization_submission.csv")
    
    logger.info("Fast optimization pipeline completed!")

def example_model_analysis():
    """Example showing model analysis and feature importance."""
    logger.info("=== Model Analysis Example ===")
    
    # Initialize baseline
    baseline = EnhancedSensorBaseline()
    
    # Process and train
    X_train, y_train, X_test, sequence_ids = baseline.process_pipeline()
    baseline.train_model(X_train, y_train, use_feature_selection=True)
    
    # Analyze model
    if baseline.model is not None:
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': baseline.model.feature_names_in_,
            'importance': baseline.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Model parameters
        logger.info(f"Model parameters: {baseline.best_params}")
        
        # Feature selection info
        if baseline.feature_selector is not None:
            logger.info(f"Selected {len(baseline.feature_selector.get_feature_names_out())} features")
    
    logger.info("Model analysis completed!")

def main():
    """Run different examples."""
    logger.info("Enhanced XGBoost Baseline Examples")
    logger.info("===================================")
    
    # Choose which example to run
    examples = {
        '1': example_basic_usage,
        '2': example_custom_processing,
        '3': example_without_feature_selection,
        '4': example_fast_optimization,
        '5': example_model_analysis
    }
    
    print("\nAvailable examples:")
    print("1. Basic usage (complete pipeline)")
    print("2. Custom processing (step by step)")
    print("3. No feature selection (faster)")
    print("4. Fast optimization (fewer trials)")
    print("5. Model analysis (feature importance)")
    
    choice = input("\nEnter example number (1-5): ").strip()
    
    if choice in examples:
        try:
            examples[choice]()
            logger.info("Example completed successfully!")
        except Exception as e:
            logger.error(f"Example failed: {str(e)}")
    else:
        logger.error("Invalid choice. Please select 1-5.")

if __name__ == '__main__':
    main() 