"""
Test script for the Enhanced XGBoost Baseline
Validates functionality with sample data and provides usage examples.
"""

import numpy as np
import pandas as pd
import polars as pl
import logging
from enhanced_xgboost_baseline import EnhancedSensorBaseline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data for testing."""
    logger.info("Creating sample data for testing")
    
    # Sample sequence data
    n_sequences = 10
    n_timesteps = 50
    
    sample_data = []
    for seq_id in range(n_sequences):
        for timestep in range(n_timesteps):
            row = {
                'sequence_id': seq_id,
                'sequence_counter': timestep,
                'subject': seq_id % 5,  # 5 subjects
                'behavior': np.random.choice(['Transition', 'Pause', 'Gesture']),
                'acc_x': np.random.normal(0, 1),
                'acc_y': np.random.normal(0, 1),
                'acc_z': np.random.normal(0, 1),
                'rot_w': np.random.normal(0, 1),
                'rot_x': np.random.normal(0, 1),
                'rot_y': np.random.normal(0, 1),
                'rot_z': np.random.normal(0, 1),
            }
            
            # Add thermopile sensors (with some nulls)
            for i in range(1, 6):
                row[f'thm_{i}'] = np.random.normal(25, 5) if np.random.random() > 0.1 else None
            
            # Add ToF sensors (with some nulls)
            for sensor in range(1, 6):
                for pix in range(64):
                    row[f'tof_{sensor}_v{pix}'] = np.random.normal(100, 20) if np.random.random() > 0.05 else None
            
            # Add gesture for training data
            if timestep == 0:  # Only add once per sequence
                row['gesture'] = np.random.choice(['gesture_A', 'gesture_B', 'gesture_C'])
            
            sample_data.append(row)
    
    return pl.DataFrame(sample_data)

def create_sample_demographics():
    """Create sample demographics data."""
    logger.info("Creating sample demographics data")
    
    demo_data = []
    for subject in range(5):
        demo_data.append({
            'subject': subject,
            'adult_child': np.random.choice(['adult', 'child']),
            'age': np.random.randint(8, 65),
            'sex': np.random.choice(['M', 'F']),
            'handedness': np.random.choice(['right', 'left']),
            'height_cm': np.random.normal(160, 20),
            'shoulder_to_wrist_cm': np.random.normal(60, 10),
            'elbow_to_wrist_cm': np.random.normal(30, 5)
        })
    
    return pl.DataFrame(demo_data)

def test_data_processing():
    """Test the data processing pipeline."""
    logger.info("Testing data processing pipeline")
    
    # Create sample data
    train_data = create_sample_data()
    demo_data = create_sample_demographics()
    
    # Initialize baseline
    baseline = EnhancedSensorBaseline()
    
    # Test data loading and preprocessing
    processed_data = baseline.load_and_preprocess(train_data, is_train=True)
    logger.info(f"Processed data shape: {processed_data.shape}")
    
    # Test time-series features
    data_with_ts = baseline.add_time_series_features(processed_data)
    logger.info(f"Data with time-series features shape: {data_with_ts.shape}")
    
    # Test sensor failure features
    data_with_failures = baseline.add_sensor_failure_features(data_with_ts)
    logger.info(f"Data with failure features shape: {data_with_failures.shape}")
    
    # Test aggregation
    aggregated_data = baseline.aggregate_sequence_features(data_with_failures, is_train=True)
    logger.info(f"Aggregated data shape: {aggregated_data.shape}")
    
    # Test demographics merge
    final_data = baseline.merge_demographics(aggregated_data, demo_data)
    logger.info(f"Final data shape: {final_data.shape}")
    
    return final_data

def test_feature_selection():
    """Test feature selection functionality."""
    logger.info("Testing feature selection")
    
    # Create sample features and target
    n_samples = 100
    n_features = 50
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.choice([0, 1, 2], n_samples))
    
    # Initialize baseline
    baseline = EnhancedSensorBaseline()
    
    # Test feature selection
    X_selected = baseline.feature_selection(X, y, k=20)
    logger.info(f"Original features: {X.shape[1]}")
    logger.info(f"Selected features: {X_selected.shape[1]}")
    
    return X_selected

def test_hyperparameter_optimization():
    """Test hyperparameter optimization (with reduced trials for speed)."""
    logger.info("Testing hyperparameter optimization")
    
    # Create sample data
    n_samples = 200
    n_features = 30
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.choice([0, 1, 2], n_samples))
    
    # Initialize baseline
    baseline = EnhancedSensorBaseline()
    
    # Test optimization with few trials
    best_params = baseline.optimize_hyperparameters(X, y, n_trials=5)
    logger.info(f"Best parameters: {best_params}")
    
    return best_params

def test_model_training():
    """Test model training functionality."""
    logger.info("Testing model training")
    
    # Create sample data
    n_samples = 300
    n_features = 40
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.choice([0, 1, 2], n_samples))
    
    # Initialize baseline
    baseline = EnhancedSensorBaseline()
    
    # Train model
    baseline.train_model(X, y, use_feature_selection=False)  # Skip feature selection for speed
    
    # Test prediction
    predictions = baseline.predict(X)
    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Unique predictions: {np.unique(predictions)}")
    
    return baseline.model

def main():
    """Run all tests."""
    logger.info("Starting enhanced baseline tests")
    
    try:
        # Test data processing
        final_data = test_data_processing()
        logger.info("✓ Data processing test passed")
        
        # Test feature selection
        X_selected = test_feature_selection()
        logger.info("✓ Feature selection test passed")
        
        # Test hyperparameter optimization
        best_params = test_hyperparameter_optimization()
        logger.info("✓ Hyperparameter optimization test passed")
        
        # Test model training
        model = test_model_training()
        logger.info("✓ Model training test passed")
        
        logger.info("All tests passed successfully!")
        
        # Print summary
        logger.info("\n=== Test Summary ===")
        logger.info(f"Final data shape: {final_data.shape}")
        logger.info(f"Selected features: {X_selected.shape[1]}")
        logger.info(f"Best parameters found: {len(best_params)}")
        logger.info(f"Model trained: {model is not None}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 