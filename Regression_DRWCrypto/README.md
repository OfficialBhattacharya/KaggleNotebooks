# XGBoost-LightGBM Ensemble Pipeline

A modular, well-organized pipeline for regression tasks using XGBoost and LightGBM with ensemble modeling, time-based weighting, and flexible data loading capabilities.

## 🚀 Features

- **Modular Architecture**: Separate classes for data processing, model training, and ensemble management
- **Flexible Data Loading**: Support for multiple file sources with automatic merging
- **Dual Algorithm Support**: XGBoost and LightGBM models with GPU acceleration
- **Multi-GPU Support**: Parallel training across multiple GPUs for faster processing
- **Time-Based Weighting**: Exponential decay weighting for temporal data
- **Ensemble Modeling**: Multiple models with different data subsets (100%, 75%, 50% of recent data)
- **Comprehensive Logging**: Detailed progress tracking and performance monitoring
- **Automatic Feature Selection**: Configurable feature filtering and selection
- **Cross-Validation**: Built-in K-fold cross-validation with proper time handling
- **Memory Optimization**: Automatic data type optimization reducing memory usage by 50-80%
- **Memory Management**: Comprehensive cleanup before training with real-time monitoring
- **Feature Engineering**: Market microstructure feature generation from basic market data
- **Parallel Fold Training**: Distribute cross-validation folds across available GPUs

## 📁 File Structure

```
Regression_DRWCrypto/
├── XGBoost_Modeller.py              # Main pipeline implementation
├── example_usage.py                 # Basic usage examples and configuration templates
├── example_mathematical_match.py    # Configuration to match reference implementation
├── example_feature_engineering.py   # Feature engineering and memory optimization examples
└── README.md                       # This documentation
```

## 🏗️ Architecture

### Core Classes

1. **`DatasetConfig`**: Configuration for individual dataset files
2. **`ModelConfig`**: Main configuration class for the entire pipeline
3. **`WeightCalculator`**: Time-based sample weight generation
4. **`DataProcessor`**: Data loading, merging, and preprocessing
5. **`ModelTrainer`**: Model training with different algorithms and data subsets
6. **`EnsembleManager`**: Ensemble creation and evaluation
7. **`XGBoostLightGBMPipeline`**: Main orchestrator class

### Key Features

- **Flexible Dataset Configuration**: Support for multiple files with automatic merging
- **Feature Selection**: Specify exact features to keep from each file
- **Time-Based Weights**: Exponential decay to emphasize recent data
- **Multiple Model Variants**: Train models on different data subsets
- **Ensemble Strategies**: Simple average and performance-weighted ensembles

## 📋 Quick Start

### Basic Usage

```python
from XGBoost_Modeller import ModelConfig, DatasetConfig, XGBoostLightGBMPipeline

# Define your parameters
lgbm_params = {
    "boosting_type": "gbdt",
    "learning_rate": 0.029,
    "n_estimators": 126,
    "random_state": 42,
    # ... other parameters
}

xgb_params = {
    "tree_method": "gpu_hist",
    "learning_rate": 0.022,
    "n_estimators": 1667,
    "random_state": 42,
    # ... other parameters
}

# Define features to use
selected_features = [
    "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603",
    "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume",
    # ... add your features
]

# Configure the pipeline
config = ModelConfig(
    TRAIN_DATASETS=[
        DatasetConfig(
            file_path="data/train_features.parquet",  # Parquet for performance
            feature_columns=selected_features,
            id_columns=["timestamp"],
            dataset_name="Train Features"
        ),
        DatasetConfig(
            file_path="data/train_labels.csv",        # CSV also supported
            feature_columns=["label"],
            id_columns=["timestamp"],
            dataset_name="Train Labels"
        )
    ],
    TEST_DATASETS=[
        DatasetConfig(
            file_path="data/test_features.parquet",   # Mixed formats work fine
            feature_columns=selected_features,
            id_columns=["ID"],
            dataset_name="Test Features"
        )
    ],
    XGB_PARAMS=xgb_params,
    LGBM_PARAMS=lgbm_params,
    SELECTED_FEATURES=selected_features
)

# Run the pipeline
pipeline = XGBoostLightGBMPipeline(config)
results = pipeline.run_pipeline()
```

## ⚙️ Configuration Options

### DatasetConfig Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | str | Path to CSV, Parquet, or TSV file |
| `feature_columns` | List[str] | Specific columns to keep from this file |
| `id_columns` | List[str] | ID/merge columns (timestamp, ID, etc.) |
| `dataset_name` | str | Human-readable name for logging |
| `is_required` | bool | Whether this dataset is required |

### ModelConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `TRAIN_DATASETS` | List[DatasetConfig] | [] | Train dataset configurations |
| `TEST_DATASETS` | List[DatasetConfig] | [] | Test dataset configurations |
| `TARGET_COLUMN` | str | "label" | Target column name |
| `ID_COLUMN` | str | "ID" | ID column name |
| `TIMESTAMP_COLUMN` | str | "timestamp" | Timestamp column name |
| `N_FOLDS` | int | 5 | Number of CV folds |
| `RANDOM_STATE` | int | 42 | Random seed |
| `DECAY_FACTOR` | float | 0.95 | Time weight decay factor |
| `XGB_PARAMS` | Dict | {} | XGBoost parameters |
| `LGBM_PARAMS` | Dict | {} | LightGBM parameters |
| `SELECTED_FEATURES` | List[str] | [] | Features to use in modeling |
| `REDUCE_MEMORY_USAGE` | bool | True | Enable automatic memory optimization |
| `ADD_ENGINEERED_FEATURES` | bool | False | Enable market microstructure feature engineering |
| `USE_MULTI_GPU` | bool | True | Enable multi-GPU training |
| `GPU_DEVICES` | List[int] | [0, 1] | GPU device IDs to use |
| `PARALLEL_FOLD_TRAINING` | bool | True | Train folds in parallel across GPUs |

## 🔧 Advanced Configuration

### Supported File Formats

The pipeline automatically detects and loads multiple file formats:

```python
config = ModelConfig(
    TRAIN_DATASETS=[
        DatasetConfig(
            file_path="large_features.parquet",    # Recommended for large datasets
            feature_columns=features_list,
            id_columns=["timestamp"],
            dataset_name="Parquet Features"
        ),
        DatasetConfig(
            file_path="labels.csv",                # Traditional CSV format
            feature_columns=["target"],
            id_columns=["timestamp"],
            dataset_name="CSV Labels" 
        ),
        DatasetConfig(
            file_path="additional.tsv",            # Tab-separated values
            feature_columns=["extra_feature"],
            id_columns=["timestamp"],
            dataset_name="TSV Additional"
        )
    ]
)
```

**📁 File Format Benefits:**
- **Parquet**: Faster loading, smaller file size, better compression
- **CSV**: Universal compatibility, human-readable
- **TSV**: Tab-separated, good for certain data sources
- **Auto-detection**: Format determined by file extension
- **Fallback Support**: Automatic retry with different formats if needed

### Multiple File Merging

```python
config = ModelConfig(
    TRAIN_DATASETS=[
        DatasetConfig(
            file_path="features_part1.parquet",  # Parquet for faster loading
            feature_columns=["X1", "X2", "X3"],
            id_columns=["timestamp"],
            dataset_name="Features Part 1"
        ),
        DatasetConfig(
            file_path="features_part2.csv",      # CSV also supported
            feature_columns=["X4", "X5", "X6"],
            id_columns=["timestamp"],
            dataset_name="Features Part 2"
        ),
        DatasetConfig(
            file_path="targets.tsv",             # TSV files supported too
            feature_columns=["label"],
            id_columns=["timestamp"],
            dataset_name="Targets"
        )
    ],
    # ... rest of configuration
)
```

### Custom Ensemble Strategy

```python
config = ModelConfig(
    MODEL_CONFIGS=[
        {"name": "Full Dataset", "percent": 1.00, "priority": 1},
        {"name": "Recent 90%", "percent": 0.90, "priority": 2},
        {"name": "Recent 70%", "percent": 0.70, "priority": 3},
        {"name": "Recent 50%", "percent": 0.50, "priority": 4},
        {"name": "Recent 30%", "percent": 0.30, "priority": 5},
    ],
    # ... rest of configuration
)
```

### Multi-GPU Configuration

```python
config = ModelConfig(
    # Enable multi-GPU training
    USE_MULTI_GPU=True,
    GPU_DEVICES=[0, 1],  # Use both T4 GPUs
    PARALLEL_FOLD_TRAINING=True,
    
    # XGBoost parameters for GPU
    XGB_PARAMS={
        "tree_method": "gpu_hist",  # GPU-accelerated training
        "device": "cuda:0",         # Will be automatically set per fold
        # ... other parameters
    },
    
    # LightGBM parameters for GPU  
    LGBM_PARAMS={
        "device": "gpu",            # Enable GPU training
        "gpu_device_id": 0,         # Will be automatically set per fold
        # ... other parameters
    },
    
    # ... rest of configuration
)
```

**🔥 GPU Training Benefits:**
- **2x Speed Improvement**: Parallel fold training across both GPUs
- **Automatic Load Balancing**: Folds distributed evenly across available GPUs  
- **Memory Efficiency**: Each GPU handles subset of folds independently
- **Fault Tolerance**: Graceful fallback to sequential training if needed

**⚙️ GPU Configuration Options:**
- `USE_MULTI_GPU=True`: Enable multi-GPU support
- `GPU_DEVICES=[0, 1]`: Specify which GPU devices to use
- `PARALLEL_FOLD_TRAINING=True`: Train folds in parallel (vs sequential)
- Automatic GPU parameter adjustment for both XGBoost and LightGBM

### Memory Management

The pipeline includes comprehensive memory management throughout the entire process:

```python
# Automatic memory cleanup is built-in, but you can monitor it:
pipeline = XGBoostLightGBMPipeline(config)
results = pipeline.run_pipeline()  # Memory monitored throughout
```

**🧠 Memory Management Features:**
- **Pre-training Cleanup**: Comprehensive memory cleanup before model training starts
- **Real-time Monitoring**: Memory usage tracking at all pipeline stages
- **Automatic Garbage Collection**: Multiple GC passes for thorough cleanup
- **Variable Cleanup**: Deletion of temporary objects and variables
- **Per-model Cleanup**: Memory cleanup after each model training
- **GPU Memory Optimization**: Smart memory management for multi-GPU training

**📊 Memory Tracking Stages:**
- Pipeline start and end
- Data loading completion  
- Pre-training cleanup
- XGBoost/LightGBM training start
- Training completion
- Real-time memory usage reporting

**💡 Memory Benefits:**
- Reduced peak memory usage during training
- Better GPU memory utilization
- Faster training through optimized memory access
- Prevention of out-of-memory errors
- Automatic cleanup of temporary variables

## 📊 Output Files

The pipeline generates several output files:

1. **Submission File** (`submission_ensemble_XGB_LGB.csv`): Final predictions
2. **Results File** (`ensemble_results.csv`): Detailed model performance metrics
3. **Console Output**: Real-time progress and performance information

### Submission File Format
```csv
ID,prediction
1,0.123456
2,-0.234567
...
```

### Results File Format
```csv
model,algorithm,data_percent,pearson_correlation,weight_in_ensemble
XGB_Full_Dataset_100%,XGBoost,1.0,0.825,0.15
LGB_Full_Dataset_100%,LightGBM,1.0,0.830,0.16
...
```

## 🎯 Time-Based Weighting

The pipeline uses exponential decay weighting to emphasize recent samples:

```python
def create_time_weights(n_samples, decay_factor=0.95):
    """
    Create exponentially decaying weights based on sample position.
    More recent samples (higher indices) get higher weights.
    """
    positions = np.arange(n_samples)
    normalized_positions = positions / (n_samples - 1)
    weights = decay_factor ** (1 - normalized_positions)
    weights = weights * n_samples / weights.sum()
    return weights
```

- `decay_factor=0.95`: 5% decay per time unit
- `decay_factor=0.98`: 2% decay per time unit (stronger recent emphasis)

## 🔄 Cross-Validation Strategy

The pipeline uses KFold cross-validation with proper handling of different data subsets:

1. **Full Dataset Models**: Use all available training data
2. **Subset Models**: Use only the most recent X% of data
3. **OOF Predictions**: Properly map predictions back to original indices
4. **Ensemble Creation**: Combine predictions using simple average or performance weighting

## 💾 Memory Optimization

The pipeline includes automatic memory optimization that can reduce RAM usage by 50-80%:

```python
config = ModelConfig(
    REDUCE_MEMORY_USAGE=True,  # Enable memory optimization
    # ... other parameters
)
```

**How it works:**
- Automatically converts `int64` → `int8/int16/int32` where possible
- Automatically converts `float64` → `float16/float32` where possible
- Maintains numerical precision within safe ranges
- Provides detailed logging of memory savings

## ⚙️ Feature Engineering

Generate 30+ market microstructure features from basic market data:

```python
config = ModelConfig(
    ADD_ENGINEERED_FEATURES=True,  # Enable feature engineering
    # ... other parameters
)
```

**Requirements:** Your data must contain: `bid_qty`, `ask_qty`, `buy_qty`, `sell_qty`, `volume`

**Generated features include:**
- **Market Microstructure**: bid-ask spread, liquidity measures, trade imbalances
- **Volume-Based**: volume per trade, buy/sell volume ratios
- **Pressure Indicators**: buying/selling pressure, order imbalances
- **Activity Measures**: market competition, activity concentration
- **Advanced Features**: order flow imbalance EWMA, market making intensity

## 🚀 Performance Tips

### GPU Acceleration
```python
xgb_params = {
    "tree_method": "gpu_hist",  # Use GPU
    "device": "cuda",
    # ... other parameters
}
```

### Memory Optimization
- Enable `REDUCE_MEMORY_USAGE=True` for automatic optimization
- Use appropriate `max_bin` settings for XGBoost
- Monitor memory usage during large dataset processing
- Consider data chunking for very large datasets

### Feature Selection
- Start with a subset of most important features
- Use the `SELECTED_FEATURES` parameter to control feature usage
- Enable `ADD_ENGINEERED_FEATURES=True` for market data
- Monitor training time vs. performance trade-offs

## 🐛 Troubleshooting

### Common Issues

1. **File Not Found**: Check file paths in `DatasetConfig`
2. **Memory Issues**: Reduce `n_estimators` or use smaller data subsets
3. **GPU Issues**: Fall back to `tree_method="hist"` for CPU usage
4. **Merge Failures**: Ensure ID columns exist in all datasets
5. **Feature Mismatches**: Check that selected features exist in data files

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Model Performance

The pipeline trains multiple model variants:

1. **XGBoost Models**: Full dataset + multiple recent data subsets
2. **LightGBM Models**: Full dataset + multiple recent data subsets  
3. **Ensemble**: Combines all individual models using:
   - Simple average ensemble
   - Performance-weighted ensemble (chooses best)

Expected output:
```
Individual Model Performance:
- XGB_Full_Dataset: 0.8250 correlation
- LGB_Full_Dataset: 0.8300 correlation
- XGB_Recent_75%: 0.8180 correlation
- LGB_Recent_75%: 0.8220 correlation
...

Final Ensemble: 0.8350 correlation
```

## 🤝 Contributing

To extend the pipeline:

1. **Add New Algorithms**: Extend `ModelTrainer._train_single_model()`
2. **Custom Ensemble Methods**: Modify `EnsembleManager.create_ensemble_predictions()`
3. **New Data Sources**: Add support for different file formats in `DataProcessor`
4. **Advanced Weighting**: Extend `WeightCalculator` with new weighting schemes

## 📝 License

This code is provided as-is for educational and research purposes. Modify according to your specific requirements.

---

## 🎉 Example Results

```
🚀 XGBOOST-LIGHTGBM ENSEMBLE PIPELINE
================================================================================

📋 Data loading and merging completed successfully!
   ✅ Train shape: (50000, 29)
   ✅ Test shape: (25000, 28)

📋 Model ensemble training completed!
   ✅ XGB_Full_Dataset: 0.8245
   ✅ LGB_Full_Dataset: 0.8289
   ✅ XGB_Recent_75%: 0.8167
   ✅ LGB_Recent_75%: 0.8201
   ✅ XGB_Recent_50%: 0.8098
   ✅ LGB_Recent_50%: 0.8134

📋 Final ensemble score: 0.8356
   💾 Submission saved to: submission_ensemble_XGB_LGB.csv
   💾 Results saved to: ensemble_results.csv

🎉 Pipeline completed! Total time: 2847.3 seconds
``` 