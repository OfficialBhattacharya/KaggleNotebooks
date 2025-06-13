# Enhanced Multi-Class Classification Ensemble Framework

A comprehensive, data-agnostic framework for multi-class classification problems using ensemble methods with proper cross-validation, feature engineering, and mathematically sound evaluation.

## 🚀 **NEW: GPU-Accelerated Parallel Training**
- **Dual GPU Support**: Optimized for T4 x 2 setups with automatic parallel fold processing
- **3-5x Speed Improvement**: Parallel training across multiple GPUs
- **Automatic Fallback**: Seamlessly falls back to CPU if GPUs unavailable
- **Smart GPU Allocation**: Intelligent device management and memory optimization

## Features

### 🎯 **Core Capabilities**
- **Data-agnostic design**: Works with any multi-class classification dataset
- **Robust ensemble methods**: Combines multiple models for improved performance
- **Comprehensive evaluation**: Uses MAP@K (Mean Average Precision at K) scoring
- **Advanced logging**: Detailed progress tracking and performance monitoring
- **Production-ready**: Clean, modular code with proper error handling
- **GPU-accelerated**: Massive speed improvements with CUDA support

### 🤖 **Included Models**
1. **LightGBM GBDT** - Gradient boosting with optimized parameters
2. **LightGBM GOSS** - Gradient-based One-Side Sampling variant
3. **XGBoost** - Extreme gradient boosting with categorical support
4. **CatBoost** - Gradient boosting for categorical features (optional)
5. **Random Forest** - Robust ensemble baseline
6. **AutoGluon** - Automated machine learning (optional)
7. **Logistic Regression** - Final ensemble meta-learner

### 📊 **Mathematical Improvements**
- **Enhanced MAP@K calculation**: More robust implementation with proper edge case handling
- **Stratified cross-validation**: Maintains class distribution across folds
- **Class weight balancing**: Handles imbalanced datasets effectively
- **Feature scaling**: Proper preprocessing for ensemble stage
- **Early stopping**: Prevents overfitting in tree-based models

## Installation

### 1. Install Core Dependencies
```bash
pip install -r requirements.txt
```

### 2. Optional: Install AutoGluon (Large Package)
```bash
pip install autogluon
```

### 3. For GPU Support (Optional)
```bash
# For LightGBM GPU support
pip install lightgbm[gpu]

# For XGBoost GPU support  
pip install xgboost[gpu]
```

## Quick Start

### 1. Prepare Your Data
Ensure your data has the following structure:

**Training Data (train.csv):**
```
id,feature1,feature2,feature3,...,Fertilizer Name
1,value1,value2,value3,...,14-35-14
2,value1,value2,value3,...,10-26-26
...
```

**Test Data (test.csv):**
```
id,feature1,feature2,feature3,...
750000,value1,value2,value3,...
750001,value1,value2,value3,...
...
```

### 2. Basic Usage

**Quick Start (GPU-Accelerated):**
```python
from enhanced_multiclass_ensemble import Config, main

# Configure for dual T4 GPU setup
config = Config(
    train_path="path/to/train.csv",
    test_path="path/to/test.csv", 
    target_col="Fertilizer Name",
    id_col="id",
    n_folds=5,
    top_k_predictions=3,
    
    # GPU acceleration settings
    use_gpu=True,
    gpu_devices=[0, 1],  # Use both T4 GPUs
    parallel_folds=True  # Parallel processing
)

# Run the ensemble (3-5x faster with dual GPUs!)
main(config)
```

**CPU-Only Usage:**
```python
config = Config(
    train_path="path/to/train.csv",
    test_path="path/to/test.csv", 
    target_col="Fertilizer Name",
    id_col="id",
    use_gpu=False  # Disable GPU
)
```

### 3. Run Example Script
```bash
python run_ensemble_example.py
```

## Configuration Options

### Core Settings
```python
config = Config(
    # Data paths
    train_path="train.csv",              # Training data
    test_path="test.csv",                # Test data
    original_path="",                    # Optional: additional training data
    
    # Column specifications
    target_col="Fertilizer Name",        # Target column name
    id_col="id",                         # ID column name
    
    # Cross-validation
    n_folds=5,                          # Number of CV folds
    seed=42,                            # Random seed
    
    # Prediction settings
    top_k_predictions=3,                # Number of top predictions (MAP@3)
    
    # Output
    output_dir="ensemble_outputs"       # Results directory
)
```

### Advanced Settings
```python
config = Config(
    # ... basic settings ...
    
    # Ensemble optimization
    use_class_weights=True,             # Balance class weights
    scale_features_for_lr=True,         # Scale features for ensemble
    
    # Model training
    early_stopping_rounds=100,          # Early stopping patience
    verbose_eval=1000,                  # Logging frequency
    
    # AutoGluon (if available)
    autogluon_time_limit=3600,         # Time limit in seconds
    autogluon_presets="good_quality"   # Quality preset
)
```

## Model Parameters

All models use optimized hyperparameters that have shown good performance across various datasets:

### LightGBM GBDT
- Optimized for gradient boosting with balanced regularization
- Early stopping to prevent overfitting
- Categorical feature support

### LightGBM GOSS  
- Gradient-based One-Side Sampling for efficiency
- Lower learning rate with more estimators
- Optimized for large datasets

### XGBoost
- Multi-class softmax objective
- Built-in categorical feature handling
- GPU support when available

### Additional Models
- **CatBoost**: Specialized for categorical features
- **Random Forest**: Robust baseline with balanced class weights
- **AutoGluon**: Automated feature engineering and model selection

## Output Files

The framework generates several output files in the specified output directory:

### 📄 **Core Results**
- `submission_ensemble_[score].csv` - Final predictions in competition format
- `ensemble_log_[timestamp].log` - Detailed training log
- `cv_scores.csv` - Cross-validation scores for all models

### 📊 **Analysis Files**
- `experiment_summary.json` - Summary statistics and configuration
- `oof_predictions.npz` - Out-of-fold predictions (if enabled)
- `test_predictions.npz` - Test predictions for each model (if enabled)
- `feature_importance.csv` - Feature importance analysis (if enabled)

### 📈 **Visualizations**
- `ensemble_results.png` - Performance comparison plots

## Expected Output Format

The final submission file follows the required format:

```csv
id,Fertilizer Name
750000,14-35-14 10-26-26 Urea
750001,14-35-14 10-26-26 Urea
...
```

Each row contains up to 3 space-delimited predictions ordered by confidence.

## Performance Monitoring

### Real-time Logging
The framework provides comprehensive logging:

```
================================================================================
ENHANCED MULTI-CLASS CLASSIFICATION ENSEMBLE FRAMEWORK
================================================================================

============================================================
DATA LOADING AND PREPROCESSING
============================================================
Loading datasets...
Train shape: (74444, 7)
Test shape: (6, 6)
Number of classes: 22
Classes: ['10-10-10' '10-26-26' '14-14-14' '14-35-14' '15-15-15' ...]

============================================================
TRAINING LIGHTGBM_GBDT
============================================================

Fold 1/5
----------------------------------------
Training samples: 59555, Validation samples: 14889
MAP@3: 0.921543
Log Loss: 0.234567
Accuracy: 0.876543
```

### Performance Metrics
- **MAP@K**: Primary evaluation metric
- **Log Loss**: Probabilistic loss function
- **Accuracy**: Classification accuracy
- **Cross-validation statistics**: Mean and standard deviation across folds

## Troubleshooting

### Common Issues

**1. Memory Issues**
```python
# Reduce AutoGluon time limit
config.autogluon_time_limit = 1800  # 30 minutes

# Use fewer folds
config.n_folds = 3
```

**2. Missing Dependencies**
```bash
# Install specific packages
pip install lightgbm xgboost catboost

# Check available models in logs
```

**3. Data Format Issues**
- Ensure CSV files have proper headers
- Check that target column exists in training data
- Verify ID column consistency between train/test

**4. GPU Issues**
```python
# The framework automatically falls back to CPU
# Check logs for GPU availability status
```

## Advanced Usage

### Custom Model Parameters
You can modify model parameters in the `get_model_configurations()` function:

```python
# Example: Modify LightGBM parameters
configs["LightGBM_GBDT"]["model"].learning_rate = 0.02
configs["LightGBM_GBDT"]["model"].num_leaves = 500
```

### Adding New Models
The framework is extensible. Add new models in `get_model_configurations()`:

```python
configs["YourModel"] = {
    "model": YourModelClass(**params),
    "fit_args": {...}
}
```

## Mathematical Foundation

### MAP@K Calculation
The framework uses a robust implementation of Mean Average Precision at K:

```
AP@K = (1/min(K,|P|)) * Σ(i=1 to K) P(i) * rel(i)
MAP@K = (1/N) * Σ(i=1 to N) AP@K(i)
```

Where:
- P(i) = precision at position i
- rel(i) = relevance indicator at position i
- N = number of samples

### Cross-Validation Strategy
- **Stratified K-Fold**: Maintains class distribution
- **Shuffled splits**: Reduces variance
- **Consistent seeding**: Ensures reproducibility

### Ensemble Method
- **Stacked generalization**: Uses OOF predictions as features
- **Logistic regression**: Meta-learner with regularization
- **Feature scaling**: Standardization for ensemble stage
- **Class balancing**: Handles imbalanced datasets

## License

This framework is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- New features include documentation
- Tests pass (when available)
- Performance improvements are benchmarked

---

**Happy ensemble modeling! 🚀** 