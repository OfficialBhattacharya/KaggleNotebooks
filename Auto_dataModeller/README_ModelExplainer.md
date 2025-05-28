# ModelExplainer Class Documentation

A comprehensive model explainer class that supports XGBoost, CatBoost, and LightGBM models with both SHAP and LIME explanations. Includes memory optimization through data sampling.

## Features

- ✅ **Model Support**: XGBoost, CatBoost, and LightGBM
- ✅ **File Formats**: Load models from joblib (.pkl, .joblib) or YAML (.yaml, .yml) files
- ✅ **Explanation Methods**: SHAP (TreeExplainer, Explainer) and LIME (tabular)
- ✅ **Memory Optimization**: Data sampling to handle large datasets
- ✅ **Task Types**: Automatic detection of regression/classification
- ✅ **Comprehensive Visualization**: Multiple plot types for explanations
- ✅ **Memory Monitoring**: Track memory usage during operations
- ✅ **Performance Metrics**: Calculate and display model performance

## Installation

First, install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib pyyaml psutil
pip install shap lime  # For explanations
pip install xgboost lightgbm catboost  # For model support
```

## Quick Start

### Basic Usage

```python
from modelExplainer import ModelExplainer

# Initialize with a pre-loaded model
explainer = ModelExplainer(
    model=your_model,
    sample_fraction=0.5,  # Use 50% of data for memory efficiency
    verbose=True
)

# Or load model from file
explainer = ModelExplainer(
    model_path="path/to/your/model.joblib",  # or .yaml for CatBoost
    sample_fraction=0.3,
    verbose=True
)

# Prepare data
explainer.prepare_data(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    fitted_values=fitted_values,
    predictions=predictions
)

# Setup explainers
explainer.setup_shap_explainer()
explainer.setup_lime_explainer()

# Explain specific instances
shap_explanation = explainer.explain_instance_shap(instance_index=0)
lime_explanation = explainer.explain_instance_lime(instance_index=0)

# Compare explanations
comparison = explainer.compare_explanations(instance_index=0)

# Global feature importance
global_importance = explainer.global_feature_importance()

# Performance summary
summary = explainer.model_performance_summary()
```

## Detailed Usage

### 1. Initialization

```python
# Option 1: With pre-loaded model
explainer = ModelExplainer(
    model=your_model,
    sample_fraction=1.0,  # Use full dataset (default)
    random_state=42,
    verbose=True
)

# Option 2: Load from file
explainer = ModelExplainer(
    model_path="models/xgb_model.joblib",
    sample_fraction=0.5,  # Use 50% for memory efficiency
    verbose=True
)
```

### 2. Data Preparation

```python
explainer.prepare_data(
    X_train=X_train,           # Training features (pandas DataFrame)
    X_test=X_test,             # Test features (pandas DataFrame)
    y_train=y_train,           # Training targets (pandas Series, optional)
    y_test=y_test,             # Test targets (pandas Series, optional)
    fitted_values=fitted_vals, # Model predictions on training data (numpy array, optional)
    predictions=test_preds     # Model predictions on test data (numpy array, optional)
)
```

### 3. Setup Explainers

```python
# Setup SHAP explainer
shap_success = explainer.setup_shap_explainer(explainer_type='auto')  # 'tree', 'explainer', or 'auto'

# Setup LIME explainer
lime_success = explainer.setup_lime_explainer(kernel_width=5.0)
```

### 4. Instance Explanations

```python
# SHAP explanation for a specific instance
shap_results = explainer.explain_instance_shap(
    instance_index=0,        # Index of instance to explain
    data_source='test',      # 'train' or 'test'
    plot=True               # Whether to create plots
)

# LIME explanation for a specific instance
lime_results = explainer.explain_instance_lime(
    instance_index=0,
    data_source='test',
    num_features=15,        # Number of features to include
    plot=True
)

# Explain custom instance data
custom_instance = X_test.iloc[5].values
shap_results = explainer.explain_instance_shap(instance_data=custom_instance)
```

### 5. Compare Explanations

```python
comparison = explainer.compare_explanations(
    instance_index=0,
    data_source='test',
    num_features=10
)

print(f"Agreement score: {comparison['agreement_score']:.3f}")
print(f"Correlation: {comparison['correlation']:.3f}")
```

### 6. Global Feature Importance

```python
global_importance = explainer.global_feature_importance(plot=True)

# Access results
feature_importance = global_importance['feature_importance']
top_features = global_importance['top_10_features']
```

### 7. Model Performance Summary

```python
summary = explainer.model_performance_summary()

print(f"Model Type: {summary['model_type']}")
print(f"Task Type: {summary['task_type']}")
print(f"R² Score: {summary.get('r2_score', 'N/A')}")
print(f"Accuracy: {summary.get('accuracy', 'N/A')}")
```

## Memory Optimization

The `sample_fraction` parameter allows you to work with large datasets by using only a fraction of the data:

```python
# Use 30% of data for explanations
explainer = ModelExplainer(
    model=your_model,
    sample_fraction=0.3,  # Reduces memory usage significantly
    verbose=True
)
```

This is particularly useful when:
- Working with datasets larger than available RAM
- Running on systems with limited memory
- Wanting faster explanation generation

## Supported Model Types

### XGBoost
```python
import xgboost as xgb

# Load XGBoost model
model = xgb.XGBClassifier()
# ... train model ...

explainer = ModelExplainer(model=model)
```

### LightGBM
```python
import lightgbm as lgb

# Load LightGBM model
model = lgb.LGBMRegressor()
# ... train model ...

explainer = ModelExplainer(model=model)
```

### CatBoost
```python
from catboost import CatBoostClassifier

# Load CatBoost model from YAML
explainer = ModelExplainer(model_path="catboost_model.yaml")

# Or use pre-loaded model
model = CatBoostClassifier()
# ... train model ...
explainer = ModelExplainer(model=model)
```

## File Format Support

### Joblib/Pickle Files (.pkl, .joblib)
```python
# Save model
import joblib
joblib.dump(model, "model.joblib")

# Load with ModelExplainer
explainer = ModelExplainer(model_path="model.joblib")
```

### YAML Files (.yaml, .yml) - CatBoost
```python
# Save CatBoost model
model.save_model("catboost_model.yaml")

# Load with ModelExplainer
explainer = ModelExplainer(model_path="catboost_model.yaml")
```

## Example Output

### SHAP Explanation Results
```python
{
    'shap_values': array([...]),
    'expected_value': 0.5,
    'instance_data': array([...]),
    'feature_names': ['feature_0', 'feature_1', ...],
    'feature_importance': {'feature_0': 0.1, 'feature_1': 0.05, ...},
    'top_features': [('feature_0', 0.1), ('feature_1', -0.08), ...]
}
```

### LIME Explanation Results
```python
{
    'explanation': <lime.explanation.Explanation>,
    'feature_importance': {'feature_0': 0.12, 'feature_1': -0.06, ...},
    'top_features': [('feature_0', 0.12), ('feature_1', -0.06), ...],
    'instance_data': array([...]),
    'prediction': 0.75,
    'feature_names': ['feature_0', 'feature_1', ...]
}
```

## Error Handling

The class includes comprehensive error handling:

```python
# Check if explainers were set up successfully
if not explainer.setup_shap_explainer():
    print("SHAP not available - install with: pip install shap")

if not explainer.setup_lime_explainer():
    print("LIME not available - install with: pip install lime")

# Explanations return empty dict on error
explanation = explainer.explain_instance_shap(instance_index=0)
if not explanation:
    print("SHAP explanation failed")
```

## Performance Tips

1. **Use sampling for large datasets**:
   ```python
   explainer = ModelExplainer(model=model, sample_fraction=0.3)
   ```

2. **Disable plots for batch processing**:
   ```python
   explanation = explainer.explain_instance_shap(instance_index=0, plot=False)
   ```

3. **Use TreeExplainer for tree-based models** (automatic):
   ```python
   explainer.setup_shap_explainer(explainer_type='tree')  # Faster for tree models
   ```

4. **Monitor memory usage**:
   ```python
   summary = explainer.model_performance_summary()
   print(f"Memory usage: {summary['memory_usage']}")
   ```

## Troubleshooting

### Common Issues

1. **SHAP/LIME not installed**:
   ```bash
   pip install shap lime
   ```

2. **Model type not recognized**:
   - Ensure you're using XGBoost, CatBoost, or LightGBM
   - Check model class name in error messages

3. **Memory issues**:
   - Reduce `sample_fraction` parameter
   - Use smaller datasets for testing

4. **File loading errors**:
   - Check file path and format
   - Ensure model was saved correctly

### Debug Mode

Enable verbose output for debugging:

```python
explainer = ModelExplainer(
    model=model,
    verbose=True  # Detailed progress information
)
```

## Example Script

Run the provided example script to see the ModelExplainer in action:

```bash
cd dataModeller
python example_model_explainer_usage.py
```

This script demonstrates:
- Creating sample data and models
- Loading models from files
- Setting up explainers
- Generating explanations
- Comparing SHAP and LIME results

## API Reference

### ModelExplainer Class

#### Constructor
```python
ModelExplainer(model_path=None, model=None, sample_fraction=1.0, random_state=42, verbose=True)
```

#### Methods
- `load_model(model_path)`: Load model from file
- `prepare_data(X_train, X_test, y_train, y_test, fitted_values, predictions)`: Prepare data for explanations
- `setup_shap_explainer(explainer_type='auto')`: Initialize SHAP explainer
- `setup_lime_explainer(kernel_width=5.0)`: Initialize LIME explainer
- `explain_instance_shap(instance_index, instance_data, data_source, plot)`: SHAP explanation
- `explain_instance_lime(instance_index, instance_data, data_source, num_features, plot)`: LIME explanation
- `compare_explanations(instance_index, instance_data, data_source, num_features)`: Compare SHAP and LIME
- `global_feature_importance(plot)`: Calculate global feature importance
- `model_performance_summary()`: Generate performance summary

## License

This code is provided as-is for educational and research purposes. 