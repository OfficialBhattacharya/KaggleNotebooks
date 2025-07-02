# Enhanced XGBoost Baseline for CMI Sensor Data

## Overview
Memory-efficient baseline solution with advanced feature engineering, Optuna optimization, and comprehensive preprocessing.

## Key Features
- **Memory Optimization**: Polars for efficient data processing
- **Time-Series Features**: Slopes and FFT components
- **Sensor Failure Detection**: Pattern recognition for missing data
- **Automated Optimization**: Optuna hyperparameter tuning
- **Feature Selection**: ANOVA F-test for dimensionality reduction
- **Demographics Integration**: Subject information incorporation

## Installation
```bash
pip install -r requirements_enhanced.txt
```

## Usage
```python
from enhanced_xgboost_baseline import EnhancedSensorBaseline

# Run complete pipeline
baseline = EnhancedSensorBaseline()
baseline.run_complete_pipeline()
```

## Pipeline Components
1. **Data Loading**: Memory-optimized CSV reading
2. **Feature Engineering**: Time-series and sensor failure features
3. **Aggregation**: Statistical features per sequence
4. **Feature Selection**: Top-k features using ANOVA F-test
5. **Hyperparameter Optimization**: Optuna with stratified CV
6. **Model Training**: XGBoost with optimized parameters

## Expected Performance
- CV Score: ~0.85-0.90 Macro F1
- Training Time: 10-30 minutes
- Memory Usage: ~4-8 GB RAM
- Features: ~2000-3000 (after selection)

## Extensibility
- Modular design for easy feature additions
- Support for ensemble methods
- Custom cross-validation strategies
- Competition API integration

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce feature selection threshold or use smaller data samples
2. **Slow Training**: Reduce Optuna trials or use GPU acceleration
3. **Poor Performance**: Increase feature engineering or adjust hyperparameter ranges
4. **Data Loading Issues**: Check file paths and data format

### Performance Tuning

1. **For Better Accuracy**: Increase Optuna trials, add more features
2. **For Faster Training**: Reduce trials, use feature selection
3. **For Memory Constraints**: Reduce feature count, use data sampling

## Competition Integration

The solution is designed to work seamlessly with the Kaggle competition environment:

- Compatible with Kaggle evaluation API
- Handles train/test data separation
- Generates proper submission format
- Supports local testing and validation

## Future Enhancements

1. **Deep Learning**: Add neural network models
2. **Ensemble Methods**: Combine multiple algorithms
3. **Advanced CV**: Subject-based cross-validation
4. **Feature Importance**: Analyze and visualize feature contributions
5. **Model Interpretability**: Add SHAP explanations 