import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class LightGBMHyperparameterOptimizer:
    """
    Analyzes dataset characteristics and provides optimal starting hyperparameters for LightGBM
    """
    
    def __init__(self):
        self.dataset_stats = {}
        self.recommended_params = {}
    
    def analyze_dataset(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Analyze dataset characteristics to inform hyperparameter selection
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary containing dataset statistics
        """
        n_samples, n_features = X.shape
        
        # Basic dataset characteristics
        stats_dict = {
            'n_samples': n_samples,
            'n_features': n_features,
            'dimensionality_ratio': n_features / n_samples,
            'target_variance': np.var(y),
            'target_mean': np.mean(y),
            'target_std': np.std(y),
            'target_skewness': stats.skew(y),
            'target_kurtosis': stats.kurtosis(y),
            'feature_variance_mean': np.mean([np.var(X[col]) for col in X.columns]),
            'feature_correlation_max': np.abs(X.corr()).max().max(),
            'missing_rate': X.isnull().sum().sum() / (n_samples * n_features),
        }
        
        # Estimate noise level using residuals from simple linear model
        try:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            X_scaled = StandardScaler().fit_transform(X)
            lr.fit(X_scaled, y)
            y_pred = lr.predict(X_scaled)
            residuals = y - y_pred
            estimated_noise = np.std(residuals)
            signal_to_noise = np.std(y) / estimated_noise if estimated_noise > 0 else float('inf')
            
            stats_dict.update({
                'estimated_noise': estimated_noise,
                'signal_to_noise': signal_to_noise
            })
        except:
            stats_dict.update({
                'estimated_noise': 0.1,
                'signal_to_noise': 10.0
            })
        
        self.dataset_stats = stats_dict
        return stats_dict
    
    def get_optimal_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                  problem_type: str = 'auto') -> Dict[str, Any]:
        """
        Generate optimal starting hyperparameters based on dataset analysis
        
        Args:
            X: Feature matrix
            y: Target variable
            problem_type: 'regression', 'classification', or 'auto'
            
        Returns:
            Dictionary of recommended hyperparameters
        """
        # Analyze dataset first
        stats = self.analyze_dataset(X, y)
        
        # Determine problem type if auto
        if problem_type == 'auto':
            unique_values = len(np.unique(y))
            if unique_values <= 10 and all(isinstance(val, (int, np.integer)) for val in y):
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        
        # Base parameters
        params = {
            'objective': 'regression' if problem_type == 'regression' else 'binary',
            'metric': 'rmse' if problem_type == 'regression' else 'binary_logloss',
            'boosting_type': 'gbdt',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Learning rate based on dataset size and complexity
        if stats['n_samples'] < 1000:
            learning_rate = 0.3
        elif stats['n_samples'] < 10000:
            learning_rate = 0.1
        else:
            learning_rate = 0.05
            
        # Adjust learning rate based on signal-to-noise ratio
        if stats['signal_to_noise'] < 5:
            learning_rate *= 0.5  # Lower learning rate for noisy data
        elif stats['signal_to_noise'] > 20:
            learning_rate *= 1.5  # Higher learning rate for clean data
            
        params['learning_rate'] = min(max(learning_rate, 0.01), 0.3)
        
        # Number of estimators (inversely related to learning rate)
        base_estimators = int(1000 / params['learning_rate'])
        params['n_estimators'] = min(max(base_estimators, 100), 3000)
        
        # Max depth based on feature count and sample size
        if stats['dimensionality_ratio'] > 0.1:  # High dimensional
            max_depth = min(6, int(np.log2(stats['n_features'])) + 2)
        else:
            max_depth = min(15, int(np.log2(stats['n_samples'])) - 2)
        params['max_depth'] = max(max_depth, 3)
        
        # Number of leaves (LightGBM specific - should be less than 2^max_depth)
        params['num_leaves'] = min(2 ** params['max_depth'] - 1, 300)
        
        # Regularization based on dimensionality and sample size
        if stats['dimensionality_ratio'] > 0.05:  # High dimensional, need more regularization
            params['reg_lambda'] = 1.0
            params['reg_alpha'] = 0.1
        elif stats['n_samples'] < 1000:  # Small dataset, moderate regularization
            params['reg_lambda'] = 0.1
            params['reg_alpha'] = 0.01
        else:  # Large dataset, light regularization
            params['reg_lambda'] = 0.01
            params['reg_alpha'] = 0.001
        
        # Subsampling based on dataset size
        if stats['n_samples'] > 10000:
            subsample = 0.8
            colsample = 0.8
        else:
            subsample = 0.9
            colsample = 0.9
        
        # Feature fraction (LightGBM equivalent of colsample_bytree)
        params['feature_fraction'] = colsample
        
        # Bagging parameters
        params['bagging_fraction'] = subsample
        params['bagging_freq'] = 5 if subsample < 1.0 else 0
        
        # Min child samples based on target variance and sample size
        if stats['target_variance'] > 1 and stats['n_samples'] > 1000:
            params['min_child_samples'] = 20
        else:
            params['min_child_samples'] = 10
            
        # Min data in leaf (similar to min_child_samples but for leaves)
        params['min_data_in_leaf'] = params['min_child_samples']
        
        # Path smoothing (LightGBM specific for better generalization)
        params['path_smooth'] = 0.1 if stats['signal_to_noise'] < 10 else 0.0
        
        # Early stopping
        params['early_stopping_rounds'] = max(50, params['n_estimators'] // 20)
        
        # Additional LightGBM specific parameters for better performance
        params['force_col_wise'] = True  # Faster training for small datasets
        params['max_bin'] = 255 if stats['n_samples'] > 10000 else 127
        
        # Categorical feature handling
        params['cat_smooth'] = 10.0
        params['cat_l2'] = 10.0
        
        self.recommended_params = params
        return params
    
    def print_analysis_report(self):
        """Print a detailed analysis report"""
        print("🔄 LightGBM Hyperparameter Optimization")
        print("🔬 Analyzing dataset characteristics for automated parameterization...")
        print("\n📊 Dataset Characteristics:")
        
        for key, value in self.dataset_stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print("\n🎯 Recommended Hyperparameters:")
        for key, value in self.recommended_params.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    
    def get_lightgbm_model(self) -> lgb.LGBMRegressor:
        """
        Return a configured LightGBM model with recommended parameters
        """
        if not self.recommended_params:
            raise ValueError("Must call get_optimal_hyperparameters first")
            
        # Remove non-model parameters
        model_params = self.recommended_params.copy()
        model_params.pop('early_stopping_rounds', None)
        
        if 'binary' in str(model_params.get('objective', '')):
            return lgb.LGBMClassifier(**model_params)
        else:
            return lgb.LGBMRegressor(**model_params)


def optimize_lightgbm_hyperparameters(X: pd.DataFrame, y: pd.Series, 
                                     problem_type: str = 'auto', 
                                     verbose: bool = True) -> Tuple[Dict[str, Any], lgb.LGBMRegressor]:
    """
    Main function to get optimal LightGBM hyperparameters for a dataset
    
    Args:
        X: Feature matrix (pandas DataFrame)
        y: Target variable (pandas Series)
        problem_type: 'regression', 'classification', or 'auto'
        verbose: Whether to print analysis report
        
    Returns:
        Tuple of (hyperparameters_dict, configured_lightgbm_model)
    """
    optimizer = LightGBMHyperparameterOptimizer()
    
    # Get optimal hyperparameters
    params = optimizer.get_optimal_hyperparameters(X, y, problem_type)
    
    # Print report if requested
    if verbose:
        optimizer.print_analysis_report()
    
    # Get configured model
    model = optimizer.get_lightgbm_model()
    
    return params, model


# Example usage
if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples) * 2 + X.iloc[:, 0] * 0.5 + X.iloc[:, 1] * 0.3)
    
    # Get optimal hyperparameters
    optimal_params, lgbm_model = optimize_lightgbm_hyperparameters(X, y, verbose=True)
    
    print(f"\n✅ LightGBM model configured and ready for training!")
    print(f"📝 Use early_stopping_rounds={optimal_params.get('early_stopping_rounds', 50)} during fit")
    
    # Example of how to use the model
    print(f"\n🚀 Example usage:")
    print(f"   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
    print(f"   lgbm_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], ")
    print(f"                 eval_metric='rmse', early_stopping_rounds={optimal_params.get('early_stopping_rounds', 50)})")
