import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import catboost as cb
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class CatBoostHyperparameterOptimizer:
    """
    Analyzes dataset characteristics and provides optimal starting hyperparameters for CatBoost
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
        
        # Detect categorical features
        categorical_features = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(col)
            elif X[col].nunique() <= 20 and X[col].dtype in ['int64', 'int32']:
                # Potential categorical feature (low cardinality integer)
                categorical_features.append(col)
        
        stats_dict['n_categorical_features'] = len(categorical_features)
        stats_dict['categorical_ratio'] = len(categorical_features) / n_features
        
        # Estimate noise level using residuals from simple linear model
        try:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            
            # Handle categorical features for noise estimation
            X_numeric = X.select_dtypes(include=[np.number])
            if X_numeric.shape[1] > 0:
                X_scaled = StandardScaler().fit_transform(X_numeric)
                lr.fit(X_scaled, y)
                y_pred = lr.predict(X_scaled)
                residuals = y - y_pred
                estimated_noise = np.std(residuals)
                signal_to_noise = np.std(y) / estimated_noise if estimated_noise > 0 else float('inf')
            else:
                estimated_noise = np.std(y) * 0.1
                signal_to_noise = 10.0
            
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
            'loss_function': 'RMSE' if problem_type == 'regression' else 'Logloss',
            'eval_metric': 'RMSE' if problem_type == 'regression' else 'Logloss',
            'random_seed': 42,
            'thread_count': -1,
            'verbose': False,
            'allow_writing_files': False
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
        
        # Number of iterations (inversely related to learning rate)
        base_iterations = int(1000 / params['learning_rate'])
        params['iterations'] = min(max(base_iterations, 100), 3000)
        
        # Depth based on feature count and sample size
        if stats['dimensionality_ratio'] > 0.1:  # High dimensional
            depth = min(6, int(np.log2(stats['n_features'])) + 2)
        else:
            depth = min(10, int(np.log2(stats['n_samples'])) - 2)
        params['depth'] = max(depth, 4)
        
        # L2 regularization based on dimensionality and sample size
        if stats['dimensionality_ratio'] > 0.05:  # High dimensional, need more regularization
            params['l2_leaf_reg'] = 10.0
        elif stats['n_samples'] < 1000:  # Small dataset, moderate regularization
            params['l2_leaf_reg'] = 3.0
        else:  # Large dataset, light regularization
            params['l2_leaf_reg'] = 1.0
        
        # Subsampling based on dataset size
        if stats['n_samples'] > 10000:
            params['subsample'] = 0.8
        else:
            params['subsample'] = 0.9
        
        # Feature sampling (CatBoost equivalent of colsample)
        if stats['n_features'] > 50:
            params['rsm'] = 0.8  # Random subspace method
        else:
            params['rsm'] = 1.0
        
        # Border count (number of splits for numerical features)
        if stats['n_samples'] > 50000:
            params['border_count'] = 254
        elif stats['n_samples'] > 10000:
            params['border_count'] = 128
        else:
            params['border_count'] = 64
        
        # Bagging temperature (controls randomness in bagging)
        if stats['signal_to_noise'] < 10:
            params['bagging_temperature'] = 1.0  # More randomness for noisy data
        else:
            params['bagging_temperature'] = 0.5  # Less randomness for clean data
        
        # Random strength (amount of randomness for scoring splits)
        if stats['n_features'] > 100:
            params['random_strength'] = 1.0
        else:
            params['random_strength'] = 0.5
        
        # One hot max size (threshold for one-hot encoding categorical features)
        params['one_hot_max_size'] = min(25, max(2, int(np.log2(stats['n_samples']))))
        
        # Min data in leaf based on target variance and sample size
        if stats['target_variance'] > 1 and stats['n_samples'] > 1000:
            params['min_data_in_leaf'] = 10
        else:
            params['min_data_in_leaf'] = 5
        
        # Leaf estimation method
        if problem_type == 'regression':
            params['leaf_estimation_method'] = 'Newton'
        else:
            params['leaf_estimation_method'] = 'Newton'
        
        # Boosting type
        if stats['n_samples'] > 100000:
            params['boosting_type'] = 'Plain'  # Faster for large datasets
        else:
            params['boosting_type'] = 'Ordered'  # Better quality for smaller datasets
        
        # Early stopping
        params['early_stopping_rounds'] = max(50, params['iterations'] // 20)
        
        # CatBoost specific optimizations
        params['leaf_estimation_iterations'] = 3
        params['max_ctr_complexity'] = 4
        
        # Task type specific parameters
        if problem_type == 'classification':
            if len(np.unique(y)) == 2:
                params['loss_function'] = 'Logloss'
                params['eval_metric'] = 'AUC'
            else:
                params['loss_function'] = 'MultiClass'
                params['eval_metric'] = 'MultiClass'
        
        # GPU settings (will be handled separately in the enhancer)
        params['task_type'] = 'CPU'  # Default to CPU, GPU will be set if available
        
        self.recommended_params = params
        return params
    
    def print_analysis_report(self):
        """Print a detailed analysis report"""
        print("🔄 CatBoost Hyperparameter Optimization")
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
    
    def get_catboost_model(self) -> cb.CatBoostRegressor:
        """
        Return a configured CatBoost model with recommended parameters
        """
        if not self.recommended_params:
            raise ValueError("Must call get_optimal_hyperparameters first")
            
        # Remove non-model parameters
        model_params = self.recommended_params.copy()
        model_params.pop('early_stopping_rounds', None)
        
        if 'Logloss' in str(model_params.get('loss_function', '')) or 'MultiClass' in str(model_params.get('loss_function', '')):
            return cb.CatBoostClassifier(**model_params)
        else:
            return cb.CatBoostRegressor(**model_params)


def optimize_catboost_hyperparameters(X: pd.DataFrame, y: pd.Series, 
                                     problem_type: str = 'auto', 
                                     verbose: bool = True) -> Tuple[Dict[str, Any], cb.CatBoostRegressor]:
    """
    Main function to get optimal CatBoost hyperparameters for a dataset
    
    Args:
        X: Feature matrix (pandas DataFrame)
        y: Target variable (pandas Series)
        problem_type: 'regression', 'classification', or 'auto'
        verbose: Whether to print analysis report
        
    Returns:
        Tuple of (hyperparameters_dict, configured_catboost_model)
    """
    optimizer = CatBoostHyperparameterOptimizer()
    
    # Get optimal hyperparameters
    params = optimizer.get_optimal_hyperparameters(X, y, problem_type)
    
    # Print report if requested
    if verbose:
        optimizer.print_analysis_report()
    
    # Get configured model
    model = optimizer.get_catboost_model()
    
    return params, model


# Example usage
if __name__ == "__main__":
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    
    # Add some categorical features
    X['cat_feature_1'] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    X['cat_feature_2'] = np.random.choice([1, 2, 3, 4, 5], n_samples)
    
    y = pd.Series(np.random.randn(n_samples) * 2 + X.iloc[:, 0] * 0.5 + X.iloc[:, 1] * 0.3)
    
    # Get optimal hyperparameters
    optimal_params, catboost_model = optimize_catboost_hyperparameters(X, y, verbose=True)
    
    print(f"\n✅ CatBoost model configured and ready for training!")
    print(f"📝 Use early_stopping_rounds={optimal_params.get('early_stopping_rounds', 50)} during fit")
    
    # Example of how to use the model
    print(f"\n🚀 Example usage:")
    print(f"   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
    print(f"   # Identify categorical features")
    print(f"   cat_features = ['cat_feature_1', 'cat_feature_2']")
    print(f"   catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test),")
    print(f"                     cat_features=cat_features, early_stopping_rounds={optimal_params.get('early_stopping_rounds', 50)})")
    
    print(f"\n🎯 CatBoost-Specific Features:")
    print(f"   • Automatic categorical feature handling")
    print(f"   • Built-in overfitting detection")
    print(f"   • GPU acceleration support")
    print(f"   • Robust to hyperparameter choices")
    print(f"   • No need for extensive preprocessing")
