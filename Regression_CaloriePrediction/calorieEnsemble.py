import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import catboost as ctb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class EnsembleRegressor:
    def __init__(self, model_list=None, model_params=None, test_size=0.2, random_state=42):
        """
        Initialize the EnsembleRegressor class.
        
        Parameters:
        -----------
        model_list : list
            List of model names to use. Available options:
            ['linear', 'cart', 'rf', 'knn', 'xgb', 'catboost', 'lgbm']
        model_params : dict
            Dictionary containing parameters for each model.
            Keys should match the model names in model_list.
        test_size : float
            Proportion of the data to use for testing.
        random_state : int
            Random state for reproducibility.
        """
        print("="*50)
        print("Initializing Ensemble Regressor")
        print("="*50)
        
        self.available_models = {
            'linear': LinearRegression,
            'cart': DecisionTreeRegressor,
            'rf': RandomForestRegressor,
            'knn': KNeighborsRegressor,
            'xgb': xgb.XGBRegressor,
            'catboost': ctb.CatBoostRegressor,
            'lgbm': lgb.LGBMRegressor
        }
        
        # Default to all models if none specified
        self.model_list = model_list if model_list else list(self.available_models.keys())
        self.model_params = model_params if model_params else {}
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize containers
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.voting_regressor = None
        self.results = {}
        
        print(f"Selected models: {', '.join(self.model_list)}")
        print(f"Test size: {self.test_size}")
        print("="*50 + "\n")
        
    def _initialize_models(self):
        """Initialize the selected models with their parameters."""
        print("Initializing models with parameters...")
        
        for model_name in self.model_list:
            if model_name not in self.available_models:
                print(f"⚠️ Warning: Model {model_name} is not available. Skipping.")
                continue
                
            # Get parameters for the current model
            params = self.model_params.get(model_name, {})
            
            # Special handling for CatBoost to suppress output
            if model_name == 'catboost':
                params['verbose'] = False
                
            # Initialize the model with parameters
            self.models[model_name] = self.available_models[model_name](**params)
            print(f"✓ Initialized {model_name} with params: {params}")
            
        print(f"✓ Successfully initialized {len(self.models)} models\n")
            
    def fit(self, X, y, sample=1.0):
        """
        Fit the ensemble model on the data.
        
        Parameters:
        -----------
        X : array-like
            Features.
        y : array-like
            Target variable.
        sample : float, default=1.0
            Proportion of training data to use for fitting the models.
        """
        start_time_total = time.time()
        
        print("="*50)
        print("STEP 1: Splitting data into train and test sets")
        print("="*50)
        
        # Split the data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"✓ Train set: {self.X_train.shape[0]} samples")
        print(f"✓ Test set: {self.X_test.shape[0]} samples")
        
        # Sample the training data if specified
        if sample < 1.0:
            print(f"\nSampling {sample*100:.1f}% of training data...")
            sample_size = int(len(self.X_train) * sample)
            indices = np.random.choice(len(self.X_train), sample_size, replace=False)
            X_train_sample = self.X_train.iloc[indices] if hasattr(self.X_train, 'iloc') else self.X_train[indices]
            y_train_sample = self.y_train.iloc[indices] if hasattr(self.y_train, 'iloc') else self.y_train[indices]
            print(f"✓ Using {sample_size} samples for training")
        else:
            X_train_sample = self.X_train
            y_train_sample = self.y_train
            print(f"✓ Using all {len(self.X_train)} samples for training")
        
        print("\n" + "="*50)
        print("STEP 2: Initializing models")
        print("="*50)
        # Initialize models
        self._initialize_models()
        
        print("\n" + "="*50)
        print("STEP 3: Training individual models")
        print("="*50)
        
        # Train individual models and track performance
        trained_models = []
        model_count = len(self.models)
        
        for idx, (name, model) in enumerate(self.models.items(), 1):
            model_start_time = time.time()
            print(f"\nTraining model {idx}/{model_count}: {name}")
            print("-" * 30)
            
            # Fit the model
            model.fit(X_train_sample, y_train_sample)
            
            # Calculate train metrics
            y_train_pred = model.predict(X_train_sample)
            train_mse = mean_squared_error(y_train_sample, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            
            # Predict on test set
            y_test_pred = model.predict(self.X_test)
            
            # Calculate test metrics
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            r2 = r2_score(self.y_test, y_test_pred)
            
            # Store results
            self.results[name] = {
                'mse': test_mse,
                'rmse': test_rmse,
                'r2': r2,
                'train_rmse': train_rmse
            }
            
            # Add to list for VotingRegressor
            trained_models.append((name, model))
            
            # Calculate elapsed time and estimate remaining time
            elapsed = time.time() - model_start_time
            remaining_models = model_count - idx
            estimated_remaining = elapsed * remaining_models
            
            print(f"✓ {name} metrics:")
            print(f"  - Train RMSE: {train_rmse:.4f}")
            print(f"  - Test RMSE: {test_rmse:.4f}")
            print(f"  - R²: {r2:.4f}")
            print(f"  - Time: {elapsed:.2f}s")
            
            if remaining_models > 0:
                print(f"Progress: {idx}/{model_count} models completed ({idx/model_count*100:.1f}%)")
                print(f"Estimated time remaining: {estimated_remaining:.2f}s")
        
        print("\n" + "="*50)
        print("STEP 4: Creating and evaluating ensemble model")
        print("="*50)
        
        # Create and train VotingRegressor
        if trained_models:
            ensemble_start_time = time.time()
            print("Training voting ensemble...")
            
            self.voting_regressor = VotingRegressor(estimators=trained_models)
            self.voting_regressor.fit(X_train_sample, y_train_sample)
            
            # Calculate train metrics for ensemble
            ensemble_train_preds = self.voting_regressor.predict(X_train_sample)
            ensemble_train_mse = mean_squared_error(y_train_sample, ensemble_train_preds)
            ensemble_train_rmse = np.sqrt(ensemble_train_mse)
            
            # Evaluate ensemble on test set
            ensemble_test_preds = self.voting_regressor.predict(self.X_test)
            ensemble_test_mse = mean_squared_error(self.y_test, ensemble_test_preds)
            ensemble_test_rmse = np.sqrt(ensemble_test_mse)
            ensemble_r2 = r2_score(self.y_test, ensemble_test_preds)
            
            self.results['ensemble'] = {
                'mse': ensemble_test_mse,
                'rmse': ensemble_test_rmse,
                'r2': ensemble_r2,
                'train_rmse': ensemble_train_rmse
            }
            
            ensemble_elapsed = time.time() - ensemble_start_time
            
            print(f"✓ Ensemble metrics:")
            print(f"  - Train RMSE: {ensemble_train_rmse:.4f}")
            print(f"  - Test RMSE: {ensemble_test_rmse:.4f}")
            print(f"  - R²: {ensemble_r2:.4f}")
            print(f"  - Time: {ensemble_elapsed:.2f}s")
            
        # Total time
        total_elapsed = time.time() - start_time_total
        print("\n" + "="*50)
        print(f"COMPLETE: Total training time: {total_elapsed:.2f}s")
        print("="*50)
            
        return self
    
    def predict(self, X):
        """
        Make predictions using the voting regressor.
        
        Parameters:
        -----------
        X : array-like
            Features to predict on.
            
        Returns:
        --------
        array-like
            Predictions.
        """
        if self.voting_regressor is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        print(f"Making predictions on {X.shape[0]} samples...")
        start_time = time.time()
        predictions = self.voting_regressor.predict(X)
        elapsed = time.time() - start_time
        
        print(f"✓ Predictions complete in {elapsed:.2f}s")
        return predictions
    
    def get_best_model(self):
        """
        Get the best performing model based on RMSE.
        
        Returns:
        --------
        tuple
            (model_name, model_object, metrics)
        """
        if not self.results:
            raise ValueError("No models have been trained yet. Call fit() first.")
        
        print("Finding the best performing model based on test RMSE...")
        best_model_name = min(self.results, key=lambda x: self.results[x]['rmse'])
        
        print(f"✓ Best model: {best_model_name}")
        print(f"  - Test RMSE: {self.results[best_model_name]['rmse']:.4f}")
        print(f"  - R²: {self.results[best_model_name]['r2']:.4f}")
        
        if best_model_name == 'ensemble':
            return 'ensemble', self.voting_regressor, self.results['ensemble']
        else:
            return best_model_name, self.models[best_model_name], self.results[best_model_name]
    
    def get_results_df(self):
        """
        Return the results as a DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            Results for all models.
        """
        print("Generating results summary...")
        
        results_df = pd.DataFrame({
            model: {
                'Train RMSE': metrics.get('train_rmse', float('nan')),
                'Test RMSE': metrics['rmse'],
                'MSE': metrics['mse'],
                'R²': metrics['r2'],
                'RMLSE': np.sqrt(np.mean(np.square(np.log1p(self.y_test) - np.log1p(self.voting_regressor.predict(self.X_test)))))
            }
            for model, metrics in self.results.items()
        }).T.sort_values('Test RMSE')
        
        print("✓ Results summary generated")
        return results_df

# Example usage of EnsembleRegressor class
'''
# Import necessary libraries for data loading and preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset (assumes X_final and y are already preprocessed)
# X_final = pd.read_csv('preprocessed_features.csv')
# y = pd.read_csv('target.csv').values.ravel()

# Or if starting from raw data:
# data = pd.read_csv('your_dataset.csv')
# X = data.drop('target_column', axis=1)
# y = data['target_column']

# Preprocess data if needed
# imputer = SimpleImputer(strategy='median')
# scaler = StandardScaler()
# X_processed = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X)), columns=X.columns)

# Initialize the ensemble regressor
ensemble = EnsembleRegressor(
    # List of models to include in the ensemble
    model_list=['linear', 'cart', 'rf', 'knn', 'xgb', 'catboost', 'lgbm'],
    
    # Parameters for each model
    model_params={
        'linear': {},  # Default parameters for LinearRegression
        'cart': {'max_depth': 10, 'min_samples_split': 5},
        'rf': {'n_estimators': 100, 'max_depth': 15, 'n_jobs': -1},
        'knn': {'n_neighbors': 5, 'weights': 'distance'},
        'xgb': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
        'catboost': {'iterations': 100, 'depth': 6, 'learning_rate': 0.1},
        'lgbm': {'n_estimators': 100, 'num_leaves': 31, 'learning_rate': 0.1}
    },
    
    # Test size for train-test split
    test_size=0.2,
    
    # Random state for reproducibility
    random_state=42
)

# Fit the ensemble using only 20% of the training data
ensemble.fit(X_final, y, sample=0.2)

# Get the best performing model
best_model_name, best_model, metrics = ensemble.get_best_model()
print(f"Best model: {best_model_name}")

# Make predictions with the voting ensemble on new data
# new_data = pd.read_csv('new_data.csv')
# new_data_processed = preprocess_new_data(new_data)  # Apply the same preprocessing
# predictions = ensemble.predict(new_data_processed)

# View detailed results in a DataFrame
results_df = ensemble.get_results_df()
print(results_df)
'''
