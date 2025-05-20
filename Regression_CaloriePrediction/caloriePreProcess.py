import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_regression, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging
import sys
import time
from joblib import Parallel, delayed
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add custom print function for more visible output
def print_step(message, symbol='=', length=80):
    """Print a message with decorative symbols for better visibility"""
    print(f"\n{symbol * length}")
    print(f"{message.center(length)}")
    print(f"{symbol * length}\n")
    sys.stdout.flush()  # Force output to display immediately

class FeatureProcessor:
    def __init__(self, X, y, cv=5, skew_threshold=0.5, vif_threshold=10.0, n_jobs=-1, selection_method='fast', sample_frac=1.0, target_features=None):
        """
        Initialize the feature processor class
        
        Parameters:
        -----------
        X : pandas DataFrame
            Features dataframe
        y : pandas Series or array-like
            Target variable
        cv : int or cross-validation generator
            Cross-validation strategy for feature selection
        skew_threshold : float
            Threshold for skewness above which Box-Cox transform is applied
        vif_threshold : float
            Threshold for VIF above which features are dropped
        n_jobs : int
            Number of jobs for parallel processing (-1 uses all available cores)
        selection_method : str
            Method for feature selection:
            - 'fast' = uses univariate selection + L1 (fastest)
            - 'rf' = Random Forest importance (fast, usually better quality)
            - 'sequential' = traditional forward/backward selection (slowest)
        sample_frac : float
            Fraction of data to use for feature selection (0.0-1.0)
            Lower values speed up processing but may reduce selection quality
        target_features : int or None
            Target number of features to select. If None, will select half of features
        """
        print_step("INITIALIZING FEATURE PROCESSOR", "=")
        self.X_original = X.copy()
        self.y = y
        self.cv = cv
        self.skew_threshold = skew_threshold
        self.vif_threshold = vif_threshold
        self.n_jobs = n_jobs
        self.selection_method = selection_method
        self.sample_frac = max(0.01, min(1.0, sample_frac))  # Ensure between 0.01 and 1.0
        self.target_features = target_features
        self.dropped_features = {
            'feature_selection': [],
            'skewness_transform_failed': [],
            'high_vif': []
        }
        self.X_processed = None
        print(f"Dataset shape: {X.shape}")
        print(f"Feature names: {', '.join(X.columns[:5])}{'...' if len(X.columns) > 5 else ''}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Configuration parameters:")
        print(f"  - Cross-validation folds: {cv}")
        print(f"  - Skewness threshold: {skew_threshold}")
        print(f"  - VIF threshold: {vif_threshold}")
        print(f"  - Selection method: {selection_method}")
        print(f"  - Parallel jobs: {n_jobs}")
        print(f"  - Data sampling fraction: {self.sample_frac}")
        print(f"  - Target number of features: {self.target_features if self.target_features else 'auto (50% of features)'}")
        logger.info(f"Initialized FeatureProcessor with {X.shape[1]} features")
        logger.info(f"Skewness threshold: {skew_threshold}, VIF threshold: {vif_threshold}")
    
    def _sample_data(self, X, y):
        """Sample data for faster processing"""
        if self.sample_frac < 1.0:
            n_samples = max(int(X.shape[0] * self.sample_frac), 10)  # At least 10 samples
            n_samples = min(n_samples, X.shape[0])  # But not more than we have
            
            # Create a random sample - use same indices for X and y
            indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
            X_sampled = X.iloc[indices]
            
            # Handle both Series and array y
            if hasattr(y, 'iloc'):
                y_sampled = y.iloc[indices]
            else:
                y_sampled = y[indices]
                
            print(f"Using {n_samples} samples ({self.sample_frac:.1%} of data) for feature selection")
            return X_sampled, y_sampled
        return X, y
    
    def select_features_fast(self, k_features=None):
        """
        Fast feature selection using a combination of univariate selection and L1 regularization
        
        Parameters:
        -----------
        k_features : int or None
            Number of features to select, if None will use self.target_features or default to half of features
            
        Returns:
        --------
        X_selected : pandas DataFrame
            Selected features
        """
        print_step("STEP 1: FAST FEATURE SELECTION", "*")
        X = self.X_original.copy()
        
        if k_features is None:
            k_features = self.target_features if self.target_features is not None else max(int(X.shape[1] * 0.5), 1)
        
        print(f"Target number of features to select: {k_features}")
        
        # Sample data for faster processing
        X_sample, y_sample = self._sample_data(X, self.y)
        
        # Phase 1: Univariate feature selection (very fast)
        print("1.1: Performing univariate feature selection...")
        k_univariate = min(k_features * 2, X.shape[1])  # Select 2x features we want for next stage
        
        start_time = time.time()
        selector = SelectKBest(score_func=f_regression, k=k_univariate)
        selector.fit(X_sample, y_sample)
        
        # Get selected feature mask and transform data
        selected_mask = selector.get_support()
        X_reduced = X.iloc[:, selected_mask]
        X_sample_reduced = X_sample.iloc[:, selected_mask]
        
        print(f"Selected {X_reduced.shape[1]} features using univariate selection")
        print(f"Univariate selection time: {time.time() - start_time:.2f} seconds")
        
        # Phase 2: L1 regularization for final feature selection
        print("\n1.2: Refining with L1 regularization (Lasso)...")
        start_time = time.time()
        
        # Temporarily suppress convergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            
            # Use cross-validation to find good alpha if we have enough samples
            if len(y_sample) > 30:
                from sklearn.model_selection import GridSearchCV
                param_grid = {'alpha': np.logspace(-4, 1, 10)}
                
                # Increase max_iter to help with convergence
                lasso = Lasso(random_state=42, max_iter=5000, tol=0.01)
                
                grid_search = GridSearchCV(lasso, param_grid, cv=min(3, self.cv), 
                                          n_jobs=self.n_jobs, scoring='neg_mean_squared_error')
                grid_search.fit(X_sample_reduced, y_sample)
                best_alpha = grid_search.best_params_['alpha']
                print(f"Selected alpha = {best_alpha:.6f} via grid search")
            else:
                best_alpha = 0.01
                print(f"Using default alpha = {best_alpha}")
                
            # Final Lasso with selected alpha and increased iterations
            lasso = Lasso(alpha=best_alpha, random_state=42, max_iter=5000, tol=0.01)
            lasso.fit(X_sample_reduced, y_sample)
        
        # Get feature importances
        importances = np.abs(lasso.coef_)
        
        # Keep only top k features based on importance
        feature_indices = np.argsort(importances)[::-1][:k_features]
        selected_features = X_reduced.columns[feature_indices]
        
        print(f"Selected {len(selected_features)} features using Lasso")
        print(f"Lasso selection time: {time.time() - start_time:.2f} seconds")
        
        # Record dropped features
        self.dropped_features['feature_selection'] = [feat for feat in X.columns if feat not in selected_features]
        
        print_step("FEATURE SELECTION RESULTS", "-")
        print(f"Selected {len(selected_features)} out of {X.shape[1]} original features")
        print(f"Dropped {len(self.dropped_features['feature_selection'])} features")
        if len(self.dropped_features['feature_selection']) > 0:
            print(f"Dropped feature examples: {', '.join(self.dropped_features['feature_selection'][:5])}")
            if len(self.dropped_features['feature_selection']) > 5:
                print(f"...and {len(self.dropped_features['feature_selection']) - 5} more")
        
        # Create new X dataframe with selected features
        self.X_processed = X[selected_features].copy()
        print(f"New dataset shape: {self.X_processed.shape}")
        return self.X_processed
    
    def select_features_rf(self, k_features=None):
        """
        Feature selection using Random Forest feature importance
        
        Parameters:
        -----------
        k_features : int or None
            Number of features to select, if None will use self.target_features or default to half of features
            
        Returns:
        --------
        X_selected : pandas DataFrame
            Selected features
        """
        print_step("STEP 1: RANDOM FOREST FEATURE SELECTION", "*")
        X = self.X_original.copy()
        
        if k_features is None:
            k_features = self.target_features if self.target_features is not None else max(int(X.shape[1] * 0.5), 1)
        
        print(f"Target number of features to select: {k_features}")
        
        # Sample data for faster processing
        X_sample, y_sample = self._sample_data(X, self.y)
        
        # Train Random Forest model
        print("Training Random Forest to determine feature importance...")
        start_time = time.time()
        
        # Set parameters for faster execution
        rf = RandomForestRegressor(
            n_estimators=100,  # Fewer trees for speed
            max_depth=None,
            min_samples_split=5,
            random_state=42,
            n_jobs=self.n_jobs
        )
        
        rf.fit(X_sample, y_sample)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Select top k features based on importance
        feature_indices = np.argsort(importances)[::-1][:k_features]
        selected_features = X.columns[feature_indices]
        
        print(f"Selected {len(selected_features)} features using Random Forest importance")
        print(f"Model fitting time: {time.time() - start_time:.2f} seconds")
        
        # Record dropped features
        self.dropped_features['feature_selection'] = [feat for feat in X.columns if feat not in selected_features]
        
        print_step("FEATURE SELECTION RESULTS", "-")
        print(f"Selected {len(selected_features)} out of {X.shape[1]} original features")
        print(f"Dropped {len(self.dropped_features['feature_selection'])} features")
        if len(self.dropped_features['feature_selection']) > 0:
            print(f"Dropped feature examples: {', '.join(self.dropped_features['feature_selection'][:5])}")
            if len(self.dropped_features['feature_selection']) > 5:
                print(f"...and {len(self.dropped_features['feature_selection']) - 5} more")
        
        # Create new X dataframe with selected features
        self.X_processed = X[selected_features].copy()
        print(f"New dataset shape: {self.X_processed.shape}")
        return self.X_processed
    
    def select_features(self, direction='both', scoring='neg_mean_squared_error', k_features=None):
        """
        Performs forward and backward feature selection using linear regression
        
        Parameters:
        -----------
        direction : str
            Direction of feature selection: 'forward', 'backward', or 'both'
        scoring : str
            Scoring metric for cross-validation
        k_features : int or None
            Number of features to select, if None will use self.target_features or default to half of features
            
        Returns:
        --------
        X_selected : pandas DataFrame
            Selected features
        """
        # Dispatch to the appropriate selection method
        if self.selection_method == 'fast':
            return self.select_features_fast(k_features)
        elif self.selection_method == 'rf':
            return self.select_features_rf(k_features)
        
        # If we're here, use the traditional slower method
        print_step("STEP 1: SEQUENTIAL FEATURE SELECTION", "*")
        print(f"Selection method: {direction.upper()}")
        print(f"Scoring metric: {scoring}")
        logger.info(f"Starting feature selection with direction: {direction}")
        X = self.X_original.copy()
        
        if k_features is None:
            k_features = self.target_features if self.target_features is not None else X.shape[1] // 2
        print(f"Target number of features to select: {k_features}")
        
        # Sample data for faster processing
        X_sample, y_sample = self._sample_data(X, self.y)
        
        selected_features = []
        
        if direction in ['forward', 'both']:
            print_step("1.1: PERFORMING FORWARD SELECTION", "-")
            logger.info("Performing forward feature selection...")
            model = LinearRegression()
            
            # Forward selection
            print("Fitting Sequential Feature Selector with forward direction...")
            sfs_forward = SequentialFeatureSelector(
                model, n_features_to_select=k_features,
                direction='forward', scoring=scoring, 
                cv=self.cv, n_jobs=self.n_jobs
            )
            
            sfs_forward.fit(X_sample, y_sample)
            forward_features = list(X.columns[sfs_forward.get_support()])
            
            print(f"Forward selection complete - Selected {len(forward_features)} features")
            logger.info(f"Forward selection selected {len(forward_features)} features")
            selected_features.extend(forward_features)
        
        if direction in ['backward', 'both']:
            print_step("1.2: PERFORMING BACKWARD SELECTION", "-")
            logger.info("Performing backward feature selection...")
            model = LinearRegression()
            
            # Backward selection
            print("Fitting Sequential Feature Selector with backward direction...")
            sfs_backward = SequentialFeatureSelector(
                model, n_features_to_select=k_features,
                direction='backward', scoring=scoring, 
                cv=self.cv, n_jobs=self.n_jobs
            )
            
            sfs_backward.fit(X_sample, y_sample)
            backward_features = list(X.columns[sfs_backward.get_support()])
            
            print(f"Backward selection complete - Selected {len(backward_features)} features")
            logger.info(f"Backward selection selected {len(backward_features)} features")
            
            # If both directions are used, take the intersection of features
            if direction == 'both':
                print("Taking intersection of forward and backward selection results...")
                selected_features = list(set(forward_features).intersection(set(backward_features)))
                print(f"Intersection resulted in {len(selected_features)} features")
                logger.info(f"Intersection of forward and backward selection: {len(selected_features)} features")
            else:
                selected_features = backward_features
        
        # Record dropped features
        self.dropped_features['feature_selection'] = [feat for feat in X.columns if feat not in selected_features]
        
        print_step("FEATURE SELECTION RESULTS", "-")
        print(f"Selected {len(selected_features)} out of {X.shape[1]} original features")
        print(f"Dropped {len(self.dropped_features['feature_selection'])} features")
        if len(self.dropped_features['feature_selection']) > 0:
            print(f"Dropped feature examples: {', '.join(self.dropped_features['feature_selection'][:5])}")
            if len(self.dropped_features['feature_selection']) > 5:
                print(f"...and {len(self.dropped_features['feature_selection']) - 5} more")
        logger.info(f"Feature selection complete. Dropped {len(self.dropped_features['feature_selection'])} features.")
        logger.info(f"Dropped features: {', '.join(self.dropped_features['feature_selection'])}")
        
        # Create new X dataframe with selected features
        self.X_processed = X[selected_features].copy()
        print(f"New dataset shape: {self.X_processed.shape}")
        return self.X_processed
    
    def fix_skewness(self):
        """
        Checks skewness and kurtosis of features and applies Box-Cox transformation
        to features with skewness above the threshold
        
        Returns:
        --------
        X_transformed : pandas DataFrame
            Data with skewness corrected
        """
        print_step("STEP 2: SKEWNESS CORRECTION", "*")
        print(f"Skewness threshold: {self.skew_threshold}")
        logger.info("Starting skewness checking and correction...")
        if self.X_processed is None:
            print("No processed data found from previous step. Using original data.")
            logger.warning("No processed data found. Using original data.")
            self.X_processed = self.X_original.copy()
        
        X = self.X_processed.copy()
        
        print("2.1: Calculating skewness and kurtosis for all features...")
        # Calculate skewness and kurtosis for each column - vectorized for speed
        skew_data = {}
        kurt_data = {}
        
        # Vectorized skewness calculation
        skew_values = X.skew()
        kurt_values = X.kurtosis()
        
        for col in X.columns:
            skew_data[col] = skew_values[col]
            kurt_data[col] = kurt_values[col]
        
        # Print skewness and kurtosis values
        skew_df = pd.DataFrame({'Skewness': skew_data, 'Kurtosis': kurt_data})
        print("\nSkewness summary:")
        print(f"Mean skewness: {skew_df['Skewness'].mean():.4f}")
        print(f"Max skewness: {skew_df['Skewness'].max():.4f}")
        print(f"Features with high skewness: {sum(abs(skew_df['Skewness']) > self.skew_threshold)}")
        
        logger.info("\nSkewness and Kurtosis before transformation:")
        logger.info(f"\n{skew_df}")
        
        # Apply Box-Cox transformation to highly skewed features
        print("\n2.2: Applying Box-Cox transformation to highly skewed features...")
        transformed_count = 0
        failed_count = 0
        
        # Get columns requiring transformation
        high_skew_cols = [col for col in X.columns if abs(skew_data[col]) > self.skew_threshold]
        print(f"Found {len(high_skew_cols)} features with skewness > {self.skew_threshold}")
        
        if len(high_skew_cols) > 0:
            # Define function for parallel processing
            def transform_column(col):
                try:
                    col_data = X[col].copy()
                    skew_val = skew_data[col]
                    
                    # Box-Cox requires positive values
                    if col_data.min() <= 0:
                        shift = abs(col_data.min()) + 1
                        col_data = col_data + shift
                        shift_msg = f" - shifted by {shift}"
                    else:
                        shift_msg = ""
                    
                    # Apply Box-Cox
                    transformed_data, _ = stats.boxcox(col_data)
                    return {
                        'col': col,
                        'status': 'success',
                        'message': f"Transformed {col} (skewness: {skew_val:.4f}){shift_msg}",
                        'data': transformed_data
                    }
                except Exception as e:
                    return {
                        'col': col,
                        'status': 'failed',
                        'message': f"Failed to transform {col} (skewness: {skew_val:.4f}): {str(e)}",
                        'data': None
                    }
            
            # Apply transformations in parallel
            if len(high_skew_cols) > 10 and self.n_jobs != 1:  # Only use parallel for many columns
                print(f"Processing {len(high_skew_cols)} columns in parallel...")
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(transform_column)(col) for col in high_skew_cols
                )
            else:
                print(f"Processing {len(high_skew_cols)} columns sequentially...")
                results = [transform_column(col) for col in high_skew_cols]
            
            # Process results
            for result in results:
                print(f"  - {result['message']}")
                
                if result['status'] == 'success':
                    X[result['col']] = result['data']
                    transformed_count += 1
                else:
                    self.dropped_features['skewness_transform_failed'].append(result['col'])
                    X = X.drop(columns=[result['col']])
                    failed_count += 1
        
        print(f"\nTransformed {transformed_count} features successfully")
        print(f"Failed to transform {failed_count} features (these were dropped)")
                
        # Calculate skewness and kurtosis after transformation
        print("\n2.3: Recalculating skewness after transformation...")
        # Vectorized skewness calculation for speed
        skew_after_values = X.skew()
        kurt_after_values = X.kurtosis()
        
        skew_after = {}
        kurt_after = {}
        for col in X.columns:
            skew_after[col] = skew_after_values[col]
            kurt_after[col] = kurt_after_values[col]
        
        skew_after_df = pd.DataFrame({'Skewness': skew_after, 'Kurtosis': kurt_after})
        print("\nSkewness summary after transformation:")
        print(f"Mean skewness: {skew_after_df['Skewness'].mean():.4f}")
        print(f"Max skewness: {skew_after_df['Skewness'].max():.4f}")
        print(f"Features with high skewness: {sum(abs(skew_after_df['Skewness']) > self.skew_threshold)}")
        
        logger.info("\nSkewness and Kurtosis after transformation:")
        logger.info(f"\n{skew_after_df}")
        
        logger.info(f"Skewness correction complete. Dropped {len(self.dropped_features['skewness_transform_failed'])} features.")
        if self.dropped_features['skewness_transform_failed']:
            logger.info(f"Dropped features: {', '.join(self.dropped_features['skewness_transform_failed'])}")
        
        self.X_processed = X
        print(f"\nNew dataset shape after skewness correction: {X.shape}")
        return self.X_processed
    
    def check_vif(self):
        """
        Checks Variance Inflation Factor (VIF) for features and drops those
        with VIF above the threshold
        
        Returns:
        --------
        X_vif : pandas DataFrame
            Data with high VIF features removed
        """
        print_step("STEP 3: MULTICOLLINEARITY CHECK (VIF)", "*")
        print(f"VIF threshold: {self.vif_threshold}")
        logger.info("Starting VIF calculation and multicollinearity check...")
        if self.X_processed is None:
            print("No processed data found from previous step. Using original data.")
            logger.warning("No processed data found. Using original data.")
            self.X_processed = self.X_original.copy()
        
        X = self.X_processed.copy()
        
        # Fast exit for small number of features
        if X.shape[1] <= 2:
            print("Dataset has 2 or fewer features, skipping VIF calculation")
            return X
        
        print("3.1: Calculating VIF for all features...")
        
        # Function to calculate VIF for a single column
        def calculate_vif(i, col_name, X_data):
            try:
                return {
                    'col': col_name,
                    'vif': variance_inflation_factor(X_data.values, i)
                }
            except:
                return {
                    'col': col_name,
                    'vif': float('inf')
                }
        
        # Parallel processing for VIF calculation if many columns
        print(f"Processing {len(X.columns)} features:")
        
        # Monitor progress
        chunk_size = max(1, len(X.columns) // 10)
        
        # Calculate VIF in parallel if we have many columns
        if len(X.columns) > 10 and self.n_jobs != 1:
            print("Using parallel processing for VIF calculation...")
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(calculate_vif)(i, col, X) 
                for i, col in enumerate(X.columns)
            )
        else:
            results = []
            for i, col in enumerate(X.columns):
                print(f"  - Calculating VIF for feature {i+1}/{len(X.columns)}: {col}", end="\r")
                results.append(calculate_vif(i, col, X))
        
        # Process results
        vif_data = {result['col']: result['vif'] for result in results}
        
        # Print VIF values
        vif_df = pd.DataFrame({'VIF': vif_data}).sort_values(by='VIF', ascending=False)
        print("\nTop 10 features by VIF value:")
        print(vif_df.head(10))
        
        logger.info("\nVIF values:")
        logger.info(f"\n{vif_df}")
        
        # Identify and drop features with high VIF
        high_vif_cols = [col for col in vif_data if vif_data[col] > self.vif_threshold]
        
        print("\n3.2: Removing features with high multicollinearity...")
        if high_vif_cols:
            print(f"Identified {len(high_vif_cols)} features with VIF > {self.vif_threshold}")
            for col in high_vif_cols:
                print(f"  - Dropping {col} (VIF: {vif_data[col]:.2f})")
            X = X.drop(columns=high_vif_cols)
            self.dropped_features['high_vif'] = high_vif_cols
            logger.info(f"Dropping {len(high_vif_cols)} features with VIF > {self.vif_threshold}")
            logger.info(f"Dropped features: {', '.join(high_vif_cols)}")
        else:
            print("No features with high VIF detected")
            logger.info("No features with high VIF detected")
        
        self.X_processed = X
        print(f"\nNew dataset shape after VIF check: {X.shape}")
        return self.X_processed
    
    def get_final_features(self):
        """
        Returns the final processed features and reports all dropped features
        
        Returns:
        --------
        X_final : pandas DataFrame
            Final processed features
        report : dict
            Dictionary of dropped features by stage
        """
        print_step("STEP 4: FINAL PROCESSING REPORT", "*")
        logger.info("\n=== Final Processing Report ===")
        
        if self.X_processed is None:
            print("No processing has been done. Returning original data.")
            logger.warning("No processing has been done. Returning original data.")
            return self.X_original, self.dropped_features
        
        total_dropped = sum(len(dropped) for dropped in self.dropped_features.values())
        print(f"Original features: {self.X_original.shape[1]}")
        print(f"Final features: {self.X_processed.shape[1]}")
        print(f"Total features dropped: {total_dropped}")
        logger.info(f"Original number of features: {self.X_original.shape[1]}")
        logger.info(f"Final number of features: {self.X_processed.shape[1]}")
        logger.info(f"Total features dropped: {total_dropped}")
        
        # Detailed report by stage
        for stage, features in self.dropped_features.items():
            if features:
                print(f"\nFeatures dropped during {stage.replace('_', ' ')} ({len(features)}):")
                print(f"{', '.join(features[:10])}")
                if len(features) > 10:
                    print(f"...and {len(features) - 10} more")
                logger.info(f"\nFeatures dropped during {stage} ({len(features)}):")
                logger.info(f"{', '.join(features)}")
        
        print("\nRemaining Features:")
        remaining_features = list(self.X_processed.columns)
        print(f"{', '.join(remaining_features[:10])}")
        if len(remaining_features) > 10:
            print(f"...and {len(remaining_features) - 10} more")
        logger.info("\n=== Remaining Features ===")
        logger.info(f"{', '.join(self.X_processed.columns)}")
        
        return self.X_processed, self.dropped_features

    def process_pipeline(self, direction='both', scoring='neg_mean_squared_error', k_features=None):
        """
        Run the complete feature processing pipeline
        
        Parameters:
        -----------
        direction : str
            Direction for feature selection
        scoring : str
            Scoring metric for cross-validation
        k_features : int or None
            Number of features to select
            
        Returns:
        --------
        X_final : pandas DataFrame
            Final processed features
        report : dict
            Dictionary of dropped features by stage
        """
        print_step("STARTING COMPLETE FEATURE PROCESSING PIPELINE", "#")
        print("Pipeline overview:")
        print("1. Feature Selection")
        print("2. Skewness Correction")
        print("3. Multicollinearity Check (VIF)")
        print("4. Final Report")
        logger.info("Starting complete feature processing pipeline...")
        
        start_time = time.time()
        
        # Step 1: Feature Selection
        print("\nExecuting Step 1...")
        self.select_features(direction=direction, scoring=scoring, k_features=k_features)
        
        # Step 2: Fix Skewness
        print("\nExecuting Step 2...")
        self.fix_skewness()
        
        # Step 3: Check VIF
        print("\nExecuting Step 3...")
        self.check_vif()
        
        # Step 4: Get Final Report
        print("\nExecuting Step 4...")
        result = self.get_final_features()
        
        elapsed_time = time.time() - start_time
        print_step(f"PIPELINE COMPLETE - Execution time: {elapsed_time:.2f} seconds", "#")
        
        return result

def transform_test_data(X_test, processor, apply_skew_transform=True):
    """
    Transform test data to match the preprocessing applied to training data
    
    Parameters:
    -----------
    X_test : pandas DataFrame
        Test data to be transformed
    processor : FeatureProcessor
        Fitted FeatureProcessor instance that was used to process training data
    apply_skew_transform : bool
        Whether to apply skewness transformation to test data
        
    Returns:
    --------
    X_test_processed : pandas DataFrame
        Processed test data with same features and transformations as training data
    """
    print_step("TRANSFORMING TEST DATA", "=")
    
    if processor.X_processed is None:
        raise ValueError("The processor has not been fitted on training data. Run the pipeline first.")
    
    # Step 1: Filter features to match those in the processed training data
    selected_features = list(processor.X_processed.columns)
    print(f"Filtering test data to include {len(selected_features)} selected features")
    
    # Check if we have all required features in X_test
    missing_features = [feat for feat in selected_features if feat not in X_test.columns]
    if missing_features:
        raise ValueError(f"Test data is missing {len(missing_features)} required features: {', '.join(missing_features[:5])}" + 
                        ("..." if len(missing_features) > 5 else ""))
    
    # Select only features that were kept in the training data
    X_test_filtered = X_test[selected_features].copy()
    
    # Step 2: Apply Box-Cox transformation to handle skewness (if requested)
    if apply_skew_transform:
        print("\nApplying skewness transformation to test data...")
        # Calculate skewness for each column
        skew_values = X_test_filtered.skew()
        transformed_count = 0
        
        for col in X_test_filtered.columns:
            skew_val = skew_values[col]
            
            # Apply transformation only if skewness is above threshold
            if abs(skew_val) > processor.skew_threshold:
                print(f"  - Transforming {col} (skewness: {skew_val:.4f})", end="")
                
                try:
                    # Handle negative values (Box-Cox requires positive data)
                    if X_test_filtered[col].min() <= 0:
                        shift = abs(X_test_filtered[col].min()) + 1
                        X_test_filtered[col] = X_test_filtered[col] + shift
                        print(f" - shifted by {shift}", end="")
                    
                    # Apply Box-Cox transformation
                    X_test_filtered[col], _ = stats.boxcox(X_test_filtered[col])
                    transformed_count += 1
                    print(f" - ✓ Success")
                except Exception as e:
                    # If transformation fails, log warning but keep the column untransformed
                    print(f" - ✗ Failed: {str(e)}")
                    print(f"    Warning: Could not apply Box-Cox to column {col} in test data.")
                    print(f"    Using original values instead.")
        
        print(f"\nTransformed {transformed_count} features in test data")
    
    print(f"\nFinal test data shape: {X_test_filtered.shape}")
    return X_test_filtered

# Example usage:
"""
# Assuming X is your feature dataframe and y is your target series:
# X = pd.read_csv('features.csv')
# y = X.pop('target')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize with a specific target number of features
processor = FeatureProcessor(
    X=X,
    y=y,
    target_features=124,  # Specify the desired number of features
    selection_method='fast',
    cv=5,
    skew_threshold=0.5,
    vif_threshold=10.0,
    n_jobs=-1,
    sample_frac=1.0
)

# Run the complete pipeline
X_processed, report = processor.process_pipeline()

# Process the test data using the same transformations
X_test_processed = transform_test_data(X_test, processor)

# Now you can use these processed datasets for modeling
model = LinearRegression()
model.fit(X_train_processed, y_train)
predictions = model.predict(X_test_processed)
"""
