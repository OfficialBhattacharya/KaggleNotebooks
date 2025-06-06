from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class FeatureManager:
    """Manages feature generation and tracking"""
    def __init__(self):
        self.feature_pipelines: Dict[str, Dict] = {}
        self.datasets: Dict[str, pd.DataFrame] = {}

    def _format_time(self, seconds: float) -> str:
        """Helper method to format time in a readable way"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {mins}m {secs:.1f}s"

    def _log_progress(self, step: str, step_num: int, total_steps: int, start_time: float):
        """Helper method to log progress with timing information"""
        current_time = time.time()
        elapsed = current_time - start_time
        
        if step_num > 0:
            avg_time_per_step = elapsed / step_num
            remaining_steps = total_steps - step_num
            estimated_remaining = avg_time_per_step * remaining_steps
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Step {step_num}/{total_steps}: {step}")
            print(f"  ⏱️  Elapsed: {self._format_time(elapsed)} | Estimated remaining: {self._format_time(estimated_remaining)}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {step}")

    def add_feature_CrossTerms(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Multiplies pairwise from specified columns"""
        step_start = time.time()
        print(f"  🔄 Creating cross-term features from {len(features)} features...")
        
        df_processed = df.copy()
        total_combinations = len(features) * (len(features) - 1) // 2
        created_features = 0
        
        for i in range(len(features)):
            for j in range(i + 1, len(features)):  
                feature1 = features[i]
                feature2 = features[j]
                cross_term_name = f"{feature1}_x_{feature2}"
                df_processed[cross_term_name] = df_processed[feature1] * df_processed[feature2]
                created_features += 1
        
        step_time = time.time() - step_start
        print(f"  ✅ Created {created_features} cross-term features in {self._format_time(step_time)}")
        return df_processed
        
    def add_feature_divisionTerms(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Divides pairwise from specified columns"""
        step_start = time.time()
        print(f"  🔄 Creating division-term features from {len(features)} features...")
        
        df_processed = df.copy()
        total_combinations = len(features) * (len(features) - 1) // 2
        created_features = 0
        
        for i in range(len(features)):
            for j in range(i + 1, len(features)):  
                feature1 = features[i]
                feature2 = features[j]
                div_term_name = f"{feature1}_div_{feature2}"
                df_processed[div_term_name] = df_processed[feature1] / (df_processed[feature2] + 1e-6)
                created_features += 1
        
        step_time = time.time() - step_start
        print(f"  ✅ Created {created_features} division-term features in {self._format_time(step_time)}")
        return df_processed   
        
    def add_feature_squareCubeTerms(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Square and Cube of each feature from specified columns"""
        step_start = time.time()
        print(f"  🔄 Creating square/cube features from {len(features)} features...")
        
        df_processed = df.copy()
        created_features = 0
        
        for feature in features:
            df_processed[f'{feature}_squared'] = df_processed[feature] ** 2
            df_processed[f'{feature}_cubed'] = df_processed[feature] ** 3
            created_features += 2
        
        step_time = time.time() - step_start
        print(f"  ✅ Created {created_features} polynomial features in {self._format_time(step_time)}")
        return df_processed    

    def add_feature_logTerms(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Natural logarithm of each feature from specified columns"""
        step_start = time.time()
        print(f"  🔄 Creating log features from {len(features)} features...")
        
        df_processed = df.copy()
        created_features = 0
        
        for feature in features:
            # Add small epsilon to avoid log(0) and handle negative values
            # Use log(abs(x) + 1e-8) to handle negative values and zeros
            df_processed[f'{feature}_log'] = np.log(np.abs(df_processed[feature]) + 1e-8)
            created_features += 1
        
        step_time = time.time() - step_start
        print(f"  ✅ Created {created_features} logarithmic features in {self._format_time(step_time)}")
        return df_processed

    def add_feature_exponentialTerms(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Exponential (e^x) of each feature from specified columns"""
        step_start = time.time()
        print(f"  🔄 Creating exponential features from {len(features)} features...")
        
        df_processed = df.copy()
        created_features = 0
        
        for feature in features:
            # Clip values to prevent overflow (exp can grow very large)
            # Limit to reasonable range to avoid numerical overflow
            clipped_values = np.clip(df_processed[feature], -50, 50)
            df_processed[f'{feature}_exp'] = np.exp(clipped_values)
            created_features += 1
        
        step_time = time.time() - step_start
        print(f"  ✅ Created {created_features} exponential features in {self._format_time(step_time)}")
        return df_processed

    def add_feature_sqrtTerms(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Square root of each feature from specified columns"""
        step_start = time.time()
        print(f"  🔄 Creating square root features from {len(features)} features...")
        
        df_processed = df.copy()
        created_features = 0
        
        for feature in features:
            # Use sqrt of absolute value to handle negative numbers
            df_processed[f'{feature}_sqrt'] = np.sqrt(np.abs(df_processed[feature]))
            created_features += 1
        
        step_time = time.time() - step_start
        print(f"  ✅ Created {created_features} square root features in {self._format_time(step_time)}")
        return df_processed

    def add_feature_subjectKnowledge(self, df: pd.DataFrame, columns_to_add: list[str]) -> pd.DataFrame:
        """
        Create subject knowledge features based on the provided list of column names.

        Args:
            df (pd.DataFrame): The input DataFrame containing raw data.
                               Expected columns: 'Weight', 'Height', 'Heart_Rate', 'Age', 'Duration', 'Body_Temp'.
            columns_to_add (list[str]): A list of strings specifying which new feature columns
                                        to add to the DataFrame. Valid options are:
                                        'BMIscaled', 'BMR', 'Metabolic_Efficiency', 'Cardio_Stress',
                                        'Thermic_Effect', 'Power_Output'.

        Returns:
            pd.DataFrame: A new DataFrame containing only the selected new feature columns.
                          If `columns_to_add` is empty or contains unrecognized names,
                          an empty DataFrame with the same index as the input will be returned.
        """
        step_start = time.time()
        print(f"  🔄 Creating {len(columns_to_add)} subject knowledge features...")
        
        df_processed = df.copy()
        created_features = []

        # --- Calculate base features that might be dependencies for others ---
        # BMI and BMR are calculated first as Metabolic_Efficiency depends on BMR.
        if 'Weight' in df_processed.columns and 'Height' in df_processed.columns:
            if 'BMIscaled' in columns_to_add:
                df_processed['BMIscaled'] = df_processed['Weight'] / (df_processed['Height'] / 100) ** 2 + df_processed['Height'] / (df_processed['Weight'] / 100) ** 2
                created_features.append('BMIscaled')
            if 'BMR' in columns_to_add or 'Metabolic_Efficiency' in columns_to_add:
                df_processed['BMR'] = df_processed['Weight'] / ((df_processed['Height'] / 100) ** 2)
                if 'BMR' in columns_to_add:
                    created_features.append('BMR')
        else:
            print("  ⚠️  Warning: 'Weight' and/or 'Height' columns are missing, cannot calculate BMIscaled/BMR.")


        # Metabolic Efficiency Index
        if 'Metabolic_Efficiency' in columns_to_add:
            # Ensure BMR is available; it should have been calculated above if needed.
            if 'BMR' in df_processed.columns and 'Heart_Rate' in df_processed.columns:
                # Handle potential division by zero or NaN in median if BMR is all zeros/NaT
                bmr_median = df_processed['BMR'].median()
                if not pd.isna(bmr_median) and bmr_median != 0:
                    df_processed['Metabolic_Efficiency'] = df_processed['BMR'] * (df_processed['Heart_Rate'] / bmr_median)
                    created_features.append('Metabolic_Efficiency')
                else:
                    df_processed['Metabolic_Efficiency'] = float('nan') # Assign NaN if median is problematic
                    created_features.append('Metabolic_Efficiency')
            else:
                print("  ⚠️  Warning: 'BMR' and/or 'Heart_Rate' columns are missing, cannot calculate Metabolic_Efficiency.")

        # Cardiovascular Stress
        if 'Cardio_Stress' in columns_to_add:
            if 'Heart_Rate' in df_processed.columns and 'Age' in df_processed.columns and 'Duration' in df_processed.columns:
                # Avoid division by zero if (220 - Age) is zero or negative
                denominator = (220 - df_processed['Age'])
                df_processed['Cardio_Stress'] = (df_processed['Heart_Rate'] / denominator.replace(0, float('nan'))) * df_processed['Duration']
                created_features.append('Cardio_Stress')
            else:
                print("  ⚠️  Warning: 'Heart_Rate', 'Age', and/or 'Duration' columns are missing, cannot calculate Cardio_Stress.")

        # Thermic Effect Ratio
        if 'Thermic_Effect' in columns_to_add:
            if 'Body_Temp' in df_processed.columns and 'Weight' in df_processed.columns:
                # Avoid division by zero or negative weight for power 0.5
                df_processed['Thermic_Effect'] = (df_processed['Body_Temp'] * 100) / (df_processed['Weight'].apply(lambda x: x**0.5 if x > 0 else float('nan')))
                created_features.append('Thermic_Effect')
            else:
                print("  ⚠️  Warning: 'Body_Temp' and/or 'Weight' columns are missing, cannot calculate Thermic_Effect.")

        # Power Output Estimate
        if 'Power_Output' in columns_to_add:
            if 'Weight' in df_processed.columns and 'Duration' in df_processed.columns and 'Heart_Rate' in df_processed.columns:
                df_processed['Power_Output'] = df_processed['Weight'] * df_processed['Duration'] * (df_processed['Heart_Rate'] / 1000)
                created_features.append('Power_Output')
            else:
                print("  ⚠️  Warning: 'Weight', 'Duration', and/or 'Heart_Rate' columns are missing, cannot calculate Power_Output.")

        # --- Prepare the final output DataFrame ---
        # Filter df_processed to include only the explicitly requested new columns that were successfully calculated.
        final_output_columns = [col for col in columns_to_add if col in df_processed.columns]

        step_time = time.time() - step_start
        print(f"  ✅ Created {len(created_features)} subject knowledge features in {self._format_time(step_time)}")
        print(f"      Features: {', '.join(created_features) if created_features else 'None'}")

        if not final_output_columns:
            # If no valid columns were requested or could be calculated, return the original DataFrame.
            return df
        else:
            # Concatenate the original DataFrame with the newly calculated columns.
            return pd.concat([df, df_processed[final_output_columns]], axis=1)

    def add_feature_outlierDetection(self, df: pd.DataFrame, features: List[str] = None, score_method: str = 'z_score') -> pd.DataFrame:
        """
        Calculates outlier scores for each specified numerical feature individually and adds them as new columns.

        This function computes outlier scores for each feature separately using either z-score or 
        modified z-score methods. Higher absolute scores indicate potential outliers for that specific feature.

        Args:
            df (pd.DataFrame): The input DataFrame.
            features (List[str]): List of feature names to use for outlier detection.
                                If None, uses all numerical columns.
            score_method (str): Method to calculate outlier scores. Options:
                              - 'z_score': Standard z-score (mean-based)
                              - 'modified_z_score': Modified z-score using median (more robust)
                              Defaults to 'z_score'.

        Returns:
            pd.DataFrame: The original DataFrame with new outlier score columns appended.
                         Each feature gets a column named '{feature}_Outlier_Score'.
        """
        step_start = time.time()
        
        df_copy = df.copy()

        # Validate score_method parameter
        valid_methods = ['z_score', 'modified_z_score']
        if score_method not in valid_methods:
            print(f"  ⚠️  Error: score_method must be one of {valid_methods}. Got: {score_method}")
            step_time = time.time() - step_start
            print(f"  ❌ Outlier detection feature creation failed in {self._format_time(step_time)}")
            return df_copy

        # Determine which features to use
        if features is None:
            # Use all numerical columns if no specific features provided
            numerical_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
            # Exclude target column if present
            if 'Calories' in numerical_cols:
                numerical_cols.remove('Calories')
        else:
            # Use specified features, but verify they exist and are numerical
            available_features = df_copy.select_dtypes(include=np.number).columns.tolist()
            numerical_cols = [col for col in features if col in available_features]
            missing_features = [col for col in features if col not in available_features]
            if missing_features:
                print(f"  ⚠️  Warning: Features not found or not numerical: {', '.join(missing_features)}")

        print(f"  🔄 Creating individual outlier detection features using '{score_method}' method...")
        print(f"      Processing {len(numerical_cols)} features: {', '.join(numerical_cols)}")

        if not numerical_cols:
            print(f"  ⚠️  Warning: No numerical columns found. Returning original DataFrame.")
            step_time = time.time() - step_start
            print(f"  ❌ Outlier detection feature creation failed in {self._format_time(step_time)}")
            return df_copy

        # Process each feature individually
        created_features = []
        failed_features = []

        for col in numerical_cols:
            feature_start = time.time()
            
            try:
                # Get the feature data
                feature_data = df_copy[col].copy()
                
                # Check for NaN values in this specific column
                nan_count = feature_data.isna().sum()
                total_count = len(feature_data)
                
                if nan_count > 0:
                    nan_percentage = (nan_count / total_count) * 100
                    print(f"      📊 '{col}': {nan_count}/{total_count} ({nan_percentage:.1f}%) NaN values")
                    
                    # Handle NaN values based on percentage
                    if nan_percentage == 100:
                        # All values are NaN - fill with 0 and set outlier score to 0
                        print(f"         🔧 Column '{col}' is 100% NaN - setting outlier scores to 0")
                        df_copy[f'{col}_Outlier_Score'] = 0.0
                        created_features.append(f'{col}_Outlier_Score')
                        continue
                    elif nan_percentage > 50:
                        # More than 50% NaN - fill with median
                        median_val = feature_data.median()
                        if pd.isna(median_val):
                            median_val = 0
                        print(f"         🔧 Filling {nan_percentage:.1f}% NaN values with median ({median_val})")
                        feature_data = feature_data.fillna(median_val)
                    else:
                        # Less than 50% NaN - fill with mean
                        mean_val = feature_data.mean()
                        if pd.isna(mean_val):
                            mean_val = 0
                        print(f"         🔧 Filling {nan_percentage:.1f}% NaN values with mean ({mean_val:.2f})")
                        feature_data = feature_data.fillna(mean_val)

                # Calculate outlier scores based on method
                if score_method == 'z_score':
                    # Standard z-score: (x - mean) / std
                    mean_val = feature_data.mean()
                    std_val = feature_data.std()
                    
                    if std_val == 0 or pd.isna(std_val):
                        # No variation in the feature - all values are the same
                        print(f"         ⚠️  No variation in '{col}' (std = {std_val}) - setting outlier scores to 0")
                        outlier_scores = pd.Series(0.0, index=df_copy.index)
                    else:
                        outlier_scores = np.abs((feature_data - mean_val) / std_val)
                        print(f"         ✅ Z-score outlier detection: mean={mean_val:.2f}, std={std_val:.2f}")
                
                elif score_method == 'modified_z_score':
                    # Modified z-score: 0.6745 * (x - median) / MAD
                    # More robust to outliers than standard z-score
                    median_val = feature_data.median()
                    mad = np.median(np.abs(feature_data - median_val))  # Median Absolute Deviation
                    
                    if mad == 0 or pd.isna(mad):
                        # No variation in the feature
                        print(f"         ⚠️  No variation in '{col}' (MAD = {mad}) - setting outlier scores to 0")
                        outlier_scores = pd.Series(0.0, index=df_copy.index)
                    else:
                        outlier_scores = 0.6745 * np.abs((feature_data - median_val) / mad)
                        print(f"         ✅ Modified z-score outlier detection: median={median_val:.2f}, MAD={mad:.2f}")

                # Add the outlier score column
                outlier_col_name = f'{col}_Outlier_Score'
                df_copy[outlier_col_name] = outlier_scores
                created_features.append(outlier_col_name)
                
                feature_time = time.time() - feature_start
                print(f"         ⏱️  Processed in {self._format_time(feature_time)}")

            except Exception as e:
                print(f"         ❌ Error processing '{col}': {str(e)}")
                failed_features.append(col)
                continue

        # Summary
        step_time = time.time() - step_start
        success_count = len(created_features)
        total_count = len(numerical_cols)
        
        print(f"  ✅ Created {success_count} individual outlier detection features from {success_count}/{total_count} features in {self._format_time(step_time)}")
        
        if created_features:
            print(f"      ✅ Successfully created outlier detection features:")
            for feature in created_features:
                original_feature = feature.replace('_Outlier_Score', '')
                print(f"         - {original_feature}: 1 outlier score")
        
        if failed_features:
            print(f"      ❌ Failed features: {', '.join(failed_features)}")
        
        return df_copy

    def add_feature_clustering(self, df: pd.DataFrame, features: List[str] = None, n_clusters: int = 3) -> pd.DataFrame:
        """
        Performs K-Means clustering on pairs of specified numerical features and adds cluster statistics as new columns.

        This function performs clustering on each pair of features separately and creates cluster-based 
        statistics (mean, max, min, median, mean error, standard error) for each pair, providing comprehensive cluster information.

        Args:
            df (pd.DataFrame): The input DataFrame.
            features (List[str]): List of feature names to use for pairwise clustering.
                                If None, uses all numerical columns.
            n_clusters (int): The number of clusters (K) to form for each pair. Defaults to 3.

        Returns:
            pd.DataFrame: The original DataFrame with new cluster statistic columns appended.
                         For each feature pair, creates 6 columns: '{feature1}_{feature2}_Cluster_Mean',
                         '{feature1}_{feature2}_Cluster_Max', '{feature1}_{feature2}_Cluster_Min', 
                         '{feature1}_{feature2}_Cluster_Median', '{feature1}_{feature2}_Cluster_MeanError',
                         '{feature1}_{feature2}_Cluster_StandardError'.
        """
        step_start = time.time()
        
        df_copy = df.copy()

        # Validate n_clusters parameter
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            print(f"  ⚠️  Error: n_clusters must be a positive integer. Got: {n_clusters}")
            step_time = time.time() - step_start
            print(f"  ❌ Pairwise clustering feature creation failed in {self._format_time(step_time)}")
            return df_copy

        # Determine which features to use
        if features is None:
            # Use all numerical columns if no specific features provided
            numerical_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
            # Exclude target column if present
            if 'Calories' in numerical_cols:
                numerical_cols.remove('Calories')
        else:
            # Use specified features, but verify they exist and are numerical
            available_features = df_copy.select_dtypes(include=np.number).columns.tolist()
            numerical_cols = [col for col in features if col in available_features]
            missing_features = [col for col in features if col not in available_features]
            if missing_features:
                print(f"  ⚠️  Warning: Features not found or not numerical: {', '.join(missing_features)}")

        if len(numerical_cols) < 2:
            print(f"  ⚠️  Warning: Need at least 2 numerical columns for pairwise clustering. Found {len(numerical_cols)}.")
            step_time = time.time() - step_start
            print(f"  ❌ Pairwise clustering feature creation failed in {self._format_time(step_time)}")
            return df_copy

        # Calculate total number of pairs
        total_pairs = len(numerical_cols) * (len(numerical_cols) - 1) // 2
        print(f"  🔄 Creating pairwise clustering features with {n_clusters} clusters...")
        print(f"      Processing {total_pairs} feature pairs from {len(numerical_cols)} features: {', '.join(numerical_cols)}")

        # Process each pair of features
        created_features = []
        failed_pairs = []
        pair_count = 0

        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                pair_count += 1
                feature1 = numerical_cols[i]
                feature2 = numerical_cols[j]
                pair_start = time.time()
                
                try:
                    print(f"      📊 Pair {pair_count}/{total_pairs}: ({feature1}, {feature2})")
                    
                    # Get the feature pair data
                    pair_data = df_copy[[feature1, feature2]].copy()
                    
                    # Check for NaN values in this specific pair
                    nan_info = {}
                    for col in [feature1, feature2]:
                        nan_count = pair_data[col].isna().sum()
                        total_count = len(pair_data[col])
                        if nan_count > 0:
                            nan_percentage = (nan_count / total_count) * 100
                            nan_info[col] = {'count': nan_count, 'percentage': nan_percentage}
                    
                    if nan_info:
                        print(f"         ⚠️  Found NaN values:")
                        for col, info in nan_info.items():
                            print(f"            - {col}: {info['count']} ({info['percentage']:.1f}%) NaN values")
                    
                    # Handle NaN values for each feature in the pair
                    for col in [feature1, feature2]:
                        if col in nan_info:
                            nan_percentage = nan_info[col]['percentage']
                            if nan_percentage == 100:
                                print(f"         🔧 {col} is 100% NaN - filling with 0")
                                pair_data[col] = 0
                            elif nan_percentage > 50:
                                median_val = pair_data[col].median()
                                if pd.isna(median_val):
                                    median_val = 0
                                print(f"         🔧 {col}: filling {nan_percentage:.1f}% NaN with median ({median_val})")
                                pair_data[col] = pair_data[col].fillna(median_val)
                            else:
                                mean_val = pair_data[col].mean()
                                if pd.isna(mean_val):
                                    mean_val = 0
                                print(f"         🔧 {col}: filling {nan_percentage:.1f}% NaN with mean ({mean_val:.2f})")
                                pair_data[col] = pair_data[col].fillna(mean_val)
                    
                    # Final check for any remaining NaN values in the pair
                    remaining_nans = pair_data.isna().sum().sum()
                    if remaining_nans > 0:
                        print(f"         🔧 Filling {remaining_nans} remaining NaN values with 0")
                        pair_data = pair_data.fillna(0)
                    
                    # Check if there are enough samples for clustering
                    if len(pair_data) < n_clusters:
                        print(f"         ⚠️  Not enough samples ({len(pair_data)}) for {n_clusters} clusters")
                        failed_pairs.append((feature1, feature2))
                        continue
                    
                    # Scale the feature pair
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(pair_data)
                    scaled_df = pd.DataFrame(scaled_data, columns=[feature1, feature2], index=df_copy.index)
                    
                    # Perform K-Means clustering on the pair
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                    cluster_labels = kmeans.fit_predict(scaled_df)
                    
                    # Calculate distance of each point from its cluster centroid
                    centroids = kmeans.cluster_centers_
                    distances = []
                    for idx, label in enumerate(cluster_labels):
                        point = scaled_df.iloc[idx].values
                        centroid = centroids[label]
                        distance = np.sqrt(np.sum((point - centroid) ** 2))
                        distances.append(distance)
                    
                    distances = np.array(distances)
                    
                    # For each cluster, calculate statistics of distances
                    cluster_stats = {}
                    for cluster_id in range(n_clusters):
                        cluster_mask = cluster_labels == cluster_id
                        if np.any(cluster_mask):
                            cluster_distances = distances[cluster_mask]
                            cluster_mean = np.mean(cluster_distances)
                            cluster_std = np.std(cluster_distances, ddof=1) if len(cluster_distances) > 1 else 0.0
                            cluster_count = len(cluster_distances)
                            
                            cluster_stats[cluster_id] = {
                                'mean': cluster_mean,
                                'max': np.max(cluster_distances),
                                'min': np.min(cluster_distances),
                                'median': np.median(cluster_distances),
                                'mean_error': np.mean(np.abs(cluster_distances - cluster_mean)),  # Mean absolute deviation
                                'standard_error': cluster_std / np.sqrt(cluster_count) if cluster_count > 0 else 0.0  # Standard error of the mean
                            }
                        else:
                            # Empty cluster - shouldn't happen with sklearn, but just in case
                            cluster_stats[cluster_id] = {
                                'mean': 0.0, 'max': 0.0, 'min': 0.0, 'median': 0.0, 'mean_error': 0.0, 'standard_error': 0.0
                            }
                    
                    # Assign cluster statistics to each row
                    pair_prefix = f"{feature1}_{feature2}"
                    
                    mean_values = np.array([cluster_stats[label]['mean'] for label in cluster_labels])
                    max_values = np.array([cluster_stats[label]['max'] for label in cluster_labels])
                    min_values = np.array([cluster_stats[label]['min'] for label in cluster_labels])
                    median_values = np.array([cluster_stats[label]['median'] for label in cluster_labels])
                    mean_error_values = np.array([cluster_stats[label]['mean_error'] for label in cluster_labels])
                    standard_error_values = np.array([cluster_stats[label]['standard_error'] for label in cluster_labels])
                    
                    df_copy[f'{pair_prefix}_Cluster_Mean'] = mean_values
                    df_copy[f'{pair_prefix}_Cluster_Max'] = max_values
                    df_copy[f'{pair_prefix}_Cluster_Min'] = min_values
                    df_copy[f'{pair_prefix}_Cluster_Median'] = median_values
                    df_copy[f'{pair_prefix}_Cluster_MeanError'] = mean_error_values
                    df_copy[f'{pair_prefix}_Cluster_StandardError'] = standard_error_values
                    
                    created_features.extend([
                        f'{pair_prefix}_Cluster_Mean',
                        f'{pair_prefix}_Cluster_Max', 
                        f'{pair_prefix}_Cluster_Min',
                        f'{pair_prefix}_Cluster_Median',
                        f'{pair_prefix}_Cluster_MeanError',
                        f'{pair_prefix}_Cluster_StandardError'
                    ])
                    
                    pair_time = time.time() - pair_start
                    print(f"         ✅ Created 6 cluster features in {self._format_time(pair_time)}")
                    print(f"            Cluster distribution: {np.bincount(cluster_labels)}")
                    
                except Exception as e:
                    print(f"         ❌ Error processing pair ({feature1}, {feature2}): {str(e)}")
                    failed_pairs.append((feature1, feature2))
                    continue

        # Summary
        step_time = time.time() - step_start
        success_pairs = (total_pairs - len(failed_pairs))
        total_features_created = len(created_features)
        
        print(f"  ✅ Created {total_features_created} pairwise clustering features from {success_pairs}/{total_pairs} pairs in {self._format_time(step_time)}")
        
        if created_features:
            # Group features by pair for better readability
            feature_pairs = {}
            for feature in created_features:
                # Extract pair name (everything before the last underscore part)
                parts = feature.split('_')
                if len(parts) >= 4:  # feature1_feature2_Cluster_Stat
                    pair_name = '_'.join(parts[:-2])  # feature1_feature2
                    if pair_name not in feature_pairs:
                        feature_pairs[pair_name] = []
                    feature_pairs[pair_name].append(feature)
            
            print(f"      ✅ Successfully created features for pairs:")
            for pair_name, pair_features in feature_pairs.items():
                print(f"         - {pair_name}: {len(pair_features)} features")
        
        if failed_pairs:
            print(f"      ❌ Failed pairs: {', '.join([f'({f1}, {f2})' for f1, f2 in failed_pairs])}")
        
        return df_copy

    def add_feature_pcaFeatures(self, df: pd.DataFrame, features: List[str] = None, n_components: int = 3) -> pd.DataFrame:
        """
        Performs Principal Component Analysis (PCA) on the specified numerical features 
        and adds the principal components as new features.

        This function scales the specified features, performs PCA to extract the most 
        important components, and adds them as new features for dimensionality reduction.

        Args:
            df (pd.DataFrame): The input DataFrame.
            features (List[str]): List of feature names to use for PCA.
                                If None, uses all numerical columns.
            n_components (int): The number of principal components to extract.
                              Must be less than or equal to the number of features. Defaults to 3.

        Returns:
            pd.DataFrame: The original DataFrame with the new PCA feature columns appended.
        """
        step_start = time.time()
        
        df_copy = df.copy()

        # Determine which features to use
        if features is None:
            # Use all numerical columns if no specific features provided
            numerical_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
            # Exclude target column if present
            if 'Calories' in numerical_cols:
                numerical_cols.remove('Calories')
        else:
            # Use specified features, but verify they exist and are numerical
            available_features = df_copy.select_dtypes(include=np.number).columns.tolist()
            numerical_cols = [col for col in features if col in available_features]
            missing_features = [col for col in features if col not in available_features]
            if missing_features:
                print(f"  ⚠️  Warning: Features not found or not numerical: {', '.join(missing_features)}")

        print(f"  🔄 Creating PCA features with {n_components} components from {len(numerical_cols)} features...")
        print(f"      Features: {', '.join(numerical_cols)}")

        if not numerical_cols:
            print(f"  ⚠️  Warning: No numerical columns found. Returning original DataFrame.")
            step_time = time.time() - step_start
            print(f"  ❌ PCA feature creation failed in {self._format_time(step_time)}")
            return df_copy

        # Validate n_components parameter
        if not isinstance(n_components, int) or n_components <= 0:
            print(f"  ⚠️  Error: n_components must be a positive integer. Got: {n_components}")
            step_time = time.time() - step_start
            print(f"  ❌ PCA feature creation failed in {self._format_time(step_time)}")
            return df_copy

        # Adjust n_components if it's greater than available features
        if n_components > len(numerical_cols):
            print(f"  ⚠️  Warning: n_components ({n_components}) > available features ({len(numerical_cols)}). Using {len(numerical_cols)} components.")
            n_components = len(numerical_cols)

        if n_components == 0:
            print(f"  ⚠️  Warning: No principal components to extract. Returning original DataFrame.")
            step_time = time.time() - step_start
            print(f"  ❌ PCA feature creation failed in {self._format_time(step_time)}")
            return df_copy

        # Select only numerical columns for PCA
        df_numerical = df_copy[numerical_cols]

        # Check for NaN values in each column before processing
        nan_info = {}
        for col in numerical_cols:
            nan_count = df_numerical[col].isna().sum()
            total_count = len(df_numerical[col])
            if nan_count > 0:
                nan_percentage = (nan_count / total_count) * 100
                nan_info[col] = {'count': nan_count, 'percentage': nan_percentage, 'total': total_count}

        if nan_info:
            print(f"  ⚠️  Found NaN values in {len(nan_info)} columns:")
            for col, info in nan_info.items():
                print(f"      - {col}: {info['count']}/{info['total']} ({info['percentage']:.1f}%) NaN values")

        # Handle potential NaNs in numerical columns
        # Use different strategies based on the amount of missing data
        for col in numerical_cols:
            if col in nan_info:
                nan_percentage = nan_info[col]['percentage']
                if nan_percentage == 100:
                    # All values are NaN - fill with 0
                    print(f"      🔧 Column '{col}' is 100% NaN - filling with 0")
                    df_numerical[col] = 0
                elif nan_percentage > 50:
                    # More than 50% NaN - fill with median (more robust than mean)
                    median_val = df_numerical[col].median()
                    if pd.isna(median_val):
                        median_val = 0  # Fallback if median is also NaN
                    print(f"      🔧 Column '{col}' is {nan_percentage:.1f}% NaN - filling with median ({median_val})")
                    df_numerical[col] = df_numerical[col].fillna(median_val)
                else:
                    # Less than 50% NaN - fill with mean
                    mean_val = df_numerical[col].mean()
                    if pd.isna(mean_val):
                        mean_val = 0  # Fallback if mean is also NaN
                    print(f"      🔧 Column '{col}' is {nan_percentage:.1f}% NaN - filling with mean ({mean_val:.2f})")
                    df_numerical[col] = df_numerical[col].fillna(mean_val)

        # Final check for any remaining NaN values
        remaining_nans = df_numerical.isna().sum().sum()
        if remaining_nans > 0:
            print(f"  ⚠️  Warning: {remaining_nans} NaN values still remain after imputation. Filling with 0.")
            df_numerical = df_numerical.fillna(0)

        # Verify no NaN values remain before PCA
        final_nan_check = df_numerical.isna().sum().sum()
        if final_nan_check > 0:
            print(f"  ❌ Error: Still have {final_nan_check} NaN values after all imputation attempts.")
            step_time = time.time() - step_start
            print(f"  ❌ PCA feature creation failed in {self._format_time(step_time)}")
            return df_copy

        # Scale the numerical features
        # Scaling is crucial for PCA to prevent features with larger magnitudes 
        # from dominating the principal components
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numerical)
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_cols, index=df_copy.index)

        # Perform Principal Component Analysis
        try:
            pca = PCA(n_components=n_components, random_state=42)
            principal_components = pca.fit_transform(scaled_df)

            # Create new DataFrame for principal components
            pca_columns = [f'PC{i+1}' for i in range(n_components)]
            df_pca = pd.DataFrame(data=principal_components, columns=pca_columns, index=df_copy.index)

            # Concatenate the original DataFrame with the new PCA features
            df_result = pd.concat([df_copy, df_pca], axis=1)

            # Calculate explained variance for information
            explained_variance_ratio = pca.explained_variance_ratio_
            total_variance_explained = np.sum(explained_variance_ratio)

            step_time = time.time() - step_start
            print(f"  ✅ Created {n_components} PCA features in {self._format_time(step_time)}")
            print(f"      Components: {', '.join(pca_columns)}")
            print(f"      Total variance explained: {total_variance_explained:.1%}")
            
        except Exception as e:
            print(f"  ⚠️  Error during PCA: {str(e)}")
            step_time = time.time() - step_start
            print(f"  ❌ PCA feature creation failed in {self._format_time(step_time)}")
            return df_copy
        
        return df_result

    def add_feature_quantileIndicator(self, df: pd.DataFrame, features: List[str] = None, n_quantiles: int = 4) -> pd.DataFrame:
        """
        Calculates quantile-based statistics for each specified numerical feature and adds them as new columns.

        This function divides each feature into quantiles and assigns statistical measures (mean, median, max, min,
        mean error, standard error) of the quantile group that each data point belongs to, providing comprehensive quantile-based information.

        Args:
            df (pd.DataFrame): The input DataFrame.
            features (List[str]): List of feature names to create quantile statistics for.
                                If None, uses all numerical columns.
            n_quantiles (int): The number of quantiles to divide the data into.
                             Must be an integer greater than 1. Defaults to 4 (quartiles).

        Returns:
            pd.DataFrame: The original DataFrame with new quantile statistic columns appended.
                         For each feature, creates 6 columns: '{feature}_Quantile_Mean',
                         '{feature}_Quantile_Median', '{feature}_Quantile_Max', '{feature}_Quantile_Min',
                         '{feature}_Quantile_MeanError', '{feature}_Quantile_StandardError'.
        """
        step_start = time.time()
        
        df_copy = df.copy()

        # Validate n_quantiles parameter
        if not isinstance(n_quantiles, int) or n_quantiles <= 1:
            print(f"  ⚠️  Error: n_quantiles must be an integer greater than 1. Got: {n_quantiles}")
            step_time = time.time() - step_start
            print(f"  ❌ Quantile statistic feature creation failed in {self._format_time(step_time)}")
            return df_copy

        # Determine which features to use
        if features is None:
            # Use all numerical columns if no specific features provided
            numerical_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
            # Exclude target column if present
            if 'Calories' in numerical_cols:
                numerical_cols.remove('Calories')
        else:
            # Use specified features, but verify they exist and are numerical
            available_features = df_copy.select_dtypes(include=np.number).columns.tolist()
            numerical_cols = [col for col in features if col in available_features]
            missing_features = [col for col in features if col not in available_features]
            if missing_features:
                print(f"  ⚠️  Warning: Features not found or not numerical: {', '.join(missing_features)}")

        print(f"  🔄 Creating quantile statistics with {n_quantiles} quantiles for {len(numerical_cols)} features...")
        print(f"      Processing {len(numerical_cols)} features: {', '.join(numerical_cols)}")

        if not numerical_cols:
            print(f"  ⚠️  Warning: No numerical columns found. Returning original DataFrame.")
            step_time = time.time() - step_start
            print(f"  ❌ Quantile statistic feature creation failed in {self._format_time(step_time)}")
            return df_copy

        # Process each feature individually
        created_features = []
        failed_features = []

        for col in numerical_cols:
            feature_start = time.time()
            
            try:
                print(f"      📊 Processing '{col}' for quantile statistics...")
                
                # Get the feature data
                feature_data = df_copy[col].copy()
                
                # Check for NaN values
                nan_count = feature_data.isna().sum()
                total_count = len(feature_data)
                
                if nan_count > 0:
                    nan_percentage = (nan_count / total_count) * 100
                    print(f"         ⚠️  Found {nan_count}/{total_count} ({nan_percentage:.1f}%) NaN values")
                    
                    # Handle NaN values for quantile calculation
                    if nan_percentage == 100:
                        # All values are NaN - set all quantile stats to 0
                        print(f"         🔧 All values are NaN - setting quantile statistics to 0")
                        df_copy[f'{col}_Quantile_Mean'] = 0.0
                        df_copy[f'{col}_Quantile_Median'] = 0.0
                        df_copy[f'{col}_Quantile_Max'] = 0.0
                        df_copy[f'{col}_Quantile_Min'] = 0.0
                        df_copy[f'{col}_Quantile_MeanError'] = 0.0
                        df_copy[f'{col}_Quantile_StandardError'] = 0.0
                        created_features.extend([
                            f'{col}_Quantile_Mean', f'{col}_Quantile_Median', 
                            f'{col}_Quantile_Max', f'{col}_Quantile_Min',
                            f'{col}_Quantile_MeanError', f'{col}_Quantile_StandardError'
                        ])
                        continue
                
                # Use only non-NaN values for quantile calculation
                non_nan_data = feature_data.dropna()
                
                if non_nan_data.empty:
                    print(f"         ❌ No valid values for quantile calculation")
                    failed_features.append(col)
                    continue
                
                # Check if we have enough unique values for the requested quantiles
                unique_values = non_nan_data.nunique()
                if unique_values < n_quantiles:
                    print(f"         ⚠️  Only {unique_values} unique values, reducing quantiles from {n_quantiles} to {unique_values}")
                    effective_quantiles = unique_values
                else:
                    effective_quantiles = n_quantiles
                
                # Calculate quantile bins using pd.qcut
                try:
                    quantile_bins = pd.qcut(non_nan_data, q=effective_quantiles, labels=False, duplicates='drop')
                    
                    # Calculate statistics for each quantile group
                    quantile_stats = {}
                    for quantile_id in range(effective_quantiles):
                        quantile_mask = quantile_bins == quantile_id
                        if np.any(quantile_mask):
                            quantile_values = non_nan_data[quantile_mask]
                            quantile_mean = quantile_values.mean()
                            quantile_std = quantile_values.std(ddof=1) if len(quantile_values) > 1 else 0.0
                            quantile_count = len(quantile_values)
                            
                            quantile_stats[quantile_id] = {
                                'mean': quantile_mean,
                                'median': quantile_values.median(),
                                'max': quantile_values.max(),
                                'min': quantile_values.min(),
                                'mean_error': np.mean(np.abs(quantile_values - quantile_mean)),  # Mean absolute deviation
                                'standard_error': quantile_std / np.sqrt(quantile_count) if quantile_count > 0 else 0.0  # Standard error of the mean
                            }
                            print(f"         📈 Quantile {quantile_id}: {len(quantile_values)} values, "
                                  f"range [{quantile_stats[quantile_id]['min']:.2f}, {quantile_stats[quantile_id]['max']:.2f}]")
                    
                    # Initialize result columns with NaN
                    df_copy[f'{col}_Quantile_Mean'] = np.nan
                    df_copy[f'{col}_Quantile_Median'] = np.nan
                    df_copy[f'{col}_Quantile_Max'] = np.nan
                    df_copy[f'{col}_Quantile_Min'] = np.nan
                    df_copy[f'{col}_Quantile_MeanError'] = np.nan
                    df_copy[f'{col}_Quantile_StandardError'] = np.nan
                    
                    # Assign quantile statistics to non-NaN values
                    for idx, quantile_id in enumerate(quantile_bins):
                        if not pd.isna(quantile_id):
                            original_idx = non_nan_data.index[idx]
                            df_copy.loc[original_idx, f'{col}_Quantile_Mean'] = quantile_stats[quantile_id]['mean']
                            df_copy.loc[original_idx, f'{col}_Quantile_Median'] = quantile_stats[quantile_id]['median']
                            df_copy.loc[original_idx, f'{col}_Quantile_Max'] = quantile_stats[quantile_id]['max']
                            df_copy.loc[original_idx, f'{col}_Quantile_Min'] = quantile_stats[quantile_id]['min']
                            df_copy.loc[original_idx, f'{col}_Quantile_MeanError'] = quantile_stats[quantile_id]['mean_error']
                            df_copy.loc[original_idx, f'{col}_Quantile_StandardError'] = quantile_stats[quantile_id]['standard_error']
                    
                    # Handle NaN values in the result - fill with overall feature statistics
                    if nan_count > 0:
                        overall_mean = non_nan_data.mean()
                        overall_median = non_nan_data.median() 
                        overall_max = non_nan_data.max()
                        overall_min = non_nan_data.min()
                        overall_std = non_nan_data.std(ddof=1) if len(non_nan_data) > 1 else 0.0
                        overall_mean_error = np.mean(np.abs(non_nan_data - overall_mean))
                        overall_standard_error = overall_std / np.sqrt(len(non_nan_data)) if len(non_nan_data) > 0 else 0.0
                        
                        print(f"         🔧 Filling {nan_count} NaN positions with overall feature statistics")
                        df_copy[f'{col}_Quantile_Mean'] = df_copy[f'{col}_Quantile_Mean'].fillna(overall_mean)
                        df_copy[f'{col}_Quantile_Median'] = df_copy[f'{col}_Quantile_Median'].fillna(overall_median)
                        df_copy[f'{col}_Quantile_Max'] = df_copy[f'{col}_Quantile_Max'].fillna(overall_max)
                        df_copy[f'{col}_Quantile_Min'] = df_copy[f'{col}_Quantile_Min'].fillna(overall_min)
                        df_copy[f'{col}_Quantile_MeanError'] = df_copy[f'{col}_Quantile_MeanError'].fillna(overall_mean_error)
                        df_copy[f'{col}_Quantile_StandardError'] = df_copy[f'{col}_Quantile_StandardError'].fillna(overall_standard_error)
                    
                    created_features.extend([
                        f'{col}_Quantile_Mean', f'{col}_Quantile_Median', 
                        f'{col}_Quantile_Max', f'{col}_Quantile_Min',
                        f'{col}_Quantile_MeanError', f'{col}_Quantile_StandardError'
                    ])
                    
                    feature_time = time.time() - feature_start
                    print(f"         ✅ Created 6 quantile statistic features in {self._format_time(feature_time)}")
                
                except ValueError as e:
                    print(f"         ❌ Error calculating quantiles: {str(e)}")
                    print(f"            This might happen with insufficient unique values or other edge cases")
                    failed_features.append(col)
                    continue

            except Exception as e:
                print(f"         ❌ Error processing '{col}': {str(e)}")
                failed_features.append(col)
                continue

        # Summary
        step_time = time.time() - step_start
        success_count = len(created_features)
        total_count = len(numerical_cols)
        
        print(f"  ✅ Created {success_count} individual outlier detection features from {success_count}/{total_count} features in {self._format_time(step_time)}")
        
        if created_features:
            print(f"      ✅ Successfully created outlier detection features:")
            for feature in created_features:
                original_feature = feature.replace('_Outlier_Score', '')
                print(f"         - {original_feature}: 1 outlier score")
        
        if failed_features:
            print(f"      ❌ Failed features: {', '.join(failed_features)}")
        
        return df_copy

    def create_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_types: List[str] = None, 
                       crossTerms_features: List[str] = None, divisionTerms_features: List[str] = None,
                       squareCubeTerms_features: List[str] = None, subjectKnowledge_features: List[str] = None,
                       outlierDetection_features: List[str] = None, outlier_score_method: str = 'z_score',
                       clustering_features: List[str] = None, n_clusters: int = 3,
                       pca_features: List[str] = None, n_components: int = 3,
                       quantile_features: List[str] = None, n_quantiles: int = 4,
                       logTerm_features: List[str] = None, exponentialTerm_features: List[str] = None,
                       sqrtTerm_features: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create features based on specified configuration
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            feature_types: List of feature types to include (e.g., ['cross_terms', 'division_terms', 'square_cube', 'subject_knowledge', 'outlier_detection', 'clustering', 'pca', 'quantile_indicators', 'log_terms', 'exponential_terms', 'sqrt_terms'])
            crossTerms_features: List of features for cross term multiplication
            divisionTerms_features: List of features for division terms
            squareCubeTerms_features: List of features for square and cube terms
            subjectKnowledge_features: List of subject knowledge features to create
            outlierDetection_features: List of features for outlier detection calculation
            outlier_score_method: Method for outlier score calculation ('z_score' or 'modified_z_score', default: 'z_score')
            clustering_features: List of features for clustering analysis
            n_clusters: Number of clusters for K-Means clustering (default: 3)
            pca_features: List of features for PCA analysis
            n_components: Number of principal components to extract (default: 3)
            quantile_features: List of features for quantile indicator creation
            n_quantiles: Number of quantiles to create (default: 4)
            logTerm_features: List of features for log term creation
            exponentialTerm_features: List of features for exponential term creation
            sqrtTerm_features: List of features for square root term creation
            
        Returns:
            Tuple of processed training and test dataframes
        """
        # Start timing and setup
        overall_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"🚀 FEATURE ENGINEERING PIPELINE STARTED")
        print(f"{'='*80}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Input shapes - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Create copies
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        # Count total steps
        total_steps = 1  # Basic preprocessing
        if feature_types:
            total_steps += len([ft for ft in feature_types if ft in ['cross_terms', 'division_terms', 'square_cube', 'subject_knowledge', 'outlier_detection', 'clustering', 'pca', 'quantile_indicators', 'log_terms', 'exponential_terms', 'sqrt_terms']])
        
        current_step = 0
        
        preprocessing_start = time.time()
        
        preprocessing_time = time.time() - preprocessing_start
        print(f"  ✅ Basic preprocessing completed in {self._format_time(preprocessing_time)}")
        current_step += 1
        
        # If no feature types specified, return basic preprocessing only
        if not feature_types:
            total_time = time.time() - overall_start_time
            print(f"\n📊 FEATURE ENGINEERING COMPLETED in {self._format_time(total_time)}")
            print(f"Final shapes - Train: {train_processed.shape}, Test: {test_processed.shape}")
            return train_processed, test_processed
        
        print(f"\n📋 Feature engineering plan: {', '.join(feature_types)}")
        
        # Apply feature generation based on configuration
        if 'cross_terms' in feature_types:
            self._log_progress("Cross-term features", current_step, total_steps, overall_start_time)
            # Default to all numerical features if not specified
            if crossTerms_features is None:
                crossTerms_features = [col for col in train_processed.select_dtypes(include=['int64', 'float64']).columns 
                                     if col not in ['Calories']]  # Exclude target if present
            
            print(f"    📝 Using features: {', '.join(crossTerms_features)}")
            train_processed = self.add_feature_CrossTerms(train_processed, crossTerms_features)
            test_processed = self.add_feature_CrossTerms(test_processed, crossTerms_features)
            current_step += 1
        
        if 'division_terms' in feature_types:
            self._log_progress("Division-term features", current_step, total_steps, overall_start_time)
            # Default to all numerical features if not specified
            if divisionTerms_features is None:
                divisionTerms_features = [col for col in train_processed.select_dtypes(include=['int64', 'float64']).columns 
                                        if col not in ['Calories']]  # Exclude target if present
            
            print(f"    📝 Using features: {', '.join(divisionTerms_features)}")
            train_processed = self.add_feature_divisionTerms(train_processed, divisionTerms_features)
            test_processed = self.add_feature_divisionTerms(test_processed, divisionTerms_features)
            current_step += 1
        
        if 'square_cube' in feature_types:
            self._log_progress("Polynomial (square/cube) features", current_step, total_steps, overall_start_time)
            # Default to all numerical features if not specified
            if squareCubeTerms_features is None:
                squareCubeTerms_features = [col for col in train_processed.select_dtypes(include=['int64', 'float64']).columns 
                                          if col not in ['Calories']]  # Exclude target if present
            
            print(f"    📝 Using features: {', '.join(squareCubeTerms_features)}")
            train_processed = self.add_feature_squareCubeTerms(train_processed, squareCubeTerms_features)
            test_processed = self.add_feature_squareCubeTerms(test_processed, squareCubeTerms_features)
            current_step += 1
        
        if 'subject_knowledge' in feature_types:
            self._log_progress("Subject knowledge features", current_step, total_steps, overall_start_time)
            # Default subject knowledge features if not provided
            if subjectKnowledge_features is None:
                subjectKnowledge_features = ['BMIscaled', 'BMR', 'Metabolic_Efficiency', 'Cardio_Stress', 'Thermic_Effect', 'Power_Output']
            
            print(f"    📝 Requested features: {', '.join(subjectKnowledge_features)}")
            train_processed = self.add_feature_subjectKnowledge(train_processed, subjectKnowledge_features)
            test_processed = self.add_feature_subjectKnowledge(test_processed, subjectKnowledge_features)
            current_step += 1
        
        if 'outlier_detection' in feature_types:
            self._log_progress("Outlier detection features", current_step, total_steps, overall_start_time)
            # Default to all numerical features if not specified
            if outlierDetection_features is None:
                outlierDetection_features = [col for col in train_processed.select_dtypes(include=['int64', 'float64']).columns 
                                           if col not in ['Calories']]  # Exclude target if present
            
            print(f"    📝 Using features: {', '.join(outlierDetection_features)}")
            print(f"    📝 Outlier score method: '{outlier_score_method}'")
            train_processed = self.add_feature_outlierDetection(train_processed, outlierDetection_features, outlier_score_method)
            test_processed = self.add_feature_outlierDetection(test_processed, outlierDetection_features, outlier_score_method)
            current_step += 1
        
        if 'clustering' in feature_types:
            self._log_progress("Clustering features", current_step, total_steps, overall_start_time)
            # Default to all numerical features if not specified
            if clustering_features is None:
                clustering_features = [col for col in train_processed.select_dtypes(include=['int64', 'float64']).columns 
                                     if col not in ['Calories']]  # Exclude target if present
            
            print(f"    📝 Using features: {', '.join(clustering_features)}")
            print(f"    📝 Number of clusters: {n_clusters}")
            train_processed = self.add_feature_clustering(train_processed, clustering_features, n_clusters)
            test_processed = self.add_feature_clustering(test_processed, clustering_features, n_clusters)
            current_step += 1
        
        if 'pca' in feature_types:
            self._log_progress("PCA features", current_step, total_steps, overall_start_time)
            # Default to all numerical features if not specified
            if pca_features is None:
                pca_features = [col for col in train_processed.select_dtypes(include=['int64', 'float64']).columns 
                              if col not in ['Calories']]  # Exclude target if present
            
            print(f"    📝 Using features: {', '.join(pca_features)}")
            print(f"    📝 Number of components: {n_components}")
            train_processed = self.add_feature_pcaFeatures(train_processed, pca_features, n_components)
            test_processed = self.add_feature_pcaFeatures(test_processed, pca_features, n_components)
            current_step += 1
        
        if 'quantile_indicators' in feature_types:
            self._log_progress("Quantile indicator features", current_step, total_steps, overall_start_time)
            # Default to all numerical features if not specified
            if quantile_features is None:
                quantile_features = [col for col in train_processed.select_dtypes(include=['int64', 'float64']).columns 
                                   if col not in ['Calories']]  # Exclude target if present
            
            print(f"    📝 Using features: {', '.join(quantile_features)}")
            print(f"    📝 Number of quantiles: {n_quantiles}")
            train_processed = self.add_feature_quantileIndicator(train_processed, quantile_features, n_quantiles)
            test_processed = self.add_feature_quantileIndicator(test_processed, quantile_features, n_quantiles)
            current_step += 1
        
        if 'log_terms' in feature_types:
            self._log_progress("Logarithmic features", current_step, total_steps, overall_start_time)
            # Default to all numerical features if not specified
            if logTerm_features is None:
                logTerm_features = [col for col in train_processed.select_dtypes(include=['int64', 'float64']).columns 
                                  if col not in ['Calories']]  # Exclude target if present
            
            print(f"    📝 Using features: {', '.join(logTerm_features)}")
            train_processed = self.add_feature_logTerms(train_processed, logTerm_features)
            test_processed = self.add_feature_logTerms(test_processed, logTerm_features)
            current_step += 1
        
        if 'exponential_terms' in feature_types:
            self._log_progress("Exponential features", current_step, total_steps, overall_start_time)
            # Default to all numerical features if not specified
            if exponentialTerm_features is None:
                exponentialTerm_features = [col for col in train_processed.select_dtypes(include=['int64', 'float64']).columns 
                                          if col not in ['Calories']]  # Exclude target if present
            
            print(f"    📝 Using features: {', '.join(exponentialTerm_features)}")
            train_processed = self.add_feature_exponentialTerms(train_processed, exponentialTerm_features)
            test_processed = self.add_feature_exponentialTerms(test_processed, exponentialTerm_features)
            current_step += 1
        
        if 'sqrt_terms' in feature_types:
            self._log_progress("Square root features", current_step, total_steps, overall_start_time)
            # Default to all numerical features if not specified
            if sqrtTerm_features is None:
                sqrtTerm_features = [col for col in train_processed.select_dtypes(include=['int64', 'float64']).columns 
                                   if col not in ['Calories']]  # Exclude target if present
            
            print(f"    📝 Using features: {', '.join(sqrtTerm_features)}")
            train_processed = self.add_feature_sqrtTerms(train_processed, sqrtTerm_features)
            test_processed = self.add_feature_sqrtTerms(test_processed, sqrtTerm_features)
            current_step += 1
        
        # Final summary
        total_time = time.time() - overall_start_time
        original_features = len(train_df.columns)
        new_features = len(train_processed.columns) - original_features
        
        print(f"\n{'='*80}")
        print(f"🎉 FEATURE ENGINEERING COMPLETED!")
        print(f"{'='*80}")
        print(f"⏱️  Total time: {self._format_time(total_time)}")
        print(f"📊 Original features: {original_features}")
        print(f"🆕 New features created: {new_features}")
        print(f"📈 Final feature count: {len(train_processed.columns)}")
        print(f"📏 Final shapes - Train: {train_processed.shape}, Test: {test_processed.shape}")
        print(f"{'='*80}\n")
        
        return train_processed, test_processed
    
    def create_final_dataset(self, df: pd.DataFrame, selected_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create final dataset with selected features and return both selected and complete feature datasets
        
        Args:
            df: Input dataframe
            selected_features: List of feature names to include
            
        Returns:
            Tuple of (selected_features_df, complete_features_df)
            - selected_features_df: DataFrame with only selected features
            - complete_features_df: DataFrame with all available features
        """
        available_features = [col for col in df.columns if col in selected_features]
        if not available_features:
            raise ValueError("None of the selected features are available in the dataset")
        
        selected_df = df[available_features]
        complete_df = df.copy()
        
        print(f"Created final dataset with {len(available_features)} selected features out of {len(df.columns)} total features")
        
        return selected_df, complete_df
    
    def track_pipeline(self, pipeline_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_types: List[str] = None):
        """
        Track feature pipeline and datasets
        
        Args:
            pipeline_name: Name of the pipeline
            train_df: Training dataframe
            test_df: Test dataframe
            feature_types: List of feature types used
        """
        self.feature_pipelines[pipeline_name] = {
            'feature_types': feature_types or [],
            'feature_count': len(train_df.columns),
            'feature_names': list(train_df.columns)
        }
        
        self.datasets[f"{pipeline_name}_train"] = train_df
        self.datasets[f"{pipeline_name}_test"] = test_df
        
        print(f"Pipeline '{pipeline_name}' tracked with {len(train_df.columns)} features")

def process_features_with_config(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_types: List[str],
    crossTerms_features: Optional[List[str]] = None,
    divisionTerms_features: Optional[List[str]] = None,
    squareCubeTerms_features: Optional[List[str]] = None,
    subjectKnowledge_features: Optional[List[str]] = None,
    outlierDetection_features: Optional[List[str]] = None,
    outlier_score_method: str = 'z_score',
    clustering_features: Optional[List[str]] = None,
    n_clusters: int = 3,
    pca_features: Optional[List[str]] = None,
    n_components: int = 3,
    quantile_features: Optional[List[str]] = None,
    n_quantiles: int = 4,
    logTerm_features: Optional[List[str]] = None,
    exponentialTerm_features: Optional[List[str]] = None,
    sqrtTerm_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process features based on provided configuration
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        feature_types: List of feature types to include (e.g., ['cross_terms', 'division_terms', 'square_cube', 'subject_knowledge', 'outlier_detection', 'clustering', 'pca', 'quantile_indicators', 'log_terms', 'exponential_terms', 'sqrt_terms'])
        crossTerms_features: List of features for cross term multiplication
        divisionTerms_features: List of features for division terms
        squareCubeTerms_features: List of features for square and cube terms
        subjectKnowledge_features: List of subject knowledge features to create
        outlierDetection_features: List of features for outlier detection calculation
        outlier_score_method: Method for outlier score calculation ('z_score' or 'modified_z_score', default: 'z_score')
        clustering_features: List of features for clustering analysis
        n_clusters: Number of clusters for K-Means clustering (default: 3)
        pca_features: List of features for PCA analysis
        n_components: Number of principal components to extract (default: 3)
        quantile_features: List of features for quantile indicator creation
        n_quantiles: Number of quantiles to create (default: 4)
        logTerm_features: List of features for log term creation
        exponentialTerm_features: List of features for exponential term creation
        sqrtTerm_features: List of features for square root term creation
        
    Returns:
        Tuple of (processed_train_df, processed_test_df)
    """
    wrapper_start_time = time.time()
    print(f"\n🔧 PROCESS_FEATURES_WITH_CONFIG - Starting pipeline wrapper")
    
    # Create feature manager and process features
    feature_manager = FeatureManager()
    train_processed, test_processed = feature_manager.create_features(
        train_df, test_df, feature_types, crossTerms_features, divisionTerms_features, 
        squareCubeTerms_features, subjectKnowledge_features, outlierDetection_features, outlier_score_method,
        clustering_features, n_clusters, pca_features, n_components,
        quantile_features, n_quantiles, logTerm_features, exponentialTerm_features, sqrtTerm_features
    )
    
    # Track the pipeline
    feature_manager.track_pipeline("configured_pipeline", train_processed, test_processed, feature_types)
    
    wrapper_time = time.time() - wrapper_start_time
    print(f"🔧 WRAPPER COMPLETED in {feature_manager._format_time(wrapper_time)}")
    print(f"📈 Pipeline tracked as 'configured_pipeline' with {len(train_processed.columns)} features\n")
    
    return train_processed, test_processed



'''
# Define separate feature lists
crossTerms_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp','Sex']
divisionTerms_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
squareCubeTerms_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
subjectKnowledge_features = ['BMIscaled', 'BMR', 'Metabolic_Efficiency', 'Cardio_Stress', 'Thermic_Effect', 'Power_Output']
outlierDetection_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
clustering_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
pca_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
quantile_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
logTerm_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
exponentialTerm_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
sqrtTerm_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


# Simple usage with FeatureManager
feature_manager = FeatureManager()
train_processed, test_processed = feature_manager.create_features(
    train.sample(frac=0.01), test.sample(frac=0.01), 
    feature_types=['cross_terms', 'division_terms', 'square_cube', 'subject_knowledge', 'outlier_detection', 'clustering', 'pca', 'quantile_indicators', 'log_terms', 'exponential_terms', 'sqrt_terms'],
    crossTerms_features=crossTerms_features,
    divisionTerms_features=divisionTerms_features,
    squareCubeTerms_features=squareCubeTerms_features,
    subjectKnowledge_features=subjectKnowledge_features,
    outlierDetection_features=outlierDetection_features,
    outlier_score_method='z_score',
    clustering_features=clustering_features,
    n_clusters=10,  # Custom number of clusters
    pca_features=pca_features,
    n_components=2,  # Extract 2 principal components
    quantile_features=quantile_features,
    n_quantiles=10,  # Create deciles (10 quantiles)
    logTerm_features=logTerm_features,
    exponentialTerm_features=exponentialTerm_features,
    sqrtTerm_features=sqrtTerm_features
)
'''