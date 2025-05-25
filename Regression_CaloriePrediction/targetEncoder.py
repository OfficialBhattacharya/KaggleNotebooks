import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quantile_target_encoder(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                           target_col: str, target_encode: List[str], 
                           n_quantiles: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs quantile-based target encoding for specified features.
    
    Args:
        train_df: Training dataframe with target column
        test_df: Test dataframe without target column
        target_col: Name of the target column
        target_encode: List of feature names to encode
        n_quantiles: Number of quantiles to create (default: 5)
    
    Returns:
        Tuple of (encoded_train_df, encoded_test_df)
    """
    
    print("="*60)
    print("🎯 QUANTILE TARGET ENCODER STARTED")
    print("="*60)
    
    logger.info(f"Starting quantile target encoding for {len(target_encode)} features")
    logger.info(f"Target column: {target_col}")
    logger.info(f"Features to encode: {target_encode}")
    logger.info(f"Number of quantiles: {n_quantiles}")
    
    print(f"📊 Processing {len(target_encode)} features for target encoding")
    print(f"🎯 Target column: {target_col}")
    print(f"📈 Number of quantiles: {n_quantiles}")
    print(f"🔢 Train data shape: {train_df.shape}")
    print(f"🔢 Test data shape: {test_df.shape}")
    print()
    
    # Create copies to avoid modifying original dataframes
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # Verify target column exists in train_df
    if target_col not in train_df.columns:
        error_msg = f"Target column '{target_col}' not found in training data"
        print(f"❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Verify all features exist in both dataframes
    missing_train = [col for col in target_encode if col not in train_df.columns]
    missing_test = [col for col in target_encode if col not in test_df.columns]
    
    if missing_train:
        error_msg = f"Features missing in train_df: {missing_train}"
        print(f"❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    if missing_test:
        error_msg = f"Features missing in test_df: {missing_test}"
        print(f"❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    logger.info("All required columns found in both dataframes")
    print("✅ All required columns found in both dataframes")
    print()
    
    for i, feature in enumerate(target_encode, 1):
        print(f"🔄 Processing feature {i}/{len(target_encode)}: {feature}")
        print("-" * 50)
        
        logger.info(f"\n--- Processing feature: {feature} ---")
        
        # Step 1: Create quantile groups for the entire dataset (train + test)
        print(f"📊 Step 1: Creating quantile groups for {feature}")
        logger.info(f"Step 1: Creating quantile groups for {feature}")
        
        # Combine train and test data for consistent quantile boundaries
        # Reset indices to ensure proper alignment
        train_feature = train_df[feature].reset_index(drop=True)
        test_feature = test_df[feature].reset_index(drop=True)
        combined_feature_data = pd.concat([train_feature, test_feature], ignore_index=True)
        
        print(f"   📈 Combined data shape: {len(combined_feature_data)}")
        print(f"   📊 Feature range: {combined_feature_data.min():.2f} to {combined_feature_data.max():.2f}")
        
        # Create quantiles with better error handling
        try:
            quantiles = pd.qcut(combined_feature_data, q=n_quantiles, labels=False, duplicates='drop')
        except Exception as e:
            print(f"   ⚠️  Warning: pd.qcut failed, using pd.cut instead. Error: {e}")
            logger.warning(f"pd.qcut failed for {feature}, using pd.cut instead: {e}")
            # Fallback to pd.cut if qcut fails
            quantiles = pd.cut(combined_feature_data, bins=n_quantiles, labels=False, duplicates='drop')
        
        # Check for NaN values in quantiles
        nan_count = quantiles.isna().sum()
        if nan_count > 0:
            print(f"   ⚠️  Warning: {nan_count} NaN values in quantiles, filling with mode")
            logger.warning(f"{nan_count} NaN values in quantiles for {feature}")
            # Fill NaN values with the most common quantile
            mode_quantile = quantiles.mode().iloc[0] if len(quantiles.mode()) > 0 else 0
            quantiles = quantiles.fillna(mode_quantile)
        
        # Ensure quantiles are integers
        quantiles = quantiles.astype(int)
        
        # Split quantiles back to train and test with proper indexing
        train_quantiles = quantiles[:len(train_df)].values
        test_quantiles = quantiles[len(train_df):].values
        
        # Verify lengths match
        assert len(train_quantiles) == len(train_df), f"Train quantiles length mismatch: {len(train_quantiles)} vs {len(train_df)}"
        assert len(test_quantiles) == len(test_df), f"Test quantiles length mismatch: {len(test_quantiles)} vs {len(test_df)}"
        
        # Add quantile group columns with proper index alignment
        temp_col_name = f'{feature}_Quantile_Group'
        train_encoded[temp_col_name] = train_quantiles
        test_encoded[temp_col_name] = test_quantiles
        
        # Verify no NaN values in the final columns
        train_nan_count = train_encoded[temp_col_name].isna().sum()
        test_nan_count = test_encoded[temp_col_name].isna().sum()
        
        if train_nan_count > 0 or test_nan_count > 0:
            print(f"   ❌ ERROR: NaN values found in final quantile columns (train: {train_nan_count}, test: {test_nan_count})")
            logger.error(f"NaN values in final quantile columns for {feature}")
        
        n_groups = len(np.unique(train_quantiles))
        logger.info(f"Created {n_groups} quantile groups for {feature}")
        print(f"   ✅ Created {n_groups} quantile groups")
        print(f"   📊 Quantile distribution: {np.bincount(train_quantiles)}")
        
        # Step 2: Calculate statistical measures for each quantile group
        print(f"🧮 Step 2: Calculating {target_col} statistics for each quantile group")
        logger.info(f"Step 2: Calculating {target_col} statistics for each quantile group")
        
        # Debug: Check the quantile group column before groupby
        print(f"   🔍 Debug: Quantile group column stats:")
        print(f"      - Unique values: {sorted(train_encoded[temp_col_name].unique())}")
        print(f"      - Data type: {train_encoded[temp_col_name].dtype}")
        print(f"      - NaN count: {train_encoded[temp_col_name].isna().sum()}")
        
        # Calculate multiple statistical measures for each quantile group using training data
        try:
            grouped = train_encoded.groupby(temp_col_name)[target_col]
            
            # Calculate all statistics
            quantile_means = grouped.mean()
            quantile_medians = grouped.median()
            quantile_stds = grouped.std()
            quantile_mad = grouped.apply(lambda x: np.mean(np.abs(x - x.mean())))  # Mean Absolute Deviation
            
            # Ensure we have valid statistics
            for stat_name, stat_values in [('means', quantile_means), ('medians', quantile_medians), 
                                         ('stds', quantile_stds), ('mad', quantile_mad)]:
                if stat_values.isna().any():
                    print(f"   ⚠️  Warning: Some quantile {stat_name} are NaN")
                    logger.warning(f"Some quantile {stat_name} are NaN for {feature}")
                    if stat_name == 'stds':
                        # For std, fill with overall std
                        stat_values.fillna(train_df[target_col].std(), inplace=True)
                    elif stat_name == 'mad':
                        # For MAD, fill with overall MAD
                        overall_mad = np.mean(np.abs(train_df[target_col] - train_df[target_col].mean()))
                        stat_values.fillna(overall_mad, inplace=True)
                    else:
                        # For mean and median, fill with overall values
                        overall_val = train_df[target_col].mean() if stat_name == 'means' else train_df[target_col].median()
                        stat_values.fillna(overall_val, inplace=True)
                
        except Exception as e:
            print(f"   ❌ ERROR in groupby operation: {e}")
            logger.error(f"Groupby failed for {feature}: {e}")
            # Fallback: use overall statistics for all groups
            unique_groups = train_encoded[temp_col_name].unique()
            overall_mean = train_df[target_col].mean()
            overall_median = train_df[target_col].median()
            overall_std = train_df[target_col].std()
            overall_mad = np.mean(np.abs(train_df[target_col] - overall_mean))
            
            quantile_means = pd.Series([overall_mean] * len(unique_groups), index=unique_groups)
            quantile_medians = pd.Series([overall_median] * len(unique_groups), index=unique_groups)
            quantile_stds = pd.Series([overall_std] * len(unique_groups), index=unique_groups)
            quantile_mad = pd.Series([overall_mad] * len(unique_groups), index=unique_groups)
        
        # Display statistics
        logger.info(f"Quantile statistics for {feature}:")
        print(f"   📈 Quantile statistics for {feature}:")
        for q_group in sorted(quantile_means.index):
            mean_val = quantile_means[q_group]
            median_val = quantile_medians[q_group]
            std_val = quantile_stds[q_group]
            mad_val = quantile_mad[q_group]
            
            logger.info(f"  Quantile {q_group}: Mean={mean_val:.4f}, Median={median_val:.4f}, Std={std_val:.4f}, MAD={mad_val:.4f}")
            print(f"      Quantile {q_group}: Mean={mean_val:.4f}, Median={median_val:.4f}, Std={std_val:.4f}, MAD={mad_val:.4f}")
        
        # Verify that we have different statistics for different quantile groups
        unique_means = len(quantile_means.unique())
        unique_medians = len(quantile_medians.unique())
        unique_stds = len(quantile_stds.unique())
        unique_mad = len(quantile_mad.unique())
        
        print(f"   🔍 Verification: {unique_means} unique means, {unique_medians} unique medians, {unique_stds} unique stds, {unique_mad} unique MADs for {n_groups} quantile groups")
        logger.info(f"Verification: {unique_means} unique means, {unique_medians} unique medians, {unique_stds} unique stds, {unique_mad} unique MADs for {n_groups} quantile groups")
        
        # Create new feature names
        mean_feature_name = f'{feature}_QuantileMean_{target_col}'
        median_feature_name = f'{feature}_QuantileMedian_{target_col}'
        std_feature_name = f'{feature}_QuantileStd_{target_col}'
        mad_feature_name = f'{feature}_QuantileMAD_{target_col}'
        
        # Map statistics to both train and test data with better error handling
        try:
            # Map all statistics
            train_encoded[mean_feature_name] = train_encoded[temp_col_name].map(quantile_means)
            test_encoded[mean_feature_name] = test_encoded[temp_col_name].map(quantile_means)
            
            train_encoded[median_feature_name] = train_encoded[temp_col_name].map(quantile_medians)
            test_encoded[median_feature_name] = test_encoded[temp_col_name].map(quantile_medians)
            
            train_encoded[std_feature_name] = train_encoded[temp_col_name].map(quantile_stds)
            test_encoded[std_feature_name] = test_encoded[temp_col_name].map(quantile_stds)
            
            train_encoded[mad_feature_name] = train_encoded[temp_col_name].map(quantile_mad)
            test_encoded[mad_feature_name] = test_encoded[temp_col_name].map(quantile_mad)
            
            # Handle any missing values for all statistics
            overall_mean = train_df[target_col].mean()
            overall_median = train_df[target_col].median()
            overall_std = train_df[target_col].std()
            overall_mad = np.mean(np.abs(train_df[target_col] - overall_mean))
            
            # Fill NaN values for each statistic
            for df_name, df, stat_name, feature_name, overall_val in [
                ('train', train_encoded, 'mean', mean_feature_name, overall_mean),
                ('test', test_encoded, 'mean', mean_feature_name, overall_mean),
                ('train', train_encoded, 'median', median_feature_name, overall_median),
                ('test', test_encoded, 'median', median_feature_name, overall_median),
                ('train', train_encoded, 'std', std_feature_name, overall_std),
                ('test', test_encoded, 'std', std_feature_name, overall_std),
                ('train', train_encoded, 'mad', mad_feature_name, overall_mad),
                ('test', test_encoded, 'mad', mad_feature_name, overall_mad)
            ]:
                nan_count = df[feature_name].isna().sum()
                if nan_count > 0:
                    print(f"   ⚠️  Filling {nan_count} NaN values in {df_name} {stat_name} with overall {stat_name}: {overall_val:.4f}")
                    df[feature_name].fillna(overall_val, inplace=True)
                
        except Exception as e:
            print(f"   ❌ ERROR in mapping quantile statistics: {e}")
            logger.error(f"Mapping failed for {feature}: {e}")
            # Fallback: use overall statistics for all rows
            overall_mean = train_df[target_col].mean()
            overall_median = train_df[target_col].median()
            overall_std = train_df[target_col].std()
            overall_mad = np.mean(np.abs(train_df[target_col] - overall_mean))
            
            train_encoded[mean_feature_name] = overall_mean
            test_encoded[mean_feature_name] = overall_mean
            train_encoded[median_feature_name] = overall_median
            test_encoded[median_feature_name] = overall_median
            train_encoded[std_feature_name] = overall_std
            test_encoded[std_feature_name] = overall_std
            train_encoded[mad_feature_name] = overall_mad
            test_encoded[mad_feature_name] = overall_mad
        
        logger.info(f"Created new features: {mean_feature_name}, {median_feature_name}, {std_feature_name}, {mad_feature_name}")
        print(f"   ✅ Created new features:")
        print(f"      - {mean_feature_name}")
        print(f"      - {median_feature_name}")
        print(f"      - {std_feature_name}")
        print(f"      - {mad_feature_name}")
        
        # Final verification for all statistics
        all_features = [mean_feature_name, median_feature_name, std_feature_name, mad_feature_name]
        for feat_name in all_features:
            final_train_nan = train_encoded[feat_name].isna().sum()
            final_test_nan = test_encoded[feat_name].isna().sum()
            if final_train_nan > 0 or final_test_nan > 0:
                print(f"   ❌ ERROR: {feat_name} still has NaN values (train: {final_train_nan}, test: {final_test_nan})")
            else:
                print(f"   ✅ {feat_name} has no NaN values")
        
        # Step 3: Keep the quantile group column as a feature (don't remove it)
        print(f"📌 Step 3: Keeping quantile group column {temp_col_name} as a feature")
        logger.info(f"Step 3: Keeping quantile group column {temp_col_name} as a feature")
        
        logger.info(f"Completed processing for {feature}")
        print(f"   ✅ Completed processing for {feature}")
        print(f"   📊 Added 5 new features: {temp_col_name}, {mean_feature_name}, {median_feature_name}, {std_feature_name}, {mad_feature_name}")
        print()
    
    print("="*60)
    print("🎉 TARGET ENCODING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    logger.info(f"\n=== Target encoding completed ===")
    logger.info(f"Original train shape: {train_df.shape}")
    logger.info(f"Encoded train shape: {train_encoded.shape}")
    logger.info(f"Original test shape: {test_df.shape}")
    logger.info(f"Encoded test shape: {test_encoded.shape}")
    logger.info(f"Added {len(target_encode) * 5} new features ({len(target_encode)} quantile means, {len(target_encode)} quantile medians, {len(target_encode)} quantile stds, {len(target_encode)} quantile MADs)")
    
    print(f"📊 SUMMARY:")
    print(f"   Original train shape: {train_df.shape}")
    print(f"   Encoded train shape: {train_encoded.shape}")
    print(f"   Original test shape: {test_df.shape}")
    print(f"   Encoded test shape: {test_encoded.shape}")
    print(f"   ✨ Added {len(target_encode) * 5} new features ({len(target_encode)} quantile means, {len(target_encode)} quantile medians, {len(target_encode)} quantile stds, {len(target_encode)} quantile MADs)")
    
    print(f"\n🆕 New features created:")
    # Show quantile group features
    quantile_group_features = [col for col in train_encoded.columns if 'Quantile_Group' in col]
    quantile_mean_features = [col for col in train_encoded.columns if 'QuantileMean' in col]
    quantile_median_features = [col for col in train_encoded.columns if 'QuantileMedian' in col]
    quantile_std_features = [col for col in train_encoded.columns if 'QuantileStd' in col]
    quantile_mad_features = [col for col in train_encoded.columns if 'QuantileMAD' in col]
    
    print(f"   📊 Quantile Group Features ({len(quantile_group_features)}):")
    for j, feature in enumerate(quantile_group_features, 1):
        print(f"      {j:2d}. {feature}")
    
    print(f"   📈 Quantile Mean Features ({len(quantile_mean_features)}):")
    for j, feature in enumerate(quantile_mean_features, 1):
        print(f"      {j:2d}. {feature}")
    
    print(f"   📊 Quantile Median Features ({len(quantile_median_features)}):")
    for j, feature in enumerate(quantile_median_features, 1):
        print(f"      {j:2d}. {feature}")
    
    print(f"   📈 Quantile Std Features ({len(quantile_std_features)}):")
    for j, feature in enumerate(quantile_std_features, 1):
        print(f"      {j:2d}. {feature}")
    
    print(f"   📊 Quantile MAD Features ({len(quantile_mad_features)}):")
    for j, feature in enumerate(quantile_mad_features, 1):
        print(f"      {j:2d}. {feature}")
    
    print("\n" + "="*60)
    
    return train_encoded, test_encoded


def cluster_target_encoder(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                          target_col: str, cluster_features: List[str], 
                          n_clusters: int = 8, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs cluster-based target encoding for specified features.
    
    Args:
        train_df: Training dataframe with target column
        test_df: Test dataframe without target column
        target_col: Name of the target column
        cluster_features: List of feature names to use for clustering
        n_clusters: Number of clusters to create (default: 8)
        random_state: Random state for reproducibility (default: 42)
    
    Returns:
        Tuple of (encoded_train_df, encoded_test_df)
    """
    
    print("="*60)
    print("🎯 CLUSTER TARGET ENCODER STARTED")
    print("="*60)
    
    logger.info(f"Starting cluster target encoding with {len(cluster_features)} features")
    logger.info(f"Target column: {target_col}")
    logger.info(f"Cluster features: {cluster_features}")
    logger.info(f"Number of clusters: {n_clusters}")
    
    print(f"🔬 Performing clustering on {len(cluster_features)} features")
    print(f"🎯 Target column: {target_col}")
    print(f"📊 Number of clusters: {n_clusters}")
    print(f"🔢 Train data shape: {train_df.shape}")
    print(f"🔢 Test data shape: {test_df.shape}")
    print()
    
    # Create copies to avoid modifying original dataframes
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # Verify target column exists in train_df
    if target_col not in train_df.columns:
        error_msg = f"Target column '{target_col}' not found in training data"
        print(f"❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Verify all cluster features exist in both dataframes
    missing_train = [col for col in cluster_features if col not in train_df.columns]
    missing_test = [col for col in cluster_features if col not in test_df.columns]
    
    if missing_train:
        error_msg = f"Cluster features missing in train_df: {missing_train}"
        print(f"❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    if missing_test:
        error_msg = f"Cluster features missing in test_df: {missing_test}"
        print(f"❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    logger.info("All required columns found in both dataframes")
    print("✅ All required columns found in both dataframes")
    print()
    
    try:
        # Step 1: Prepare data for clustering
        print("🔄 Step 1: Preparing data for clustering")
        logger.info("Step 1: Preparing data for clustering")
        
        # Extract clustering features from both datasets
        train_cluster_data = train_df[cluster_features].copy()
        test_cluster_data = test_df[cluster_features].copy()
        
        # Combine for consistent preprocessing
        combined_cluster_data = pd.concat([train_cluster_data, test_cluster_data], ignore_index=True)
        
        print(f"   📊 Combined clustering data shape: {combined_cluster_data.shape}")
        print(f"   🔍 Missing values per feature:")
        missing_counts = combined_cluster_data.isnull().sum()
        for feature, count in missing_counts.items():
            if count > 0:
                print(f"      {feature}: {count}")
        
        # Handle missing values
        if combined_cluster_data.isnull().sum().sum() > 0:
            print("   🔧 Imputing missing values with median strategy")
            logger.info("Imputing missing values in clustering features")
            imputer = SimpleImputer(strategy='median')
            combined_cluster_data_imputed = pd.DataFrame(
                imputer.fit_transform(combined_cluster_data),
                columns=combined_cluster_data.columns
            )
        else:
            combined_cluster_data_imputed = combined_cluster_data.copy()
            print("   ✅ No missing values found")
        
        # Step 2: Scale the features for clustering
        print("📏 Step 2: Scaling features for clustering")
        logger.info("Step 2: Scaling features for clustering")
        
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined_cluster_data_imputed)
        
        print(f"   ✅ Scaled {combined_scaled.shape[1]} features")
        print(f"   📊 Scaled data shape: {combined_scaled.shape}")
        
        # Step 3: Perform clustering
        print(f"🔬 Step 3: Performing K-Means clustering with {n_clusters} clusters")
        logger.info(f"Step 3: Performing K-Means clustering with {n_clusters} clusters")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(combined_scaled)
        
        # Split cluster labels back to train and test
        train_clusters = cluster_labels[:len(train_df)]
        test_clusters = cluster_labels[len(train_df):]
        
        # Verify lengths match
        assert len(train_clusters) == len(train_df), f"Train clusters length mismatch: {len(train_clusters)} vs {len(train_df)}"
        assert len(test_clusters) == len(test_df), f"Test clusters length mismatch: {len(test_clusters)} vs {len(test_df)}"
        
        # Add cluster group columns
        cluster_col_name = 'Cluster_Group'
        train_encoded[cluster_col_name] = train_clusters
        test_encoded[cluster_col_name] = test_clusters
        
        print(f"   ✅ Clustering completed successfully")
        print(f"   📊 Cluster distribution in training data:")
        cluster_counts = np.bincount(train_clusters)
        for i, count in enumerate(cluster_counts):
            print(f"      Cluster {i}: {count} samples ({count/len(train_clusters)*100:.1f}%)")
        
        logger.info(f"Clustering completed with inertia: {kmeans.inertia_:.2f}")
        
        # Step 4: Calculate statistical measures for each cluster
        print(f"🧮 Step 4: Calculating {target_col} statistics for each cluster")
        logger.info(f"Step 4: Calculating {target_col} statistics for each cluster")
        
        # Calculate multiple statistical measures for each cluster using training data
        try:
            grouped = train_encoded.groupby(cluster_col_name)[target_col]
            
            # Calculate all statistics
            cluster_means = grouped.mean()
            cluster_medians = grouped.median()
            cluster_stds = grouped.std()
            cluster_mad = grouped.apply(lambda x: np.mean(np.abs(x - x.mean())))  # Mean Absolute Deviation
            
            # Ensure we have valid statistics
            for stat_name, stat_values in [('means', cluster_means), ('medians', cluster_medians), 
                                         ('stds', cluster_stds), ('mad', cluster_mad)]:
                if stat_values.isna().any():
                    print(f"   ⚠️  Warning: Some cluster {stat_name} are NaN")
                    logger.warning(f"Some cluster {stat_name} are NaN")
                    if stat_name == 'stds':
                        # For std, fill with overall std
                        stat_values.fillna(train_df[target_col].std(), inplace=True)
                    elif stat_name == 'mad':
                        # For MAD, fill with overall MAD
                        overall_mad = np.mean(np.abs(train_df[target_col] - train_df[target_col].mean()))
                        stat_values.fillna(overall_mad, inplace=True)
                    else:
                        # For mean and median, fill with overall values
                        overall_val = train_df[target_col].mean() if stat_name == 'means' else train_df[target_col].median()
                        stat_values.fillna(overall_val, inplace=True)
                        
        except Exception as e:
            print(f"   ❌ ERROR in cluster groupby operation: {e}")
            logger.error(f"Cluster groupby failed: {e}")
            # Fallback: use overall statistics for all clusters
            unique_clusters = train_encoded[cluster_col_name].unique()
            overall_mean = train_df[target_col].mean()
            overall_median = train_df[target_col].median()
            overall_std = train_df[target_col].std()
            overall_mad = np.mean(np.abs(train_df[target_col] - overall_mean))
            
            cluster_means = pd.Series([overall_mean] * len(unique_clusters), index=unique_clusters)
            cluster_medians = pd.Series([overall_median] * len(unique_clusters), index=unique_clusters)
            cluster_stds = pd.Series([overall_std] * len(unique_clusters), index=unique_clusters)
            cluster_mad = pd.Series([overall_mad] * len(unique_clusters), index=unique_clusters)
        
        # Display statistics
        print(f"   📈 Cluster statistics for {target_col}:")
        logger.info(f"Cluster statistics:")
        for cluster_id in sorted(cluster_means.index):
            cluster_size = (train_clusters == cluster_id).sum()
            mean_val = cluster_means[cluster_id]
            median_val = cluster_medians[cluster_id]
            std_val = cluster_stds[cluster_id]
            mad_val = cluster_mad[cluster_id]
            
            logger.info(f"  Cluster {cluster_id}: Mean={mean_val:.4f}, Median={median_val:.4f}, Std={std_val:.4f}, MAD={mad_val:.4f} (n={cluster_size})")
            print(f"      Cluster {cluster_id}: Mean={mean_val:.4f}, Median={median_val:.4f}, Std={std_val:.4f}, MAD={mad_val:.4f} (n={cluster_size})")
        
        # Verify that we have different statistics for different clusters
        unique_means = len(cluster_means.unique())
        unique_medians = len(cluster_medians.unique())
        unique_stds = len(cluster_stds.unique())
        unique_mad = len(cluster_mad.unique())
        
        print(f"   🔍 Verification: {unique_means} unique means, {unique_medians} unique medians, {unique_stds} unique stds, {unique_mad} unique MADs for {n_clusters} clusters")
        logger.info(f"Verification: {unique_means} unique means, {unique_medians} unique medians, {unique_stds} unique stds, {unique_mad} unique MADs for {n_clusters} clusters")
        
        # Step 5: Create cluster statistical features
        print(f"📊 Step 5: Creating cluster statistical features")
        logger.info("Step 5: Creating cluster statistical features")
        
        cluster_mean_col_name = f'ClusterMean_{target_col}'
        cluster_median_col_name = f'ClusterMedian_{target_col}'
        cluster_std_col_name = f'ClusterStd_{target_col}'
        cluster_mad_col_name = f'ClusterMAD_{target_col}'
        
        # Map cluster statistics to both train and test data
        try:
            # Map all statistics
            train_encoded[cluster_mean_col_name] = train_encoded[cluster_col_name].map(cluster_means)
            test_encoded[cluster_mean_col_name] = test_encoded[cluster_col_name].map(cluster_means)
            
            train_encoded[cluster_median_col_name] = train_encoded[cluster_col_name].map(cluster_medians)
            test_encoded[cluster_median_col_name] = test_encoded[cluster_col_name].map(cluster_medians)
            
            train_encoded[cluster_std_col_name] = train_encoded[cluster_col_name].map(cluster_stds)
            test_encoded[cluster_std_col_name] = test_encoded[cluster_col_name].map(cluster_stds)
            
            train_encoded[cluster_mad_col_name] = train_encoded[cluster_col_name].map(cluster_mad)
            test_encoded[cluster_mad_col_name] = test_encoded[cluster_col_name].map(cluster_mad)
            
            # Handle any missing values for all statistics
            overall_mean = train_df[target_col].mean()
            overall_median = train_df[target_col].median()
            overall_std = train_df[target_col].std()
            overall_mad = np.mean(np.abs(train_df[target_col] - overall_mean))
            
            # Fill NaN values for each statistic
            for df_name, df, stat_name, feature_name, overall_val in [
                ('train', train_encoded, 'mean', cluster_mean_col_name, overall_mean),
                ('test', test_encoded, 'mean', cluster_mean_col_name, overall_mean),
                ('train', train_encoded, 'median', cluster_median_col_name, overall_median),
                ('test', test_encoded, 'median', cluster_median_col_name, overall_median),
                ('train', train_encoded, 'std', cluster_std_col_name, overall_std),
                ('test', test_encoded, 'std', cluster_std_col_name, overall_std),
                ('train', train_encoded, 'mad', cluster_mad_col_name, overall_mad),
                ('test', test_encoded, 'mad', cluster_mad_col_name, overall_mad)
            ]:
                nan_count = df[feature_name].isna().sum()
                if nan_count > 0:
                    print(f"   ⚠️  Filling {nan_count} NaN values in {df_name} {stat_name} with overall {stat_name}: {overall_val:.4f}")
                    df[feature_name].fillna(overall_val, inplace=True)
                    
        except Exception as e:
            print(f"   ❌ ERROR in mapping cluster statistics: {e}")
            logger.error(f"Cluster mapping failed: {e}")
            # Fallback: use overall statistics for all rows
            overall_mean = train_df[target_col].mean()
            overall_median = train_df[target_col].median()
            overall_std = train_df[target_col].std()
            overall_mad = np.mean(np.abs(train_df[target_col] - overall_mean))
            
            train_encoded[cluster_mean_col_name] = overall_mean
            test_encoded[cluster_mean_col_name] = overall_mean
            train_encoded[cluster_median_col_name] = overall_median
            test_encoded[cluster_median_col_name] = overall_median
            train_encoded[cluster_std_col_name] = overall_std
            test_encoded[cluster_std_col_name] = overall_std
            train_encoded[cluster_mad_col_name] = overall_mad
            test_encoded[cluster_mad_col_name] = overall_mad
        
        print(f"   ✅ Created cluster statistical features:")
        print(f"      - {cluster_mean_col_name}")
        print(f"      - {cluster_median_col_name}")
        print(f"      - {cluster_std_col_name}")
        print(f"      - {cluster_mad_col_name}")
        logger.info(f"Created cluster statistical features: {cluster_mean_col_name}, {cluster_median_col_name}, {cluster_std_col_name}, {cluster_mad_col_name}")
        
        # Final verification for all cluster statistics
        all_cluster_features = [cluster_mean_col_name, cluster_median_col_name, cluster_std_col_name, cluster_mad_col_name]
        for feat_name in all_cluster_features:
            final_train_nan_cluster = train_encoded[cluster_col_name].isna().sum()
            final_test_nan_cluster = test_encoded[cluster_col_name].isna().sum()
            final_train_nan_stat = train_encoded[feat_name].isna().sum()
            final_test_nan_stat = test_encoded[feat_name].isna().sum()
            
            if any([final_train_nan_cluster, final_test_nan_cluster, final_train_nan_stat, final_test_nan_stat]):
                print(f"   ❌ ERROR: {feat_name} or cluster groups still have NaN values")
                print(f"      Cluster groups - train: {final_train_nan_cluster}, test: {final_test_nan_cluster}")
                print(f"      {feat_name} - train: {final_train_nan_stat}, test: {final_test_nan_stat}")
            else:
                print(f"   ✅ {feat_name} has no NaN values")
        
    except Exception as e:
        error_msg = f"Error in cluster target encoding: {str(e)}"
        print(f"❌ ERROR: {error_msg}")
        logger.error(error_msg)
        raise
    
    print("="*60)
    print("🎉 CLUSTER TARGET ENCODING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    logger.info(f"\n=== Cluster target encoding completed ===")
    logger.info(f"Original train shape: {train_df.shape}")
    logger.info(f"Encoded train shape: {train_encoded.shape}")
    logger.info(f"Original test shape: {test_df.shape}")
    logger.info(f"Encoded test shape: {test_encoded.shape}")
    logger.info(f"Added 5 new features (cluster groups + cluster means, medians, stds, MADs)")
    
    print(f"📊 SUMMARY:")
    print(f"   Original train shape: {train_df.shape}")
    print(f"   Encoded train shape: {train_encoded.shape}")
    print(f"   Original test shape: {test_df.shape}")
    print(f"   Encoded test shape: {test_encoded.shape}")
    print(f"   ✨ Added 5 new features (cluster groups + cluster means, medians, stds, MADs)")
    
    print(f"\n🆕 New features created:")
    print(f"   🔬 Cluster Group Feature: {cluster_col_name}")
    print(f"   📈 Cluster Mean Feature: {cluster_mean_col_name}")
    print(f"   📊 Cluster Median Feature: {cluster_median_col_name}")
    print(f"   📈 Cluster Std Feature: {cluster_std_col_name}")
    print(f"   📊 Cluster MAD Feature: {cluster_mad_col_name}")
    
    print(f"\n🔬 Clustering Details:")
    print(f"   📊 Features used for clustering: {len(cluster_features)}")
    print(f"   🎯 Number of clusters: {n_clusters}")
    print(f"   📈 Cluster inertia: {kmeans.inertia_:.2f}")
    print(f"   🔄 Random state: {random_state}")
    
    print("\n" + "="*60)
    
    return train_encoded, test_encoded


def create_interaction_features(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                               interaction_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create interaction features with conditionally activated features and Sex-based interactions.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        interaction_features: List of feature names to use for interactions
                             Expected: ['Duration', 'Age', 'Heart_Rate', 'Body_Temp']
    
    Returns:
        Tuple of (enhanced_train_df, enhanced_test_df)
    """
    
    print("="*60)
    print("🔗 INTERACTION FEATURES CREATOR STARTED")
    print("="*60)
    
    logger.info(f"Starting interaction feature creation with {len(interaction_features)} base features")
    logger.info(f"Interaction features: {interaction_features}")
    
    print(f"🔗 Creating interaction features from {len(interaction_features)} base features")
    print(f"📊 Base features: {interaction_features}")
    print(f"🔢 Train data shape: {train_df.shape}")
    print(f"🔢 Test data shape: {test_df.shape}")
    print()
    
    # Create copies to avoid modifying original dataframes
    train_enhanced = train_df.copy()
    test_enhanced = test_df.copy()
    
    # Verify all required features exist in both dataframes
    required_features = interaction_features + ['Sex']  # Sex is required for interactions
    missing_train = [col for col in required_features if col not in train_df.columns]
    missing_test = [col for col in required_features if col not in test_df.columns]
    
    if missing_train:
        error_msg = f"Required features missing in train_df: {missing_train}"
        print(f"❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    if missing_test:
        error_msg = f"Required features missing in test_df: {missing_test}"
        print(f"❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    logger.info("All required columns found in both dataframes")
    print("✅ All required columns found in both dataframes")
    print()
    
    try:
        # Combine train and test for consistent unique value extraction
        combined_df = pd.concat([train_df[interaction_features], test_df[interaction_features]], ignore_index=True)
        
        feature_count = 0
        
        # Step 1: Duration-based conditionally activated features
        if 'Duration' in interaction_features and 'Heart_Rate' in interaction_features and 'Body_Temp' in interaction_features:
            print("🔄 Step 1: Creating Duration-based conditionally activated features")
            logger.info("Step 1: Creating Duration-based conditionally activated features")
            
            durations = sorted(combined_df['Duration'].unique())
            print(f"   📊 Found {len(durations)} unique Duration values: {durations[:10]}{'...' if len(durations) > 10 else ''}")
            
            for dur in durations:
                # Heart Rate features for specific duration
                hr_feature_name = f'HR_Dur_{int(dur)}'
                train_enhanced[hr_feature_name] = np.where(train_enhanced['Duration'] == dur, 
                                                         train_enhanced['Heart_Rate'], 0)
                test_enhanced[hr_feature_name] = np.where(test_enhanced['Duration'] == dur, 
                                                        test_enhanced['Heart_Rate'], 0)
                
                # Body Temperature features for specific duration
                temp_feature_name = f'Temp_Dur_{int(dur)}'
                train_enhanced[temp_feature_name] = np.where(train_enhanced['Duration'] == dur, 
                                                           train_enhanced['Body_Temp'], 0)
                test_enhanced[temp_feature_name] = np.where(test_enhanced['Duration'] == dur, 
                                                          test_enhanced['Body_Temp'], 0)
                
                feature_count += 2
            
            print(f"   ✅ Created {len(durations) * 2} Duration-based features")
            logger.info(f"Created {len(durations) * 2} Duration-based features")
        
        # Step 2: Age-based conditionally activated features
        if 'Age' in interaction_features and 'Heart_Rate' in interaction_features and 'Body_Temp' in interaction_features:
            print("🔄 Step 2: Creating Age-based conditionally activated features")
            logger.info("Step 2: Creating Age-based conditionally activated features")
            
            ages = sorted(combined_df['Age'].unique())
            print(f"   📊 Found {len(ages)} unique Age values: {ages[:10]}{'...' if len(ages) > 10 else ''}")
            
            for age in ages:
                # Heart Rate features for specific age
                hr_feature_name = f'HR_Age_{int(age)}'
                train_enhanced[hr_feature_name] = np.where(train_enhanced['Age'] == age, 
                                                         train_enhanced['Heart_Rate'], 0)
                test_enhanced[hr_feature_name] = np.where(test_enhanced['Age'] == age, 
                                                        test_enhanced['Heart_Rate'], 0)
                
                # Body Temperature features for specific age
                temp_feature_name = f'Temp_Age_{int(age)}'
                train_enhanced[temp_feature_name] = np.where(train_enhanced['Age'] == age, 
                                                           train_enhanced['Body_Temp'], 0)
                test_enhanced[temp_feature_name] = np.where(test_enhanced['Age'] == age, 
                                                          test_enhanced['Body_Temp'], 0)
                
                feature_count += 2
            
            print(f"   ✅ Created {len(ages) * 2} Age-based features")
            logger.info(f"Created {len(ages) * 2} Age-based features")
        
        # Step 3: Sex-based pairwise interaction features
        print("🔄 Step 3: Creating Sex-based pairwise interaction features")
        logger.info("Step 3: Creating Sex-based pairwise interaction features")
        
        # Core activity-related features for Sex interactions
        core_features = ['Duration', 'Heart_Rate', 'Body_Temp']
        available_core_features = [f for f in core_features if f in interaction_features]
        
        print(f"   📊 Core features for Sex interactions: {available_core_features}")
        
        for feature in available_core_features:
            # Interaction with Sex (original)
            sex_feature_name = f'{feature}_x_Sex'
            train_enhanced[sex_feature_name] = train_enhanced[feature] * train_enhanced['Sex']
            test_enhanced[sex_feature_name] = test_enhanced[feature] * test_enhanced['Sex']
            
            # Interaction with reversed Sex (1 - Sex)
            rev_sex_feature_name = f'{feature}_x_RevSex'
            train_enhanced[rev_sex_feature_name] = train_enhanced[feature] * (1 - train_enhanced['Sex'])
            test_enhanced[rev_sex_feature_name] = test_enhanced[feature] * (1 - test_enhanced['Sex'])
            
            feature_count += 2
            
            print(f"   ✅ Created Sex interactions for {feature}: {sex_feature_name}, {rev_sex_feature_name}")
            logger.info(f"Created Sex interactions for {feature}")
        
        # Step 4: Additional pairwise interactions between core features
        print("🔄 Step 4: Creating additional pairwise interactions between core features")
        logger.info("Step 4: Creating additional pairwise interactions between core features")
        
        # Create interactions between pairs of core features
        for i, feat1 in enumerate(available_core_features):
            for j, feat2 in enumerate(available_core_features):
                if i < j:  # Avoid duplicate pairs and self-interactions
                    interaction_name = f'{feat1}_x_{feat2}'
                    train_enhanced[interaction_name] = train_enhanced[feat1] * train_enhanced[feat2]
                    test_enhanced[interaction_name] = test_enhanced[feat1] * test_enhanced[feat2]
                    
                    feature_count += 1
                    print(f"   ✅ Created interaction: {interaction_name}")
                    logger.info(f"Created interaction: {interaction_name}")
        
        # Final verification
        print("🔍 Step 5: Final verification")
        logger.info("Step 5: Final verification")
        
        # Check for any NaN values in new features
        new_features = [col for col in train_enhanced.columns if col not in train_df.columns]
        
        train_nan_counts = train_enhanced[new_features].isnull().sum()
        test_nan_counts = test_enhanced[new_features].isnull().sum()
        
        total_train_nans = train_nan_counts.sum()
        total_test_nans = test_nan_counts.sum()
        
        if total_train_nans > 0 or total_test_nans > 0:
            print(f"   ⚠️  Warning: Found NaN values in new features")
            print(f"      Train NaNs: {total_train_nans}, Test NaNs: {total_test_nans}")
            logger.warning(f"Found NaN values in new features: Train={total_train_nans}, Test={total_test_nans}")
            
            # Fill NaN values with 0 for interaction features
            for feature in new_features:
                if train_enhanced[feature].isnull().any():
                    train_enhanced[feature].fillna(0, inplace=True)
                if test_enhanced[feature].isnull().any():
                    test_enhanced[feature].fillna(0, inplace=True)
            
            print(f"   ✅ Filled NaN values with 0")
        else:
            print(f"   ✅ No NaN values found in new features")
        
    except Exception as e:
        error_msg = f"Error in interaction feature creation: {str(e)}"
        print(f"❌ ERROR: {error_msg}")
        logger.error(error_msg)
        raise
    
    print("="*60)
    print("🎉 INTERACTION FEATURES CREATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    logger.info(f"\n=== Interaction feature creation completed ===")
    logger.info(f"Original train shape: {train_df.shape}")
    logger.info(f"Enhanced train shape: {train_enhanced.shape}")
    logger.info(f"Original test shape: {test_df.shape}")
    logger.info(f"Enhanced test shape: {test_enhanced.shape}")
    logger.info(f"Added {feature_count} new interaction features")
    
    print(f"📊 SUMMARY:")
    print(f"   Original train shape: {train_df.shape}")
    print(f"   Enhanced train shape: {train_enhanced.shape}")
    print(f"   Original test shape: {test_df.shape}")
    print(f"   Enhanced test shape: {test_enhanced.shape}")
    print(f"   ✨ Added {feature_count} new interaction features")
    
    # Show feature breakdown
    new_features = [col for col in train_enhanced.columns if col not in train_df.columns]
    
    duration_features = [f for f in new_features if 'Dur_' in f]
    age_features = [f for f in new_features if 'Age_' in f]
    sex_features = [f for f in new_features if '_x_Sex' in f or '_x_RevSex' in f]
    other_interactions = [f for f in new_features if f not in duration_features + age_features + sex_features]
    
    print(f"\n🆕 New features breakdown:")
    print(f"   🕒 Duration-based features: {len(duration_features)}")
    print(f"   👤 Age-based features: {len(age_features)}")
    print(f"   ⚥ Sex-based interactions: {len(sex_features)}")
    print(f"   🔗 Other interactions: {len(other_interactions)}")
    
    if len(new_features) <= 20:  # Show all if not too many
        print(f"\n📋 All new features created:")
        for i, feature in enumerate(new_features, 1):
            print(f"   {i:2d}. {feature}")
    else:
        print(f"\n📋 Sample of new features created (showing first 10):")
        for i, feature in enumerate(new_features[:10], 1):
            print(f"   {i:2d}. {feature}")
        print(f"   ... and {len(new_features) - 10} more features")
    
    print("\n" + "="*60)
    
    return train_enhanced, test_enhanced


# Configuration
targetCol = 'Calories'
targetEncode = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMIscaled',
                'BMR', 'Metabolic_Efficiency', 'Cardio_Stress', 'Thermic_Effect', 'Power_Output']

clusterMean = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMIscaled', 'Sex', 
               'BMR', 'Metabolic_Efficiency', 'Cardio_Stress', 'Thermic_Effect', 'Power_Output']

interaction_features = ['Duration', 'Age', 'Heart_Rate', 'Body_Temp']

