# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Memory management
import gc
import psutil
import os

# Data manipulation
import pandas as pd
import numpy as np

# Modeling
import xgboost as xgb
from sklearn.decomposition import IncrementalPCA

# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt

# Display
from IPython.display import display
import time
import logging

from hmmlearn import hmm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"💾 Current memory usage: {memory_mb:.2f} MB")
    logger.info(f"Memory usage: {memory_mb:.2f} MB")

def check_gpu_availability():
    """Check for GPU availability and return device type"""
    try:
        # Check if XGBoost was compiled with GPU support
        import xgboost as xgb
        # Try to create a GPU-enabled booster
        dtrain = xgb.DMatrix(np.random.rand(10, 5), label=np.random.rand(10))
        params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        bst = xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
        print("🚀 GPU detected and will be used for training")
        logger.info("GPU detected and will be used for training")
        return 'gpu'
    except Exception as e:
        print("⚠️  GPU not available, using CPU with multi-threading")
        print(f"   GPU error: {str(e)}")
        logger.warning(f"GPU not available, using CPU. Error: {str(e)}")
        return 'cpu'

def get_gpu_count():
    """Get number of available GPUs"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_count = len(result.stdout.strip().split('\n'))
            print(f"🎮 Found {gpu_count} GPU(s)")
            logger.info(f"Found {gpu_count} GPU(s)")
            return gpu_count
        else:
            return 0
    except:
        return 0

def reduce_mem_usage(dataframe, dataset):    
    print('Reducing memory usage for:', dataset)
    logger.info(f'Starting memory reduction for: {dataset}')
    initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
    
    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        c_min = dataframe[col].min()
        c_max = dataframe[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                dataframe[col] = dataframe[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                dataframe[col] = dataframe[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                dataframe[col] = dataframe[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                dataframe[col] = dataframe[col].astype(np.int64)
        else:
            try:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dataframe[col] = dataframe[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dataframe[col] = dataframe[col].astype(np.float32)
                else:
                    dataframe[col] = dataframe[col].astype(np.float64)
            except:
                pass

    final_mem_usage = dataframe.memory_usage().sum() / 1024**2
    print('--- Memory usage before: {:.2f} MB'.format(initial_mem_usage))
    print('--- Memory usage after: {:.2f} MB'.format(final_mem_usage))
    print('--- Decreased memory usage by {:.1f}%\n'.format(100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage))
    logger.info(f'Memory reduction for {dataset}: {initial_mem_usage:.2f} MB -> {final_mem_usage:.2f} MB ({100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage:.1f}% reduction)')

    return dataframe

def check_regime_balance(regimes, max_imbalance=0.4, tolerance=0.1):
    """
    Check if regimes are reasonably balanced
    
    Args:
        regimes: array of regime predictions
        max_imbalance: maximum allowed imbalance ratio (minority/majority)
        tolerance: tolerance for balance check
    
    Returns:
        bool: True if balanced, False otherwise
    """
    regime_counts = np.bincount(regimes)
    min_count = regime_counts.min()
    max_count = regime_counts.max()
    balance_ratio = min_count / max_count
    
    logger.info(f"Regime balance check: min={min_count}, max={max_count}, ratio={balance_ratio:.3f}")
    print(f"📊 Regime balance: min={min_count}, max={max_count}, ratio={balance_ratio:.3f}")
    
    return balance_ratio >= max_imbalance

def fit_hmm_with_balance_check(observations, max_attempts=5, n_components_range=(2, 5)):
    """
    Fit HMM model with balance checking and re-iteration
    """
    logger.info("Starting HMM model fitting with balance checking")
    print("🎯 Starting HMM model fitting with balance checking...")
    
    best_model = None
    best_balance_ratio = 0
    best_regimes = None
    
    for attempt in range(max_attempts):
        logger.info(f"HMM fitting attempt {attempt + 1}/{max_attempts}")
        print(f"🔄 Attempt {attempt + 1}/{max_attempts}")
        
        for n_components in range(n_components_range[0], n_components_range[1] + 1):
            logger.info(f"Trying HMM with {n_components} components")
            print(f"   📈 Testing {n_components} regimes...")
            
            try:
                # Create model with different parameters each attempt
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type="diag" if attempt % 2 == 0 else "spherical",
                    n_iter=200 + attempt * 50,
                    tol=1e-4,
                    random_state=42 + attempt * 10,
                    init_params="stmc"
                )
                
                logger.info(f"Fitting HMM model with {n_components} components...")
                print(f"      🔧 Fitting model...")
                model.fit(observations)
                
                # Predict regimes
                predicted_regimes = model.predict(observations)
                
                # Check balance
                regime_counts = np.bincount(predicted_regimes)
                min_count = regime_counts.min()
                max_count = regime_counts.max()
                balance_ratio = min_count / max_count
                
                logger.info(f"Model with {n_components} components: balance ratio = {balance_ratio:.3f}")
                print(f"      📊 Balance ratio: {balance_ratio:.3f}")
                
                # Update best model if this one is more balanced
                if balance_ratio > best_balance_ratio:
                    best_balance_ratio = balance_ratio
                    best_model = model
                    best_regimes = predicted_regimes
                    logger.info(f"New best model found with balance ratio: {balance_ratio:.3f}")
                    print(f"      ✅ New best model! Balance: {balance_ratio:.3f}")
                
                # If we found a well-balanced model, use it
                if check_regime_balance(predicted_regimes):
                    logger.info(f"Found well-balanced model with {n_components} components")
                    print(f"      🎉 Well-balanced model found!")
                    return model, predicted_regimes
                    
            except Exception as e:
                logger.warning(f"HMM fitting failed for {n_components} components: {e}")
                print(f"      ⚠️ Failed with {n_components} components: {e}")
                continue
    
    # Return best model found
    if best_model is not None:
        logger.info(f"Returning best model with balance ratio: {best_balance_ratio:.3f}")
        print(f"🏆 Using best model found with balance ratio: {best_balance_ratio:.3f}")
        return best_model, best_regimes
    else:
        raise ValueError("Failed to fit any HMM model")

def fit_classifier_with_balance_check(X_train, y_train, max_attempts=5):
    """
    Fit classifier with balance checking and re-iteration
    """
    logger.info("Starting classifier fitting with balance checking")
    print("🎯 Starting classifier model fitting with balance checking...")
    
    best_clf = None
    best_auc = 0
    best_balance_ratio = 0
    
    for attempt in range(max_attempts):
        logger.info(f"Classifier fitting attempt {attempt + 1}/{max_attempts}")
        print(f"🔄 Classifier attempt {attempt + 1}/{max_attempts}")
        
        try:
            # Try different classifier parameters
            clf = RandomForestClassifier(
                n_estimators=100 + attempt * 50,
                max_depth=10 + attempt * 2,
                min_samples_split=2 + attempt,
                random_state=42 + attempt * 10,
                class_weight='balanced' if attempt > 2 else None
            )
            
            logger.info("Fitting Random Forest classifier...")
            print("   🌲 Fitting Random Forest...")
            clf.fit(X_train, y_train)
            
            # Predict and evaluate
            train_preds = clf.predict(X_train)
            
            # Check balance of predictions
            pred_counts = np.bincount(train_preds)
            min_count = pred_counts.min()
            max_count = pred_counts.max()
            balance_ratio = min_count / max_count
            
            # Calculate ROC AUC (for multiclass, use ovr strategy)
            try:
                if len(np.unique(y_train)) > 2:
                    train_proba = clf.predict_proba(X_train)
                    auc_score = roc_auc_score(y_train, train_proba, multi_class='ovr', average='weighted')
                else:
                    train_proba = clf.predict_proba(X_train)[:, 1]
                    auc_score = roc_auc_score(y_train, train_proba)
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                auc_score = 0
            
            logger.info(f"Classifier attempt {attempt + 1}: AUC={auc_score:.4f}, balance={balance_ratio:.3f}")
            print(f"   📊 AUC: {auc_score:.4f}, Balance: {balance_ratio:.3f}")
            
            # Update best classifier based on both AUC and balance
            combined_score = auc_score * 0.7 + balance_ratio * 0.3
            best_combined = best_auc * 0.7 + best_balance_ratio * 0.3
            
            if combined_score > best_combined:
                best_clf = clf
                best_auc = auc_score
                best_balance_ratio = balance_ratio
                logger.info(f"New best classifier found: AUC={auc_score:.4f}, balance={balance_ratio:.3f}")
                print(f"   ✅ New best classifier!")
            
            # If we have good balance and AUC, stop
            if balance_ratio >= 0.3 and auc_score >= 0.7:
                logger.info("Found satisfactory classifier with good balance and AUC")
                print(f"   🎉 Satisfactory classifier found!")
                return clf, auc_score, balance_ratio
                
        except Exception as e:
            logger.warning(f"Classifier fitting failed in attempt {attempt + 1}: {e}")
            print(f"   ⚠️ Attempt {attempt + 1} failed: {e}")
            continue
    
    if best_clf is not None:
        logger.info(f"Returning best classifier: AUC={best_auc:.4f}, balance={best_balance_ratio:.3f}")
        print(f"🏆 Using best classifier found: AUC={best_auc:.4f}, balance={best_balance_ratio:.3f}")
        return best_clf, best_auc, best_balance_ratio
    else:
        raise ValueError("Failed to fit any classifier")

def add_features(df):
    data = df.copy()
    features_df = pd.DataFrame(index=data.index)
    
    features_df['bid_ask_spread_proxy'] = data['ask_qty'] - data['bid_qty']
    features_df['total_liquidity'] = data['bid_qty'] + data['ask_qty']
    features_df['trade_imbalance'] = data['buy_qty'] - data['sell_qty']
    features_df['total_trades'] = data['buy_qty'] + data['sell_qty']
    
    features_df['volume_per_trade'] = data['volume'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    features_df['buy_volume_ratio'] = data['buy_qty'] / (data['volume'] + 1e-8)
    features_df['sell_volume_ratio'] = data['sell_qty'] / (data['volume'] + 1e-8)
    
    features_df['buying_pressure'] = data['buy_qty'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    features_df['selling_pressure'] = data['sell_qty'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    
    features_df['order_imbalance'] = (data['bid_qty'] - data['ask_qty']) / (data['bid_qty'] + data['ask_qty'] + 1e-8)
    features_df['order_imbalance_abs'] = np.abs(features_df['order_imbalance'])
    features_df['bid_liquidity_ratio'] = data['bid_qty'] / (data['volume'] + 1e-8)
    features_df['ask_liquidity_ratio'] = data['ask_qty'] / (data['volume'] + 1e-8)
    features_df['depth_imbalance'] = features_df['total_trades'] - data['volume']
    
    features_df['buy_sell_ratio'] = data['buy_qty'] / (data['sell_qty'] + 1e-8)
    features_df['bid_ask_ratio'] = data['bid_qty'] / (data['ask_qty'] + 1e-8)
    features_df['volume_liquidity_ratio'] = data['volume'] / (data['bid_qty'] + data['ask_qty'] + 1e-8)

    features_df['buy_volume_product'] = data['buy_qty'] * data['volume']
    features_df['sell_volume_product'] = data['sell_qty'] * data['volume']
    features_df['bid_ask_product'] = data['bid_qty'] * data['ask_qty']
    
    features_df['market_competition'] = (data['buy_qty'] * data['sell_qty']) / ((data['buy_qty'] + data['sell_qty']) + 1e-8)
    features_df['liquidity_competition'] = (data['bid_qty'] * data['ask_qty']) / ((data['bid_qty'] + data['ask_qty']) + 1e-8)
    
    total_activity = data['buy_qty'] + data['sell_qty'] + data['bid_qty'] + data['ask_qty']
    features_df['market_activity'] = total_activity
    features_df['activity_concentration'] = data['volume'] / (total_activity + 1e-8)
    
    features_df['info_arrival_rate'] = (data['buy_qty'] + data['sell_qty']) / (data['volume'] + 1e-8)
    features_df['market_making_intensity'] = (data['bid_qty'] + data['ask_qty']) / (data['buy_qty'] + data['sell_qty'] + 1e-8)
    features_df['effective_spread_proxy'] = np.abs(data['buy_qty'] - data['sell_qty']) / (data['volume'] + 1e-8)
    
    lambda_decay = 0.95
    ofi = data['buy_qty'] - data['sell_qty']
    features_df['order_flow_imbalance_ewm'] = ofi.ewm(alpha=1-lambda_decay).mean()

    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    return features_df

def select_top_k_features_fast(X_data, y_data, feature_columns, k):
    """
    Fast in-memory feature selection based on absolute correlation with target.
    
    Args:
        X_data: DataFrame with features
        y_data: Series with target values  
        feature_columns: List of feature column names
        k: Number of top features to select
    
    Returns:
        list: Top k feature names
    """
    logger.info(f"Starting fast feature selection: selecting top {k} features")
    print(f"🚀 Fast feature selection: selecting top {k} features...")
    
    # Ensure all features are numeric
    print("   🔧 Ensuring numeric data types...")
    X_numeric = X_data[feature_columns].copy()
    
    # Convert to numeric, coercing errors to NaN
    for col in feature_columns:
        X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
    
    # Fill any remaining NaN with 0
    X_numeric = X_numeric.fillna(0)
    
    # Calculate correlations using pandas (much faster)
    print("   📊 Computing correlations with target...")
    logger.info("Computing correlations with vectorized operations")
    
    correlations = {}
    y_numeric = pd.to_numeric(y_data, errors='coerce').fillna(0)
    
    # Use pandas correlation which is optimized
    correlation_series = X_numeric.corrwith(y_numeric)
    
    # Convert to absolute values and handle NaN
    for feature in feature_columns:
        corr_val = correlation_series.get(feature, 0.0)
        if pd.isna(corr_val):
            corr_val = 0.0
        correlations[feature] = abs(corr_val)
    
    # Select top k features
    print("   🎯 Selecting top features...")
    top_features = sorted(correlations.keys(), key=lambda x: correlations[x], reverse=True)[:k]
    
    logger.info(f"Feature selection completed: selected {len(top_features)} features")
    print(f"   ✅ Selected {len(top_features)} features")
    
    # Log top 10 correlations for debugging
    top_10_correlations = [(feat, correlations[feat]) for feat in top_features[:10]]
    logger.info(f"Top 10 feature correlations: {top_10_correlations}")
    print(f"   📈 Top 5 correlations: {[(f, f'{c:.4f}') for f, c in top_10_correlations[:5]]}")
    
    return top_features

def create_component_features(input_file, output_file, target_col, feature_columns=None, n_components=5, chunksize=50000):
    """
    Creates principal component features from a large dataset and writes the result to a CSV.
    Only saves timestamp/ID, target column (if present), and PCA components.
    Optimized version with faster processing and better memory management.
    
    Args:
        input_file (str): Path to input CSV file.
        output_file (str): Path to output CSV file.
        target_col (str): Name of the target column.
        feature_columns (list): List of feature column names to use for PCA. If None, uses all columns except target.
        n_components (int): Number of principal components to create (default: 5).
        chunksize (int): Number of rows per processing chunk (default: 50000).
    """
    logger.info(f"Starting optimized PCA component creation: {n_components} components from {input_file}")
    print(f"\n🚀 Creating {n_components} PCA components (optimized version)...")
    
    # Initialize memory tracking
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"   💾 Initial memory usage: {initial_memory:.1f} MB")
    
    # First pass: Determine feature columns and compute statistics
    print("   📊 Pass 1/3: Computing global statistics...")
    logger.info("PCA Pass 1: Computing global statistics")
    
    first_chunk = True
    n_samples = 0
    running_mean = None
    running_var = None
    chunk_count = 0
    total_chunks = sum(1 for _ in pd.read_csv(input_file, chunksize=chunksize))
    id_col = None  # To store ID column name if present
    
    print(f"      Total chunks to process: {total_chunks}")
    
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk_count += 1
        
        if first_chunk:
            # Determine ID column (either 'timestamp' or 'ID')
            if 'timestamp' in chunk.columns:
                id_col = 'timestamp'
            elif 'ID' in chunk.columns:
                id_col = 'ID'
            
            if feature_columns is None:
                # Exclude target and ID columns from features
                exclude_cols = [target_col] if target_col in chunk.columns else []
                if id_col:
                    exclude_cols.append(id_col)
                feature_columns = [col for col in chunk.columns if col not in exclude_cols]
            
            n_features = len(feature_columns)
            running_mean = np.zeros(n_features)
            running_var = np.zeros(n_features)
            print(f"      Using {n_features} features")
            print(f"      ID column: {id_col}")
            print(f"      Target column: {target_col if target_col in chunk.columns else 'Not present'}")
            logger.info(f"Using {n_features} features for PCA, ID column: {id_col}")
            first_chunk = False
        
        if chunk_count % 5 == 0:
            progress = (chunk_count / total_chunks) * 100
            print(f"      ⏳ Progress: {progress:.1f}% (chunk {chunk_count}/{total_chunks})")
            logger.info(f"Pass 1: Processing chunk {chunk_count}/{total_chunks}")
        
        X_chunk = chunk[feature_columns].values.astype(np.float32)  # Use float32 for memory efficiency
        chunk_size = X_chunk.shape[0]
        
        # Update running statistics using Welford's online algorithm
        if n_samples == 0:
            running_mean = np.mean(X_chunk, axis=0)
            running_var = np.var(X_chunk, axis=0)
        else:
            chunk_mean = np.mean(X_chunk, axis=0)
            chunk_var = np.var(X_chunk, axis=0)
            delta = chunk_mean - running_mean
            running_mean += delta * chunk_size / (n_samples + chunk_size)
            running_var = (n_samples * running_var + chunk_size * chunk_var + 
                         delta * delta * n_samples * chunk_size / (n_samples + chunk_size)) / (n_samples + chunk_size)
        
        n_samples += chunk_size
        
        # Clear memory
        del X_chunk
        gc.collect()
    
    global_std = np.sqrt(running_var)
    global_std[global_std == 0] = 1.0
    
    current_memory = process.memory_info().rss / 1024 / 1024
    print(f"      💾 Memory usage after Pass 1: {current_memory:.1f} MB")
    print(f"      ✅ Pass 1 completed: {n_samples:,} samples processed")
    
    # Initialize Incremental PCA with reduced components
    print("   🔧 Pass 2/3: Training PCA model...")
    logger.info("PCA Pass 2: Training Incremental PCA")
    
    ipca = IncrementalPCA(n_components=n_components)
    chunk_count = 0
    
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk_count += 1
        if chunk_count % 5 == 0:
            progress = (chunk_count / total_chunks) * 100
            print(f"      ⏳ Progress: {progress:.1f}% (chunk {chunk_count}/{total_chunks})")
            logger.info(f"Pass 2: Processing chunk {chunk_count}/{total_chunks}")
        
        X_chunk = chunk[feature_columns].values.astype(np.float32)
        X_scaled = (X_chunk - running_mean) / global_std
        ipca.partial_fit(X_scaled)
        
        del X_chunk, X_scaled
        gc.collect()
    
    explained_var = [f"{var:.3f}" for var in ipca.explained_variance_ratio_]
    print(f"      📈 Explained variance ratios: {explained_var}")
    logger.info(f"PCA components explain: {explained_var} of variance")
    
    current_memory = process.memory_info().rss / 1024 / 1024
    print(f"      💾 Memory usage after Pass 2: {current_memory:.1f} MB")
    
    # Third pass: Transform data and write to output
    print("   💾 Pass 3/3: Transforming data and saving...")
    logger.info("PCA Pass 3: Transforming and saving data")
    
    first_chunk = True
    chunk_count = 0
    
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk_count += 1
        if chunk_count % 5 == 0:
            progress = (chunk_count / total_chunks) * 100
            print(f"      ⏳ Progress: {progress:.1f}% (chunk {chunk_count}/{total_chunks})")
            logger.info(f"Pass 3: Processing chunk {chunk_count}/{total_chunks}")
        
        X_chunk = chunk[feature_columns].values.astype(np.float32)
        X_scaled = (X_chunk - running_mean) / global_std
        components = ipca.transform(X_scaled)
        
        comp_cols = [f'pca_{i}' for i in range(components.shape[1])]
        components_df = pd.DataFrame(components, columns=comp_cols)
        
        # Create output with only ID/timestamp, target (if present), and PCA components
        output_columns = []
        output_chunk = pd.DataFrame()
        
        # Add ID column if present
        if id_col and id_col in chunk.columns:
            output_chunk[id_col] = chunk[id_col]
            output_columns.append(id_col)
        
        # Add target column if present
        if target_col in chunk.columns:
            output_chunk[target_col] = chunk[target_col]
            output_columns.append(target_col)
        
        # Add PCA components
        output_chunk = pd.concat([output_chunk, components_df], axis=1)
        
        output_chunk.to_csv(output_file, mode='a', header=first_chunk, index=False)
        first_chunk = False
        
        del X_chunk, X_scaled, components, components_df, output_chunk
        gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_change = final_memory - initial_memory
    print(f"\n   💾 Final memory usage: {final_memory:.1f} MB (change: {memory_change:+.1f} MB)")
    print(f"   ✅ PCA components saved to {output_file}")
    print(f"   📊 Output contains: {id_col if id_col else 'No ID'}, {'target' if target_col else 'No target'}, {n_components} PCA components")
    logger.info(f"PCA completed: {n_components} components saved to {output_file}")

# Check system resources
print("🔍 Checking system resources...")
logger.info("Starting system resource check")
device_type = check_gpu_availability()
cpu_count = os.cpu_count()
gpu_count = get_gpu_count() if device_type == 'gpu' else 0
print(f"💻 Available CPU cores: {cpu_count}")
logger.info(f"Available CPU cores: {cpu_count}")
if gpu_count > 0:
    print(f"🎮 Available GPUs: {gpu_count}")
    logger.info(f"Available GPUs: {gpu_count}")
print_memory_usage()
print()

# --- Step 0: Load Data and Optimize dtypes for lesser memory usage ---
logger.info("Loading and optimizing data...")
print("📂 Loading data...")
train_path = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
test_path = '/kaggle/input/drw-crypto-market-prediction/test.parquet'
train = pd.read_parquet(train_path).reset_index(drop=False)
test = pd.read_parquet(test_path).reset_index(drop=False)
logger.info(f"Loaded train data: {train.shape}, test data: {test.shape}")
train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")

# --- Step 1: Create X_train, y_train, X_test ---
logger.info("Preparing training and test sets...")
print("🎯 Preparing training and test sets...")
target = 'label'
X_train = train.copy()
y_train = train[target]
X_test = test.drop(target, axis=1)
del train,test
print(f"Train data: {len(X_train)} samples")
print(f"Test data: {len(X_test)} samples")
logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# --- Step 2: Add extra derived features & Popular features in discussions---
logger.info("Adding derived features...")
print("🔧 Adding derived features...")
X_train = pd.concat([add_features(X_train), X_train], axis=1)
X_test = pd.concat([add_features(X_test), X_test], axis=1)

# --- Step 2.5: Create Enhanced Features with PCA and Feature Selection ---
logger.info("Creating enhanced features with PCA and feature selection...")
print("🚀 Creating two enhanced datasets: PCA components and Top 500 features...")

# Define feature lists for regime detection
regimeFeatures_HMM = ['info_arrival_rate', 'market_making_intensity', 'effective_spread_proxy', 'order_flow_imbalance_ewm']
regimeFeatures_CLF = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'trade_imbalance']

# Get all available features (excluding timestamp/ID and target)
all_features = [col for col in X_train.columns if col not in ['timestamp', target]]
print(f"   📊 Total available features: {len(all_features)}")
logger.info(f"Total available features for processing: {len(all_features)}")

# --- Comprehensive Data Cleaning ---
print("\n🧹 Comprehensive data cleaning...")
logger.info("Starting comprehensive data cleaning")

# Check for NaN and infinite values in training data
print("   🔍 Checking training data...")
train_nan_count = X_train[all_features].isna().sum().sum()
train_inf_count = np.isinf(X_train[all_features].select_dtypes(include=[np.number]).values).sum()
print(f"      - Training NaN values: {train_nan_count}")
print(f"      - Training infinite values: {train_inf_count}")
logger.info(f"Training data: {train_nan_count} NaN, {train_inf_count} infinite values")

# Check for NaN and infinite values in test data
print("   🔍 Checking test data...")
test_nan_count = X_test[all_features].isna().sum().sum()
test_inf_count = np.isinf(X_test[all_features].select_dtypes(include=[np.number]).values).sum()
print(f"      - Test NaN values: {test_nan_count}")
print(f"      - Test infinite values: {test_inf_count}")
logger.info(f"Test data: {test_nan_count} NaN, {test_inf_count} infinite values")

# Clean training data
print("   🔧 Cleaning training data...")
# Replace infinite values with NaN first
X_train[all_features] = X_train[all_features].replace([np.inf, -np.inf], np.nan)

# Fill NaN values with column medians (more robust than mean)
for col in all_features:
    if X_train[col].isna().any():
        median_val = X_train[col].median()
        if pd.isna(median_val):  # If median is also NaN, use 0
            median_val = 0.0
        X_train[col] = X_train[col].fillna(median_val)
        logger.info(f"Filled NaN in training column {col} with {median_val}")

# Clean test data using training data statistics
print("   🔧 Cleaning test data...")
# Replace infinite values with NaN first
X_test[all_features] = X_test[all_features].replace([np.inf, -np.inf], np.nan)

# Fill NaN values with corresponding training medians
for col in all_features:
    if X_test[col].isna().any():
        # Use the same median from training data for consistency
        median_val = X_train[col].median()
        if pd.isna(median_val):  # If median is still NaN, use 0
            median_val = 0.0
        X_test[col] = X_test[col].fillna(median_val)
        logger.info(f"Filled NaN in test column {col} with training median {median_val}")

# Final verification
print("   ✅ Final verification...")
final_train_nan = X_train[all_features].isna().sum().sum()
final_train_inf = np.isinf(X_train[all_features].select_dtypes(include=[np.number]).values).sum()
final_test_nan = X_test[all_features].isna().sum().sum()
final_test_inf = np.isinf(X_test[all_features].select_dtypes(include=[np.number]).values).sum()

print(f"      - Final training NaN: {final_train_nan}, infinite: {final_train_inf}")
print(f"      - Final test NaN: {final_test_nan}, infinite: {final_test_inf}")
logger.info(f"Final cleanup - Train: {final_train_nan} NaN, {final_train_inf} inf; Test: {final_test_nan} NaN, {final_test_inf} inf")

if final_train_nan > 0 or final_train_inf > 0 or final_test_nan > 0 or final_test_inf > 0:
    print("      ⚠️ Warning: Still have NaN/infinite values, applying final fallback...")
    # Fallback: replace any remaining NaN/inf with 0
    X_train[all_features] = X_train[all_features].fillna(0).replace([np.inf, -np.inf], 0)
    X_test[all_features] = X_test[all_features].fillna(0).replace([np.inf, -np.inf], 0)
    logger.warning("Applied fallback: replaced remaining NaN/inf with 0")

print("   🎉 Data cleaning completed!")
logger.info("Data cleaning completed successfully")

# --- Dataset 1: PCA Components Dataset ---
print("\n🔧 Creating Dataset 1: PCA Components...")
logger.info("Creating PCA components dataset")

# Define feature columns for PCA (exclude timestamp and target)
pca_feature_columns = all_features
logger.info(f"PCA will use {len(pca_feature_columns)} feature columns")
print(f"   🎯 Using {len(pca_feature_columns)} features for PCA")

# Prepare data for PCA (directly from memory)
print("   🔧 Preparing data for PCA processing...")
X_train_features = X_train[pca_feature_columns].values.astype(np.float32)
X_test_features = X_test[pca_feature_columns].values.astype(np.float32)

# Fit and transform with Incremental PCA
print("   🔧 Fitting PCA model...")
logger.info("Fitting Incremental PCA on training data")
ipca = IncrementalPCA(n_components=20)

# Fit PCA on training data in chunks if needed
chunk_size = 10000
n_samples = X_train_features.shape[0]
for i in range(0, n_samples, chunk_size):
    end_idx = min(i + chunk_size, n_samples)
    chunk = X_train_features[i:end_idx]
    ipca.partial_fit(chunk)

# Transform both datasets
print("   🔄 Transforming datasets with PCA...")
X_train_pca_components = ipca.transform(X_train_features)
X_test_pca_components = ipca.transform(X_test_features)

# Create PCA component column names
pca_component_names = [f'pca_{i}' for i in range(20)]
explained_var = [f"{var:.3f}" for var in ipca.explained_variance_ratio_]
print(f"   📈 Explained variance ratios: {explained_var}")
logger.info(f"PCA components explain: {explained_var} of variance")

# Create final PCA datasets
print("   💾 Creating PCA datasets...")
pca_train_file = 'pca_train_dataset.csv'
pca_test_file = 'pca_test_dataset.csv'

# Train PCA dataset: timestamp + target + PCA components
pca_train_df = pd.DataFrame(X_train_pca_components, columns=pca_component_names)
pca_train_df.insert(0, 'timestamp', X_train['timestamp'].values)
pca_train_df.insert(1, target, X_train[target].values)
pca_train_df.to_csv(pca_train_file, index=False)

# Test PCA dataset: ID + PCA components
pca_test_df = pd.DataFrame(X_test_pca_components, columns=pca_component_names)
pca_test_df.insert(0, 'ID', X_test['ID'].values)
pca_test_df.to_csv(pca_test_file, index=False)

print(f"   📊 PCA train dataset: {pca_train_df.shape}")
print(f"   📊 PCA test dataset: {pca_test_df.shape}")
print(f"   ✅ PCA datasets saved: {pca_train_file}, {pca_test_file}")
logger.info(f"PCA datasets created - Train: {pca_train_df.shape}, Test: {pca_test_df.shape}")

# Clean up PCA variables
del X_train_features, X_test_features, X_train_pca_components, X_test_pca_components, pca_train_df, pca_test_df
gc.collect()

# --- Dataset 2: Top 500 Features Dataset ---
print("\n🎯 Creating Dataset 2: Top 500 Features...")
logger.info("Creating top 500 features dataset")

# Select top 500 features from all available features (directly from memory)
print(f"   📊 Selecting top 500 from {len(all_features)} available features...")

# Use fast in-memory feature selection
top_500_features = select_top_k_features_fast(
    X_data=X_train, 
    y_data=y_train, 
    feature_columns=all_features,
    k=min(500, len(all_features))
)

print(f"   ✅ Selected {len(top_500_features)} features (showing first 5): {top_500_features[:5]}")
logger.info(f"Selected {len(top_500_features)} top features")

# Create top 500 features datasets (directly from memory)
top500_train_file = 'top500_train_dataset.csv'
top500_test_file = 'top500_test_dataset.csv'

print("   💾 Creating top 500 features datasets...")

# For train data: timestamp, target, top 500 features
train_top500_columns = ['timestamp', target] + top_500_features
X_train_top500 = X_train[train_top500_columns]
X_train_top500.to_csv(top500_train_file, index=False)

# For test data: ID, top 500 features
test_top500_columns = ['ID'] + top_500_features
X_test_top500 = X_test[test_top500_columns]
X_test_top500.to_csv(top500_test_file, index=False)

print(f"   📊 Top 500 train dataset: {X_train_top500.shape}")
print(f"   📊 Top 500 test dataset: {X_test_top500.shape}")
logger.info(f"Top 500 datasets created - Train: {X_train_top500.shape}, Test: {X_test_top500.shape}")

# --- Memory Cleanup ---
print("\n🧹 Cleaning up memory...")
logger.info("Starting memory cleanup")

# Delete large intermediate dataframes
del X_train_top500, X_test_top500
gc.collect()

print_memory_usage()

# --- Load Final Datasets for Processing ---
print("\n📂 Loading final datasets for regime detection...")
logger.info("Loading final datasets for regime detection")

# Load PCA dataset for HMM training (will use PCA components + some original features)
print("   📂 Loading PCA dataset for HMM...")
X_train_pca_hmm = pd.read_csv(pca_train_file)
logger.info(f"Loaded PCA dataset for HMM: {X_train_pca_hmm.shape}")

# Load top 500 dataset for classifier training
print("   📂 Loading top 500 training dataset...")
X_train_top500_loaded = pd.read_csv(top500_train_file)
logger.info(f"Loaded top 500 train dataset: {X_train_top500_loaded.shape}")

# Update regime features for classifier (use original classifier features from top 500)
regimeFeatures_CLF_enhanced = [f for f in regimeFeatures_CLF if f in X_train_top500_loaded.columns]

print(f"   🌲 Classifier features: {len(regimeFeatures_CLF_enhanced)} original features")
logger.info(f"Classifier feature set: {len(regimeFeatures_CLF_enhanced)} features")

# Clean up original large dataframes
del X_train, X_test
gc.collect()

print("   ✅ Memory cleanup completed")
logger.info("Memory cleanup completed")
print_memory_usage()

# Use the loaded datasets for further processing
X_train = X_train_top500_loaded  # Use top 500 dataset as main training data
y_train = X_train[target]

# Extract PCA components and target for observations
pca_features_for_hmm = [f'pca_{i}' for i in range(20)]
y_train_pca = X_train_pca_hmm[target]

print(f"   🎯 Using {len(pca_features_for_hmm)} PCA components for HMM regime detection")
logger.info(f"Using {len(pca_features_for_hmm)} PCA components for HMM")

# Prepare observation sequence (PCA features + returns)
observations = pd.concat([X_train_pca_hmm[pca_features_for_hmm], y_train_pca], axis=1)

# Clean observations data to handle NaN and infinite values
print("🧹 Cleaning observations data for HMM...")
logger.info("Cleaning observations data for HMM")
print(f"   - Original shape: {observations.shape}")

# Check for NaN and infinite values
nan_count = observations.isna().sum().sum()
inf_count = np.isinf(observations.values).sum()
print(f"   - NaN values found: {nan_count}")
print(f"   - Infinite values found: {inf_count}")
logger.info(f"Data cleaning: {nan_count} NaN values, {inf_count} infinite values")

# Replace infinite values with NaN first
observations = observations.replace([np.inf, -np.inf], np.nan)

# Fill NaN values with column means
observations = observations.fillna(observations.mean())

# Final check and fallback to zeros if mean is still NaN
observations = observations.fillna(0)

# Verify data is clean
final_nan_count = observations.isna().sum().sum()
final_inf_count = np.isinf(observations.values).sum()
print(f"   - Final NaN count: {final_nan_count}")
print(f"   - Final infinite count: {final_inf_count}")
print(f"   - Clean shape: {observations.shape}")
logger.info(f"Data cleaned: {final_nan_count} NaN, {final_inf_count} infinite values remaining")

# Standardize the data to prevent numerical issues
print("🔧 Standardizing observations for numerical stability...")
logger.info("Standardizing observations for HMM")
scaler = StandardScaler()
observations_scaled = scaler.fit_transform(observations)

# Add small noise to prevent singular covariance matrices
print("🎲 Adding small regularization noise...")
logger.info("Adding regularization noise to prevent singular matrices")
np.random.seed(42)
regularization_noise = np.random.normal(0, 1e-6, observations_scaled.shape)
observations_scaled += regularization_noise

# Convert to numpy array
observations = observations_scaled

print(f"   - Standardized data stats: mean={observations.mean():.6f}, std={observations.std():.6f}")
print(f"   - Data range: [{observations.min():.6f}, {observations.max():.6f}]")
logger.info(f"Standardized data: mean={observations.mean():.6f}, std={observations.std():.6f}")

# Fit Gaussian HMM with balance checking
logger.info("Fitting HMM model with balance checking...")
model, predicted_regimes = fit_hmm_with_balance_check(observations)

# Add predicted regimes to both datasets (PCA and top500)
X_train_pca_hmm['predicted_regime'] = predicted_regimes
X_train['predicted_regime'] = predicted_regimes  # Also add to main training dataset

print(f"📊 Final regime distribution: {np.bincount(predicted_regimes)}")
print(f"📈 Number of regimes detected: {len(np.unique(predicted_regimes))}")
logger.info(f"HMM regime distribution: {dict(enumerate(np.bincount(predicted_regimes)))}")

# --- Step 3.5: Analyze Regime Properties ---
logger.info("Analyzing regime properties...")
print("📈 Analyzing regime properties...")

# Map HMM states to consistent regime numbering using PCA dataset
regime_mapping = {}
for regime_id in range(len(np.unique(predicted_regimes))):
    regime_data = X_train_pca_hmm[X_train_pca_hmm['predicted_regime'] == regime_id]
    mean_return = regime_data[target].mean()
    std_return = regime_data[target].std()
    duration = len(regime_data)
    regime_mapping[regime_id] = (mean_return, std_return, duration)

print("\nDetected Regime Properties:")
logger.info("Regime properties analysis:")
for regime_id, (mean, std, dur) in regime_mapping.items():
    print(f"Regime {regime_id}: μ={mean:.6f}, σ={std:.6f}, duration={dur}s")
    logger.info(f"Regime {regime_id}: μ={mean:.6f}, σ={std:.6f}, duration={dur}s")

# Clean up PCA HMM dataset to save memory
del X_train_pca_hmm
gc.collect()

# --- Step 4: Train Regime Classifier ---
logger.info("Training regime classifier with balance checking using PCA features...")
print("\n🌲 Training regime classifier with balance checking using PCA features...")

# Load PCA dataset again for classifier training (we deleted it earlier for memory)
print("   📂 Reloading PCA dataset for classifier training...")
X_train_pca_classifier = pd.read_csv(pca_train_file)
logger.info(f"Reloaded PCA dataset for classifier: {X_train_pca_classifier.shape}")

# Use PCA features (without target) for classifier training
X_RegimeTrain = X_train_pca_classifier[pca_features_for_hmm].values
y_RegimeTrain = X_train['predicted_regime']  # Use HMM predicted regimes as labels

print(f"   🎯 Using {len(pca_features_for_hmm)} PCA components for classifier training")
logger.info(f"Classifier training with {len(pca_features_for_hmm)} PCA features")

# Standardize the PCA features for classifier
scaler_classifier = StandardScaler()
X_RegimeTrain_scaled = scaler_classifier.fit_transform(X_RegimeTrain)

# Fit classifier with balance checking
clf, best_auc, best_balance = fit_classifier_with_balance_check(X_RegimeTrain_scaled, y_RegimeTrain)

print(f"✅ Final classifier - ROC AUC: {best_auc:.4f}, Balance ratio: {best_balance:.4f}")
logger.info(f"Final classifier performance: ROC AUC={best_auc:.4f}, Balance ratio={best_balance:.4f}")

# --- Step 5.00: Predict Regimes for Train and Test Data ---
logger.info("Predicting regimes for train and test data using PCA features...")
print("🔮 Predicting regimes for train and test data using PCA features...")

# For training data (using classifier with PCA features)
X_train_pca_scaled = scaler_classifier.transform(X_train_pca_classifier[pca_features_for_hmm])
X_train['predicted_regime_Est'] = clf.predict(X_train_pca_scaled)

# For test data (load the PCA test dataset)
print("   📂 Loading PCA test dataset for regime prediction...")
X_test_pca = pd.read_csv(pca_test_file)
logger.info(f"Loaded PCA test dataset for prediction: {X_test_pca.shape}")

# Predict regimes for test data using PCA features
X_test_pca_scaled = scaler_classifier.transform(X_test_pca[pca_features_for_hmm])
X_test_pca['predicted_regime_Est'] = clf.predict(X_test_pca_scaled)

# --- Step 5.25: Analyze Regime Properties ---
logger.info("Analyzing final regime distributions...")
print("\n📊 Final Regime Analysis:")

print("\nTrain Data Original HMM Regime Distribution:")
orig_dist = X_train['predicted_regime'].value_counts()
print(orig_dist)
logger.info(f"Train original regime distribution: {orig_dist.to_dict()}")

print("\nTrain Data Classifier Regime Distribution:")
train_dist = X_train['predicted_regime_Est'].value_counts()
print(train_dist)
logger.info(f"Train classifier regime distribution: {train_dist.to_dict()}")

print("\nTest Data Regime Distribution:")
test_dist = X_test_pca['predicted_regime_Est'].value_counts()
print(test_dist)
logger.info(f"Test regime distribution: {test_dist.to_dict()}")

# Create regime dataframes for output
logger.info("Creating output regime dataframes...")
print("💾 Creating output files...")

train_regimes_df = X_train[['timestamp', 'predicted_regime', 'predicted_regime_Est']].copy()
test_regimes_df = X_test_pca[['ID', 'predicted_regime_Est']].copy()

# --- Step 5.75: Output Regime DataFrames ---
# Save to CSV files
train_regimes_df.to_csv('train_regimes.csv', index=False)
test_regimes_df.to_csv('test_regimes.csv', index=False)

print("\n✅ Output saved to train_regimes.csv and test_regimes.csv")
print(f"\n📊 Final Dataset Summary:")
print(f"   🔧 PCA Train Dataset: {pca_train_file}")
print(f"   🔧 PCA Test Dataset: {pca_test_file}")
print(f"   🎯 Top 500 Train Dataset: {top500_train_file}")
print(f"   🎯 Top 500 Test Dataset: {top500_test_file}")
print(f"   📋 Regime Predictions: train_regimes.csv, test_regimes.csv")
print(f"   🎯 Classifier uses: {len(pca_features_for_hmm)} PCA components (same as HMM)")
logger.info("Regime detection pipeline completed successfully with PCA features")


