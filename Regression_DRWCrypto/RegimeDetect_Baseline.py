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

def select_top_k_features(file_path, target_col, k, chunksize=10000):
    """
    Select top k features based on Pearson correlation with the target variable.
    
    Args:
        file_path (str): Path to CSV file containing the dataset.
        target_col (str): Name of the target column.
        k (int): Number of top features to select.
        chunksize (int): Number of rows per chunk for processing.
    
    Returns:
        list: Top k feature names.
    """
    logger.info(f"Starting feature selection: selecting top {k} features from {file_path}")
    print(f"🔍 Selecting top {k} features based on correlation with {target_col}...")
    
    # Initialize statistics
    n = 0
    sum_y = 0.0
    sum_y2 = 0.0
    stats = {}
    
    # Process data in chunks
    chunk_count = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunk_count += 1
        chunk_size = len(chunk)
        n += chunk_size
        
        if chunk_count % 10 == 0:
            print(f"   Processing chunk {chunk_count}, total samples: {n}")
            logger.info(f"Feature selection: processed chunk {chunk_count}, total samples: {n}")
        
        # Update target statistics
        y_vals = chunk[target_col].values
        sum_y += np.sum(y_vals)
        sum_y2 += np.sum(y_vals ** 2)
        
        # Update feature statistics
        features = [col for col in chunk.columns if col != target_col]
        for feat in features:
            x_vals = chunk[feat].values
            if feat not in stats:
                stats[feat] = {'sum_x': 0.0, 'sum_x2': 0.0, 'sum_xy': 0.0}
            stats[feat]['sum_x'] += np.sum(x_vals)
            stats[feat]['sum_x2'] += np.sum(x_vals ** 2)
            stats[feat]['sum_xy'] += np.sum(x_vals * y_vals)
    
    # Handle edge case: no data processed
    if n == 0:
        logger.warning("No data processed during feature selection")
        return []
    
    logger.info(f"Feature selection: processed {n} total samples across {chunk_count} chunks")
    print(f"   📊 Processed {n} samples, calculating correlations...")
    
    # Compute correlation for each feature
    correlations = {}
    for feat, vals in stats.items():
        sum_x = vals['sum_x']
        sum_x2 = vals['sum_x2']
        sum_xy = vals['sum_xy']
        
        # Calculate numerator and denominators
        numerator = n * sum_xy - sum_x * sum_y
        denom_x = n * sum_x2 - sum_x ** 2
        denom_y = n * sum_y2 - sum_y ** 2
        
        # Avoid division by zero
        if denom_x <= 0 or denom_y <= 0:
            corr = 0.0
        else:
            corr = numerator / (np.sqrt(denom_x) * np.sqrt(denom_y))
        correlations[feat] = abs(corr)
    
    # Select top k features
    top_features = sorted(correlations.keys(), key=lambda x: correlations[x], reverse=True)[:k]
    
    logger.info(f"Feature selection completed: selected {len(top_features)} features")
    print(f"   ✅ Selected top {len(top_features)} features")
    
    # Log top 10 correlations for debugging
    top_10_correlations = [(feat, correlations[feat]) for feat in top_features[:10]]
    logger.info(f"Top 10 feature correlations: {top_10_correlations}")
    print(f"   📈 Top 5 correlations: {[(f, f'{c:.4f}') for f, c in top_10_correlations[:5]]}")
    
    return top_features

def create_component_features(input_file, output_file, target_col, n_components=10, chunksize=10000, keep_original=True):
    """
    Creates principal component features from a large dataset and writes the result to a CSV.
    
    Args:
        input_file (str): Path to input CSV file.
        output_file (str): Path to output CSV file.
        target_col (str): Name of the target column.
        n_components (int): Number of principal components to create.
        chunksize (int): Number of rows per processing chunk.
        keep_original (bool): Whether to keep original features in the output.
    """
    logger.info(f"Starting PCA component creation: {n_components} components from {input_file}")
    print(f"🧮 Creating {n_components} PCA components...")
    
    # First pass: Compute global mean and standard deviation
    print("   📊 Pass 1: Computing global statistics...")
    logger.info("PCA Pass 1: Computing global mean and standard deviation")
    
    feature_columns = None
    n_samples = 0
    sum_x = None
    sum_x2 = None
    chunk_count = 0

    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk_count += 1
        if feature_columns is None:
            feature_columns = [col for col in chunk.columns if col != target_col]
            n_features = len(feature_columns)
            sum_x = np.zeros(n_features)
            sum_x2 = np.zeros(n_features)
            logger.info(f"PCA: Found {n_features} feature columns")
            print(f"      Found {n_features} features")
        
        if chunk_count % 10 == 0:
            print(f"      Processing chunk {chunk_count}...")
            logger.info(f"PCA Pass 1: processing chunk {chunk_count}")
        
        X_chunk = chunk[feature_columns].values.astype(np.float64)
        n_chunk = X_chunk.shape[0]
        n_samples += n_chunk
        sum_x += np.sum(X_chunk, axis=0)
        sum_x2 += np.sum(X_chunk ** 2, axis=0)
    
    if n_samples == 0:
        logger.error("No data found in the input file for PCA")
        raise ValueError("No data found in the input file.")
    
    global_mean = sum_x / n_samples
    global_std = np.sqrt((sum_x2 / n_samples) - (global_mean ** 2))
    global_std[global_std == 0] = 1.0  # Avoid division by zero
    
    logger.info(f"PCA Pass 1 completed: {n_samples} samples, {len(feature_columns)} features")
    print(f"      ✅ Pass 1 completed: {n_samples} samples")

    # Initialize Incremental PCA
    print("   🔧 Pass 2: Training PCA model...")
    logger.info("PCA Pass 2: Training Incremental PCA")
    
    ipca = IncrementalPCA(n_components=n_components)
    chunk_count = 0
    
    # Second pass: Train PCA incrementally
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"      Training on chunk {chunk_count}...")
            logger.info(f"PCA Pass 2: training on chunk {chunk_count}")
        
        X_chunk = chunk[feature_columns].values.astype(np.float64)
        X_scaled = (X_chunk - global_mean) / global_std
        ipca.partial_fit(X_scaled)
    
    logger.info(f"PCA training completed: explained variance ratio = {ipca.explained_variance_ratio_[:5]}")
    print(f"      ✅ PCA training completed")
    print(f"      📈 Top 5 components explain: {[f'{r:.3f}' for r in ipca.explained_variance_ratio_[:5]]}")
    
    # Third pass: Transform data and write to output
    print("   💾 Pass 3: Transforming data and saving...")
    logger.info("PCA Pass 3: Transforming data and writing to output")
    
    first_chunk = True
    chunk_count = 0
    
    for chunk in pd.read_csv(input_file, chunksize=chunksize):
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"      Transforming chunk {chunk_count}...")
            logger.info(f"PCA Pass 3: transforming chunk {chunk_count}")
        
        X_chunk = chunk[feature_columns].values.astype(np.float64)
        X_scaled = (X_chunk - global_mean) / global_std
        components = ipca.transform(X_scaled)
        
        comp_cols = [f'pca_{i}' for i in range(components.shape[1])]
        components_df = pd.DataFrame(components, columns=comp_cols)
        
        if not keep_original:
            chunk = chunk[[target_col]].join(components_df)
        else:
            chunk = pd.concat([chunk, components_df], axis=1)
        
        chunk.to_csv(output_file, mode='a', header=first_chunk, index=False)
        first_chunk = False
    
    logger.info(f"PCA component creation completed: output saved to {output_file}")
    print(f"   ✅ PCA components saved to {output_file}")

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

features = [
    'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'trade_imbalance', 
    'total_trades', 'volume_per_trade', 'selling_pressure', 'ask_liquidity_ratio',
    'order_imbalance', 'depth_imbalance', 'market_activity', 'activity_concentration',
    'info_arrival_rate', 'market_making_intensity', 'effective_spread_proxy', 'order_flow_imbalance_ewm',
    'buy_volume_product', 'sell_volume_product', 'volume_liquidity_ratio', 'effective_spread_proxy',
    'X2', 'X10', 'X11', 'X13', 'X14', 'X18', 'X19', 'X20', 'X26', 'X27', 'X40', 
    'X42', 'X60', 'X63', 'X64', 'X81', 'X82', 'X86', 'X87', 'X88', 'X99', 'X100',
    'X119', 'X120', 'X124', 'X125', 'X135', 'X143', 'X145', 'X157', 'X160', 'X161', 
    'X264', 'X265', 'X267', 'X269', 'X270', 'X289', 'X309', 'X314', 'X320', 'X325',
    'X331', 'X342', 'X480', 'X511', 'X534', 'X579', 'X580', 'X582', 'X602', 'X612',
    'X614', 'X764', 'X782', 'X808', 'X811', 'X863', 'X865', 'X866', 'X868', 'X877'
]
regimeFeatures_HMM = ['info_arrival_rate', 'market_making_intensity', 'effective_spread_proxy', 'order_flow_imbalance_ewm']

regimeFeatures_CLF = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'trade_imbalance']

X_train, X_test = X_train[features+['timestamp']+[target]], X_test[features+['ID']]

# --- Step 2.5: Create Enhanced Features with PCA and Feature Selection ---
logger.info("Creating enhanced features with PCA and feature selection...")
print("🚀 Creating enhanced features with PCA and feature selection...")

# Save current data to temporary files for processing
temp_train_file = 'temp_train_features.csv'
temp_enhanced_train_file = 'temp_enhanced_train.csv'

print("   💾 Saving training data for feature enhancement...")
logger.info("Saving training data to temporary file for feature enhancement")
X_train.to_csv(temp_train_file, index=False)

# Create PCA components (keep original features)
logger.info("Creating PCA components for feature enhancement")
create_component_features(
    input_file=temp_train_file, 
    output_file=temp_enhanced_train_file, 
    target_col=target, 
    n_components=20,  # Create 20 PCA components
    chunksize=5000,
    keep_original=True
)

# Load enhanced data
print("   📂 Loading enhanced training data...")
logger.info("Loading enhanced training data with PCA components")
X_train_enhanced = pd.read_csv(temp_enhanced_train_file)

# Select top 50 features for HMM training
print("   🎯 Selecting top 50 features for HMM training...")
logger.info("Selecting top 50 features for HMM training")

# Create a temporary file with just the features we want to consider for selection
hmm_features_file = 'temp_hmm_features.csv'
hmm_candidate_features = regimeFeatures_HMM + [f'pca_{i}' for i in range(20)]  # Original HMM features + PCA components
X_hmm_candidates = X_train_enhanced[hmm_candidate_features + [target]]
X_hmm_candidates.to_csv(hmm_features_file, index=False)

# Select top 50 features
top_50_features = select_top_k_features(
    file_path=hmm_features_file, 
    target_col=target, 
    k=50, 
    chunksize=5000
)

logger.info(f"Selected top 50 features for HMM: {top_50_features[:10]}...")  # Log first 10
print(f"   ✅ Selected top 50 features (showing first 5): {top_50_features[:5]}")

# Update regime features for HMM with selected features
regimeFeatures_HMM_enhanced = top_50_features

# Use enhanced training data
X_train = X_train_enhanced
logger.info(f"Enhanced training data shape: {X_train.shape}")
print(f"   📊 Enhanced training data: {X_train.shape}")

# Clean up temporary files
import os
try:
    os.remove(temp_train_file)
    os.remove(temp_enhanced_train_file)
    os.remove(hmm_features_file)
    logger.info("Temporary files cleaned up")
except:
    logger.warning("Could not clean up all temporary files")

# --- Step 3: Regime Detection with HMM ---
logger.info("Starting regime detection with HMM using enhanced features...")
print("🎲 Starting HMM-based regime detection with enhanced features...")

# Prepare observation sequence (enhanced features + returns)
observations = pd.concat([X_train[regimeFeatures_HMM_enhanced], y_train], axis=1)

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

X_train['predicted_regime'] = predicted_regimes

print(f"📊 Final regime distribution: {np.bincount(predicted_regimes)}")
print(f"📈 Number of regimes detected: {len(np.unique(predicted_regimes))}")
logger.info(f"HMM regime distribution: {dict(enumerate(np.bincount(predicted_regimes)))}")

# --- Step 3.5: Analyze Regime Properties ---
logger.info("Analyzing regime properties...")
print("📈 Analyzing regime properties...")

# Map HMM states to consistent regime numbering
regime_mapping = {}
for regime_id in range(len(np.unique(predicted_regimes))):
    regime_data = X_train[X_train['predicted_regime'] == regime_id]
    mean_return = regime_data[target].mean()
    std_return = regime_data[target].std()
    duration = len(regime_data)
    regime_mapping[regime_id] = (mean_return, std_return, duration)

print("\nDetected Regime Properties:")
logger.info("Regime properties analysis:")
for regime_id, (mean, std, dur) in regime_mapping.items():
    print(f"Regime {regime_id}: μ={mean:.6f}, σ={std:.6f}, duration={dur}s")
    logger.info(f"Regime {regime_id}: μ={mean:.6f}, σ={std:.6f}, duration={dur}s")

# --- Step 4: Train Regime Classifier ---
logger.info("Training regime classifier with balance checking...")
print("\n🌲 Training regime classifier with balance checking...")
X_RegimeTrain = scaler.fit_transform(X_train[regimeFeatures_CLF])
y_RegimeTrain = X_train['predicted_regime']

# Fit classifier with balance checking
clf, best_auc, best_balance = fit_classifier_with_balance_check(X_RegimeTrain, y_RegimeTrain)

print(f"✅ Final classifier - ROC AUC: {best_auc:.4f}, Balance ratio: {best_balance:.4f}")
logger.info(f"Final classifier performance: ROC AUC={best_auc:.4f}, Balance ratio={best_balance:.4f}")

# --- Step 5.00: Predict Regimes for Train and Test Data ---
logger.info("Predicting regimes for train and test data...")
print("🔮 Predicting regimes for train and test data...")

# For training data (using classifier, not HMM directly)
X_train['predicted_regime_Est'] = clf.predict(X_train[regimeFeatures_CLF])

# For test data (only features available)
X_test['predicted_regime_Est'] = clf.predict(X_test[regimeFeatures_CLF])

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
test_dist = X_test['predicted_regime_Est'].value_counts()
print(test_dist)
logger.info(f"Test regime distribution: {test_dist.to_dict()}")

# Create regime dataframes for output
logger.info("Creating output regime dataframes...")
print("💾 Creating output files...")

train_regimes_df = X_train[['timestamp', 'predicted_regime', 'predicted_regime_Est']].copy()
test_regimes_df = X_test[['ID', 'predicted_regime_Est']].copy()

# --- Step 5.75: Output Regime DataFrames ---
# Save to CSV files
train_regimes_df.to_csv('train_regimes.csv', index=False)
test_regimes_df.to_csv('test_regimes.csv', index=False)

print("\n✅ Output saved to train_regimes.csv and test_regimes.csv")
logger.info("Regime detection pipeline completed successfully")


