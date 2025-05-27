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
import catboost as cb
from catboost import CatBoostRegressor, Pool

# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
# Display
from IPython.display import display
import time

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"💾 Current memory usage: {memory_mb:.2f} MB")

def check_gpu_availability():
    """Check for GPU availability and return device type"""
    try:
        # Check if CatBoost can use GPU
        import catboost as cb
        # Try to create a GPU-enabled model
        temp_model = cb.CatBoostRegressor(
            iterations=1,
            task_type='GPU',
            devices='0',
            verbose=False
        )
        # Test with dummy data
        temp_model.fit(
            np.random.rand(10, 5), 
            np.random.rand(10),
            verbose=False
        )
        print("🚀 GPU detected and will be used for training")
        return 'GPU'
    except Exception as e:
        print("⚠️  GPU not available, using CPU with multi-threading")
        print(f"   GPU error: {str(e)}")
        return 'CPU'

def get_gpu_count():
    """Get number of available GPUs"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_count = len(result.stdout.strip().split('\n'))
            print(f"🎮 Found {gpu_count} GPU(s)")
            return gpu_count
        else:
            return 0
    except:
        return 0

print("🔄 Starting DRW Crypto Market Prediction with CatBoost")
print("=" * 60)

# Check system resources
print("🔍 Checking system resources...")
device_type = check_gpu_availability()
cpu_count = os.cpu_count()
gpu_count = get_gpu_count() if device_type == 'GPU' else 0
print(f"💻 Available CPU cores: {cpu_count}")
if gpu_count > 0:
    print(f"🎮 Available GPUs: {gpu_count}")
print_memory_usage()
print()

print("📊 Loading training data...")
start_time = time.time()
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
print(f"✅ Training data loaded in {time.time() - start_time:.2f} seconds")
print(f"📈 Training data shape: {train.shape}")
print_memory_usage()
print()

def preprocess_train(df):
    """Preprocess training data with feature engineering"""
    print("🔧 Starting feature engineering on training data...")
    
    # Basic features
    print("   - Creating imbalance features...")
    df['imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-6)
    
    print("   - Creating spread features...")
    df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-6)
    
    # Additional features for better performance
    print("   - Creating volume-based features...")
    df['volume_imbalance'] = df['volume'] * df['imbalance']
    df['price_pressure'] = (df['buy_qty'] - df['sell_qty']) / df['volume']
    
    print("   - Creating momentum features...")
    df['bid_ask_mid'] = (df['bid_qty'] + df['ask_qty']) / 2
    df['order_flow'] = df['buy_qty'] - df['sell_qty']
    
    # CatBoost-specific features (it handles interactions automatically)
    print("   - Creating ratio features...")
    df['volume_to_imbalance_ratio'] = df['volume'] / (np.abs(df['imbalance']) + 1e-6)
    df['order_intensity'] = (df['buy_qty'] + df['sell_qty']) / df['volume']
    
    print("   - Handling infinite values and NaNs...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Define feature columns
    base_features = [f'X{i}' for i in range(1, 891)]
    market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    engineered_features = [
        'imbalance', 'bid_ask_spread', 'buy_sell_ratio', 'volume_imbalance',
        'price_pressure', 'bid_ask_mid', 'order_flow', 'volume_to_imbalance_ratio',
        'order_intensity'
    ]
    
    features = base_features + market_features + engineered_features
    
    # CatBoost can handle raw features well, but we'll still scale for consistency
    print("   - Scaling features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].astype(np.float32))
    y = df['label'].astype(np.float32).values

    print(f"✅ Feature engineering completed. Final feature count: {len(features)}")
    return X, y, features, scaler

# Preprocess training data
X, y, features, scaler = preprocess_train(train)
print_memory_usage()

# Delete training dataframe to free memory
print("🗑️  Deleting training dataframe to free memory...")
del train
gc.collect()
print_memory_usage()
print()

print("🤖 Setting up CatBoost model...")

# CatBoost parameters optimized for financial data and GPU usage
if device_type == 'GPU':
    catboost_params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 8,
        'l2_leaf_reg': 3,
        'random_strength': 1,
        'bagging_temperature': 1,
        'od_type': 'Iter',
        'od_wait': 50,
        'random_seed': 42,
        'allow_writing_files': False,
        'task_type': 'GPU',
        'devices': '0',
        'gpu_ram_part': 0.8,
        'verbose': 100,
        'eval_metric': 'RMSE',
        'loss_function': 'RMSE'
    }
else:
    catboost_params = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 8,
        'l2_leaf_reg': 3,
        'random_strength': 1,
        'bagging_temperature': 1,
        'od_type': 'Iter',
        'od_wait': 50,
        'random_seed': 42,
        'allow_writing_files': False,
        'task_type': 'CPU',
        'thread_count': cpu_count,
        'verbose': 100,
        'eval_metric': 'RMSE',
        'loss_function': 'RMSE'
    }

print(f"📋 CatBoost parameters: {catboost_params}")

# Define correlation evaluation metric for CatBoost
class CorrelationMetric:
    """Custom correlation evaluation metric for CatBoost"""
    
    def get_final_error(self, error, weight):
        return error
    
    def is_max_optimal(self):
        return True
    
    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        
        y_pred = np.array(approxes[0])
        y_true = np.array(target)
        
        # Calculate correlation
        correlation = np.corrcoef(y_pred, y_true)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
            
        return correlation, 1.0

# Create CatBoost Pool for efficient memory usage
print("🔄 Creating CatBoost Pool...")
train_pool = Pool(
    data=X,
    label=y,
    feature_names=[f'feature_{i}' for i in range(X.shape[1])]
)
print_memory_usage()

# Delete X, y to free memory before training
print("🗑️  Deleting training arrays to free memory...")
del X, y
gc.collect()
print_memory_usage()
print()

# Create and train model
print("🏋️  Training CatBoost model...")
start_time = time.time()

model = CatBoostRegressor(**catboost_params)

# Fit the model with custom correlation metric
model.fit(
    train_pool,
    eval_set=train_pool,
    use_best_model=True,
    plot=False
)

training_time = time.time() - start_time
print(f"✅ Model training completed in {training_time:.2f} seconds")
print(f"🎯 Best iteration: {model.get_best_iteration()}")
print(f"🎯 Best score: {model.get_best_score()}")
print_memory_usage()

# Delete training pool to free memory
print("🗑️  Deleting training pool to free memory...")
del train_pool
gc.collect()
print_memory_usage()
print()

print("📊 Loading test data...")
start_time = time.time()
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
print(f"✅ Test data loaded in {time.time() - start_time:.2f} seconds")
print(f"📈 Test data shape: {test.shape}")
print_memory_usage()

def preprocess_test(df_test, features, scaler):
    """Preprocess test data with same feature engineering as training"""
    print("🔧 Starting feature engineering on test data...")
    
    # Apply same feature engineering as training
    print("   - Creating imbalance features...")
    df_test['imbalance'] = (df_test['buy_qty'] - df_test['sell_qty']) / (df_test['buy_qty'] + df_test['sell_qty'] + 1e-6)
    
    print("   - Creating spread features...")
    df_test['bid_ask_spread'] = df_test['ask_qty'] - df_test['bid_qty']
    df_test['buy_sell_ratio'] = df_test['buy_qty'] / (df_test['sell_qty'] + 1e-6)
    
    print("   - Creating volume-based features...")
    df_test['volume_imbalance'] = df_test['volume'] * df_test['imbalance']
    df_test['price_pressure'] = (df_test['buy_qty'] - df_test['sell_qty']) / df_test['volume']
    
    print("   - Creating momentum features...")
    df_test['bid_ask_mid'] = (df_test['bid_qty'] + df_test['ask_qty']) / 2
    df_test['order_flow'] = df_test['buy_qty'] - df_test['sell_qty']
    
    # CatBoost-specific features
    print("   - Creating ratio features...")
    df_test['volume_to_imbalance_ratio'] = df_test['volume'] / (np.abs(df_test['imbalance']) + 1e-6)
    df_test['order_intensity'] = (df_test['buy_qty'] + df_test['sell_qty']) / df_test['volume']
    
    print("   - Handling infinite values and NaNs...")
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.fillna(0, inplace=True)

    print("   - Scaling features...")
    X_test = scaler.transform(df_test[features].astype(np.float32))
    
    print("✅ Test data preprocessing completed")
    return X_test

# Preprocess test data
X_test = preprocess_test(test, features, scaler)
print_memory_usage()

# Delete test dataframe to free memory (keep only what we need for submission)
print("🗑️  Cleaning up test data to free memory...")
test_ids = test.index if 'id' not in test.columns else test['id']
del test
gc.collect()
print_memory_usage()
print()

print("🔮 Making predictions on test data...")
start_time = time.time()

# Create Pool for test data
test_pool = Pool(
    data=X_test,
    feature_names=[f'feature_{i}' for i in range(X_test.shape[1])]
)

# Make predictions
test_preds = model.predict(test_pool)

prediction_time = time.time() - start_time
print(f"✅ Predictions completed in {prediction_time:.2f} seconds")
print(f"📊 Prediction statistics:")
print(f"   - Min: {test_preds.min():.6f}")
print(f"   - Max: {test_preds.max():.6f}")
print(f"   - Mean: {test_preds.mean():.6f}")
print(f"   - Std: {test_preds.std():.6f}")

# Delete test features and model to free memory
print("🗑️  Deleting test features and model to free memory...")
del X_test, test_pool, model, scaler
gc.collect()
print_memory_usage()
print()

print("📝 Creating submission file...")
# Load sample submission to get the correct format
submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')
print(f"📋 Submission template shape: {submission.shape}")

# Update predictions
submission['prediction'] = test_preds

# Save submission
submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")

# Final cleanup
del submission, test_preds
gc.collect()

print()
print("🎉 Process completed successfully!")
print("=" * 60)
print_memory_usage()
print(f"📁 Submission file: submission.csv")
print("🚀 Ready for submission to Kaggle!") 