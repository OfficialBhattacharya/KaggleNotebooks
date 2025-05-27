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
import lightgbm as lgb

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
        import lightgbm as lgb
        # Check if GPU is available
        gpu_available = lgb.LGBMRegressor(device='gpu', gpu_use_dp=True, objective='regression')
        print("🚀 GPU detected and will be used for training")
        return 'gpu'
    except:
        print("⚠️  GPU not available, using CPU with multi-threading")
        return 'cpu'

print("🔄 Starting DRW Crypto Market Prediction with LightGBM")
print("=" * 60)

# Check system resources
print("🔍 Checking system resources...")
device_type = check_gpu_availability()
cpu_count = os.cpu_count()
print(f"💻 Available CPU cores: {cpu_count}")
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
    
    print("   - Handling infinite values and NaNs...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Define feature columns
    base_features = [f'X{i}' for i in range(1, 891)]
    market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
    engineered_features = [
        'imbalance', 'bid_ask_spread', 'buy_sell_ratio', 'volume_imbalance',
        'price_pressure', 'bid_ask_mid', 'order_flow'
    ]
    
    features = base_features + market_features + engineered_features
    
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

print("🤖 Setting up LightGBM model...")

# LightGBM parameters optimized for performance and GPU usage
lgb_params = {
    'objective': 'regression',
    'metric': 'None',  # We'll use custom correlation metric
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': cpu_count if device_type == 'cpu' else 1,
    'device': device_type,
}

if device_type == 'gpu':
    lgb_params.update({
        'gpu_use_dp': True,
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
    })

print(f"📋 LightGBM parameters: {lgb_params}")

# Define correlation evaluation metric
def lgb_correlation_eval(y_true, y_pred):
    """Custom correlation evaluation metric for LightGBM"""
    correlation = np.corrcoef(y_pred, y_true)[0, 1]
    # Return name, eval_result, is_higher_better
    return 'correlation', correlation, True

# Create and train model
print("🏋️  Training LightGBM model...")
start_time = time.time()

model = lgb.LGBMRegressor(
    n_estimators=1000,
    early_stopping_rounds=50,
    **lgb_params
)

# Fit the model
model.fit(
    X, y,
    eval_set=[(X, y)],
    eval_metric=lgb_correlation_eval,
    callbacks=[lgb.log_evaluation(period=100)]
)

training_time = time.time() - start_time
print(f"✅ Model training completed in {training_time:.2f} seconds")
print_memory_usage()

# Delete training features to free memory
print("🗑️  Deleting training features to free memory...")
del X, y
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
test_preds = model.predict(X_test)
prediction_time = time.time() - start_time
print(f"✅ Predictions completed in {prediction_time:.2f} seconds")
print(f"📊 Prediction statistics:")
print(f"   - Min: {test_preds.min():.6f}")
print(f"   - Max: {test_preds.max():.6f}")
print(f"   - Mean: {test_preds.mean():.6f}")
print(f"   - Std: {test_preds.std():.6f}")

# Delete test features to free memory
print("🗑️  Deleting test features to free memory...")
del X_test, model, scaler
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