import pandas as pd
import numpy as np
import logging
import time
from cuml.ensemble import RandomForestClassifier as cuRFClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def print_and_log(message, level='info'):
    """Helper function to both print and log messages"""
    print(f"\n{'='*80}\n{message}\n{'='*80}")
    if level == 'info':
        logger.info(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)

def optimize_memory(df):
    """Optimize memory usage of dataframe"""
    print_and_log("Starting memory optimization...")
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2  # in MB
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2  # in MB
    memory_saved = initial_memory - final_memory
    print_and_log(f"Memory optimization complete. Saved {memory_saved:.2f} MB")
    return df

# Step 1: Load and optimize data
print_and_log("Starting data processing pipeline...")
start_time = time.time()

# Assuming train_data is already loaded
features = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type',
            'Nitrogen', 'Potassium', 'Phosphorous']
target = 'Fertilizer Name'

print_and_log(f"Selected features: {', '.join(features)}")
print_and_log(f"Target variable: {target}")

X = train_data[features].copy()
y = train_data[target].copy()

print_and_log(f"Initial data shape: {X.shape}")
print_and_log(f"Number of unique target classes: {y.nunique()}")

# Optimize memory usage
X = optimize_memory(X)
logger.info(f"Memory optimization completed in {time.time() - start_time:.2f} seconds")

# Step 2: Fast encoding of categorical variables
print_and_log("Starting categorical variable encoding...")
encode_start = time.time()

# Convert categorical columns to codes directly (faster than LabelEncoder)
X['Soil Type'] = X['Soil Type'].cat.codes
X['Crop Type'] = X['Crop Type'].cat.codes

# For target variable, we still need LabelEncoder for inverse transform later
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print_and_log(f"Encoding completed in {time.time() - encode_start:.2f} seconds")
print_and_log(f"Encoded target classes: {le_target.classes_}")

# Step 3: Split data
print_and_log("Splitting data into train/test sets...")
split_start = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print_and_log(f"Train set shape: {X_train.shape}")
print_and_log(f"Test set shape: {X_test.shape}")
logger.info(f"Data split completed in {time.time() - split_start:.2f} seconds")

# Step 4: Train GPU Random Forest
print_and_log("Initializing GPU Random Forest model...")
print_and_log("Model parameters:")
print(f"- Number of estimators: 750")
print(f"- Number of GPUs: 2")
print(f"- Random state: 42")

train_start = time.time()

# Initialize and train GPU Random Forest
rf = cuRFClassifier(
    n_estimators=750,
    random_state=42,
    n_streams=2  # Utilize both T4 GPUs
)

print_and_log("Starting model training...")
rf.fit(X_train, y_train)

training_time = time.time() - train_start
print_and_log(f"Model training completed in {training_time:.2f} seconds")
print_and_log(f"Average time per tree: {training_time/750:.4f} seconds")

# Step 5: Evaluate
print_and_log("Starting model evaluation...")
eval_start = time.time()

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print_and_log("Model Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

logger.info(f"Evaluation completed in {time.time() - eval_start:.2f} seconds")

# Final summary
total_time = time.time() - start_time
print_and_log("Pipeline Execution Summary:")
print(f"Total execution time: {total_time:.2f} seconds")
print(f"Memory optimization time: {encode_start - start_time:.2f} seconds")
print(f"Encoding time: {split_start - encode_start:.2f} seconds")
print(f"Training time: {training_time:.2f} seconds")
print(f"Evaluation time: {time.time() - eval_start:.2f} seconds")