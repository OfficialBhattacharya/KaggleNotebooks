"""
Enhanced XGBoost Baseline for CMI Sensor Data Competition

WORKFLOW & USAGE INSTRUCTIONS
-----------------------------
This script is designed for both OFFLINE (batch) and ONLINE (API/server) Kaggle submissions.

1. Training & Artifact Creation:
   - When you run this script (with correct data paths), it will:
     * Train the XGBoost model on the provided data (using GPU if available).
     * Apply comprehensive feature engineering including time-series features (slopes, FFT, ranges).
     * Store the trained model, label encoder, and feature selector in memory for efficient inference.
     * Create a submission.csv file for offline submission.
     * You can configure the number of Optuna trials (n_trials) in main().

2. Feature Engineering:
   - Statistical features: mean, std, var, quantiles, min, max, skew, kurtosis for all sensors
   - Time-series features: slopes using linear regression, ranges (max-min) for IMU sensors
   - FFT features: mean and standard deviation of first 5 frequency components for accelerometer
   - Sensor failure patterns: count and detection of missing/null sensor readings

3. Offline Submission:
   - After running, upload the generated submission.csv to Kaggle for evaluation.

4. API/Server Submission:
   - The script includes a predict() function and API server code compatible with Kaggle's evaluation API.
   - When submitted for API evaluation, Kaggle will call your predict() function for each batch.
   - The predict() function uses the in-memory trained artifacts for fast predictions.
   - No file I/O during inference - everything is kept in memory for optimal performance.

5. How to Use:
   - Set all configuration (data paths, file names, hyperparameter search space, n_trials) in the main() function.
   - Optionally provide initial_params from previous best trials to warm-start Optuna optimization.
   - Run the script once to train and generate all necessary files.
   - For API submission, do NOT retrain in the predict() function; it only uses in-memory models for inference.

6. Optuna Optimization:
   - Supports warm-start optimization: provide known good parameters as initial_params to start from.
   - The first trial will use your provided parameters, subsequent trials will explore around them.
   - Set initial_params = None in main() to start optimization from scratch.

7. Notes:
   - The script is self-contained: no need for separate training/inference scripts.
   - If you change the model or features, rerun the script to update the in-memory artifacts.
   - XGBoost is configured to use GPU (tree_method='hist', device='cuda') for faster training on supported hardware.
   - All artifacts are stored in memory for maximum inference performance - no disk I/O during predictions.
"""
import polars as pl
import pandas as pd
import numpy as np
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
import logging
import warnings
from scipy import stats
from scipy.fft import fft
import os
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Constants
SENSOR_DTYPES = {
    'acc': pl.Float64,
    'rot': pl.Float64,
    'thm': pl.Float64,
    'tof': pl.Float64
}
BEHAVIOR_CATEGORIES = ['Transition', 'Pause', 'Gesture']
RANDOM_SEED = 42

# Global storage for trained artifacts (in-memory storage for inference)
TRAINED_MODEL = None
TRAINED_LABEL_ENCODER = None
TRAINED_FEATURE_SELECTOR = None
TRAINED_PRESELECT_FEATURE_COLUMNS = None
TRAINED_FEATURE_COLUMNS = None

class EnhancedSensorBaseline:
    """Enhanced baseline solution with memory optimization and advanced features."""
    
    def __init__(self, data_path: str, xgb_search_space: dict):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.model = None
        self.best_params = None
        self.xgb_search_space = xgb_search_space
        
    def load_and_preprocess(self, file_name: str, is_train: bool = True) -> pl.DataFrame:
        """Load and preprocess data using Polars with memory optimization."""
        logger.info(f"Loading data from: {file_name}")
        print(f"[STEP] Loading data from: {file_name}")
        
        file_path = os.path.join(self.data_path, file_name)
        
        # Read only necessary columns to save memory
        columns = [
            'sequence_id', 'sequence_counter', 'subject',
            *[f"{pre}_{suf}" for pre in ['acc'] for suf in ['x', 'y', 'z']],
            *[f"rot_{s}" for s in ['w', 'x', 'y', 'z']],
            *[f"thm_{i}" for i in range(1, 6)],
            *[f"tof_{sensor}_v{pix}" for sensor in range(1, 6) for pix in range(64)]
        ]
        if is_train:
            columns.append('gesture')
        
        # Load data with specified dtypes
        df = pl.read_csv(file_path, columns=columns)
        
        # Fill nulls in sensor data
        logger.info("Handling missing values")
        print("[STEP] Handling missing values")
        for col in df.columns:
            if col.startswith('thm_') or col.startswith('tof_'):
                df = df.with_columns(pl.col(col).fill_null(-1.0))
        
        # Cast sensor columns to float64
        for prefix, dtype in SENSOR_DTYPES.items():
            sensor_cols = [col for col in df.columns if col.startswith(prefix)]
            df = df.with_columns([pl.col(col).cast(dtype) for col in sensor_cols])
        
        # Only process behavior for train data if present
        if is_train and 'behavior' in df.columns:
            logger.info("Encoding behavior features")
            print("[STEP] Encoding behavior features")
            behavior_expr = pl.col('behavior').cast(pl.Categorical)
            df = df.with_columns(behavior_expr.alias('behavior'))
            df = df.to_dummies(columns=['behavior'])
        
        return df
    

    
    def add_sensor_failure_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add features to capture sensor failure patterns."""
        logger.info("Adding sensor failure pattern features")
        print("[STEP] Adding sensor failure pattern features")

        # Thermopile failure patterns
        thm_cols = [col for col in df.columns if col.startswith('thm_')]
        if thm_cols:
            df = df.with_columns(
                thm_failure_count=pl.sum_horizontal([pl.col(col).is_null().cast(pl.Int64) for col in thm_cols])
            )

        # ToF failure patterns
        tof_cols = [col for col in df.columns if col.startswith('tof_')]
        if tof_cols:
            df = df.with_columns(
                tof_failure_count=pl.sum_horizontal([pl.col(col).is_null().cast(pl.Int64) for col in tof_cols])
            )

        # Overall sensor health
        all_sensor_cols = thm_cols + tof_cols
        if all_sensor_cols:
            df = df.with_columns(
                total_sensor_failures=pl.sum_horizontal([pl.col(col).is_null().cast(pl.Int64) for col in all_sensor_cols])
            )

        return df
    
    def aggregate_sequence_features(self, df: pl.DataFrame, is_train: bool = True) -> pl.DataFrame:
        """Aggregate sequence data with comprehensive statistical and time-series features."""
        logger.info("Aggregating sequence features with time-series analysis")
        print("[STEP] Aggregating sequence features with time-series analysis")
        
        # Define sensor columns
        sensor_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in SENSOR_DTYPES.keys())]
        behavior_cols = [col for col in df.columns if col.startswith('behavior_')]
        
        # Build aggregation expressions
        agg_exprs = []
        
        # Statistical features for sensor columns
        for col in sensor_cols:
            agg_exprs.extend([
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).std().alias(f"{col}_std"),
                pl.col(col).var().alias(f"{col}_var"),
                pl.col(col).quantile(0.25).alias(f"{col}_q25"),
                pl.col(col).median().alias(f"{col}_q50"),
                pl.col(col).quantile(0.75).alias(f"{col}_q75"),
                pl.col(col).max().alias(f"{col}_max"),
                pl.col(col).min().alias(f"{col}_min"),
                pl.col(col).skew().alias(f"{col}_skew"),
                pl.col(col).kurtosis().alias(f"{col}_kurt"),
            ])
        
        # Time-series features for key sensor columns
        for sensor in ['acc', 'rot']:
            for axis in ['x', 'y', 'z']:
                col = f"{sensor}_{axis}"
                if col in sensor_cols:
                    # Slope calculation using list operations
                    agg_exprs.append(
                        pl.col(col).map_elements(
                            lambda values: stats.linregress(range(len(values)), values)[0] if len(values) > 1 else 0.0,
                            return_dtype=pl.Float64
                        ).alias(f"{col}_slope")
                    )
                    
                    # Range (max - min) as a simple trend indicator
                    agg_exprs.append(
                        (pl.col(col).max() - pl.col(col).min()).alias(f"{col}_range")
                    )
        
        # FFT features for accelerometer data
        for sensor in ['acc_x', 'acc_y', 'acc_z']:
            if sensor in sensor_cols:
                # FFT mean of first 5 components
                agg_exprs.append(
                    pl.col(sensor).map_elements(
                        lambda values: np.abs(fft(values))[:5].mean() if len(values) > 4 else 0.0,
                        return_dtype=pl.Float64
                    ).alias(f"{sensor}_fft_mean")
                )
                
                # FFT standard deviation of first 5 components
                agg_exprs.append(
                    pl.col(sensor).map_elements(
                        lambda values: np.abs(fft(values))[:5].std() if len(values) > 4 else 0.0,
                        return_dtype=pl.Float64
                    ).alias(f"{sensor}_fft_std")
                )
        
        # Sum features for behavior columns
        for col in behavior_cols:
            agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
        
        # Sequence-level features
        agg_exprs.extend([
            pl.col('sequence_counter').count().alias('sequence_length'),
            pl.col('subject').first().alias('subject'),
        ])
        
        if is_train:
            agg_exprs.append(pl.col('gesture').first().alias('gesture'))
        
        return df.group_by('sequence_id').agg(agg_exprs)
    
    def load_demographics(self, file_name: str) -> pl.DataFrame:
        """Load demographics data."""
        logger.info(f"Loading demographics from: {file_name}")
        print(f"[STEP] Loading demographics from: {file_name}")
        file_path = os.path.join(self.data_path, file_name)
        return pl.read_csv(file_path)
    
    def merge_demographics(self, sequence_df: pl.DataFrame, demo_df: pl.DataFrame) -> pl.DataFrame:
        """Merge sequence data with demographics."""
        logger.info("Merging demographics data")
        print("[STEP] Merging demographics data")
        return sequence_df.join(demo_df, on="subject", how="left")
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 1000) -> pd.DataFrame:
        """Perform feature selection using ANOVA F-test."""
        logger.info(f"Performing feature selection (selecting top {k} features)")
        print(f"[STEP] Performing feature selection (selecting top {k} features)")
        
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        logger.info(f"Selected {len(selected_features)} features")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50, initial_params: dict = None) -> dict:
        """Optimize XGBoost parameters using Optuna."""
        logger.info("Starting hyperparameter optimization")
        print(f"[STEP] Starting hyperparameter optimization with {n_trials} trials")
        if initial_params:
            logger.info(f"Using initial parameters: {initial_params}")
            print(f"[STEP] Using initial parameters as first trial: {initial_params}")
        
        search_space = self.xgb_search_space
        def objective(trial):
            params = {}
            for param, spec in search_space.items():
                if spec['type'] == 'int':
                    params[param] = trial.suggest_int(param, spec['low'], spec['high'])
                elif spec['type'] == 'float':
                    params[param] = trial.suggest_float(param, spec['low'], spec['high'], log=spec.get('log', False))
                elif spec['type'] == 'categorical':
                    params[param] = trial.suggest_categorical(param, spec['choices'])
                else:
                    params[param] = spec['value']
            # Add fixed params
            params['random_state'] = RANDOM_SEED
            params['n_jobs'] = -1
            params['enable_categorical'] = False
            params['tree_method'] = 'hist'  # Use hist with device for GPU
            params['device'] = 'cuda'  # Use CUDA for GPU training
            params['eval_metric'] = 'mlogloss'
            
            # Use stratified k-fold for more robust evaluation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            scores = []
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model = XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)
                preds = model.predict(X_val)
                score = f1_score(y_val, preds, average='macro')
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        
        # If initial parameters are provided, enqueue them as the first trial
        if initial_params:
            # Filter initial_params to only include parameters that are in the search space
            filtered_initial_params = {}
            for param, value in initial_params.items():
                if param in search_space:
                    filtered_initial_params[param] = value
            
            if filtered_initial_params:
                study.enqueue_trial(filtered_initial_params)
                logger.info(f"Enqueued initial trial with parameters: {filtered_initial_params}")
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Best CV score: {study.best_value:.4f}")
        
        return self.best_params
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, use_feature_selection: bool = True, n_trials: int = 50, initial_params: dict = None) -> None:
        """Train the final model with optimized parameters."""
        global TRAINED_MODEL, TRAINED_LABEL_ENCODER, TRAINED_FEATURE_SELECTOR, TRAINED_PRESELECT_FEATURE_COLUMNS, TRAINED_FEATURE_COLUMNS
        
        logger.info("Training final model")
        print("[STEP] Training final model")
        
        # Store pre-selection feature columns in memory
        TRAINED_PRESELECT_FEATURE_COLUMNS = list(X.columns)
        
        if use_feature_selection:
            X = self.feature_selection(X, y)
        
        # Store feature columns after feature selection in memory
        TRAINED_FEATURE_COLUMNS = list(X.columns)
        
        # Optimize hyperparameters if not already done
        if self.best_params is None:
            self.optimize_hyperparameters(X, y, n_trials=n_trials, initial_params=initial_params)
        
        # Train final model with updated GPU configuration
        final_params = self.best_params.copy()
        final_params['random_state'] = RANDOM_SEED
        self.model = XGBClassifier(**final_params)
        self.model.fit(X, y)
        
        # Store trained artifacts in global memory
        TRAINED_MODEL = self.model
        TRAINED_LABEL_ENCODER = self.label_encoder
        TRAINED_FEATURE_SELECTOR = self.feature_selector
        
        logger.info("Model training completed and stored in memory")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        global TRAINED_PRESELECT_FEATURE_COLUMNS, TRAINED_FEATURE_COLUMNS
        
        print("[STEP] Making predictions with trained model")
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Align features to match training data before feature selection using in-memory data
        if TRAINED_PRESELECT_FEATURE_COLUMNS is not None:
            # Ensure all expected columns are present before feature selection
            for col in TRAINED_PRESELECT_FEATURE_COLUMNS:
                if col not in X.columns:
                    X[col] = 0
            X = X[TRAINED_PRESELECT_FEATURE_COLUMNS]
        
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.feature_selector.get_feature_names_out())
        
        # Final feature alignment after feature selection using in-memory data
        if TRAINED_FEATURE_COLUMNS is not None:
            # Ensure all expected columns are present for final model
            for col in TRAINED_FEATURE_COLUMNS:
                if col not in X.columns:
                    X[col] = 0
            X = X[TRAINED_FEATURE_COLUMNS]
        
        return self.model.predict(X)
    
    def process_pipeline(self, train_file: str, test_file: str,
                        train_demo_file: str, test_demo_file: str) -> tuple:
        """Complete data processing pipeline."""
        logger.info("Starting complete data processing pipeline")
        print("[STEP] Starting complete data processing pipeline")
        
        # Load and preprocess training data
        train_seq = self.load_and_preprocess(train_file, is_train=True)
        train_seq = self.add_sensor_failure_features(train_seq)
        train_seq = self.aggregate_sequence_features(train_seq, is_train=True)
        
        # Load and merge demographics
        train_demo = self.load_demographics(train_demo_file)
        train_data = self.merge_demographics(train_seq, train_demo)
        
        # Load and preprocess test data
        test_seq = self.load_and_preprocess(test_file, is_train=False)
        test_seq = self.add_sensor_failure_features(test_seq)
        test_seq = self.aggregate_sequence_features(test_seq, is_train=False)
        
        # Load and merge test demographics
        test_demo = self.load_demographics(test_demo_file)
        test_data = self.merge_demographics(test_seq, test_demo)
        
        # Convert to pandas
        train_pd = train_data.to_pandas()
        test_pd = test_data.to_pandas()
        
        # Prepare features and target
        feature_cols = [col for col in train_pd.columns 
                       if col not in ['sequence_id', 'gesture', 'subject', 'behavior'] and not col.startswith('behavior_')]
        
        X_train = train_pd[feature_cols].fillna(0)
        y_train = train_pd['gesture']
        X_test = test_pd[feature_cols].fillna(0)
        
        # Encode target
        y_train_encoded = pd.Series(self.label_encoder.fit_transform(y_train), index=y_train.index)
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train, y_train_encoded, X_test, test_pd['sequence_id']
    
    def create_submission(self, predictions: np.ndarray, sequence_ids: pd.Series, 
                         output_file: str) -> None:
        """Create submission file."""
        logger.info("Creating submission file")
        print(f"[STEP] Creating submission file: {output_file}")
        
        # Convert predictions back to original labels
        predicted_gestures = self.label_encoder.inverse_transform(predictions)
        
        # Create submission DataFrame
        submission = pl.DataFrame({
            'sequence_id': sequence_ids,
            'gesture': predicted_gestures
        })
        
        submission.write_csv(output_file)
        logger.info(f"Submission file created: {output_file}")
    
    def store_artifacts_in_memory(self):
        """Store trained artifacts in global memory for inference."""
        global TRAINED_MODEL, TRAINED_LABEL_ENCODER, TRAINED_FEATURE_SELECTOR
        print("[STEP] Storing model, label encoder, and feature selector in memory")
        TRAINED_MODEL = self.model
        TRAINED_LABEL_ENCODER = self.label_encoder
        TRAINED_FEATURE_SELECTOR = self.feature_selector
        logger.info("Artifacts stored in memory: model, label encoder, feature selector")
    
    def run_complete_pipeline(self, train_file: str, test_file: str,
                             train_demo_file: str, test_demo_file: str,
                             output_file: str, n_trials: int = 50, initial_params: dict = None) -> None:
        """Run the complete pipeline from data loading to submission."""
        logger.info("Running complete pipeline")
        print("[STEP] Running complete pipeline")
        
        # Process data
        X_train, y_train, X_test, sequence_ids = self.process_pipeline(
            train_file, test_file, train_demo_file, test_demo_file)
        
        # Train model
        self.train_model(X_train, y_train, use_feature_selection=True, n_trials=n_trials, initial_params=initial_params)
        
        # Make predictions (feature alignment now handled in predict method)
        predictions = self.predict(X_test)
        
        # Create submission
        self.create_submission(predictions, sequence_ids, output_file)
        
        logger.info("Pipeline completed successfully! Models stored in memory for inference.")


def main(n_trials: int = 50):
    """Main execution function. All configuration is set here."""
    print("[STEP] Starting main() function with configuration and n_trials =", n_trials)
    # === CONFIGURATION ===
    data_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/"
    train_file = "train.csv"
    test_file = "test.csv"
    train_demo_file = "train_demographics.csv"
    test_demo_file = "test_demographics.csv"
    output_file = "submission.csv"
    
    # Optional: Initial parameters from previous best trial
    # Set to None to start fresh, or provide your best known parameters
    initial_params = {
        'max_depth': 10,
        'learning_rate': 0.01747075162907032,
        'n_estimators': 455,
        'min_child_weight': 5,
        'gamma': 0.45697105250958436,
        'subsample': 0.96545122198772,
        'colsample_bytree': 0.7832853868908577,
        'reg_alpha': 5.025556098362407,
        'reg_lambda': 9.092051584827171,
        'tree_method': 'hist',
        'device': 'cuda'
    }
    # Set to None if you want to start optimization from scratch:
    # initial_params = None
    
    # XGBoost hyperparameter search space
    xgb_search_space = {
        'max_depth': {'type': 'int', 'low': 3, 'high': 12},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
        'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
        'gamma': {'type': 'float', 'low': 0, 'high': 0.5},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'reg_alpha': {'type': 'float', 'low': 0, 'high': 10},
        'reg_lambda': {'type': 'float', 'low': 0, 'high': 10},
        'tree_method': {'type': 'categorical', 'choices': ['hist']},  # Use hist with device for GPU
        'device': {'type': 'categorical', 'choices': ['cuda']},  # Always use CUDA for GPU
        # You can add more or change these as needed
    }
    # =====================

    # Initialize the enhanced baseline
    print("[STEP] Initializing EnhancedSensorBaseline class")
    baseline = EnhancedSensorBaseline(data_path=data_path, xgb_search_space=xgb_search_space)
    
    # Run complete pipeline
    print("[STEP] Running the complete pipeline (training, prediction, saving artifacts)")
    baseline.run_complete_pipeline(
        train_file=train_file,
        test_file=test_file,
        train_demo_file=train_demo_file,
        test_demo_file=test_demo_file,
        output_file=output_file,
        n_trials=n_trials,
        initial_params=initial_params
    )
    
    # Print model information
    if baseline.model is not None:
        print("[STEP] Printing top 10 feature importances")
        logger.info(f"Final model feature importance (top 10):")
        feature_importance = pd.DataFrame({
            'feature': baseline.model.feature_names_in_,
            'importance': baseline.model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        for _, row in feature_importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            print(f"  {row['feature']}: {row['importance']:.4f}")


# ========== API SERVER CODE FOR KAGGLE INFERENCE ==========
# This section enables compatibility with the Kaggle evaluation API server
import os
import polars as pl
import pandas as pd
import numpy as np

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    API predict function for Kaggle server.
    Receives a batch (sequence, demographics), uses in-memory model and encoders, applies feature engineering, and returns prediction.
    """
    global TRAINED_MODEL, TRAINED_LABEL_ENCODER, TRAINED_FEATURE_SELECTOR, TRAINED_PRESELECT_FEATURE_COLUMNS, TRAINED_FEATURE_COLUMNS
    
    print("[API STEP] Running predict() for a batch of data using in-memory models")
    
    # Use in-memory artifacts (no file loading needed)
    model = TRAINED_MODEL
    label_encoder = TRAINED_LABEL_ENCODER
    feature_selector = TRAINED_FEATURE_SELECTOR

    # Merge sequence and demographics on 'subject'
    if 'subject' in sequence.columns and 'subject' in demographics.columns:
        merged = sequence.join(demographics, on="subject", how="left")
    else:
        merged = sequence

    # --- Sensor failure features ---
    thm_cols = [col for col in merged.columns if col.startswith('thm_')]
    if thm_cols:
        merged = merged.with_columns(
            thm_failure_count=pl.sum_horizontal([pl.col(col).is_null().cast(pl.Int64) for col in thm_cols])
        )
    tof_cols = [col for col in merged.columns if col.startswith('tof_')]
    if tof_cols:
        merged = merged.with_columns(
            tof_failure_count=pl.sum_horizontal([pl.col(col).is_null().cast(pl.Int64) for col in tof_cols])
        )
    all_sensor_cols = thm_cols + tof_cols
    if all_sensor_cols:
        merged = merged.with_columns(
            total_sensor_failures=pl.sum_horizontal([pl.col(col).is_null().cast(pl.Int64) for col in all_sensor_cols])
        )

    # --- Aggregate sequence features ---
    SENSOR_DTYPES = {'acc': pl.Float64, 'rot': pl.Float64, 'thm': pl.Float64, 'tof': pl.Float64}
    sensor_cols = [col for col in merged.columns if any(col.startswith(prefix) for prefix in SENSOR_DTYPES.keys())]
    behavior_cols = []
    agg_exprs = []
    for col in sensor_cols:
        agg_exprs.extend([
            pl.col(col).mean().alias(f"{col}_mean"),
            pl.col(col).std().alias(f"{col}_std"),
            pl.col(col).var().alias(f"{col}_var"),
            pl.col(col).quantile(0.25).alias(f"{col}_q25"),
            pl.col(col).median().alias(f"{col}_q50"),
            pl.col(col).quantile(0.75).alias(f"{col}_q75"),
            pl.col(col).max().alias(f"{col}_max"),
            pl.col(col).min().alias(f"{col}_min"),
            pl.col(col).skew().alias(f"{col}_skew"),
            pl.col(col).kurtosis().alias(f"{col}_kurt"),
        ])
    
    # Time-series features for key sensor columns
    for sensor in ['acc', 'rot']:
        for axis in ['x', 'y', 'z']:
            col = f"{sensor}_{axis}"
            if col in sensor_cols:
                # Slope calculation using list operations
                agg_exprs.append(
                    pl.col(col).map_elements(
                        lambda values: stats.linregress(range(len(values)), values)[0] if len(values) > 1 else 0.0,
                        return_dtype=pl.Float64
                    ).alias(f"{col}_slope")
                )
                
                # Range (max - min) as a simple trend indicator
                agg_exprs.append(
                    (pl.col(col).max() - pl.col(col).min()).alias(f"{col}_range")
                )
    
    # FFT features for accelerometer data
    for sensor in ['acc_x', 'acc_y', 'acc_z']:
        if sensor in sensor_cols:
            # FFT mean of first 5 components
            agg_exprs.append(
                pl.col(sensor).map_elements(
                    lambda values: np.abs(fft(values))[:5].mean() if len(values) > 4 else 0.0,
                    return_dtype=pl.Float64
                ).alias(f"{sensor}_fft_mean")
            )
            
            # FFT standard deviation of first 5 components
            agg_exprs.append(
                pl.col(sensor).map_elements(
                    lambda values: np.abs(fft(values))[:5].std() if len(values) > 4 else 0.0,
                    return_dtype=pl.Float64
                ).alias(f"{sensor}_fft_std")
            )
    
    for col in behavior_cols:
        agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
    agg_exprs.extend([
        pl.col('sequence_counter').count().alias('sequence_length'),
        pl.col('subject').first().alias('subject'),
    ])
    # No gesture column in test
    features_df = merged.group_by('sequence_id').agg(agg_exprs)
    features_pd = features_df.to_pandas()

    # Prepare features (drop non-feature columns)
    feature_cols = [col for col in features_pd.columns if col not in ['sequence_id', 'subject', 'behavior'] and not col.startswith('behavior_')]
    X = features_pd[feature_cols].fillna(0)

    # 1. Align to pre-selection columns for feature selector using in-memory data
    if TRAINED_PRESELECT_FEATURE_COLUMNS is not None:
        # Robust column guarantee - ensure all expected columns are present
        for col in TRAINED_PRESELECT_FEATURE_COLUMNS:
            if col not in X.columns:
                X[col] = 0
        X = X[TRAINED_PRESELECT_FEATURE_COLUMNS]
    
    # 2. Feature selection (if used)
    if feature_selector is not None:
        X = feature_selector.transform(X)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=feature_selector.get_feature_names_out())
    
    # 3. Align to post-selection columns for model using in-memory data
    if TRAINED_FEATURE_COLUMNS is not None:
        # Robust column guarantee for final model features
        for col in TRAINED_FEATURE_COLUMNS:
            if col not in X.columns:
                X[col] = 0
        X = X[TRAINED_FEATURE_COLUMNS]

    # Predict
    preds = model.predict(X)
    pred_label = label_encoder.inverse_transform(preds)[0]
    print("[API STEP] Finished predict() for a batch of data using in-memory models")
    return pred_label

# API server code
try:
    import kaggle_evaluation.cmi_inference_server
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
            )
        )
except ImportError:
    pass

# ========== END API SERVER CODE ==========


if __name__ == '__main__':
    main() 