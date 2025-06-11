import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import time
import gc
import warnings
import psutil
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
import logging
import os
from joblib import Parallel, delayed
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_header(title: str, width: int = 80):
    """Print a formatted header for better visibility."""
    print(f"\n{'='*width}")
    print(f"🚀 {title}")
    print(f"{'='*width}")

def print_step(step: str, substep: str = None):
    """Print current step with clear formatting."""
    if substep:
        print(f"\n   └── {substep}")
    else:
        print(f"\n📋 {step}")

def print_progress(current: int, total: int, item_name: str = "items", eta: str = None):
    """Print progress with percentage and ETA."""
    percentage = (current / total) * 100 if total > 0 else 0
    eta_str = f" | ETA: {eta}" if eta else ""
    print(f"   🔄 Progress: {current}/{total} {item_name} ({percentage:.1f}%){eta_str}")

def print_result(metric_name: str, value: float, format_str: str = ".6f"):
    """Print a result metric with proper formatting."""
    print(f"   ✅ {metric_name}: {value:{format_str}}")

def print_error(error_msg: str):
    """Print error message with clear formatting."""
    print(f"   ❌ ERROR: {error_msg}")

def print_warning(warning_msg: str):
    """Print warning message with clear formatting."""
    print(f"   ⚠️  WARNING: {warning_msg}")

def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)  # Convert to GB
    except:
        return 0.0

def print_memory_usage(stage: str):
    """Print current memory usage for a specific stage."""
    memory_gb = get_memory_usage()
    print(f"   🧠 Memory usage at {stage}: {memory_gb:.2f} GB")

def cleanup_memory(variables_to_delete: List[str] = None, local_vars: dict = None):
    """
    Comprehensive memory cleanup function.
    
    Args:
        variables_to_delete: List of variable names to delete from local scope
        local_vars: Local variables dictionary (usually locals())
    """
    print_step("Memory cleanup before model training")
    
    # Get initial memory usage
    initial_memory = get_memory_usage()
    print_memory_usage("cleanup start")
    
    # Delete specified variables if provided
    if variables_to_delete and local_vars:
        for var_name in variables_to_delete:
            if var_name in local_vars:
                print(f"   🗑️  Deleting variable: {var_name}")
                del local_vars[var_name]
    
    # Force garbage collection multiple times for thorough cleanup
    print("   🧹 Running garbage collection...")
    collected_objects = 0
    for i in range(3):  # Multiple passes for thorough cleanup
        collected = gc.collect()
        collected_objects += collected
        if collected > 0:
            print(f"     ✅ GC pass {i+1}: collected {collected} objects")
    
    print(f"   📊 Total objects collected: {collected_objects}")
    
    # Clear any remaining unreferenced objects
    gc.collect()
    
    # Get final memory usage
    final_memory = get_memory_usage()
    memory_freed = initial_memory - final_memory
    
    print_memory_usage("cleanup complete")
    if memory_freed > 0:
        print(f"   ✅ Memory freed: {memory_freed:.2f} GB")
    else:
        print(f"   ℹ️  Memory usage stable (difference: {abs(memory_freed):.2f} GB)")
    
    logger.info(f"Memory cleanup completed: {initial_memory:.2f}GB → {final_memory:.2f}GB")

def analyze_dataframe_columns(df: pd.DataFrame, dataset_name: str = "DataFrame"):
    """
    Analyze and report on DataFrame columns for debugging purposes.
    
    Args:
        df: DataFrame to analyze
        dataset_name: Name for reporting
    """
    print(f"\n🔍 Column Analysis for {dataset_name}:")
    print(f"   📊 Shape: {df.shape}")
    print(f"   📋 Total columns: {len(df.columns)}")
    
    # Group columns by type
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    print(f"   🔢 Numeric columns: {len(numeric_cols)}")
    print(f"   📝 Object columns: {len(object_cols)}")
    print(f"   📅 Datetime columns: {len(datetime_cols)}")
    
    if datetime_cols:
        print(f"      📅 Datetime: {datetime_cols}")
    
    # Check for potential ID/timestamp columns
    potential_id_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in ['id', 'timestamp', 'time', 'date', 'datetime', 'ts']):
            potential_id_cols.append(col)
    
    if potential_id_cols:
        print(f"   🆔 Potential ID/timestamp columns: {potential_id_cols}")
    
    # Show first 20 columns
    if len(df.columns) > 20:
        print(f"   📋 First 20 columns: {list(df.columns[:20])}")
        print(f"   📋 ... and {len(df.columns) - 20} more")
    else:
        print(f"   📋 All columns: {list(df.columns)}")
    
    # Check index
    if hasattr(df.index, 'name') and df.index.name:
        print(f"   📅 Index: {df.index.name} (type: {df.index.dtype})")
    
    return {
        'numeric_cols': numeric_cols,
        'object_cols': object_cols, 
        'datetime_cols': datetime_cols,
        'potential_id_cols': potential_id_cols
    }

@dataclass
class DatasetConfig:
    """Configuration for individual dataset files (supports CSV and Parquet formats)."""
    file_path: str             # Path to CSV or Parquet file
    feature_columns: List[str]  # Specific columns to keep from this file
    id_columns: List[str]       # ID/merge columns (timestamp, ID, etc.)
    dataset_name: str          # Human readable name for logging
    is_required: bool = True   # Whether this dataset is required for pipeline to continue

@dataclass
class ModelConfig:
    """Configuration class for model parameters and settings."""
    
    # Dataset configurations
    TRAIN_DATASETS: List[DatasetConfig] = field(default_factory=list)
    TEST_DATASETS: List[DatasetConfig] = field(default_factory=list)
    
    # Column names
    TARGET_COLUMN: str = "label"
    ID_COLUMN: str = "ID"
    TIMESTAMP_COLUMN: str = "timestamp"
    
    # Cross-validation settings
    N_FOLDS: int = 5
    RANDOM_STATE: int = 42
    
    # Time decay settings
    DECAY_FACTOR: float = 0.95
    
    # Model parameters
    XGB_PARAMS: Dict[str, Any] = field(default_factory=dict)
    LGBM_PARAMS: Dict[str, Any] = field(default_factory=dict)
    
    # Feature selection
    SELECTED_FEATURES: List[str] = field(default_factory=list)
    
    # Model ensemble configurations
    MODEL_CONFIGS: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "Full Dataset (100%)", "percent": 1.00, "priority": 1},
        {"name": "Recent Data (75%)", "percent": 0.75, "priority": 2},
        {"name": "Recent Data (50%)", "percent": 0.50, "priority": 3}
    ])
    
    # Output settings
    SUBMISSION_FILENAME: str = "submission_ensemble_XGB_LGB.csv"
    RESULTS_FILENAME: str = "ensemble_results.csv"
    
    # Performance optimization settings
    REDUCE_MEMORY_USAGE: bool = True        # Enable memory usage reduction
    ADD_ENGINEERED_FEATURES: bool = False   # Enable feature engineering
    
    # Multi-GPU settings
    USE_MULTI_GPU: bool = True              # Enable multi-GPU training
    GPU_DEVICES: List[int] = field(default_factory=lambda: [0, 1])  # GPU device IDs
    PARALLEL_FOLD_TRAINING: bool = True     # Train folds in parallel across GPUs
    
    # Ensemble weight configurations
    CUSTOM_ENSEMBLE_WEIGHTS: List[float] = field(default_factory=list)  # Custom weights for final ensemble [XGB_weight, LGB_weight]
    INDIVIDUAL_MODEL_WEIGHTS: Dict[str, float] = field(default_factory=dict)  # Custom weights for individual models
    ENSEMBLE_STRATEGY: str = "learner_level"  # "learner_level", "individual_models", or "performance_based"
    
    def validate_weights(self) -> Tuple[bool, str]:
        """
        Validate ensemble weight configurations.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if self.ENSEMBLE_STRATEGY == "individual_models":
            if not self.INDIVIDUAL_MODEL_WEIGHTS:
                return False, "INDIVIDUAL_MODEL_WEIGHTS must be specified when ENSEMBLE_STRATEGY is 'individual_models'"
            
            # Expected model names based on MODEL_CONFIGS
            expected_models = []
            for algorithm in ['XGB', 'LGB']:
                for model_config in self.MODEL_CONFIGS:
                    model_name = f"{algorithm}_{model_config['name'].replace(' ', '_').replace('(', '').replace(')', '')}"
                    expected_models.append(model_name)
            
            # Check if all expected models have weights
            missing_weights = [model for model in expected_models if model not in self.INDIVIDUAL_MODEL_WEIGHTS]
            if missing_weights:
                return False, f"Missing weights for models: {missing_weights}"
            
            # Check for extra weights
            extra_weights = [model for model in self.INDIVIDUAL_MODEL_WEIGHTS if model not in expected_models]
            if extra_weights:
                return False, f"Unexpected model weights specified: {extra_weights}. Expected models: {expected_models}"
            
            # Check that weights are positive
            negative_weights = [model for model, weight in self.INDIVIDUAL_MODEL_WEIGHTS.items() if weight < 0]
            if negative_weights:
                return False, f"Negative weights not allowed for models: {negative_weights}"
        
        elif self.ENSEMBLE_STRATEGY == "learner_level":
            if self.CUSTOM_ENSEMBLE_WEIGHTS:
                if len(self.CUSTOM_ENSEMBLE_WEIGHTS) != 2:
                    return False, "CUSTOM_ENSEMBLE_WEIGHTS must contain exactly 2 weights [XGB_weight, LGB_weight]"
                
                if any(w < 0 for w in self.CUSTOM_ENSEMBLE_WEIGHTS):
                    return False, "Negative weights not allowed in CUSTOM_ENSEMBLE_WEIGHTS"
        
        elif self.ENSEMBLE_STRATEGY == "performance_based":
            # Performance-based ensemble doesn't require any additional configuration
            # Weights are automatically calculated based on CV scores
            pass
        
        else:
            return False, f"Invalid ENSEMBLE_STRATEGY: {self.ENSEMBLE_STRATEGY}. Must be 'learner_level', 'individual_models', or 'performance_based'"
        
        return True, ""

class WeightCalculator:
    """Calculate time-based sample weights with exponential decay."""
    
    @staticmethod
    def create_time_weights(n_samples: int, decay_factor: float = 0.95) -> np.ndarray:
        """
        Create exponentially decaying weights based on sample position.
        More recent samples (higher indices) get higher weights.
        
        Args:
            n_samples: Number of samples
            decay_factor: Controls the rate of decay (0.95 = 5% decay per time unit)
            
        Returns:
            Array of sample weights normalized to sum to n_samples
        """
        if n_samples == 0:
            return np.array([])
        
        print(f"     ⚖️  Creating time weights for {n_samples:,} samples (decay: {decay_factor})")
        
        positions = np.arange(n_samples)
        # Normalize positions to [0, 1] range
        normalized_positions = positions / max(1, n_samples - 1)
        # Apply exponential weighting
        weights = decay_factor ** (1 - normalized_positions)
        # Normalize weights to sum to n_samples (maintains scale)
        weights = weights * n_samples / weights.sum()
        
        print(f"     ✅ Weight range: [{weights.min():.4f}, {weights.max():.4f}], mean: {weights.mean():.4f}")
        
        return weights

class DataProcessor:
    """Handle data loading, merging, and preprocessing with flexible dataset configurations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        print_step("Data Processor Initialized")
        print(f"   📊 Train datasets configured: {len(self.config.TRAIN_DATASETS)}")
        print(f"   📊 Test datasets configured: {len(self.config.TEST_DATASETS)}")
        print(f"   🎯 Selected features: {len(self.config.SELECTED_FEATURES)}")
        print(f"   🧠 Memory optimization: {self.config.REDUCE_MEMORY_USAGE}")
        print(f"   ⚙️  Feature engineering: {self.config.ADD_ENGINEERED_FEATURES}")
        logger.info(f"DataProcessor initialized with {len(self.config.TRAIN_DATASETS)} train and {len(self.config.TEST_DATASETS)} test datasets")
    
    def reduce_mem_usage(self, dataframe: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Reduce memory usage of dataframe by optimizing ONLY numeric data types.
        COMPLETELY preserves object/string columns (including string timestamps) and datetime columns.
        
        Args:
            dataframe: DataFrame to optimize
            dataset_name: Name for logging purposes
            
        Returns:
            Memory-optimized dataframe with preserved non-numeric columns
        """
        if not self.config.REDUCE_MEMORY_USAGE:
            return dataframe
            
        print(f"     🧠 Reducing memory usage for: {dataset_name}")
        initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
        
        # Identify columns to preserve (anything non-numeric)
        preserved_columns = set()
        numeric_columns = []
        
        # Add configured ID and timestamp columns to preserved
        if self.config.ID_COLUMN:
            preserved_columns.add(self.config.ID_COLUMN)
        if self.config.TIMESTAMP_COLUMN:
            preserved_columns.add(self.config.TIMESTAMP_COLUMN)
        if self.config.TARGET_COLUMN:
            preserved_columns.add(self.config.TARGET_COLUMN)
        
        # Categorize all columns by type
        for col in dataframe.columns:
            col_type = dataframe[col].dtype
            
            # Preserve object/string columns (including string timestamps)
            if col_type == 'object':
                preserved_columns.add(col)
                continue
            
            # Preserve datetime columns
            if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                preserved_columns.add(col)
                continue
                
            # Preserve category and bool columns
            if col_type in ['category', 'bool']:
                preserved_columns.add(col)
                continue
            
            # Only optimize numeric columns (int and float)
            if str(col_type)[:3] in ['int', 'flo']:
                numeric_columns.append(col)
        
        print(f"       🛡️  Preserved columns: {len(preserved_columns)} (object/datetime/category/bool)")
        print(f"       🔢 Numeric columns to optimize: {len(numeric_columns)}")
        
        if preserved_columns:
            # Show types of preserved columns for debugging
            preserved_types = {}
            for col in list(preserved_columns)[:5]:  # Show first 5 for brevity
                if col in dataframe.columns:
                    preserved_types[col] = str(dataframe[col].dtype)
            if preserved_types:
                print(f"       📝 Sample preserved column types: {preserved_types}")
        
        optimized_count = 0
        for col in numeric_columns:
            col_type = dataframe[col].dtype
            
            try:
                c_min = dataframe[col].min()
                c_max = dataframe[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        dataframe[col] = dataframe[col].astype(np.int8)
                        optimized_count += 1
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        dataframe[col] = dataframe[col].astype(np.int16)
                        optimized_count += 1
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        dataframe[col] = dataframe[col].astype(np.int32)
                        optimized_count += 1
                elif str(col_type)[:3] == 'flo':
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        dataframe[col] = dataframe[col].astype(np.float16)
                        optimized_count += 1
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        dataframe[col] = dataframe[col].astype(np.float32)
                        optimized_count += 1
            except Exception as e:
                print(f"       ⚠️  Could not optimize numeric column {col}: {str(e)}")
                continue

        final_mem_usage = dataframe.memory_usage().sum() / 1024**2
        reduction_pct = 100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage
        
        print(f"       📈 Memory usage before: {initial_mem_usage:.2f} MB")
        print(f"       📉 Memory usage after: {final_mem_usage:.2f} MB")
        print(f"       ✅ Optimized {optimized_count}/{len(numeric_columns)} numeric columns")
        print(f"       🛡️  Preserved {len(preserved_columns)} non-numeric columns unchanged")
        print(f"       📊 Overall memory reduction: {reduction_pct:.1f}%")
        
        logger.info(f"Memory optimization for {dataset_name}: {initial_mem_usage:.2f}MB → {final_mem_usage:.2f}MB ({reduction_pct:.1f}% reduction), preserved {len(preserved_columns)} columns")
        
        return dataframe
    
    def add_features(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Add engineered features based on market microstructure data.
        
        Args:
            df: Input dataframe with basic market data
            dataset_name: Name for logging purposes
            
        Returns:
            Dataframe with additional engineered features
        """
        if not self.config.ADD_ENGINEERED_FEATURES:
            return df
            
        print(f"     ⚙️  Engineering features for: {dataset_name}")
        
        # Check if required columns exist
        required_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print_warning(f"Missing columns for feature engineering: {missing_cols}. Skipping feature engineering.")
            return df
        
        data = df.copy()
        initial_features = len(data.columns)
        
        # Market microstructure features
        data['bid_ask_spread_proxy'] = data['ask_qty'] - data['bid_qty']
        data['total_liquidity'] = data['bid_qty'] + data['ask_qty']
        data['trade_imbalance'] = data['buy_qty'] - data['sell_qty']
        data['total_trades'] = data['buy_qty'] + data['sell_qty']
        
        # Volume-based features
        data['volume_per_trade'] = data['volume'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
        data['buy_volume_ratio'] = data['buy_qty'] / (data['volume'] + 1e-8)
        data['sell_volume_ratio'] = data['sell_qty'] / (data['volume'] + 1e-8)
        
        # Pressure and imbalance features
        data['buying_pressure'] = data['buy_qty'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
        data['selling_pressure'] = data['sell_qty'] / (data['buy_qty'] + data['sell_qty'] + 1e-8)
        
        data['order_imbalance'] = (data['bid_qty'] - data['ask_qty']) / (data['bid_qty'] + data['ask_qty'] + 1e-8)
        data['order_imbalance_abs'] = np.abs(data['order_imbalance'])
        data['bid_liquidity_ratio'] = data['bid_qty'] / (data['volume'] + 1e-8)
        data['ask_liquidity_ratio'] = data['ask_qty'] / (data['volume'] + 1e-8)
        data['depth_imbalance'] = data['total_trades'] - data['volume']
        
        # Ratio features
        data['buy_sell_ratio'] = data['buy_qty'] / (data['sell_qty'] + 1e-8)
        data['bid_ask_ratio'] = data['bid_qty'] / (data['ask_qty'] + 1e-8)
        data['volume_liquidity_ratio'] = data['volume'] / (data['bid_qty'] + data['ask_qty'] + 1e-8)

        # Product features
        data['buy_volume_product'] = data['buy_qty'] * data['volume']
        data['sell_volume_product'] = data['sell_qty'] * data['volume']
        data['bid_ask_product'] = data['bid_qty'] * data['ask_qty']
        
        # Competition and activity features
        data['market_competition'] = (data['buy_qty'] * data['sell_qty']) / ((data['buy_qty'] + data['sell_qty']) + 1e-8)
        data['liquidity_competition'] = (data['bid_qty'] * data['ask_qty']) / ((data['bid_qty'] + data['ask_qty']) + 1e-8)
        
        total_activity = data['buy_qty'] + data['sell_qty'] + data['bid_qty'] + data['ask_qty']
        data['market_activity'] = total_activity
        data['activity_concentration'] = data['volume'] / (total_activity + 1e-8)
        
        # Advanced microstructure features
        data['info_arrival_rate'] = (data['buy_qty'] + data['sell_qty']) / (data['volume'] + 1e-8)
        data['market_making_intensity'] = (data['bid_qty'] + data['ask_qty']) / (data['buy_qty'] + data['sell_qty'] + 1e-8)
        data['effective_spread_proxy'] = np.abs(data['buy_qty'] - data['sell_qty']) / (data['volume'] + 1e-8)
        
        # Exponentially weighted moving average of order flow imbalance
        lambda_decay = 0.95
        ofi = data['buy_qty'] - data['sell_qty']
        data['order_flow_imbalance_ewm'] = ofi.ewm(alpha=1-lambda_decay).mean()

        # Clean up infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        final_features = len(data.columns)
        new_features = final_features - initial_features
        
        print(f"       ✅ Added {new_features} engineered features")
        print(f"       📊 Total features: {initial_features} → {final_features}")
        
        logger.info(f"Feature engineering for {dataset_name}: added {new_features} features ({initial_features} → {final_features})")
        
        return data
    
    def load_and_merge_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and merge multiple datasets efficiently based on configuration.
        
        Returns:
            Tuple of (merged_train_df, merged_test_df)
        """
        print_header("FLEXIBLE DATA LOADING AND MERGING")
        start_time = time.time()
        
        try:
            # Load and process train datasets
            print_step("Loading and merging train datasets")
            merged_train = self._load_and_merge_dataset_group(
                self.config.TRAIN_DATASETS, "TRAIN"
            )
            
            # Load and process test datasets  
            print_step("Loading and merging test datasets")
            merged_test = self._load_and_merge_dataset_group(
                self.config.TEST_DATASETS, "TEST"
            )
            
            # Apply feature selection
            print_step("Applying feature selection")
            merged_train, merged_test = self._apply_feature_selection(merged_train, merged_test)
            
            # Final memory optimization after merging and feature selection
            print_step("Final memory optimization")
            merged_train = self.reduce_mem_usage(merged_train, "Final Merged Train")
            merged_test = self.reduce_mem_usage(merged_test, "Final Merged Test")
            
            # Validate final datasets
            print_step("Validating final merged datasets")
            self._validate_final_datasets(merged_train, merged_test)
            
            # Display final dataset information
            print_step("Final dataset information")
            print(f"     📊 Train shape: {merged_train.shape}")
            print(f"     📊 Test shape: {merged_test.shape}")
            print(f"     📊 Train columns: {list(merged_train.columns)}")
            print(f"     📊 Test columns: {list(merged_test.columns)}")
            print(f"     🎯 Target column '{self.config.TARGET_COLUMN}' present in train: {self.config.TARGET_COLUMN in merged_train.columns}")
            
            # Comprehensive memory cleanup after data loading
            print_step("Post-processing memory cleanup")
            print_memory_usage("before data cleanup")
            
            # Clean up any temporary variables from data loading
            variables_to_cleanup = [
                'loaded_datasets', 'dataset_names', 'dataset_info', 'dataset_df',
                'raw_df', 'processed_df', 'before_shape', 'after_shape'
            ]
            
            # Multiple garbage collection passes
            collected_total = 0
            for i in range(3):
                collected = gc.collect()
                collected_total += collected
                if collected > 0:
                    print(f"     🗑️  GC pass {i+1}: {collected} objects collected")
            
            print(f"   ✅ Total cleanup: {collected_total} objects collected")
            print_memory_usage("after data cleanup")
            
            total_time = time.time() - start_time
            print_result("Total loading time", total_time, ".1f")
            print("✨ Data loading and merging completed successfully!")
            logger.info(f"Data loading completed in {total_time:.1f}s - Train: {merged_train.shape}, Test: {merged_test.shape}")
            
            return merged_train, merged_test
            
        except Exception as e:
            print_error(f"Data loading failed: {str(e)}")
            logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def _load_and_merge_dataset_group(self, dataset_configs: List[DatasetConfig], 
                                    group_name: str) -> pd.DataFrame:
        """
        Load and merge a group of datasets (either train or test).
        
        Args:
            dataset_configs: List of DatasetConfig objects
            group_name: Name for logging (e.g., "TRAIN", "TEST")
            
        Returns:
            Merged dataframe
        """
        print(f"\n   🚀 Processing {group_name} dataset group ({len(dataset_configs)} files)")
        
        loaded_datasets = {}
        
        # Step 1: Load individual datasets
        total_files = len(dataset_configs)
        for i, dataset_config in enumerate(dataset_configs, 1):
            print_progress(i, total_files, "files")
            print(f"     📂 Loading {dataset_config.dataset_name}...")
            print(f"       📍 Path: {dataset_config.file_path}")
            
            try:
                load_start = time.time()
                
                # Check if file exists
                if not os.path.exists(dataset_config.file_path):
                    if dataset_config.is_required:
                        raise FileNotFoundError(f"Required file not found: {dataset_config.file_path}")
                    else:
                        print_warning(f"Optional file not found, skipping: {dataset_config.file_path}")
                        continue
                
                # Load the dataset (supports both CSV and Parquet)
                file_extension = dataset_config.file_path.lower().split('.')[-1]
                try:
                    if file_extension == 'parquet':
                        raw_df = pd.read_parquet(dataset_config.file_path)
                        file_type = "Parquet"
                    elif file_extension in ['csv', 'tsv']:
                        # For CSV, try to automatically parse datetime columns
                        raw_df = pd.read_csv(dataset_config.file_path)
                        file_type = "CSV"
                        
                        # Try to detect and parse datetime columns
                        datetime_cols_detected = []
                        for col in raw_df.columns:
                            # Check if column name suggests it's a datetime
                            col_lower = col.lower()
                            if any(pattern in col_lower for pattern in ['timestamp', 'time', 'date', 'datetime', 'ts']):
                                try:
                                    raw_df[col] = pd.to_datetime(raw_df[col])
                                    datetime_cols_detected.append(col)
                                except:
                                    pass  # If conversion fails, keep original type
                        
                        if datetime_cols_detected:
                            print(f"       📅 Auto-detected datetime columns: {datetime_cols_detected}")
                            
                    else:
                        # Try CSV as default
                        print_warning(f"Unknown file extension '.{file_extension}', trying CSV format")
                        raw_df = pd.read_csv(dataset_config.file_path)
                        file_type = "CSV (default)"
                except Exception as file_error:
                    if file_extension == 'parquet':
                        print_warning(f"Failed to read as Parquet, trying CSV: {str(file_error)}")
                        raw_df = pd.read_csv(dataset_config.file_path)
                        file_type = "CSV (fallback)"
                    else:
                        raise file_error
                
                load_time = time.time() - load_start
                
                print(f"       ✅ Raw shape: {raw_df.shape} loaded in {load_time:.1f}s ({file_type})")
                
                # DETAILED COLUMN AND INDEX ANALYSIS
                print(f"       📋 Regular columns ({len(raw_df.columns)}): {list(raw_df.columns)[:10]}...")
                
                # Check if any required ID column is in the index
                if hasattr(raw_df.index, 'name') and raw_df.index.name:
                    print(f"       📅 Index column: {raw_df.index.name} (type: {raw_df.index.dtype})")
                    
                    # Check if the index name matches any of our required ID columns (regardless of data type)
                    index_name = raw_df.index.name
                    if index_name in dataset_config.id_columns:
                        print(f"       🔄 Index '{index_name}' is a required ID column - resetting index")
                        raw_df = raw_df.reset_index()
                        print(f"       ✅ After reset_index: shape={raw_df.shape}, columns={list(raw_df.columns)[:10]}...")
                    else:
                        print(f"       ℹ️  Index '{index_name}' is not in required ID columns: {dataset_config.id_columns}")
                
                print(f"       📊 Final columns after index handling: {len(raw_df.columns)}")
                print(f"       📋 All columns: {list(raw_df.columns)}")
                
                # Check for datetime columns in regular columns
                datetime_cols = [col for col in raw_df.columns if pd.api.types.is_datetime64_any_dtype(raw_df[col])]
                if datetime_cols:
                    print(f"       📅 Datetime columns found: {datetime_cols}")
                
                # Verify required ID columns are now present
                missing_id_cols_check = [col for col in dataset_config.id_columns if col not in raw_df.columns]
                if missing_id_cols_check:
                    print(f"       ⚠️  Still missing ID columns after index reset: {missing_id_cols_check}")
                else:
                    print(f"       ✅ All required ID columns found: {dataset_config.id_columns}")
                
                # Analyze columns for debugging if there might be issues
                if not all(col in raw_df.columns for col in dataset_config.id_columns):
                    analyze_dataframe_columns(raw_df, f"{dataset_config.dataset_name} (Raw)")
                
                # Process columns according to configuration
                processed_df = self._process_dataset_columns(raw_df, dataset_config)
                
                # Apply memory optimization
                processed_df = self.reduce_mem_usage(processed_df, dataset_config.dataset_name)
                
                # Apply feature engineering if enabled
                processed_df = self.add_features(processed_df, dataset_config.dataset_name)
                
                loaded_datasets[dataset_config.dataset_name] = {
                    'dataframe': processed_df,
                    'config': dataset_config
                }
                
                print(f"       ✅ Final shape: {processed_df.shape}")
                print(f"       📊 Final columns: {list(processed_df.columns)[:10]}..." if len(processed_df.columns) > 10 else f"       📊 Final columns: {list(processed_df.columns)}")
                
            except Exception as e:
                if dataset_config.is_required:
                    print_error(f"Failed to load required dataset '{dataset_config.dataset_name}': {str(e)}")
                    raise
                else:
                    print_warning(f"Failed to load optional dataset '{dataset_config.dataset_name}': {str(e)}")
                    continue
        
        if not loaded_datasets:
            raise ValueError(f"No datasets successfully loaded for {group_name} group")
        
        # Step 2: Merge datasets
        print(f"\n   🔗 Merging {len(loaded_datasets)} {group_name} datasets...")
        
        # Start with the first dataset
        dataset_names = list(loaded_datasets.keys())
        merged_df = loaded_datasets[dataset_names[0]]['dataframe'].copy()
        merge_info = [f"Base: {dataset_names[0]} ({merged_df.shape})"]
        
        # Merge remaining datasets
        for dataset_name in dataset_names[1:]:
            dataset_info = loaded_datasets[dataset_name]
            dataset_df = dataset_info['dataframe']
            merge_columns = dataset_info['config'].id_columns
            
            print(f"     🔗 Merging {dataset_name} on {merge_columns}...")
            
            # Validate merge columns exist
            missing_cols_left = [col for col in merge_columns if col not in merged_df.columns]
            missing_cols_right = [col for col in merge_columns if col not in dataset_df.columns]
            
            if missing_cols_left:
                raise KeyError(f"Merge columns {missing_cols_left} not found in merged dataset")
            if missing_cols_right:
                raise KeyError(f"Merge columns {missing_cols_right} not found in {dataset_name}")
            
            # Perform merge
            before_shape = merged_df.shape
            merged_df = pd.merge(
                merged_df, 
                dataset_df, 
                on=merge_columns, 
                how='inner'
            )
            after_shape = merged_df.shape
            
            merge_info.append(f"+ {dataset_name}: {before_shape} → {after_shape}")
            print(f"       ✅ Merge result: {before_shape} → {after_shape}")
        
        print(f"\n   📊 {group_name} merge summary:")
        for info in merge_info:
            print(f"     {info}")
        
        print(f"   ✅ Final {group_name} dataset: {merged_df.shape}")
        logger.info(f"{group_name} datasets merged successfully: {merged_df.shape}")
        
        return merged_df
    
    def _process_dataset_columns(self, df: pd.DataFrame, 
                               config: DatasetConfig) -> pd.DataFrame:
        """
        Process dataset columns according to configuration.
        ALWAYS preserves ID columns regardless of their data type.
        
        Args:
            df: Raw dataframe
            config: Dataset configuration
            
        Returns:
            Processed dataframe with only desired columns
        """
        # Validate that ID columns exist
        missing_id_cols = [col for col in config.id_columns if col not in df.columns]
        if missing_id_cols:
            print(f"       ❌ Missing ID columns: {missing_id_cols}")
            print(f"       📋 Available columns: {list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''}")
            print(f"       📊 Total columns: {len(df.columns)}")
            
            # Check for similar column names
            similar_cols = []
            for missing_col in missing_id_cols:
                for available_col in df.columns:
                    if missing_col.lower() in available_col.lower() or available_col.lower() in missing_col.lower():
                        similar_cols.append((missing_col, available_col))
            
            if similar_cols:
                print(f"       💡 Similar columns found: {similar_cols}")
                print(f"       💡 Consider updating your DatasetConfig id_columns")
            
            # Check for datetime columns that might have been parsed
            datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
            if datetime_cols:
                print(f"       📅 Available datetime columns: {datetime_cols}")
            
            # Check for object columns that might be string timestamps
            object_cols = [col for col in df.columns if df[col].dtype == 'object']
            potential_timestamp_cols = [col for col in object_cols if any(pattern in col.lower() 
                                      for pattern in ['timestamp', 'time', 'date', 'datetime', 'ts'])]
            if potential_timestamp_cols:
                print(f"       🕐 Potential string timestamp columns: {potential_timestamp_cols}")
            
            raise KeyError(f"ID columns {missing_id_cols} not found in {config.dataset_name}. "
                          f"Available columns: {list(df.columns)}")
        
        # ALWAYS preserve ID columns regardless of configuration
        columns_to_keep = set(config.id_columns)
        
        # Initialize existing_feature_cols to prevent UnboundLocalError
        existing_feature_cols = []
        
        # Add feature columns if specified
        if config.feature_columns:
            # Validate that specified feature columns exist
            missing_feature_cols = [col for col in config.feature_columns if col not in df.columns]
            if missing_feature_cols:
                print_warning(f"Feature columns {missing_feature_cols} not found in {config.dataset_name}")
            
            # Keep only existing feature columns
            existing_feature_cols = [col for col in config.feature_columns if col in df.columns]
            columns_to_keep.update(existing_feature_cols)
            
            print(f"       🎯 Feature columns specified: {len(config.feature_columns)}")
            print(f"       ✅ Feature columns found: {len(existing_feature_cols)}")
        else:
            # Keep all columns (ID columns already included)
            columns_to_keep.update(df.columns)
        
        # Convert to list and select columns
        columns_to_keep = list(columns_to_keep)
        processed_df = df[columns_to_keep].copy()
        
        print(f"       📋 Kept {len(columns_to_keep)} columns from {len(df.columns)} available")
        print(f"       🔗 ID columns preserved: {config.id_columns}")
        
        # Show data types of ID columns for debugging
        for id_col in config.id_columns:
            if id_col in processed_df.columns:
                print(f"       📝 ID column '{id_col}' type: {processed_df[id_col].dtype}")
        
        return processed_df
    
    def _apply_feature_selection(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply feature selection based on SELECTED_FEATURES."""
        if not self.config.SELECTED_FEATURES:
            print("     ℹ️  No feature selection specified, keeping all features")
            return train_df, test_df
        
        print(f"     🎯 Applying feature selection: {len(self.config.SELECTED_FEATURES)} features")
        
        # Identify which columns to keep (selected features + required columns)
        required_cols = {self.config.TARGET_COLUMN, self.config.ID_COLUMN, self.config.TIMESTAMP_COLUMN}
        required_cols = {col for col in required_cols if col}  # Remove None values
        
        # For train data: keep target + ID/timestamp + selected features
        train_cols_to_keep = []
        for col in required_cols:
            if col in train_df.columns:
                train_cols_to_keep.append(col)
        
        # Add selected features that exist in train data
        for feature in self.config.SELECTED_FEATURES:
            if feature in train_df.columns and feature not in train_cols_to_keep:
                train_cols_to_keep.append(feature)
            elif feature not in train_df.columns:
                print_warning(f"Selected feature '{feature}' not found in train data")
        
        # For test data: keep ID + selected features (no target)
        test_cols_to_keep = []
        for col in [self.config.ID_COLUMN, self.config.TIMESTAMP_COLUMN]:
            if col and col in test_df.columns:
                test_cols_to_keep.append(col)
        
        # Add selected features that exist in test data
        for feature in self.config.SELECTED_FEATURES:
            if feature in test_df.columns and feature not in test_cols_to_keep:
                test_cols_to_keep.append(feature)
            elif feature not in test_df.columns:
                print_warning(f"Selected feature '{feature}' not found in test data")
        
        # Apply selection
        train_selected = train_df[train_cols_to_keep].copy()
        test_selected = test_df[test_cols_to_keep].copy()
        
        print(f"     ✅ Train features: {train_df.shape[1]} → {train_selected.shape[1]} columns")
        print(f"     ✅ Test features: {test_df.shape[1]} → {test_selected.shape[1]} columns")
        
        return train_selected, test_selected
    
    def _validate_final_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Validate that final merged datasets have required columns."""
        
        # Check for target column in train data
        if self.config.TARGET_COLUMN and self.config.TARGET_COLUMN not in train_df.columns:
            available_cols = list(train_df.columns)
            raise KeyError(f"Target column '{self.config.TARGET_COLUMN}' not found in merged train data. "
                          f"Available columns: {available_cols}")
        
        # Check that test data has ID column if specified
        if self.config.ID_COLUMN and self.config.ID_COLUMN not in test_df.columns:
            available_cols = list(test_df.columns)
            raise KeyError(f"ID column '{self.config.ID_COLUMN}' not found in test data. "
                          f"Available columns: {available_cols}")
        
        print(f"     ✅ All required columns validated")
        if self.config.TARGET_COLUMN:
            print(f"       🎯 Target column: {self.config.TARGET_COLUMN}")
        if self.config.ID_COLUMN:
            print(f"       🆔 ID column: {self.config.ID_COLUMN}")

class ModelTrainer:
    """Handle individual model training with different data subsets and algorithms."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.weight_calculator = WeightCalculator()
        print_step("Model Trainer Initialized")
        print(f"   🎯 Number of model configurations: {len(self.config.MODEL_CONFIGS)}")
        print(f"   📊 Cross-validation folds: {self.config.N_FOLDS}")
        print(f"   🔥 Multi-GPU training: {self.config.USE_MULTI_GPU}")
        if self.config.USE_MULTI_GPU:
            print(f"   🎮 GPU devices: {self.config.GPU_DEVICES}")
            print(f"   ⚡ Parallel fold training: {self.config.PARALLEL_FOLD_TRAINING}")
        logger.info(f"ModelTrainer initialized with {len(self.config.MODEL_CONFIGS)} model configs")
    
    def _get_gpu_device_for_fold(self, fold_idx: int) -> int:
        """Get GPU device ID for a specific fold."""
        if not self.config.USE_MULTI_GPU:
            return 0
        return self.config.GPU_DEVICES[fold_idx % len(self.config.GPU_DEVICES)]
    
    def _prepare_gpu_params(self, base_params: Dict[str, Any], gpu_device: int) -> Dict[str, Any]:
        """Prepare GPU-specific parameters for XGBoost/LightGBM."""
        params = base_params.copy()
        
        if 'tree_method' in params:  # XGBoost
            if self.config.USE_MULTI_GPU:
                params['device'] = f'cuda:{gpu_device}'
                params['tree_method'] = 'gpu_hist'
            else:
                params['tree_method'] = 'hist'
        
        if 'device' in params and params['device'] == 'gpu':  # LightGBM
            if self.config.USE_MULTI_GPU:
                params['device'] = f'gpu'
                params['gpu_device_id'] = gpu_device
            else:
                params['device'] = 'cpu'
        
        return params
    
    def _train_single_fold(self, fold_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Train a single fold on a specific GPU device.
        
        Args:
            fold_data: Dictionary containing fold training data and parameters
            
        Returns:
            Tuple of (fold_oof_predictions, fold_test_predictions, fold_score)
        """
        fold_idx = fold_data['fold_idx']
        X_fold_train = fold_data['X_fold_train']
        y_fold_train = fold_data['y_fold_train']
        X_fold_valid = fold_data['X_fold_valid']
        y_fold_valid = fold_data['y_fold_valid']
        X_test = fold_data['X_test']
        fold_weights = fold_data['fold_weights']
        algorithm = fold_data['algorithm']
        model_name = fold_data['model_name']
        gpu_device = fold_data['gpu_device']
        
        print(f"         🎮 Fold {fold_idx + 1} training on GPU {gpu_device}")
        
        # Prepare GPU-specific parameters
        if algorithm == 'xgb':
            params = self._prepare_gpu_params(self.config.XGB_PARAMS, gpu_device)
            model = XGBRegressor(**params)
        else:  # lgb
            params = self._prepare_gpu_params(self.config.LGBM_PARAMS, gpu_device)
            model = LGBMRegressor(**params)
        
        # Train model
        try:
            if algorithm == 'xgb':
                model.fit(
                    X_fold_train, y_fold_train,
                    sample_weight=fold_weights,
                    eval_set=[(X_fold_valid, y_fold_valid)],
                    early_stopping_rounds=25,
                    verbose=False
                )
            else:  # lgb
                model.fit(
                    X_fold_train, y_fold_train,
                    sample_weight=fold_weights,
                    eval_set=[(X_fold_valid, y_fold_valid)],
                    callbacks=[lgb.early_stopping(25), lgb.log_evaluation(0)]
                )
            
            # Make predictions
            valid_preds = model.predict(X_fold_valid)
            test_preds = model.predict(X_test)
            
            # Calculate fold score
            fold_score = pearsonr(y_fold_valid, valid_preds)[0]
            
            print(f"         ✅ Fold {fold_idx + 1} completed on GPU {gpu_device}: {fold_score:.6f}")
            
            # Quick cleanup after fold completion
            del model
            gc.collect()
            
            return valid_preds, test_preds, fold_score
            
        except Exception as e:
            print(f"         ❌ Error in fold {fold_idx + 1} on GPU {gpu_device}: {str(e)}")
            raise
    
    def train_model_ensemble(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ensemble of models with different data subsets and algorithms.
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            
        Returns:
            Dictionary containing OOF predictions, test predictions, and scores
        """
        print_header("MODEL ENSEMBLE TRAINING")
        start_time = time.time()
        
        # Prepare features and target
        feature_cols = [col for col in train_df.columns 
                       if col not in {self.config.TARGET_COLUMN, self.config.ID_COLUMN, self.config.TIMESTAMP_COLUMN}]
        
        print_step("Feature preparation")
        print(f"   🎯 Feature columns: {len(feature_cols)}")
        print(f"   📋 Features: {feature_cols}")
        
        X_train = train_df[feature_cols].copy()
        y_train = train_df[self.config.TARGET_COLUMN].copy()
        X_test = test_df[feature_cols].copy()
        
        print(f"   📊 Training data shape: {X_train.shape}")
        print(f"   📊 Test data shape: {X_test.shape}")
        print(f"   📊 Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        
        # COMPREHENSIVE MEMORY CLEANUP BEFORE MODEL TRAINING
        print_step("Pre-training memory cleanup")
        print_memory_usage("before training cleanup")
        
        # Clear temporary variables from feature preparation
        variables_to_cleanup = [
            'feature_cols', 'train_df', 'test_df'
        ]
        
        # Delete original dataframes to free memory (we have copies)
        del train_df, test_df
        
        # Force multiple garbage collection passes
        print("   🧹 Performing thorough memory cleanup...")
        collected_total = 0
        for i in range(4):  # Extra passes before training
            collected = gc.collect()
            collected_total += collected
            if collected > 0:
                print(f"     🗑️  GC pass {i+1}: {collected} objects collected")
        
        print(f"   ✅ Pre-training cleanup: {collected_total} objects collected")
        print_memory_usage("after training cleanup")
        print("   🚀 Memory optimized for model training!")
        
        # Initialize cross-validation (matching reference implementation)
        kf = KFold(n_splits=self.config.N_FOLDS, shuffle=False)
        
        # Initialize prediction storage
        model_results = {}
        
        # Train XGBoost models with different data subsets
        print_step("Training XGBoost models")
        print_memory_usage("XGBoost training start")
        full_dataset_oof_xgb = None
        
        for model_config in self.config.MODEL_CONFIGS:
            model_name = f"XGB_{model_config['name'].replace(' ', '_').replace('(', '').replace(')', '')}"
            print_step(f"Training {model_name}")
            
            oof_preds, test_preds, score = self._train_single_model(
                X_train, y_train, X_test, kf, 
                model_config['percent'], 'xgb', model_name
            )
            
            # Store full dataset OOF for filling subset models
            if model_config['percent'] == 1.0:
                full_dataset_oof_xgb = oof_preds.copy()
            
            model_results[model_name] = {
                'oof_preds': oof_preds,
                'test_preds': test_preds,
                'score': score,
                'algorithm': 'XGBoost',
                'data_percent': model_config['percent']
            }
            
            # Quick memory cleanup after each model
            gc.collect()
        
        # Train LightGBM models with different data subsets
        print_step("Training LightGBM models")
        print_memory_usage("LightGBM training start")
        full_dataset_oof_lgb = None
        
        for model_config in self.config.MODEL_CONFIGS:
            model_name = f"LGB_{model_config['name'].replace(' ', '_').replace('(', '').replace(')', '')}"
            print_step(f"Training {model_name}")
            
            oof_preds, test_preds, score = self._train_single_model(
                X_train, y_train, X_test, kf, 
                model_config['percent'], 'lgb', model_name
            )
            
            # Store full dataset OOF for filling subset models
            if model_config['percent'] == 1.0:
                full_dataset_oof_lgb = oof_preds.copy()
            
            model_results[model_name] = {
                'oof_preds': oof_preds,
                'test_preds': test_preds,
                'score': score,
                'algorithm': 'LightGBM',
                'data_percent': model_config['percent']
            }
            
            # Quick memory cleanup after each model
            gc.collect()
        
        # Fill missing OOF predictions for subset models using full dataset predictions
        print_step("Filling missing OOF predictions for subset models")
        for model_name, results in model_results.items():
            if results['data_percent'] < 1.0:
                cutoff_idx = int(len(X_train) * (1 - results['data_percent']))
                if 'XGB' in model_name and full_dataset_oof_xgb is not None:
                    # Fill samples before cutoff with full dataset XGB predictions
                    mask = results['oof_preds'][:cutoff_idx] == 0
                    results['oof_preds'][:cutoff_idx][mask] = full_dataset_oof_xgb[:cutoff_idx][mask]
                elif 'LGB' in model_name and full_dataset_oof_lgb is not None:
                    # Fill samples before cutoff with full dataset LGB predictions
                    mask = results['oof_preds'][:cutoff_idx] == 0
                    results['oof_preds'][:cutoff_idx][mask] = full_dataset_oof_lgb[:cutoff_idx][mask]
        
        # Add target for ensemble creation
        model_results['y_true'] = y_train
        
        # Final memory cleanup and reporting
        print_step("Final training memory cleanup")
        collected = gc.collect()
        if collected > 0:
            print(f"   🗑️  Final cleanup: {collected} objects collected")
        print_memory_usage("training complete")
        
        total_time = time.time() - start_time
        print_result("Total training time", total_time, ".1f")
        print("✨ Model ensemble training completed successfully!")
        logger.info(f"Model ensemble training completed in {total_time:.1f}s with {len(model_results)-1} models")
        
        return model_results
    
    def _train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                          kf: KFold, data_percent: float, algorithm: str, model_name: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Train a single model with specified data percentage and algorithm.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            kf: KFold object for cross-validation
            data_percent: Percentage of recent data to use
            algorithm: 'xgb' or 'lgb'
            model_name: Name for logging
            
        Returns:
            Tuple of (oof_predictions, test_predictions, cv_score)
        """
        print(f"     🎯 Model: {model_name} ({algorithm.upper()}) - {data_percent*100:.0f}% recent data")
        
        # Calculate data cutoff
        cutoff_idx = int(len(X_train) * (1 - data_percent)) if data_percent < 1.0 else 0
        
        if cutoff_idx > 0:
            print(f"       📊 Using samples {cutoff_idx} to {len(X_train)-1} ({len(X_train) - cutoff_idx:,} samples)")
            train_subset = X_train.iloc[cutoff_idx:].reset_index(drop=True)
            target_subset = y_train.iloc[cutoff_idx:].reset_index(drop=True)
        else:
            print(f"       📊 Using all {len(X_train):,} samples")
            train_subset = X_train.copy()
            target_subset = y_train.copy()
        
        # Create time weights for the subset
        sample_weights = self.weight_calculator.create_time_weights(
            len(train_subset), self.config.DECAY_FACTOR
        )
        
        # Initialize prediction arrays
        oof_preds = np.zeros(len(y_train))
        test_preds = np.zeros(len(X_test))
        cv_scores = []
        
        # Prepare fold data for parallel training
        fold_data_list = []
        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_subset)):
            gpu_device = self._get_gpu_device_for_fold(fold)
            
            fold_data = {
                'fold_idx': fold,
                'train_idx': train_idx,
                'valid_idx': valid_idx,
                'X_fold_train': train_subset.iloc[train_idx],
                'y_fold_train': target_subset.iloc[train_idx],
                'X_fold_valid': train_subset.iloc[valid_idx],
                'y_fold_valid': target_subset.iloc[valid_idx],
                'X_test': X_test,
                'fold_weights': sample_weights[train_idx],
                'algorithm': algorithm,
                'model_name': model_name,
                'gpu_device': gpu_device
            }
            fold_data_list.append(fold_data)
        
        # Train folds in parallel if enabled, otherwise sequentially
        if self.config.PARALLEL_FOLD_TRAINING and self.config.USE_MULTI_GPU and len(self.config.GPU_DEVICES) > 1:
            print(f"       ⚡ Training {self.config.N_FOLDS} folds in parallel across {len(self.config.GPU_DEVICES)} GPUs")
            
            # Use ThreadPoolExecutor for GPU parallelism (better for GPU workloads)
            with ThreadPoolExecutor(max_workers=len(self.config.GPU_DEVICES)) as executor:
                # Submit all fold training jobs
                future_to_fold = {executor.submit(self._train_single_fold, fold_data): fold_data['fold_idx'] 
                                for fold_data in fold_data_list}
                
                # Collect results as they complete
                fold_results = {}
                for future in as_completed(future_to_fold):
                    fold_idx = future_to_fold[future]
                    try:
                        valid_preds, fold_test_preds, fold_score = future.result()
                        fold_results[fold_idx] = {
                            'valid_preds': valid_preds,
                            'test_preds': fold_test_preds,
                            'score': fold_score
                        }
                        cv_scores.append(fold_score)
                        print(f"         ✅ Fold {fold_idx + 1} completed: {fold_score:.6f}")
                    except Exception as e:
                        print(f"         ❌ Fold {fold_idx + 1} failed: {str(e)}")
                        raise
    
            # Process results in fold order
            for fold_idx in range(self.config.N_FOLDS):
                result = fold_results[fold_idx]
                valid_preds = result['valid_preds']
                fold_test_preds = result['test_preds']
                
                # Get the original fold validation indices from fold_data_list
                original_valid_idx = fold_data_list[fold_idx]['valid_idx']
                
                # Store out-of-fold predictions with proper index mapping
                if cutoff_idx > 0:
                    # Map subset validation indices back to full dataset indices
                    full_dataset_valid_idx = original_valid_idx + cutoff_idx
                    oof_preds[full_dataset_valid_idx] = valid_preds
                else:
                    oof_preds[original_valid_idx] = valid_preds
                
                # Accumulate test predictions
                test_preds += fold_test_preds
        else:
            # Sequential training (fallback)
            print(f"       🔄 Training {self.config.N_FOLDS} folds sequentially")
            for fold_data in fold_data_list:
                fold_idx = fold_data['fold_idx']
                print(f"       🔄 Fold {fold_idx + 1}/{self.config.N_FOLDS}")
                
                valid_preds, fold_test_preds, fold_score = self._train_single_fold(fold_data)
                
                # Store out-of-fold predictions
                valid_idx = fold_data['valid_idx']
                if cutoff_idx > 0:
                    # Map subset validation indices back to full dataset indices
                    full_dataset_valid_idx = valid_idx + cutoff_idx
                    oof_preds[full_dataset_valid_idx] = valid_preds
                else:
                    oof_preds[valid_idx] = valid_preds
                
                # Accumulate test predictions
                test_preds += fold_test_preds
                cv_scores.append(fold_score)
        
        # Average test predictions
        test_preds /= self.config.N_FOLDS
        
        # Calculate overall CV score
        cv_score = pearsonr(y_train, oof_preds)[0]
        print(f"       🎯 CV Score: {cv_score:.6f} (±{np.std(cv_scores):.6f})")
        
        return oof_preds, test_preds, cv_score

class EnsembleManager:
    """Manage ensemble creation and evaluation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        print_step("Ensemble Manager Initialized")
        logger.info("EnsembleManager initialized")
    
    def create_ensemble_predictions(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create ensemble predictions using different strategies.
        
        Args:
            model_results: Dictionary containing individual model results
            
        Returns:
            Dictionary containing ensemble results
        """
        print_header("ENSEMBLE CREATION AND EVALUATION")
        
        # Extract individual model results
        y_true = model_results['y_true']
        individual_models = {k: v for k, v in model_results.items() if k != 'y_true'}
        
        print_step("Individual model performance")
        for model_name, results in individual_models.items():
            print(f"   🎯 {model_name}: {results['score']:.6f} ({results['algorithm']}, {results['data_percent']*100:.0f}% data)")
        
        # Simple average ensemble
        print_step("Creating simple average ensemble")
        avg_oof = np.mean([results['oof_preds'] for results in individual_models.values()], axis=0)
        avg_test = np.mean([results['test_preds'] for results in individual_models.values()], axis=0)
        avg_score = pearsonr(y_true, avg_oof)[0]
        print_result("Average ensemble score", avg_score)
        
        # Weighted ensemble (performance-based)
        print_step("Creating performance-weighted ensemble")
        scores = np.array([results['score'] for results in individual_models.values()])
        weights = scores / scores.sum()
        
        weighted_oof = np.average([results['oof_preds'] for results in individual_models.values()], 
                                 weights=weights, axis=0)
        weighted_test = np.average([results['test_preds'] for results in individual_models.values()], 
                                  weights=weights, axis=0)
        weighted_score = pearsonr(y_true, weighted_oof)[0]
        print_result("Weighted ensemble score", weighted_score)
        
        print("   📊 Model weights:")
        for i, (model_name, weight) in enumerate(zip(individual_models.keys(), weights)):
            print(f"     {model_name}: {weight:.4f}")
        
        # Create per-learner ensembles first (matching reference implementation)
        print_step("Creating per-learner ensembles")
        learner_ensembles = {}
        
        # Separate models by algorithm
        xgb_models = {k: v for k, v in individual_models.items() if 'XGB' in k}
        lgb_models = {k: v for k, v in individual_models.items() if 'LGB' in k}
        
        # Create XGBoost ensemble
        if xgb_models:
            xgb_oof_simple = np.mean([v['oof_preds'] for v in xgb_models.values()], axis=0)
            xgb_test_simple = np.mean([v['test_preds'] for v in xgb_models.values()], axis=0)
            xgb_score_simple = pearsonr(y_true, xgb_oof_simple)[0]
            print(f"   🎯 XGBoost simple ensemble score: {xgb_score_simple:.6f}")
            learner_ensembles['xgb'] = {
                'oof_simple': xgb_oof_simple,
                'test_simple': xgb_test_simple
            }
        
        # Create LightGBM ensemble
        if lgb_models:
            lgb_oof_simple = np.mean([v['oof_preds'] for v in lgb_models.values()], axis=0)
            lgb_test_simple = np.mean([v['test_preds'] for v in lgb_models.values()], axis=0)
            lgb_score_simple = pearsonr(y_true, lgb_oof_simple)[0]
            print(f"   🎯 LightGBM simple ensemble score: {lgb_score_simple:.6f}")
            learner_ensembles['lgb'] = {
                'oof_simple': lgb_oof_simple,
                'test_simple': lgb_test_simple
            }
        
        # Final ensemble creation based on strategy
        print_step("Creating final ensemble")
        
        if self.config.ENSEMBLE_STRATEGY == "individual_models":
            print("   🎯 Using individual model weights strategy")
            final_oof, final_test, ensemble_type = self._create_individual_weighted_ensemble(individual_models, y_true)
        elif self.config.ENSEMBLE_STRATEGY == "performance_based":
            print("   🎯 Using performance-based ensemble strategy")
            final_oof, final_test, ensemble_type = self._create_performance_based_ensemble(individual_models, y_true)
        else:
            print("   🎯 Using learner-level ensemble strategy")
            final_oof, final_test, ensemble_type = self._create_learner_level_ensemble(learner_ensembles)
        
        final_score = pearsonr(y_true, final_oof)[0]
        
        # For backward compatibility, also compute the old ensemble scores
        avg_score = pearsonr(y_true, avg_oof)[0]
        
        # Prepare results summary based on ensemble strategy
        results_summary = self._prepare_results_summary(
            individual_models, learner_ensembles, final_score, ensemble_type, 
            weights, xgb_score_simple if 'xgb' in learner_ensembles else None,
            lgb_score_simple if 'lgb' in learner_ensembles else None
        )
        
        ensemble_results = {
            'final_predictions': final_test,
            'final_oof': final_oof,
            'final_score': final_score,
            'ensemble_type': ensemble_type,
            'results_summary': results_summary,
            'individual_models': individual_models
        }
        
        print_result("Final ensemble score", final_score)
        print("✨ Ensemble creation completed successfully!")
        logger.info(f"Ensemble created successfully with score: {final_score:.6f}")
        
        return ensemble_results
    
    def _create_individual_weighted_ensemble(self, individual_models: Dict[str, Any], y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Create ensemble using individual model weights.
        
        Args:
            individual_models: Dictionary of individual model results
            y_true: True target values for scoring
            
        Returns:
            Tuple of (final_oof, final_test, ensemble_type)
        """
        print("   🔍 Creating individual model weighted ensemble")
        
        # Get weights and normalize if needed
        model_weights = self.config.INDIVIDUAL_MODEL_WEIGHTS.copy()
        total_weight = sum(model_weights.values())
        
        if abs(total_weight - 1.0) > 1e-6:
            print_warning(f"Individual model weights sum to {total_weight:.6f}, not 1.0. Normalizing weights.")
            for model_name in model_weights:
                model_weights[model_name] /= total_weight
            ensemble_type = "individual_weighted_normalized"
            print("   ✅ Using normalized individual model weights")
        else:
            ensemble_type = "individual_weighted"
            print("   ✅ Using individual model weights")
        
        # Display weights
        print("   📊 Individual model weights:")
        for model_name, weight in model_weights.items():
            if model_name in individual_models:
                algorithm = individual_models[model_name]['algorithm']
                data_percent = individual_models[model_name]['data_percent']
                print(f"     {model_name}: {weight:.4f} ({algorithm}, {data_percent*100:.0f}% data)")
        
        # Create weighted ensemble
        weighted_oof = np.zeros(len(y_true))
        weighted_test = np.zeros(len(individual_models[list(individual_models.keys())[0]]['test_preds']))
        
        for model_name, weight in model_weights.items():
            if model_name in individual_models:
                weighted_oof += weight * individual_models[model_name]['oof_preds']
                weighted_test += weight * individual_models[model_name]['test_preds']
            else:
                print_warning(f"Model {model_name} not found in individual_models, skipping")
        
        score = pearsonr(y_true, weighted_oof)[0]
        print(f"   🎯 Individual weighted ensemble score: {score:.6f}")
        
        return weighted_oof, weighted_test, ensemble_type
    
    def _create_learner_level_ensemble(self, learner_ensembles: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Create ensemble using learner-level weights (XGBoost vs LightGBM).
        
        Args:
            learner_ensembles: Dictionary of learner ensemble results
            
        Returns:
            Tuple of (final_oof, final_test, ensemble_type)
        """
        print("   🔍 Creating learner-level ensemble")
        
        # Check if custom weights are provided
        if self.config.CUSTOM_ENSEMBLE_WEIGHTS and len(self.config.CUSTOM_ENSEMBLE_WEIGHTS) > 0:
            custom_weights = self.config.CUSTOM_ENSEMBLE_WEIGHTS
            
            # Validate custom weights
            if len(custom_weights) != len(learner_ensembles):
                print_warning(f"Custom weights length ({len(custom_weights)}) doesn't match number of learners ({len(learner_ensembles)}). Using equal weights.")
                final_oof = np.mean([le['oof_simple'] for le in learner_ensembles.values()], axis=0)
                final_test = np.mean([le['test_simple'] for le in learner_ensembles.values()], axis=0)
                ensemble_type = "simple_across_learners"
                print("   ✅ Using simple average ensemble across learners")
            elif abs(sum(custom_weights) - 1.0) > 1e-6:
                print_warning(f"Custom weights sum to {sum(custom_weights):.6f}, not 1.0. Normalizing weights.")
                # Normalize weights to sum to 1.0
                custom_weights = [w / sum(custom_weights) for w in custom_weights]
                
                # Apply custom weights
                learner_oofs = [le['oof_simple'] for le in learner_ensembles.values()]
                learner_tests = [le['test_simple'] for le in learner_ensembles.values()]
                
                final_oof = np.average(learner_oofs, weights=custom_weights, axis=0)
                final_test = np.average(learner_tests, weights=custom_weights, axis=0)
                ensemble_type = "custom_weighted_across_learners_normalized"
                
                print("   ✅ Using custom weighted ensemble across learners (normalized)")
                print(f"   📊 Normalized weights: {custom_weights}")
            else:
                # Apply custom weights
                learner_oofs = [le['oof_simple'] for le in learner_ensembles.values()]
                learner_tests = [le['test_simple'] for le in learner_ensembles.values()]
                
                final_oof = np.average(learner_oofs, weights=custom_weights, axis=0)
                final_test = np.average(learner_tests, weights=custom_weights, axis=0)
                ensemble_type = "custom_weighted_across_learners"
                
                print("   ✅ Using custom weighted ensemble across learners")
                print(f"   📊 Custom weights: {custom_weights}")
                
                # Show weight assignment
                for i, (learner_name, weight) in enumerate(zip(learner_ensembles.keys(), custom_weights)):
                    algorithm = "XGBoost" if learner_name == 'xgb' else "LightGBM"
                    print(f"     {algorithm}: {weight:.4f}")
        else:
            # Default: simple average ensemble
            final_oof = np.mean([le['oof_simple'] for le in learner_ensembles.values()], axis=0)
            final_test = np.mean([le['test_simple'] for le in learner_ensembles.values()], axis=0)
            ensemble_type = "simple_across_learners"
            print("   ✅ Using simple average ensemble across learners (default)")
        
        return final_oof, final_test, ensemble_type
    
    def _create_performance_based_ensemble(self, individual_models: Dict[str, Any], y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Create ensemble using performance-based weights (automatically calculated from CV scores).
        
        Args:
            individual_models: Dictionary of individual model results
            y_true: True target values for scoring
            
        Returns:
            Tuple of (final_oof, final_test, ensemble_type)
        """
        print("   🔍 Creating performance-based weighted ensemble")
        
        # Extract scores and create performance-based weights
        model_names = list(individual_models.keys())
        scores = np.array([individual_models[model]['score'] for model in model_names])
        
        print(f"   📊 Individual model scores:")
        for model_name, score in zip(model_names, scores):
            algorithm = individual_models[model_name]['algorithm']
            data_percent = individual_models[model_name]['data_percent']
            print(f"     {model_name}: {score:.6f} ({algorithm}, {data_percent*100:.0f}% data)")
        
        # Calculate performance-based weights (higher score = higher weight)
        # Using softmax-like transformation to convert scores to weights
        # Add small epsilon to prevent division by zero
        epsilon = 1e-8
        exp_scores = np.exp((scores - scores.max()) / 0.1)  # Temperature scaling for smoother weights
        weights = exp_scores / (exp_scores.sum() + epsilon)
        
        # Alternative: Simple proportional weights
        # weights = scores / scores.sum()
        
        print(f"   ⚖️  Performance-based weights:")
        for model_name, weight, score in zip(model_names, weights, scores):
            print(f"     {model_name}: {weight:.4f} (score: {score:.6f})")
        
        print(f"   📊 Weight statistics:")
        print(f"     Min weight: {weights.min():.4f}")
        print(f"     Max weight: {weights.max():.4f}")
        print(f"     Weight range: {weights.max() - weights.min():.4f}")
        print(f"     Weight sum: {weights.sum():.6f}")
        
        # Create weighted ensemble
        weighted_oof = np.zeros(len(y_true))
        weighted_test = np.zeros(len(individual_models[model_names[0]]['test_preds']))
        
        for model_name, weight in zip(model_names, weights):
            weighted_oof += weight * individual_models[model_name]['oof_preds']
            weighted_test += weight * individual_models[model_name]['test_preds']
        
        # Calculate ensemble score
        ensemble_score = pearsonr(y_true, weighted_oof)[0]
        print(f"   🎯 Performance-based ensemble score: {ensemble_score:.6f}")
        
        # Compare with simple average
        simple_avg_oof = np.mean([individual_models[model]['oof_preds'] for model in model_names], axis=0)
        simple_avg_score = pearsonr(y_true, simple_avg_oof)[0]
        improvement = ensemble_score - simple_avg_score
        print(f"   📈 Improvement over simple average: {improvement:+.6f}")
        
        ensemble_type = "performance_based_weighted"
        print("   ✅ Performance-based ensemble created successfully")
        
        return weighted_oof, weighted_test, ensemble_type
    
    def _prepare_results_summary(self, individual_models: Dict[str, Any], learner_ensembles: Dict[str, Any], 
                               final_score: float, ensemble_type: str, weights: np.ndarray,
                               xgb_score_simple: Optional[float], lgb_score_simple: Optional[float]) -> List[Dict[str, Any]]:
        """
        Prepare results summary based on ensemble strategy.
        
        Args:
            individual_models: Dictionary of individual model results
            learner_ensembles: Dictionary of learner ensemble results  
            final_score: Final ensemble score
            ensemble_type: Type of ensemble used
            weights: Performance-based weights for backward compatibility
            xgb_score_simple: XGBoost ensemble score (if available)
            lgb_score_simple: LightGBM ensemble score (if available)
            
        Returns:
            List of result dictionaries for summary
        """
        results_summary = []
        
        # Add individual model results with appropriate weights
        for model_name, results in individual_models.items():
            if self.config.ENSEMBLE_STRATEGY == "individual_models":
                # Use individual model weights if specified
                weight_in_ensemble = self.config.INDIVIDUAL_MODEL_WEIGHTS.get(model_name, 0.0)
                # Normalize if needed
                total_weight = sum(self.config.INDIVIDUAL_MODEL_WEIGHTS.values())
                if abs(total_weight - 1.0) > 1e-6:
                    weight_in_ensemble /= total_weight
            elif self.config.ENSEMBLE_STRATEGY == "performance_based":
                # Calculate performance-based weight for this model
                scores_array = np.array([individual_models[m]['score'] for m in individual_models.keys()])
                model_idx = list(individual_models.keys()).index(model_name)
                # Use same calculation as in _create_performance_based_ensemble
                epsilon = 1e-8
                exp_scores = np.exp((scores_array - scores_array.max()) / 0.1)
                weights_array = exp_scores / (exp_scores.sum() + epsilon)
                weight_in_ensemble = weights_array[model_idx]
            else:
                # Use equal weight for learner-level strategy
                weight_in_ensemble = 1.0 / len(individual_models)
            
            results_summary.append({
                'model': model_name,
                'algorithm': results['algorithm'],
                'data_percent': results['data_percent'],
                'pearson_correlation': results['score'],
                'weight_in_ensemble': weight_in_ensemble
            })
        
        # Add learner ensemble results if using learner-level strategy
        if self.config.ENSEMBLE_STRATEGY == "learner_level":
            # Determine learner ensemble weights
            ensemble_weights = []
            if self.config.CUSTOM_ENSEMBLE_WEIGHTS and len(self.config.CUSTOM_ENSEMBLE_WEIGHTS) == len(learner_ensembles):
                # Use custom weights (potentially normalized)
                if abs(sum(self.config.CUSTOM_ENSEMBLE_WEIGHTS) - 1.0) > 1e-6:
                    ensemble_weights = [w / sum(self.config.CUSTOM_ENSEMBLE_WEIGHTS) for w in self.config.CUSTOM_ENSEMBLE_WEIGHTS]
                else:
                    ensemble_weights = self.config.CUSTOM_ENSEMBLE_WEIGHTS
            else:
                # Use equal weights
                ensemble_weights = [1.0 / len(learner_ensembles)] * len(learner_ensembles)
            
            weight_idx = 0
            if 'xgb' in learner_ensembles and xgb_score_simple is not None:
                results_summary.append({
                    'model': 'XGBoost Simple Ensemble',
                    'algorithm': 'XGBoost Ensemble',
                    'data_percent': 1.0,
                    'pearson_correlation': xgb_score_simple,
                    'weight_in_ensemble': ensemble_weights[weight_idx]
                })
                weight_idx += 1
            
            if 'lgb' in learner_ensembles and lgb_score_simple is not None:
                results_summary.append({
                    'model': 'LightGBM Simple Ensemble',
                    'algorithm': 'LightGBM Ensemble',
                    'data_percent': 1.0,
                    'pearson_correlation': lgb_score_simple,
                    'weight_in_ensemble': ensemble_weights[weight_idx]
                })
        
        # Add final ensemble result
        ensemble_name = f"Final Ensemble ({ensemble_type.replace('_', ' ').title()})"
        results_summary.append({
            'model': ensemble_name,
            'algorithm': 'Final Ensemble',
            'data_percent': 1.0,
            'pearson_correlation': final_score,
            'weight_in_ensemble': 1.0
        })
        
        return results_summary

class XGBoostLightGBMPipeline:
    """Main pipeline orchestrator for XGBoost and LightGBM ensemble modeling."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        print_header("XGBOOST-LIGHTGBM ENSEMBLE PIPELINE")
        print(f"   🎯 Pipeline initialized with {len(self.config.TRAIN_DATASETS)} train and {len(self.config.TEST_DATASETS)} test datasets")
        print(f"   🤖 Model algorithms: XGBoost + LightGBM")
        print(f"   📊 Cross-validation folds: {self.config.N_FOLDS}")
        print(f"   🎯 Selected features: {len(self.config.SELECTED_FEATURES)}")
        print(f"   ⚖️  Ensemble strategy: {self.config.ENSEMBLE_STRATEGY}")
        
        # Validate configuration
        is_valid, error_msg = self.config.validate_weights()
        if not is_valid:
            raise ValueError(f"Configuration validation failed: {error_msg}")
        
        if self.config.ENSEMBLE_STRATEGY == "individual_models":
            print(f"   🎯 Individual model weights specified: {len(self.config.INDIVIDUAL_MODEL_WEIGHTS)} models")
        elif self.config.ENSEMBLE_STRATEGY == "performance_based":
            print(f"   🎯 Performance-based ensemble: weights calculated from CV scores")
        elif self.config.CUSTOM_ENSEMBLE_WEIGHTS:
            print(f"   🎯 Custom learner weights: {self.config.CUSTOM_ENSEMBLE_WEIGHTS}")
        else:
            print(f"   🎯 Using equal weights for ensemble")
        
        # Initialize components
        self.data_processor = DataProcessor(config)
        self.model_trainer = ModelTrainer(config)
        self.ensemble_manager = EnsembleManager(config)
        
        logger.info("XGBoostLightGBMPipeline initialized successfully")
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline from data loading to final predictions.
        
        Returns:
            Dictionary containing final results and metadata
        """
        pipeline_start = time.time()
        print(f"⏰ Pipeline start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_memory_usage("pipeline start")
        
        try:
            # Step 1: Load and merge datasets
            train_df, test_df = self.data_processor.load_and_merge_datasets()
            
            # Step 2: Train model ensemble
            model_results = self.model_trainer.train_model_ensemble(train_df, test_df)
            
            # Step 3: Create ensemble predictions
            ensemble_results = self.ensemble_manager.create_ensemble_predictions(model_results)
            
            # Step 4: Save results
            self._save_results(ensemble_results, test_df)
            
            # Final summary
            total_time = time.time() - pipeline_start
            print_header("PIPELINE COMPLETED SUCCESSFULLY")
            print_result("Total pipeline time", total_time, ".1f")
            print_result("Final ensemble score", ensemble_results['final_score'])
            print_memory_usage("pipeline end")
            print(f"   📁 Submission saved to: {self.config.SUBMISSION_FILENAME}")
            print(f"   📁 Results saved to: {self.config.RESULTS_FILENAME}")
            print("🎉 Pipeline execution completed successfully!")
            
            logger.info(f"Pipeline completed successfully in {total_time:.1f}s with final score: {ensemble_results['final_score']:.6f}")
            
            # Get final shapes before cleanup
            train_shape = train_df.shape if 'train_df' in locals() else "N/A"
            test_shape = test_df.shape if 'test_df' in locals() else "N/A"
            
            return {
                'ensemble_results': ensemble_results,
                'execution_time': total_time,
                'train_shape': train_shape,
                'test_shape': test_shape
            }
            
        except Exception as e:
            print_error(f"Pipeline failed: {str(e)}")
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _save_results(self, ensemble_results: Dict[str, Any], test_df: pd.DataFrame):
        """Save predictions and results to files."""
        print_step("Saving results to files")
        
        try:
            # Save submission file
            if self.config.ID_COLUMN in test_df.columns:
                submission_df = pd.DataFrame({
                    self.config.ID_COLUMN: test_df[self.config.ID_COLUMN],
                    'prediction': ensemble_results['final_predictions']
                })
            else:
                submission_df = pd.DataFrame({
                    'ID': range(len(ensemble_results['final_predictions'])),
                    'prediction': ensemble_results['final_predictions']
                })
            
            submission_df.to_csv(self.config.SUBMISSION_FILENAME, index=False)
            print(f"   ✅ Submission saved: {self.config.SUBMISSION_FILENAME}")
            
            # Save detailed results
            results_df = pd.DataFrame(ensemble_results['results_summary'])
            results_df.to_csv(self.config.RESULTS_FILENAME, index=False)
            print(f"   ✅ Results saved: {self.config.RESULTS_FILENAME}")
            
            # Display submission preview
            print("   📋 Submission preview:")
            print(submission_df.head().to_string(index=False))
            
            # Display results summary
            print("   📋 Results summary:")
            print(results_df.to_string(index=False))
            
        except Exception as e:
            print_error(f"Failed to save results: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # User-provided parameters
    lgbm_params = {
        "boosting_type": "gbdt",
        "colsample_bytree": 0.5625888953382505,
        "learning_rate": 0.029312951475451557,
        "min_child_samples": 63,
        "min_child_weight": 0.11456572852335424,
        "n_estimators": 150,
        "n_jobs": -1,
        "num_leaves": 37,
        "random_state": 42,
        "reg_alpha": 85.2476527854083,
        "reg_lambda": 99.38305361388907,
        "subsample": 0.450669817684892,
        "verbose": -1
    }

    xgb_params = {
        "tree_method": "gpu_hist",
        "colsample_bylevel": 0.4778015829774066,
        "colsample_bynode": 0.362764358742407,
        "colsample_bytree": 0.7107423488010493,
        "gamma": 1.7094857725240398,
        "learning_rate": 0.02213323588455387,
        "max_depth": 20,
        "max_leaves": 12,
        "min_child_weight": 16,
        "n_estimators": 1800,
        "n_jobs": -1,
        "random_state": 42,
        "reg_alpha": 39.352415706891264,
        "reg_lambda": 75.44843704068275,
        "subsample": 0.06566669853471274,
        "verbosity": 0
    }

    base_features = [
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
    ]

    additional_features = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "X888", "X421", "X333"
    ]
    
    selected_features = base_features + additional_features
    
    # Create configuration
    config = ModelConfig(
        # Dataset configurations - UPDATE THESE PATHS AS NEEDED
        TRAIN_DATASETS=[
            DatasetConfig(
                file_path="/kaggle/input/drw-crypto-market-prediction/train.parquet",
                feature_columns=selected_features + ["label"],  # Will keep these + ID columns
                id_columns=["timestamp"],
                dataset_name="Train Features",
                is_required=True
            ),
            DatasetConfig(
                file_path="/kaggle/input/drw-all-auxiliary-datasets/train_regimes.csv",  # Adjust filename
                feature_columns=["predicted_regime_Est"],  # Additional regime feature
                id_columns=["timestamp"],
                dataset_name="Train Regimes",
                is_required=True
            ),
            DatasetConfig(
                file_path="/kaggle/input/drw-all-auxiliary-datasets/pca_train_dataset.csv",  # Adjust filename
                feature_columns=["pca_0","pca_1","pca_2","pca_3","pca_4","pca_5","pca_6","pca_7","pca_8","pca_9"],  # Additional pca features
                id_columns=["timestamp"],
                dataset_name="Train PCA",
                is_required=True
            )
        ],
        TEST_DATASETS=[
            DatasetConfig(
                file_path="/kaggle/input/drw-crypto-market-prediction/test.parquet",
                feature_columns=selected_features,  # No label in test data
                id_columns=["ID"],
                dataset_name="Test Features",
                is_required=True
            ),
            DatasetConfig(
                file_path="/kaggle/input/drw-all-auxiliary-datasets/test_regimes.csv",
                feature_columns=["predicted_regime_Est"],  # Additional regime feature
                id_columns=["ID"],
                dataset_name="Test Regimes",
                is_required=True
            ),
            DatasetConfig(
                file_path="/kaggle/input/drw-all-auxiliary-datasets/pca_test_dataset.csv",
                feature_columns=["pca_0","pca_1","pca_2","pca_3","pca_4","pca_5","pca_6","pca_7","pca_8","pca_9"],  # Additional pca features
                id_columns=["ID"],
                dataset_name="Test PCA",
                is_required=True
            )
        ],
        
        # Column configuration
        TARGET_COLUMN="label",  # Target column from main train dataset
        ID_COLUMN="ID",         # ID column for test predictions
        TIMESTAMP_COLUMN="timestamp",  # Timestamp column for merging
        
        # Model parameters
        XGB_PARAMS=xgb_params,
        LGBM_PARAMS=lgbm_params,
        SELECTED_FEATURES=selected_features + ["predicted_regime_Est"],  # Include regime feature
        
        # Multi-GPU settings
        USE_MULTI_GPU=True,
        GPU_DEVICES=[0, 1],  # Use both T4 GPUs
        PARALLEL_FOLD_TRAINING=True,
        REDUCE_MEMORY_USAGE=True,  # Optimize memory usage
        ADD_ENGINEERED_FEATURES=True,
        
        # Ensemble weight configuration (choose one strategy):
        
        # STRATEGY 1: Learner-level weights (XGBoost vs LightGBM ensembles)
        # ENSEMBLE_STRATEGY="learner_level",
        # CUSTOM_ENSEMBLE_WEIGHTS=[0.6, 0.4],  # Give XGBoost 60% weight, LightGBM 40% weight
        
        # STRATEGY 2: Individual model weights (granular control over all 6 models)
        ENSEMBLE_STRATEGY="individual_models",
        INDIVIDUAL_MODEL_WEIGHTS={
            # XGBoost models
            "XGB_Full_Dataset_100%": 0.25,      # XGBoost full dataset
            "XGB_Recent_Data_75%": 0.20,        # XGBoost 75% recent data
            "XGB_Recent_Data_50%": 0.15,        # XGBoost 50% recent data
            # LightGBM models  
            "LGB_Full_Dataset_100%": 0.20,      # LightGBM full dataset
            "LGB_Recent_Data_75%": 0.15,        # LightGBM 75% recent data
            "LGB_Recent_Data_50%": 0.05,        # LightGBM 50% recent data
        },
        # Weights will be automatically normalized if they don't sum to 1.0
        
        # STRATEGY 3: Performance-based weights (automatic weight calculation)
        # ENSEMBLE_STRATEGY="performance_based",
        # No additional configuration needed - weights calculated from CV scores
        # Higher performing models automatically get higher weights
        
        # Alternative individual weights example (equal weights):
        # INDIVIDUAL_MODEL_WEIGHTS={
        #     "XGB_Full_Dataset_100%": 1/6, "XGB_Recent_Data_75%": 1/6, "XGB_Recent_Data_50%": 1/6,
        #     "LGB_Full_Dataset_100%": 1/6, "LGB_Recent_Data_75%": 1/6, "LGB_Recent_Data_50%": 1/6,
        # },
        
        # Alternative individual weights example (prefer full dataset models):
        # INDIVIDUAL_MODEL_WEIGHTS={
        #     "XGB_Full_Dataset_100%": 0.35, "XGB_Recent_Data_75%": 0.15, "XGB_Recent_Data_50%": 0.10,
        #     "LGB_Full_Dataset_100%": 0.25, "LGB_Recent_Data_75%": 0.10, "LGB_Recent_Data_50%": 0.05,
        # },
        
        # Other settings
        N_FOLDS=5,
        RANDOM_STATE=42,
        DECAY_FACTOR=0.95,
        
        SUBMISSION_FILENAME="submission.csv",
    )
    
    # Run pipeline
    pipeline = XGBoostLightGBMPipeline(config)
    results = pipeline.run_pipeline()
    
    print(f"\n Pipeline completed! Final ensemble score: {results['ensemble_results']['final_score']:.6f}")