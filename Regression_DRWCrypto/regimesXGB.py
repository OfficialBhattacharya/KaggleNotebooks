import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr
import time
import gc
import warnings
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import logging
import os

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

@dataclass
class DatasetConfig:
    """Configuration for individual dataset files."""
    file_path: str
    feature_columns: List[str]  # Specific columns to keep from this file (empty list means keep all except merge columns)
    merge_columns: List[str]    # Columns to use for merging (e.g., ['timestamp'], ['ID'])
    dataset_name: str          # Human readable name for logging
    is_required: bool = True   # Whether this dataset is required for pipeline to continue

@dataclass
class ModelConfig:
    """Configuration class for model parameters and settings."""
    
    # Dataset configurations - will be set in __post_init__
    TRAIN_DATASETS: List[DatasetConfig] = None
    TEST_DATASETS: List[DatasetConfig] = None
    
    # Column names - Updated to match actual data structure
    TARGET_COLUMN: str = "label"
    REGIME_COLUMN: str = "predicted_regime_Est"  # Fixed: Use actual column name
    ID_COLUMN: str = "ID"
    TIMESTAMP_COLUMN: str = "timestamp"
    
    # Cross-validation settings
    N_FOLDS: int = 5
    RANDOM_STATE: int = 42
    
    # Time decay settings
    DECAY_FACTOR: float = 0.95
    
    # Model ensemble configurations
    MODEL_CONFIGS: List[Dict[str, Any]] = None
    
    # XGBoost parameters optimized for dual GPU
    XGB_PARAMS: Dict[str, Any] = None
    
    # Output settings
    SUBMISSION_FILENAME: str = "submission_regimes_XGB.csv"
    
    def __post_init__(self):
        """Initialize default configurations after dataclass creation."""
        if self.TRAIN_DATASETS is None:
            # Default train dataset configurations (backward compatibility)
            self.TRAIN_DATASETS = [
                DatasetConfig(
                    file_path="/kaggle/input/drw-all-auxiliary-datasets/top500_train_dataset.csv",
                    feature_columns=[],  # Empty means keep all except merge columns
                    merge_columns=[self.TIMESTAMP_COLUMN],
                    dataset_name="Train Features",
                    is_required=True
                ),
                DatasetConfig(
                    file_path="/kaggle/input/drw-all-auxiliary-datasets/train_regimes.csv",
                    feature_columns=[self.REGIME_COLUMN],  # Only keep regime column
                    merge_columns=[self.TIMESTAMP_COLUMN],
                    dataset_name="Train Regimes",
                    is_required=True
                )
            ]
        
        if self.TEST_DATASETS is None:
            # Default test dataset configurations (backward compatibility)
            self.TEST_DATASETS = [
                DatasetConfig(
                    file_path="/kaggle/input/drw-all-auxiliary-datasets/top500_test_dataset.csv",
                    feature_columns=[],  # Empty means keep all except merge columns
                    merge_columns=[self.ID_COLUMN],
                    dataset_name="Test Features",
                    is_required=True
                ),
                DatasetConfig(
                    file_path="/kaggle/input/drw-all-auxiliary-datasets/test_regimes.csv",
                    feature_columns=[self.REGIME_COLUMN],  # Only keep regime column
                    merge_columns=[self.ID_COLUMN],
                    dataset_name="Test Regimes",
                    is_required=True
                )
            ]
        
        if self.MODEL_CONFIGS is None:
            self.MODEL_CONFIGS = [
                {"name": "Full Dataset (100%)", "percent": 1.00, "priority": 1},
                {"name": "Recent Data (90%)", "percent": 0.90, "priority": 2},
                {"name": "Recent Data (80%)", "percent": 0.80, "priority": 3},
                {"name": "Recent Data (70%)", "percent": 0.70, "priority": 4},
                {"name": "Recent Data (60%)", "percent": 0.60, "priority": 5},
                {"name": "Recent Data (50%)", "percent": 0.50, "priority": 6},
                {"name": "Recent Data (40%)", "percent": 0.40, "priority": 7}
            ]
        
        if self.XGB_PARAMS is None:
            self.XGB_PARAMS = {
                # GPU Configuration - Optimized for T4 x 2
                "tree_method": "hist",
                "device": "cuda",  # XGBoost will automatically use available GPUs
                # Removed gpu_id to avoid conflicts with device parameter
                "predictor": "gpu_predictor",
                
                # Performance Parameters
                "colsample_bylevel": 0.4778015829774066,
                "colsample_bynode": 0.362764358742407,
                "colsample_bytree": 0.7107423488010493,
                "gamma": 1.7094857725240398,
                "learning_rate": 0.02213323588455387,
                "max_depth": 20,
                "max_leaves": 12,
                "min_child_weight": 16,
                "n_estimators": 1667,
                "reg_alpha": 39.352415706891264,
                "reg_lambda": 75.44843704068275,
                "subsample": 0.06566669853471274,
                
                # System Configuration
                "n_jobs": -1,  # Use all CPU cores
                "random_state": self.RANDOM_STATE,
                "verbosity": 0,
                "objective": "reg:squarederror",
                
                # Memory Optimization
                "max_bin": 256,
                "grow_policy": "lossguide",
            }

class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
        print_step("Performance Monitor Initialized")
        print(f"   ⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def checkpoint(self, name: str) -> float:
        """Record a checkpoint and return elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.checkpoints[name] = elapsed
        print(f"   ⏱️  Checkpoint '{name}': {elapsed:.1f}s elapsed")
        return elapsed
    
    def get_eta(self, current_step: int, total_steps: int) -> str:
        """Calculate estimated time remaining."""
        if current_step == 0:
            return "Calculating..."
        
        elapsed = time.time() - self.start_time
        avg_time_per_step = elapsed / current_step
        remaining_steps = total_steps - current_step
        eta_seconds = avg_time_per_step * remaining_steps
        
        return str(timedelta(seconds=int(eta_seconds)))
    
    def memory_cleanup(self):
        """Perform garbage collection and memory cleanup."""
        print("   🧹 Performing memory cleanup...")
        gc.collect()
        print("   ✅ Memory cleanup completed")

class DataProcessor:
    """Handle data loading, merging, and preprocessing with flexible dataset configurations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        print_step("Data Processor Initialized")
        print(f"   📊 Train datasets configured: {len(self.config.TRAIN_DATASETS)}")
        print(f"   📊 Test datasets configured: {len(self.config.TEST_DATASETS)}")
    
    def load_and_merge_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and merge multiple datasets efficiently based on configuration.
        
        Returns:
            Tuple of (merged_train_df, merged_test_df)
        """
        print_header("FLEXIBLE DATA LOADING AND MERGING")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
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
            
            # Validate final datasets
            print_step("Validating final merged datasets")
            self._validate_final_datasets(merged_train, merged_test)
            
            # Display final dataset information
            print_step("Final dataset information")
            print(f"     📊 Train shape: {merged_train.shape}")
            print(f"     📊 Test shape: {merged_test.shape}")
            print(f"     📊 Train columns: {list(merged_train.columns)[:10]}... (total: {len(merged_train.columns)})")
            print(f"     📊 Test columns: {list(merged_test.columns)[:10]}... (total: {len(merged_test.columns)})")
            print(f"     🎯 Target column '{self.config.TARGET_COLUMN}' present in train: {self.config.TARGET_COLUMN in merged_train.columns}")
            print(f"     🏷️  Regime column '{self.config.REGIME_COLUMN}' present: {self.config.REGIME_COLUMN in merged_train.columns}")
            
            # Memory cleanup
            self.monitor.memory_cleanup()
            
            total_time = self.monitor.checkpoint('data_loading')
            print_result("Total loading time", total_time, ".1f")
            print("✨ Data loading and merging completed successfully!")
            
            return merged_train, merged_test
            
        except Exception as e:
            print_error(f"Data loading failed: {str(e)}")
            logger.error(f"❌ Error in data loading: {str(e)}")
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
                
                # Load the dataset
                raw_df = pd.read_csv(dataset_config.file_path)
                load_time = time.time() - load_start
                
                print(f"       ✅ Raw shape: {raw_df.shape} loaded in {load_time:.1f}s")
                print(f"       📋 Available columns: {list(raw_df.columns)[:10]}...")
                
                # Process columns according to configuration
                processed_df = self._process_dataset_columns(raw_df, dataset_config)
                
                loaded_datasets[dataset_config.dataset_name] = {
                    'dataframe': processed_df,
                    'config': dataset_config
                }
                
                print(f"       ✅ Processed shape: {processed_df.shape}")
                print(f"       📊 Final columns: {list(processed_df.columns)}")
                
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
            merge_columns = dataset_info['config'].merge_columns
            
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
        
        return merged_df
    
    def _process_dataset_columns(self, df: pd.DataFrame, 
                               config: DatasetConfig) -> pd.DataFrame:
        """
        Process dataset columns according to configuration.
        
        Args:
            df: Raw dataframe
            config: Dataset configuration
            
        Returns:
            Processed dataframe with only desired columns
        """
        # Validate that merge columns exist
        missing_merge_cols = [col for col in config.merge_columns if col not in df.columns]
        if missing_merge_cols:
            raise KeyError(f"Merge columns {missing_merge_cols} not found in {config.dataset_name}. "
                          f"Available columns: {list(df.columns)}")
        
        # Determine columns to keep
        if config.feature_columns:
            # Use specified feature columns + merge columns
            columns_to_keep = list(set(config.feature_columns + config.merge_columns))
            
            # Validate that specified feature columns exist
            missing_feature_cols = [col for col in config.feature_columns if col not in df.columns]
            if missing_feature_cols:
                print_warning(f"Feature columns {missing_feature_cols} not found in {config.dataset_name}")
                # Keep only existing columns
                existing_feature_cols = [col for col in config.feature_columns if col in df.columns]
                columns_to_keep = list(set(existing_feature_cols + config.merge_columns))
        else:
            # Keep all columns (merge columns will be included)
            columns_to_keep = list(df.columns)
        
        # Select columns
        processed_df = df[columns_to_keep].copy()
        
        print(f"       📋 Kept {len(columns_to_keep)} columns from {len(df.columns)} available")
        if config.feature_columns:
            print(f"       🎯 Feature columns: {config.feature_columns}")
        print(f"       🔗 Merge columns: {config.merge_columns}")
        
        return processed_df
    
    def _validate_final_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Validate that final merged datasets have required columns."""
        
        # Check for target column in train data
        if self.config.TARGET_COLUMN not in train_df.columns:
            available_cols = list(train_df.columns)
            raise KeyError(f"Target column '{self.config.TARGET_COLUMN}' not found in merged train data. "
                          f"Available columns: {available_cols}")
        
        # Check for regime column in both datasets
        for df_name, df in [("train", train_df), ("test", test_df)]:
            if self.config.REGIME_COLUMN not in df.columns:
                available_cols = list(df.columns)
                raise KeyError(f"Regime column '{self.config.REGIME_COLUMN}' not found in {df_name} data. "
                              f"Available columns: {available_cols}")
        
        # Check that test data has ID column
        if self.config.ID_COLUMN not in test_df.columns:
            available_cols = list(test_df.columns)
            raise KeyError(f"ID column '{self.config.ID_COLUMN}' not found in test data. "
                          f"Available columns: {available_cols}")
        
        print(f"     ✅ All required columns validated")
        print(f"       🎯 Target column: {self.config.TARGET_COLUMN}")
        print(f"       🏷️  Regime column: {self.config.REGIME_COLUMN}")
        print(f"       🆔 ID column: {self.config.ID_COLUMN}")

    def split_data_by_regimes(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Split datasets by regimes with comprehensive validation.
        
        Args:
            train_df: Merged training dataframe
            test_df: Merged test dataframe
            
        Returns:
            Dictionary containing regime-specific datasets
        """
        print_header("DATA SPLITTING BY REGIMES")
        
        try:
            # Identify regimes
            print_step("Analyzing regime distributions")
            
            # Validate regime column exists
            if self.config.REGIME_COLUMN not in train_df.columns:
                available_cols = list(train_df.columns)
                error_msg = (f"Regime column '{self.config.REGIME_COLUMN}' not found in training data. "
                           f"Available columns: {available_cols}")
                print_error(error_msg)
                raise KeyError(error_msg)
            
            if self.config.REGIME_COLUMN not in test_df.columns:
                available_cols = list(test_df.columns)
                error_msg = (f"Regime column '{self.config.REGIME_COLUMN}' not found in test data. "
                           f"Available columns: {available_cols}")
                print_error(error_msg)
                raise KeyError(error_msg)
            
            train_regimes = set(train_df[self.config.REGIME_COLUMN].unique())
            test_regimes = set(test_df[self.config.REGIME_COLUMN].unique())
            all_regimes = sorted(train_regimes.union(test_regimes))
            
            print(f"   📊 Found regimes: {all_regimes}")
            print(f"   🔍 Using regime column: '{self.config.REGIME_COLUMN}'")
            
            # Log regime distributions
            self._log_regime_distributions(train_df, test_df)
            
            # Identify feature columns
            print_step("Identifying feature columns")
            exclude_cols = {
                self.config.TARGET_COLUMN, 
                self.config.REGIME_COLUMN, 
                self.config.TIMESTAMP_COLUMN, 
                self.config.ID_COLUMN
            }
            
            # Find any additional columns to exclude (like 'predicted_regime' if it exists)
            additional_exclude = [col for col in train_df.columns 
                                if col.startswith('predicted_regime') and col != self.config.REGIME_COLUMN]
            exclude_cols.update(additional_exclude)
            
            feature_cols = [col for col in train_df.columns if col not in exclude_cols]
            
            print(f"   🎯 Feature columns identified: {len(feature_cols)}")
            print(f"   📋 Excluded columns: {sorted(list(exclude_cols))}")
            if additional_exclude:
                print(f"   🚫 Additional excluded regime columns: {additional_exclude}")
            
            # Validate feature columns exist in both datasets
            missing_in_test = [col for col in feature_cols if col not in test_df.columns]
            if missing_in_test:
                print_warning(f"Features missing in test data: {missing_in_test}")
                feature_cols = [col for col in feature_cols if col in test_df.columns]
                print(f"   🔄 Adjusted feature count: {len(feature_cols)}")
            
            # Split data by regimes
            print_step("Splitting data by regimes")
            regime_datasets = {}
            
            total_regimes = len(all_regimes)
            for i, regime in enumerate(all_regimes, 1):
                print_progress(i, total_regimes, "regimes")
                print(f"     🔄 Processing regime {regime}...")
                
                # Train data for regime
                train_mask = train_df[self.config.REGIME_COLUMN] == regime
                train_regime_data = train_df[train_mask]
                
                # Test data for regime
                test_mask = test_df[self.config.REGIME_COLUMN] == regime
                test_regime_data = test_df[test_mask]
                
                if len(train_regime_data) > 0:
                    X_train = train_regime_data[feature_cols].copy()
                    y_train = train_regime_data[self.config.TARGET_COLUMN].copy()
                    print(f"       ✅ Train data: {X_train.shape}")
                    print(f"       📊 Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
                else:
                    X_train = pd.DataFrame(columns=feature_cols)
                    y_train = pd.Series(dtype=float, name=self.config.TARGET_COLUMN)
                    print_warning("No training data for this regime")
                
                if len(test_regime_data) > 0:
                    X_test = test_regime_data[feature_cols].copy()
                    test_ids = test_regime_data[self.config.ID_COLUMN].values
                    print(f"       ✅ Test data: {X_test.shape}")
                else:
                    X_test = pd.DataFrame(columns=feature_cols)
                    test_ids = np.array([])
                    print_warning("No test data for this regime")
                
                regime_datasets[f'regime_{regime}'] = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'test_ids': test_ids,
                    'regime_id': regime
                }
            
            # Memory cleanup
            self.monitor.memory_cleanup()
            
            split_time = self.monitor.checkpoint('data_splitting')
            print_result("Data splitting time", split_time, ".1f")
            print("✨ Data splitting completed successfully!")
            
            return regime_datasets
            
        except Exception as e:
            print_error(f"Data splitting failed: {str(e)}")
            logger.error(f"❌ Error in data splitting: {str(e)}")
            raise
    
    def _log_regime_distributions(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Log detailed regime distributions."""
        print("   📈 Regime distributions:")
        
        train_counts = train_df[self.config.REGIME_COLUMN].value_counts().sort_index()
        test_counts = test_df[self.config.REGIME_COLUMN].value_counts().sort_index()
        
        print("     📊 Training data:")
        for regime, count in train_counts.items():
            pct = (count / len(train_df)) * 100
            print(f"       Regime {regime}: {count:,} samples ({pct:.1f}%)")
        
        print("     📊 Test data:")
        for regime, count in test_counts.items():
            pct = (count / len(test_df)) * 100
            print(f"       Regime {regime}: {count:,} samples ({pct:.1f}%)")

class WeightCalculator:
    """Calculate time-based sample weights."""
    
    @staticmethod
    def create_time_weights(n_samples: int, decay_factor: float = 0.95) -> np.ndarray:
        """
        Create exponentially decaying weights for temporal importance.
        
        Args:
            n_samples: Number of samples
            decay_factor: Decay rate (higher = more recent emphasis)
            
        Returns:
            Array of sample weights
        """
        if n_samples == 0:
            return np.array([])
        
        print(f"     ⚖️  Creating time weights for {n_samples:,} samples (decay: {decay_factor})")
        
        positions = np.arange(n_samples)
        normalized_positions = positions / max(1, n_samples - 1)
        weights = decay_factor ** (1 - normalized_positions)
        
        # Normalize weights to sum to n_samples
        weights = weights * n_samples / weights.sum()
        
        print(f"     ✅ Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        
        return weights

class RegimeModelTrainer:
    """Train XGBoost models for each regime with advanced optimization."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        print_step("Initializing Regime Model Trainer")
        
        # Verify GPU availability
        self._setup_gpu_environment()
    
    def _setup_gpu_environment(self):
        """Setup and verify GPU environment for dual GPU usage."""
        print_step("Setting up GPU environment for dual GPU usage")
        
        try:
            # Method 1: Try newer XGBoost API (XGBoost >= 2.0)
            gpu_available = False
            gpu_count = 0
            detected_gpus = []
            
            # First, detect available GPUs
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    gpu_lines = [line for line in result.stdout.split('\n') if 'GPU' in line]
                    gpu_count = len(gpu_lines)
                    for i, line in enumerate(gpu_lines):
                        detected_gpus.append(f"GPU {i}: {line.split(':')[1].strip()}")
                    
                    if gpu_count > 0:
                        print(f"   🎯 Detected {gpu_count} GPU(s):")
                        for gpu_info in detected_gpus:
                            print(f"      {gpu_info}")
                        
                        if gpu_count >= 2:
                            print("   🚀 Dual GPU setup available!")
                        gpu_available = True
                    else:
                        print("   📝 No GPUs found via nvidia-smi")
                else:
                    print("   📝 nvidia-smi command failed")
            except Exception as e:
                print(f"   📝 GPU detection failed: {str(e)[:50]}...")
            
            # Test XGBoost GPU compatibility
            if gpu_available:
                try:
                    # Test if we can actually use GPU with XGBoost
                    print("   🧪 Testing XGBoost GPU compatibility...")
                    
                    # Create a small test to verify GPU works
                    import numpy as np
                    test_X = np.random.rand(100, 10)
                    test_y = np.random.rand(100)
                    
                    # Try different GPU configurations
                    gpu_configs_to_test = []
                    
                    if gpu_count >= 2:
                        # Multi-GPU configurations for dual GPU setup
                        gpu_configs_to_test = [
                            {
                                "name": "Multi-GPU (all available)",
                                "params": {
                                    "device": "cuda",
                                    "tree_method": "hist",
                                    "n_estimators": 1,
                                    "verbosity": 0
                                }
                            },
                            {
                                "name": "GPU ID -1 (all GPUs)",
                                "params": {
                                    "device": "cuda:-1",  # Use all available GPUs
                                    "tree_method": "hist", 
                                    "n_estimators": 1,
                                    "verbosity": 0
                                }
                            }
                        ]
                    else:
                        # Single GPU configuration
                        gpu_configs_to_test = [
                            {
                                "name": "Single GPU",
                                "params": {
                                    "device": "cuda:0",
                                    "tree_method": "hist",
                                    "n_estimators": 1,
                                    "verbosity": 0
                                }
                            }
                        ]
                    
                    successful_config = None
                    
                    for config in gpu_configs_to_test:
                        try:
                            print(f"      Testing {config['name']}...")
                            test_model = xgb.XGBRegressor(**config['params'])
                            test_model.fit(test_X, test_y)
                            successful_config = config
                            print(f"      ✅ {config['name']} successful!")
                            break
                        except Exception as config_error:
                            print(f"      📝 {config['name']} failed: {str(config_error)[:50]}...")
                            continue
                    
                    if successful_config:
                        print("   ✅ GPU test successful!")
                        print(f"   🚀 Using configuration: {successful_config['name']}")
                        
                        # Configure XGBoost for optimal multi-GPU usage
                        if gpu_count >= 2:
                            print("   ⚡ Configuring for DUAL GPU acceleration")
                            
                            # Try the best multi-GPU configuration
                            if "cuda:-1" in successful_config['params']['device']:
                                device_setting = "cuda:-1"  # All GPUs
                                print("   🎯 Using device='cuda:-1' for all GPUs")
                            else:
                                device_setting = "cuda"  # Auto-detect all GPUs
                                print("   🎯 Using device='cuda' for auto GPU detection")
                            
                            self.config.XGB_PARAMS.update({
                                "device": device_setting,
                                "tree_method": "hist",
                                "multi_strategy": "one_output_per_tree",  # Better for multi-GPU
                                "n_jobs": 1  # Let XGBoost handle parallelization
                            })
                            
                        else:
                            print("   ⚡ Configuring for single GPU acceleration")
                            self.config.XGB_PARAMS.update({
                                "device": "cuda:0",
                                "tree_method": "hist",
                                "n_jobs": 1
                            })
                        
                        # Remove all potentially conflicting GPU parameters
                        conflicting_params = ["gpu_id", "gpu_hist", "updater"]
                        for param in conflicting_params:
                            self.config.XGB_PARAMS.pop(param, None)
                        
                        # Handle predictor parameter carefully
                        try:
                            # Test if gpu_predictor is supported with current config
                            test_params_with_predictor = self.config.XGB_PARAMS.copy()
                            test_params_with_predictor.update({
                                "predictor": "gpu_predictor",
                                "n_estimators": 1,
                                "verbosity": 0
                            })
                            test_model_predictor = xgb.XGBRegressor(**test_params_with_predictor)
                            test_model_predictor.fit(test_X[:50], test_y[:50])
                            
                            # If successful, keep gpu_predictor
                            self.config.XGB_PARAMS["predictor"] = "gpu_predictor"
                            print("   ✅ GPU predictor enabled")
                        except Exception as predictor_error:
                            # Remove predictor if it causes issues
                            self.config.XGB_PARAMS.pop("predictor", None)
                            print(f"   📝 GPU predictor disabled: {str(predictor_error)[:50]}...")
                        
                        if gpu_count >= 2:
                            print(f"   🎉 DUAL GPU SETUP COMPLETE!")
                            print(f"   ⚡ XGBoost will utilize GPU 0 and GPU 1 automatically")
                            print(f"   💡 Multi-GPU training enabled for faster processing")
                        else:
                            print("   ⚡ Single GPU configuration enabled")
                    else:
                        print("   ❌ All GPU configurations failed")
                        gpu_available = False
                        
                except Exception as gpu_test_error:
                    print(f"   ⚠️  GPU test failed: {str(gpu_test_error)[:100]}...")
                    print("   🔄 Falling back to CPU configuration")
                    gpu_available = False
            
            # Fallback to CPU if GPU not available or failed
            if not gpu_available:
                print("   💻 Configuring for CPU execution")
                self.config.XGB_PARAMS.update({
                    "device": "cpu",
                    "tree_method": "hist",
                    "n_jobs": -1  # Use all CPU cores
                })
                # Remove all GPU-specific parameters
                gpu_params_to_remove = ["gpu_id", "predictor", "gpu_hist", "updater", "multi_strategy"]
                for param in gpu_params_to_remove:
                    self.config.XGB_PARAMS.pop(param, None)
            
            # Display final configuration
            device = self.config.XGB_PARAMS["device"]
            tree_method = self.config.XGB_PARAMS["tree_method"]
            predictor = self.config.XGB_PARAMS.get("predictor", "auto")
            multi_strategy = self.config.XGB_PARAMS.get("multi_strategy", "auto")
            
            print(f"   ⚙️  Final XGBoost configuration:")
            print(f"      device: '{device}'")
            print(f"      tree_method: '{tree_method}'")
            print(f"      predictor: '{predictor}'")
            if multi_strategy != "auto":
                print(f"      multi_strategy: '{multi_strategy}'")
            print(f"      n_jobs: {self.config.XGB_PARAMS['n_jobs']}")
            
            # Performance summary
            if device.startswith("cuda"):
                if gpu_count >= 2:
                    print("   🚀 PERFORMANCE OPTIMIZED FOR DUAL GPU T4 SETUP")
                    print("   ⚡ Both GPUs will be utilized during training")
                    print("   💡 Expect significant speedup over CPU training")
                else:
                    print("   🚀 GPU acceleration enabled")
                print("   📊 XGBoost will automatically manage GPU memory and workload distribution")
            else:
                print("   💻 CPU-only training - consider checking GPU availability")
                
        except Exception as e:
            print_error(f"GPU setup error: {e}")
            print("   🔄 Using default CPU configuration as fallback")
            
            # Safe fallback configuration
            self.config.XGB_PARAMS.update({
                "device": "cpu",
                "tree_method": "hist",
                "n_jobs": -1
            })
            
            # Remove all GPU-specific parameters
            gpu_params_to_remove = ["gpu_id", "predictor", "gpu_hist", "updater", "multi_strategy"]
            for param in gpu_params_to_remove:
                self.config.XGB_PARAMS.pop(param, None)
        
        print(f"   ✅ GPU environment setup completed")
        print(f"   📋 Active device: {self.config.XGB_PARAMS['device']}")
        
        # Show final parameter summary for debugging
        print("   🔍 Key XGBoost parameters for dual GPU:")
        key_params = ["device", "tree_method", "predictor", "multi_strategy", "n_jobs"]
        for param in key_params:
            if param in self.config.XGB_PARAMS:
                print(f"      {param}: {self.config.XGB_PARAMS[param]}")
        
        # Show XGBoost version for debugging
        try:
            print(f"   📦 XGBoost version: {xgb.__version__}")
        except:
            print("   📦 XGBoost version: Unknown")
        
        # Add GPU monitoring tip
        if self.config.XGB_PARAMS["device"].startswith("cuda"):
            print("   💡 Monitor GPU usage with: watch -n 1 nvidia-smi")
    
    def train_regime_models(self, regime_datasets: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Train XGBoost models for each regime with comprehensive monitoring.
        
        Args:
            regime_datasets: Output from DataProcessor.split_data_by_regimes
            
        Returns:
            Dictionary containing predictions and metrics for each regime
        """
        print_header("REGIME-BASED XGBOOST TRAINING")
        print(f"⏰ Training start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Display configuration
        print_step("Training Configuration")
        print(f"   ⚙️  Device: {self.config.XGB_PARAMS['device']}")
        print(f"   🌳 Trees per model: {self.config.XGB_PARAMS['n_estimators']}")
        print(f"   📊 Learning rate: {self.config.XGB_PARAMS['learning_rate']}")
        print(f"   🔄 Cross-validation folds: {self.config.N_FOLDS}")
        print(f"   🎭 Ensemble models: {len(self.config.MODEL_CONFIGS)}")
        print(f"   ⚖️  Time decay factor: {self.config.DECAY_FACTOR}")
        
        regime_predictions = {}
        total_regimes = len(regime_datasets)
        regime_times = []
        overall_cv_summary = {}
        
        print_step("Regime Training Overview")
        print(f"   🎯 Total regimes to process: {total_regimes}")
        print(f"   🔄 Models per regime: {len(self.config.MODEL_CONFIGS)}")
        print(f"   📊 Total models to train: {total_regimes * len(self.config.MODEL_CONFIGS) * self.config.N_FOLDS}")
        
        for regime_idx, (regime_key, regime_data) in enumerate(regime_datasets.items()):
            regime_start_time = time.time()
            
            print_header(f"TRAINING {regime_key.upper()} ({regime_idx + 1}/{total_regimes})")
            print(f"🚀 Regime: {regime_data['regime_id']}")
            print(f"📊 Training samples: {len(regime_data['X_train']):,}")
            print(f"📊 Test samples: {len(regime_data['X_test']):,}")
            
            try:
                predictions = self._train_single_regime(
                    regime_key, regime_data, regime_idx, total_regimes
                )
                regime_predictions[regime_key] = predictions
                
                regime_time = time.time() - regime_start_time
                regime_times.append(regime_time)
                
                # Store CV summary for overall reporting
                overall_cv_summary[regime_key] = {
                    'final_score': predictions.get('pearson_score', 0),
                    'cv_scores': predictions.get('cv_scores', []),
                    'model_cv_scores': predictions.get('model_cv_scores', []),
                    'training_time': regime_time
                }
                
                print_result(f"{regime_key} training time", regime_time/60, ".1f")
                print_result(f"{regime_key} final ensemble score", predictions.get('pearson_score', 0), ".6f")
                
                # Display regime CV summary
                if 'cv_scores' in predictions and predictions['cv_scores']:
                    cv_scores = predictions['cv_scores']
                    print(f"   📊 {regime_key} CV fold scores: {[f'{score:.6f}' for score in cv_scores]}")
                    print(f"   📊 {regime_key} CV average: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}")
                
                # Calculate ETA for remaining regimes
                if regime_idx < total_regimes - 1:
                    avg_time = np.mean(regime_times)
                    remaining = total_regimes - regime_idx - 1
                    eta = str(timedelta(seconds=int(avg_time * remaining)))
                    print(f"   🕐 ETA for remaining {remaining} regimes: {eta}")
                
                # Memory cleanup after each regime
                self.monitor.memory_cleanup()
                
            except Exception as e:
                print_error(f"Training failed for {regime_key}: {str(e)}")
                # Create empty predictions to continue pipeline
                regime_predictions[regime_key] = {
                    'test_predictions': np.array([]),
                    'test_ids': regime_data.get('test_ids', np.array([])),
                    'oof_predictions': np.array([]),
                    'pearson_score': 0.0,
                    'cv_scores': [],
                    'model_cv_scores': [],
                    'error': str(e)
                }
                
                overall_cv_summary[regime_key] = {
                    'final_score': 0.0,
                    'cv_scores': [],
                    'model_cv_scores': [],
                    'training_time': 0.0,
                    'error': str(e)
                }
        
        total_time = self.monitor.checkpoint('training_complete')
        print_header("TRAINING COMPLETED")
        print_result("Total training time", total_time/60, ".1f")
        
        # Comprehensive training summary
        print_step("📊 COMPREHENSIVE TRAINING SUMMARY")
        
        # Overall performance summary
        print("\n   🏆 REGIME PERFORMANCE SUMMARY:")
        regime_scores = []
        for regime_key, summary in overall_cv_summary.items():
            final_score = summary['final_score']
            cv_scores = summary['cv_scores']
            training_time = summary['training_time']
            
            if cv_scores:
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                regime_scores.append(final_score)
                print(f"     📊 {regime_key}:")
                print(f"       🎯 Final ensemble score: {final_score:.6f}")
                print(f"       📈 CV average: {cv_mean:.6f} ± {cv_std:.6f}")
                print(f"       ⏱️  Training time: {training_time:.1f}s")
                print(f"       📋 Fold scores: {[f'{score:.6f}' for score in cv_scores]}")
            else:
                print(f"     ❌ {regime_key}: Failed or no data (score: {final_score:.6f})")
        
        # Overall statistics
        if regime_scores:
            print(f"\n   📊 OVERALL STATISTICS:")
            print(f"     🎯 Average regime score: {np.mean(regime_scores):.6f}")
            print(f"     📊 Score standard deviation: {np.std(regime_scores):.6f}")
            print(f"     🏆 Best regime score: {max(regime_scores):.6f}")
            print(f"     📉 Worst regime score: {min(regime_scores):.6f}")
            print(f"     📈 Score range: {max(regime_scores) - min(regime_scores):.6f}")
        
        # Model configuration performance
        print(f"\n   🤖 MODEL CONFIGURATION PERFORMANCE:")
        for model_idx, model_config in enumerate(self.config.MODEL_CONFIGS):
            model_scores = []
            for regime_key, summary in overall_cv_summary.items():
                model_cv_scores = summary.get('model_cv_scores', [])
                if model_idx < len(model_cv_scores) and model_cv_scores[model_idx]:
                    model_scores.extend(model_cv_scores[model_idx])
            
            if model_scores:
                avg_score = np.mean(model_scores)
                std_score = np.std(model_scores)
                print(f"     📊 {model_config['name']}: {avg_score:.6f} ± {std_score:.6f} ({len(model_scores)} folds)")
        
        # Training efficiency summary
        total_samples_processed = sum(len(regime_data['X_train']) for regime_data in regime_datasets.values())
        if total_time > 0:
            samples_per_second = total_samples_processed / total_time
            print(f"\n   ⚡ TRAINING EFFICIENCY:")
            print(f"     📊 Total samples processed: {total_samples_processed:,}")
            print(f"     🚀 Samples per second: {samples_per_second:.1f}")
            print(f"     ⏱️  Average time per regime: {np.mean(regime_times):.1f}s")
        
        return regime_predictions
    
    def _train_single_regime(self, regime_key: str, regime_data: Dict, 
                           regime_idx: int, total_regimes: int) -> Dict:
        """Train models for a single regime."""
        
        X_train = regime_data['X_train']
        y_train = regime_data['y_train']
        X_test = regime_data['X_test']
        test_ids = regime_data['test_ids']
        
        # Validate data
        if len(X_train) == 0:
            print_warning(f"No training data for {regime_key}")
            return {
                'test_predictions': np.array([]),
                'test_ids': test_ids,
                'oof_predictions': np.array([]),
                'pearson_score': 0.0
            }
        
        print_step("Dataset Overview")
        print(f"   📊 Training samples: {len(X_train):,}")
        print(f"   📊 Test samples: {len(X_test):,}")
        print(f"   📊 Features: {len(X_train.columns):,}")
        print(f"   📊 Target range: [{y_train.min():.4f}, {y_train.max():.4f}]")
        print(f"   📊 Target mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
        
        # Setup cross-validation
        print_step("Setting up cross-validation")
        cv_folds = min(self.config.N_FOLDS, len(X_train) // 10)
        if cv_folds < 2:
            print_warning(f"Insufficient data for CV, using {cv_folds} folds")
            cv_folds = 2
        
        print(f"   🔄 Using {cv_folds} cross-validation folds")
        
        # Create stratified folds
        y_binned = pd.qcut(y_train, q=min(10, len(y_train)//5), 
                          labels=False, duplicates='drop')
        kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                           random_state=self.config.RANDOM_STATE)
        
        # Initialize prediction arrays
        n_models = len(self.config.MODEL_CONFIGS)
        oof_preds = [np.zeros(len(X_train)) for _ in range(n_models)]
        test_preds = [np.zeros(len(X_test)) for _ in range(n_models)]
        
        print_step("Creating time-decay weights")
        # Calculate time weights
        sample_weights = WeightCalculator.create_time_weights(
            len(X_train), self.config.DECAY_FACTOR
        )
        
        # Cross-validation training
        print_step("Starting cross-validation training")
        total_models = cv_folds * n_models
        models_trained = 0
        
        print(f"   🎯 Total models to train: {total_models}")
        
        # Track fold scores for detailed logging
        fold_scores = []
        model_fold_scores = [[] for _ in range(n_models)]
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train, y_binned)):
            fold_start_time = time.time()
            
            print(f"\n   📁 FOLD {fold_idx + 1}/{cv_folds}")
            print(f"     📊 Train: {len(train_idx):,}, Validation: {len(val_idx):,}")
            
            X_val = X_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]
            
            print(f"     📈 Validation target range: [{y_val.min():.4f}, {y_val.max():.4f}]")
            print(f"     📊 Validation target mean: {y_val.mean():.4f}, std: {y_val.std():.4f}")
            
            fold_model_scores = []
            
            for model_idx, model_config in enumerate(self.config.MODEL_CONFIGS):
                models_trained += 1
                model_start_time = time.time()
                
                eta = self.monitor.get_eta(models_trained, total_models)
                print_progress(models_trained, total_models, "models", eta)
                
                print(f"     🤖 Training {model_config['name']}")
                
                # Prepare training data based on model configuration
                train_data = self._prepare_model_data(
                    X_train, y_train, sample_weights, train_idx, model_config
                )
                
                if train_data is None:
                    print_warning("No data available for this model configuration")
                    fold_model_scores.append(0.0)
                    continue
                
                X_fold_train, y_fold_train, fold_weights = train_data
                print(f"       📊 Using {len(X_fold_train):,} samples")
                print(f"       📊 Training target range: [{y_fold_train.min():.4f}, {y_fold_train.max():.4f}]")
                
                # Train model with detailed logging
                try:
                    model, training_metrics = self._train_xgb_model(
                        X_fold_train, y_fold_train, fold_weights, X_val, y_val,
                        fold_idx + 1, model_config['name']
                    )
                    
                    # Generate predictions
                    val_preds = model.predict(X_val)
                    oof_preds[model_idx][val_idx] = val_preds
                    
                    if len(X_test) > 0:
                        test_preds[model_idx] += model.predict(X_test)
                    
                    # Calculate fold validation score
                    fold_val_score = pearsonr(y_val, val_preds)[0]
                    fold_model_scores.append(fold_val_score)
                    model_fold_scores[model_idx].append(fold_val_score)
                    
                    training_time = time.time() - model_start_time
                    print(f"       ✅ Completed in {training_time:.1f}s")
                    print(f"       📈 Best training score: {training_metrics['best_score']:.6f}")
                    print(f"       📈 Validation score: {fold_val_score:.6f}")
                    print(f"       🌳 Best iteration: {training_metrics['best_iteration']}")
                    print(f"       📊 Prediction range: [{val_preds.min():.4f}, {val_preds.max():.4f}]")
                    
                except Exception as e:
                    print_error(f"Model training failed: {e}")
                    fold_model_scores.append(0.0)
                    model_fold_scores[model_idx].append(0.0)
                    continue
            
            # Calculate and display fold summary
            fold_avg_score = np.mean(fold_model_scores) if fold_model_scores else 0.0
            fold_scores.append(fold_avg_score)
            
            fold_time = time.time() - fold_start_time
            print(f"     ⏱️  Fold completed in {fold_time:.1f}s")
            print(f"     🏆 Fold {fold_idx + 1} average score: {fold_avg_score:.6f}")
            print(f"     📊 Fold {fold_idx + 1} model scores: {[f'{score:.6f}' for score in fold_model_scores]}")
        
        # Average test predictions across folds
        print_step("Averaging predictions across folds")
        for i in range(n_models):
            if cv_folds > 0:
                test_preds[i] /= cv_folds
        
        # Display detailed cross-validation summary
        print_step("Cross-validation summary")
        print(f"   📊 Fold scores: {[f'{score:.6f}' for score in fold_scores]}")
        print(f"   📊 Average CV score: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
        
        print("   📊 Model-wise CV scores:")
        for model_idx, model_config in enumerate(self.config.MODEL_CONFIGS):
            model_scores = model_fold_scores[model_idx]
            if model_scores:
                avg_score = np.mean(model_scores)
                std_score = np.std(model_scores)
                print(f"     {model_config['name']}: {avg_score:.6f} ± {std_score:.6f}")
                print(f"       Fold scores: {[f'{score:.6f}' for score in model_scores]}")
        
        # Create ensemble
        print_step("Creating ensemble predictions")
        ensemble_result = self._create_ensemble(y_train, oof_preds, test_preds)
        
        return {
            'test_predictions': ensemble_result['test_predictions'],
            'test_ids': test_ids,
            'oof_predictions': ensemble_result['oof_predictions'],
            'pearson_score': ensemble_result['pearson_score'],
            'individual_scores': ensemble_result['individual_scores'],
            'cv_scores': fold_scores,
            'model_cv_scores': model_fold_scores
        }
    
    def _prepare_model_data(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          sample_weights: np.ndarray, train_idx: np.ndarray,
                          model_config: Dict) -> Optional[Tuple]:
        """Prepare training data based on model configuration."""
        
        if model_config['percent'] == 1.0:
            # Use full dataset
            X_fold = X_train.iloc[train_idx]
            y_fold = y_train.iloc[train_idx]
            weights_fold = sample_weights[train_idx]
        else:
            # Use recent portion of data
            cutoff_idx = int(len(X_train) * (1 - model_config['percent']))
            recent_train_idx = train_idx[train_idx >= cutoff_idx]
            
            if len(recent_train_idx) == 0:
                return None
            
            # Adjust indices for recent data subset
            recent_adjusted_idx = recent_train_idx - cutoff_idx
            X_recent = X_train.iloc[cutoff_idx:].reset_index(drop=True)
            y_recent = y_train.iloc[cutoff_idx:].reset_index(drop=True)
            
            X_fold = X_recent.iloc[recent_adjusted_idx]
            y_fold = y_recent.iloc[recent_adjusted_idx]
            
            # Calculate weights for recent data
            recent_weights = WeightCalculator.create_time_weights(
                len(X_recent), self.config.DECAY_FACTOR
            )
            weights_fold = recent_weights[recent_adjusted_idx]
        
        return X_fold, y_fold, weights_fold
    
    def _train_xgb_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                        sample_weights: np.ndarray, X_val: pd.DataFrame, 
                        y_val: pd.Series, fold_num: int, model_name: str) -> Tuple[xgb.XGBRegressor, Dict]:
        """Train a single XGBoost model with detailed iteration progress."""
        
        print(f"       🚀 Starting training for Fold {fold_num} - {model_name}")
        print(f"       📊 Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
        
        # Create model with updated parameters for better logging
        model_params = self.config.XGB_PARAMS.copy()
        model_params['verbosity'] = 1  # Enable some logging
        
        model = xgb.XGBRegressor(**model_params)
        
        # Custom callback for iteration progress
        class IterationCallback:
            def __init__(self, fold_num, model_name, print_freq=50):
                self.fold_num = fold_num
                self.model_name = model_name
                self.print_freq = print_freq
                self.iteration = 0
                self.best_score = float('inf')
                self.best_iteration = 0
                
            def __call__(self, env):
                self.iteration += 1
                
                if hasattr(env, 'evaluation_result_list') and env.evaluation_result_list:
                    current_score = env.evaluation_result_list[-1][1]
                    
                    # Track best score
                    if current_score < self.best_score:
                        self.best_score = current_score
                        self.best_iteration = self.iteration
                        improvement = "🔥"
                    else:
                        improvement = "  "
                    
                    # Print progress at specified frequency
                    if self.iteration % self.print_freq == 0 or self.iteration <= 10:
                        print(f"         {improvement} Iter {self.iteration:4d}: "
                              f"val_score={current_score:.6f} "
                              f"(best: {self.best_score:.6f} @ {self.best_iteration})")
                    
                    # Print first few and last few iterations
                    elif self.iteration <= 5:
                        print(f"         {improvement} Iter {self.iteration:4d}: "
                              f"val_score={current_score:.6f}")
        
        # Create callback instance
        callback = IterationCallback(fold_num, model_name)
        
        print(f"       🎯 Training with early stopping (patience: 25)")
        training_start = time.time()
        
        # Training with early stopping and callback
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=['train', 'val'],
            early_stopping_rounds=25,
            verbose=False,  # We handle verbosity with our callback
            callbacks=[callback]
        )
        
        training_time = time.time() - training_start
        
        # Extract final training metrics
        best_iteration = model.get_booster().best_iteration
        best_score = model.best_score
        
        print(f"       ⏱️  Training completed in {training_time:.1f}s")
        print(f"       🏆 Final best iteration: {best_iteration}")
        print(f"       🏆 Final best score: {best_score:.6f}")
        
        # Additional model information
        booster = model.get_booster()
        feature_importance = model.feature_importances_[:5]  # Top 5 features
        top_features = X_train.columns[:5]
        
        print(f"       📊 Top 5 feature importance: {dict(zip(top_features, feature_importance))}")
        
        metrics = {
            'best_iteration': best_iteration,
            'best_score': best_score,
            'training_time': training_time,
            'total_iterations': len(model.evals_result_['val']['rmse']) if hasattr(model, 'evals_result_') else 0
        }
        
        return model, metrics
    
    def _create_ensemble(self, y_true: pd.Series, oof_preds: List[np.ndarray],
                        test_preds: List[np.ndarray]) -> Dict:
        """Create weighted ensemble from individual model predictions."""
        
        print("     🎭 Creating weighted ensemble...")
        
        # Calculate individual model scores
        individual_scores = []
        for i, preds in enumerate(oof_preds):
            if len(preds) > 0 and np.any(preds != 0):
                score = pearsonr(y_true, preds)[0]
                individual_scores.append(max(0, score))  # Ensure non-negative
            else:
                individual_scores.append(0.0)
        
        print("     📊 Individual model scores:")
        for i, (config, score) in enumerate(zip(self.config.MODEL_CONFIGS, individual_scores)):
            print(f"       {config['name']}: {score:.6f}")
        
        # Create weighted ensemble
        total_score = sum(individual_scores)
        if total_score > 0:
            weights = [score / total_score for score in individual_scores]
            
            print("     ⚖️  Model weights:")
            for i, (config, weight) in enumerate(zip(self.config.MODEL_CONFIGS, weights)):
                print(f"       {config['name']}: {weight:.4f}")
            
            ensemble_oof = np.zeros(len(y_true))
            ensemble_test = np.zeros(len(test_preds[0]) if test_preds[0] is not None else 0)
            
            for i, weight in enumerate(weights):
                ensemble_oof += weight * oof_preds[i]
                if len(test_preds[i]) > 0:
                    ensemble_test += weight * test_preds[i]
            
            ensemble_score = pearsonr(y_true, ensemble_oof)[0] if len(ensemble_oof) > 0 else 0.0
        else:
            # Fallback to simple average
            ensemble_oof = np.mean(oof_preds, axis=0) if oof_preds else np.array([])
            ensemble_test = np.mean(test_preds, axis=0) if test_preds else np.array([])
            ensemble_score = 0.0
            print_warning("Using simple average ensemble (no positive scores)")
        
        print_result("Ensemble score", ensemble_score)
        
        return {
            'oof_predictions': ensemble_oof,
            'test_predictions': ensemble_test,
            'pearson_score': ensemble_score,
            'individual_scores': individual_scores
        }

class SubmissionCreator:
    """Create and validate final submission file."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        print_step("Submission Creator Initialized")
    
    def create_submission(self, regime_predictions: Dict[str, Dict], 
                         test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create final submission by combining regime predictions.
        
        Args:
            regime_predictions: Results from RegimeModelTrainer
            test_df: Original test dataframe
            
        Returns:
            Final submission dataframe
        """
        print_header("CREATING FINAL SUBMISSION")
        
        # Initialize submission
        print_step("Initializing submission template")
        submission = pd.DataFrame({
            self.config.ID_COLUMN: test_df[self.config.ID_COLUMN],
            'prediction': 0.0
        })
        print(f"   📊 Template created with {len(submission):,} rows")
        
        total_filled = 0
        regime_stats = {}
        
        # Fill predictions for each regime
        print_step("Filling regime-specific predictions")
        for regime_key, regime_data in regime_predictions.items():
            test_ids = regime_data['test_ids']
            predictions = regime_data['test_predictions']
            
            if len(test_ids) > 0 and len(predictions) > 0:
                mask = submission[self.config.ID_COLUMN].isin(test_ids)
                matched_count = mask.sum()
                
                submission.loc[mask, 'prediction'] = predictions
                total_filled += matched_count
                
                regime_stats[regime_key] = {
                    'count': matched_count,
                    'mean_pred': predictions.mean(),
                    'std_pred': predictions.std(),
                    'score': regime_data.get('pearson_score', 0.0)
                }
                
                print(f"   ✅ {regime_key}: {matched_count:,} predictions filled")
                print(f"     📈 Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")
                print(f"     🏆 Score: {regime_data.get('pearson_score', 0.0):.6f}")
        
        # Validation and statistics
        print_step("Submission validation and statistics")
        coverage = (total_filled / len(submission)) * 100
        
        print(f"   📊 Total samples: {len(submission):,}")
        print(f"   ✅ Filled predictions: {total_filled:,} ({coverage:.1f}%)")
        print(f"   📈 Prediction range: [{submission['prediction'].min():.4f}, {submission['prediction'].max():.4f}]")
        print(f"   📊 Mean prediction: {submission['prediction'].mean():.4f}")
        print(f"   📊 Std prediction: {submission['prediction'].std():.4f}")
        
        # Check for any missing predictions
        missing_count = (submission['prediction'] == 0).sum()
        if missing_count > 0:
            print_warning(f"{missing_count:,} predictions remain as zero")
        
        # Save submission
        print_step("Saving submission file")
        submission.to_csv(self.config.SUBMISSION_FILENAME, index=False)
        print(f"   💾 File saved: {self.config.SUBMISSION_FILENAME}")
        
        print("✨ Submission creation completed successfully!")
        
        return submission

class RegimePipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.monitor = PerformanceMonitor()
        
        print_header("REGIME PIPELINE INITIALIZATION")
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.trainer = RegimeModelTrainer(self.config)
        self.submission_creator = SubmissionCreator(self.config)
        
        print("✨ Pipeline components initialized successfully!")
    
    def run(self) -> pd.DataFrame:
        """
        Execute the complete regime-based modeling pipeline.
        
        Returns:
            Final submission dataframe
        """
        print_header("REGIME-BASED XGBOOST PIPELINE EXECUTION")
        print(f"⏰ Pipeline start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Load and merge data
            print("\n🚀 STEP 1: DATA LOADING AND MERGING")
            train_df, test_df = self.data_processor.load_and_merge_datasets()
            
            # Step 2: Split by regimes
            print("\n🚀 STEP 2: DATA SPLITTING BY REGIMES")
            regime_datasets = self.data_processor.split_data_by_regimes(train_df, test_df)
            
            # Step 3: Train regime-specific models
            print("\n🚀 STEP 3: REGIME-SPECIFIC MODEL TRAINING")
            regime_predictions = self.trainer.train_regime_models(regime_datasets)
            
            # Step 4: Create submission
            print("\n🚀 STEP 4: SUBMISSION CREATION")
            submission = self.submission_creator.create_submission(regime_predictions, test_df)
            
            # Final summary
            total_time = self.monitor.checkpoint('pipeline_complete')
            
            print_header("PIPELINE EXECUTION COMPLETED")
            print_result("Total execution time", total_time/60, ".1f")
            
            # Performance summary
            print_step("Final Performance Summary")
            best_regime = max(regime_predictions.items(), 
                            key=lambda x: x[1].get('pearson_score', 0))
            
            for regime_key, regime_data in regime_predictions.items():
                score = regime_data.get('pearson_score', 0)
                print(f"   📊 {regime_key}: {score:.6f}")
            
            print(f"\n   🏆 Best performing regime: {best_regime[0]}")
            print(f"   📈 Best score: {best_regime[1].get('pearson_score', 0):.6f}")
            
            print(f"\n   📁 Submission file: {self.config.SUBMISSION_FILENAME}")
            print("   ✨ Ready for Kaggle submission!")
            
            return submission
            
        except Exception as e:
            print_error(f"Pipeline execution failed: {str(e)}")
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise

def main():
    """
    Main execution function.
    
    FLEXIBLE DATASET CONFIGURATION EXAMPLE:
    ======================================
    
    To use custom datasets with specific feature columns, create your config like this:
    
    # Example: Custom dataset configuration
    custom_config = ModelConfig()
    
    # Define train datasets
    custom_config.TRAIN_DATASETS = [
        DatasetConfig(
            file_path="/path/to/main_features.csv",
            feature_columns=["price", "volume", "volatility"],  # Only these features
            merge_columns=["timestamp"],
            dataset_name="Main Features",
            is_required=True
        ),
        DatasetConfig(
            file_path="/path/to/technical_indicators.csv", 
            feature_columns=[],                              # Keep all columns
            merge_columns=["timestamp"],
            dataset_name="Technical Indicators",
            is_required=False                               # Optional
        ),
        DatasetConfig(
            file_path="/path/to/regime_data.csv",
            feature_columns=["predicted_regime_Est"],
            merge_columns=["timestamp"],
            dataset_name="Regime Data",
            is_required=True
        )
    ]
    
    # Define test datasets  
    custom_config.TEST_DATASETS = [
        DatasetConfig(
            file_path="/path/to/test_features.csv",
            feature_columns=["feature_1", "feature_2", "feature_3"],  # Same features as train
            merge_columns=["ID"],
            dataset_name="Test Features", 
            is_required=True
        ),
        DatasetConfig(
            file_path="/path/to/test_regimes.csv",
            feature_columns=["predicted_regime_Est"], 
            merge_columns=["ID"],
            dataset_name="Test Regimes",
            is_required=True
        )
    ]
    
    # Then run: pipeline = RegimePipeline(custom_config)
    """
    
    print_header("REGIME-BASED XGBOOST PIPELINE")
    print("🎯 Optimized for GPU T4 x 2 Environment")
    print("📊 Crypto Market Prediction with Regime-Specific Models")
    print("🔧 Now supports flexible multiple dataset configurations!")
    
    # Create configuration
    print_step("Creating configuration")
    config = ModelConfig()
    
    # Override paths if running locally (for testing)
    if not os.path.exists(config.TRAIN_DATASETS[0].file_path):
        print_warning("Kaggle paths not found - may be running in local environment")
        # Add local path overrides here if needed for testing
    
    try:
        # Run pipeline
        print_step("Initializing and running pipeline")
        pipeline = RegimePipeline(config)
        submission = pipeline.run()
        
        print_header("EXECUTION SUCCESSFUL")
        print("🎉 Pipeline completed successfully!")
        print("📁 Submission file ready for Kaggle upload")
        print("✨ Good luck with your competition!")
        
        return submission
        
    except Exception as e:
        print_header("EXECUTION FAILED")
        print_error(f"Execution failed: {str(e)}")
        logger.error(f"❌ Execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Execute pipeline
    final_submission = main()
