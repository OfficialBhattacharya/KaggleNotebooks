import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
import warnings
from typing import Union, Optional, Dict, Any, Tuple, List
import logging
from pathlib import Path
import gc
import psutil
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    A comprehensive model explainer class that supports XGBoost, CatBoost, and LightGBM models
    with both SHAP and LIME explanations. Includes memory optimization through data sampling.
    
    Features:
    - Load models from joblib or yaml files
    - Support for XGBoost, CatBoost, and LightGBM
    - SHAP explanations (TreeExplainer, Explainer)
    - LIME explanations (tabular)
    - Memory optimization through sampling
    - Automatic task type detection (regression/classification)
    - Comprehensive visualization
    - Memory usage monitoring
    """
    
    def __init__(self, 
                 model_path: str = None,
                 model: Any = None,
                 sample_fraction: float = 1.0,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize the ModelExplainer.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to the saved model file (joblib or yaml)
        model : object, optional
            Pre-loaded model object
        sample_fraction : float, default=1.0
            Fraction of data to use for explanations (0.0 to 1.0)
        random_state : int, default=42
            Random state for reproducibility
        verbose : bool, default=True
            Whether to print detailed information
        """
        self.model_path = model_path
        self.model = model
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize attributes
        self.model_type = None
        self.task_type = None
        self.feature_names = None
        self.class_names = None
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Data storage
        self.X_train_sample = None
        self.X_test_sample = None
        self.y_train_sample = None
        self.y_test_sample = None
        self.fitted_values_sample = None
        self.predictions_sample = None
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
        elif model:
            self.model = model
            self._detect_model_type()
    
    def _print_info(self, message: str):
        """Print information if verbose is True."""
        if self.verbose:
            print(message)
    
    def _get_memory_usage(self) -> str:
        """Get current memory usage."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f} MB"
    
    def load_model(self, model_path: str):
        """
        Load model from file (joblib or yaml).
        
        Parameters:
        -----------
        model_path : str
            Path to the model file
        """
        self._print_info(f"🔄 Loading model from: {model_path}")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            if model_path.suffix.lower() in ['.pkl', '.joblib']:
                # Load joblib/pickle file
                self.model = joblib.load(model_path)
                self._print_info(f"✅ Model loaded from joblib file")
                
            elif model_path.suffix.lower() in ['.yaml', '.yml']:
                # Load YAML file (for CatBoost)
                try:
                    from catboost import CatBoostRegressor, CatBoostClassifier
                    
                    # Try to determine if it's regression or classification
                    # This is a simplified approach - you might need to adjust based on your YAML structure
                    with open(model_path, 'r') as f:
                        yaml_content = yaml.safe_load(f)
                    
                    # Attempt to load as classifier first, then regressor
                    try:
                        self.model = CatBoostClassifier()
                        self.model.load_model(str(model_path))
                        self._print_info(f"✅ CatBoost Classifier loaded from YAML file")
                    except:
                        self.model = CatBoostRegressor()
                        self.model.load_model(str(model_path))
                        self._print_info(f"✅ CatBoost Regressor loaded from YAML file")
                        
                except ImportError:
                    raise ImportError("CatBoost not installed. Install with: pip install catboost")
                    
            else:
                raise ValueError(f"Unsupported file format: {model_path.suffix}")
            
            self._detect_model_type()
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _detect_model_type(self):
        """Detect the type of model (XGBoost, CatBoost, LightGBM) and task type."""
        model_class = type(self.model).__name__
        
        if 'XGB' in model_class or 'xgboost' in str(type(self.model)).lower():
            self.model_type = 'xgboost'
        elif 'CatBoost' in model_class or 'catboost' in str(type(self.model)).lower():
            self.model_type = 'catboost'
        elif 'LGB' in model_class or 'lightgbm' in str(type(self.model)).lower():
            self.model_type = 'lightgbm'
        else:
            self.model_type = 'unknown'
            logger.warning(f"Unknown model type: {model_class}")
        
        # Detect task type
        if hasattr(self.model, 'predict_proba'):
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
        
        self._print_info(f"📊 Model Type: {self.model_type.title()}")
        self._print_info(f"🎯 Task Type: {self.task_type.title()}")
    
    def prepare_data(self, 
                     X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_train: pd.Series = None,
                     y_test: pd.Series = None,
                     fitted_values: np.ndarray = None,
                     predictions: np.ndarray = None):
        """
        Prepare and sample data for explanations.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame
            Test features
        y_train : pd.Series, optional
            Training targets
        y_test : pd.Series, optional
            Test targets
        fitted_values : np.ndarray, optional
            Model predictions on training data
        predictions : np.ndarray, optional
            Model predictions on test data
        """
        self._print_info(f"\n🔄 Preparing data for explanations...")
        self._print_info(f"💾 Memory usage before: {self._get_memory_usage()}")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Clean data (handle inf and nan values)
        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy()
        
        # Replace infinite values with NaN
        X_train_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaN values with mean
        X_train_clean.fillna(X_train_clean.mean(), inplace=True)
        X_test_clean.fillna(X_train_clean.mean(), inplace=True)  # Use train mean for test
        
        # Sample data if needed
        if self.sample_fraction < 1.0:
            train_sample_size = int(len(X_train_clean) * self.sample_fraction)
            test_sample_size = int(len(X_test_clean) * self.sample_fraction)
            
            # Sample training data
            train_indices = np.random.RandomState(self.random_state).choice(
                len(X_train_clean), size=train_sample_size, replace=False
            )
            self.X_train_sample = X_train_clean.iloc[train_indices].copy()
            
            if y_train is not None:
                self.y_train_sample = y_train.iloc[train_indices].copy()
            if fitted_values is not None:
                self.fitted_values_sample = fitted_values[train_indices]
            
            # Sample test data
            test_indices = np.random.RandomState(self.random_state + 1).choice(
                len(X_test_clean), size=test_sample_size, replace=False
            )
            self.X_test_sample = X_test_clean.iloc[test_indices].copy()
            
            if y_test is not None:
                self.y_test_sample = y_test.iloc[test_indices].copy()
            if predictions is not None:
                self.predictions_sample = predictions[test_indices]
            
            self._print_info(f"📊 Sampled {train_sample_size:,} training samples ({self.sample_fraction:.1%})")
            self._print_info(f"📊 Sampled {test_sample_size:,} test samples ({self.sample_fraction:.1%})")
            
        else:
            self.X_train_sample = X_train_clean.copy()
            self.X_test_sample = X_test_clean.copy()
            self.y_train_sample = y_train.copy() if y_train is not None else None
            self.y_test_sample = y_test.copy() if y_test is not None else None
            self.fitted_values_sample = fitted_values.copy() if fitted_values is not None else None
            self.predictions_sample = predictions.copy() if predictions is not None else None
            
            self._print_info(f"📊 Using full dataset: {len(X_train_clean):,} training, {len(X_test_clean):,} test samples")
        
        # Set class names for classification
        if self.task_type == 'classification':
            if self.y_train_sample is not None:
                unique_classes = sorted(self.y_train_sample.unique())
                self.class_names = [str(cls) for cls in unique_classes]
            else:
                self.class_names = ['0', '1']  # Default binary classification
        
        self._print_info(f"📋 Features: {len(self.feature_names)}")
        if self.task_type == 'classification':
            self._print_info(f"🏷️  Classes: {self.class_names}")
        
        self._print_info(f"💾 Memory usage after: {self._get_memory_usage()}")
        
        # Force garbage collection
        gc.collect()
    
    def _get_prediction_function(self):
        """Get the appropriate prediction function based on task type."""
        if self.task_type == 'classification':
            if hasattr(self.model, 'predict_proba'):
                return lambda x: self.model.predict_proba(x)
            else:
                # Fallback for models without predict_proba
                return lambda x: np.column_stack([1 - self.model.predict(x), self.model.predict(x)])
        else:
            return lambda x: self.model.predict(x)
    
    def setup_shap_explainer(self, explainer_type: str = 'auto'):
        """
        Set up SHAP explainer.
        
        Parameters:
        -----------
        explainer_type : str, default='auto'
            Type of SHAP explainer: 'tree', 'explainer', or 'auto'
        """
        try:
            import shap
            
            self._print_info(f"\n🔄 Setting up SHAP explainer...")
            
            if explainer_type == 'auto':
                # Choose explainer based on model type
                if self.model_type in ['xgboost', 'catboost', 'lightgbm']:
                    explainer_type = 'tree'
                else:
                    explainer_type = 'explainer'
            
            if explainer_type == 'tree':
                # Tree explainer for tree-based models
                self.shap_explainer = shap.TreeExplainer(self.model)
                self._print_info(f"✅ SHAP TreeExplainer initialized")
                
            elif explainer_type == 'explainer':
                # General explainer
                background_data = self.X_train_sample.values[:min(100, len(self.X_train_sample))]
                self.shap_explainer = shap.Explainer(self._get_prediction_function(), background_data)
                self._print_info(f"✅ SHAP Explainer initialized with {len(background_data)} background samples")
            
            return True
            
        except ImportError:
            logger.error("SHAP not installed. Install with: pip install shap")
            self._print_info("❌ SHAP not available. Install with: pip install shap")
            return False
        except Exception as e:
            logger.error(f"Error setting up SHAP explainer: {str(e)}")
            self._print_info(f"❌ Error setting up SHAP explainer: {str(e)}")
            return False
    
    def setup_lime_explainer(self, kernel_width: float = 5.0):
        """
        Set up LIME explainer.
        
        Parameters:
        -----------
        kernel_width : float, default=5.0
            Kernel width for LIME explainer
        """
        try:
            import lime
            import lime.lime_tabular
            
            self._print_info(f"\n🔄 Setting up LIME explainer...")
            
            # Use training data for LIME background
            background_data = self.X_train_sample.values
            
            if self.task_type == 'classification':
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    background_data,
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    kernel_width=kernel_width,
                    mode='classification'
                )
            else:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    background_data,
                    feature_names=self.feature_names,
                    kernel_width=kernel_width,
                    mode='regression'
                )
            
            self._print_info(f"✅ LIME explainer initialized with {len(background_data)} background samples")
            return True
            
        except ImportError:
            logger.error("LIME not installed. Install with: pip install lime")
            self._print_info("❌ LIME not available. Install with: pip install lime")
            return False
        except Exception as e:
            logger.error(f"Error setting up LIME explainer: {str(e)}")
            self._print_info(f"❌ Error setting up LIME explainer: {str(e)}")
            return False
    
    def explain_instance_shap(self, 
                              instance_index: int = None,
                              instance_data: np.ndarray = None,
                              data_source: str = 'test',
                              plot: bool = True) -> Dict[str, Any]:
        """
        Explain a single instance using SHAP.
        
        Parameters:
        -----------
        instance_index : int, optional
            Index of instance to explain
        instance_data : np.ndarray, optional
            Direct instance data to explain
        data_source : str, default='test'
            Source of data: 'train' or 'test'
        plot : bool, default=True
            Whether to create plots
        
        Returns:
        --------
        dict : Explanation results
        """
        if self.shap_explainer is None:
            self._print_info("❌ SHAP explainer not initialized. Call setup_shap_explainer() first.")
            return {}
        
        try:
            import shap
            
            # Get instance data
            if instance_data is not None:
                if len(instance_data.shape) == 1:
                    instance_data = instance_data.reshape(1, -1)
                instance = instance_data
                self._print_info(f"🔍 Explaining provided instance")
            else:
                if data_source == 'test':
                    data = self.X_test_sample
                else:
                    data = self.X_train_sample
                
                if instance_index is None:
                    instance_index = 0
                
                instance = data.iloc[[instance_index]].values
                self._print_info(f"🔍 Explaining {data_source} instance {instance_index}")
            
            # Calculate SHAP values
            self._print_info(f"⚡ Calculating SHAP values...")
            shap_values = self.shap_explainer.shap_values(instance)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_to_plot = shap_values
            
            # Create plots if requested
            if plot:
                self._print_info(f"🎨 Creating SHAP plots...")
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'SHAP Explanation - Instance {instance_index if instance_index is not None else "Custom"}', 
                           fontsize=16, fontweight='bold')
                
                # Waterfall plot
                plt.subplot(2, 2, 1)
                try:
                    if hasattr(shap, 'waterfall_plot'):
                        shap.waterfall_plot(
                            shap.Explanation(values=shap_values_to_plot[0], 
                                           base_values=self.shap_explainer.expected_value,
                                           data=instance[0],
                                           feature_names=self.feature_names),
                            show=False
                        )
                    else:
                        # Fallback for older SHAP versions
                        shap.plots._waterfall.waterfall_legacy(
                            self.shap_explainer.expected_value, 
                            shap_values_to_plot[0], 
                            instance[0],
                            feature_names=self.feature_names,
                            show=False
                        )
                except:
                    # Simple bar plot fallback
                    feature_importance = shap_values_to_plot[0]
                    sorted_idx = np.argsort(np.abs(feature_importance))[-10:]
                    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
                    plt.yticks(range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx])
                    plt.xlabel('SHAP Value')
                plt.title('SHAP Waterfall Plot')
                
                # Force plot
                plt.subplot(2, 2, 2)
                try:
                    shap.force_plot(
                        self.shap_explainer.expected_value,
                        shap_values_to_plot[0],
                        instance[0],
                        feature_names=self.feature_names,
                        matplotlib=True,
                        show=False
                    )
                except:
                    # Simple bar plot fallback
                    feature_importance = shap_values_to_plot[0]
                    plt.bar(range(len(feature_importance)), feature_importance)
                    plt.xticks(range(len(feature_importance)), self.feature_names, rotation=45)
                    plt.ylabel('SHAP Value')
                plt.title('SHAP Force Plot')
                
                # Bar plot
                plt.subplot(2, 2, 3)
                feature_importance = np.abs(shap_values_to_plot[0])
                sorted_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
                
                plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
                plt.yticks(range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx])
                plt.xlabel('|SHAP Value|')
                plt.title('Top 10 Feature Importance')
                
                # Feature values
                plt.subplot(2, 2, 4)
                feature_values = instance[0][sorted_idx]
                colors = ['red' if shap_values_to_plot[0][i] < 0 else 'blue' for i in sorted_idx]
                
                plt.barh(range(len(sorted_idx)), feature_values, color=colors, alpha=0.7)
                plt.yticks(range(len(sorted_idx)), [self.feature_names[i] for i in sorted_idx])
                plt.xlabel('Feature Value')
                plt.title('Feature Values (Red: Negative SHAP, Blue: Positive SHAP)')
                
                plt.tight_layout()
                plt.show()
            
            # Prepare results
            results = {
                'shap_values': shap_values,
                'expected_value': self.shap_explainer.expected_value,
                'instance_data': instance,
                'feature_names': self.feature_names,
                'feature_importance': dict(zip(self.feature_names, np.abs(shap_values_to_plot[0]))),
                'top_features': sorted(
                    zip(self.feature_names, shap_values_to_plot[0]), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:10]
            }
            
            self._print_info(f"✅ SHAP explanation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {str(e)}")
            self._print_info(f"❌ Error in SHAP explanation: {str(e)}")
            return {}
    
    def explain_instance_lime(self, 
                              instance_index: int = None,
                              instance_data: np.ndarray = None,
                              data_source: str = 'test',
                              num_features: int = 15,
                              plot: bool = True) -> Dict[str, Any]:
        """
        Explain a single instance using LIME.
        
        Parameters:
        -----------
        instance_index : int, optional
            Index of instance to explain
        instance_data : np.ndarray, optional
            Direct instance data to explain
        data_source : str, default='test'
            Source of data: 'train' or 'test'
        num_features : int, default=15
            Number of features to include in explanation
        plot : bool, default=True
            Whether to create plots
        
        Returns:
        --------
        dict : Explanation results
        """
        if self.lime_explainer is None:
            self._print_info("❌ LIME explainer not initialized. Call setup_lime_explainer() first.")
            return {}
        
        try:
            # Get instance data
            if instance_data is not None:
                if len(instance_data.shape) > 1:
                    instance_data = instance_data.flatten()
                instance = instance_data
                self._print_info(f"🔍 Explaining provided instance with LIME")
            else:
                if data_source == 'test':
                    data = self.X_test_sample
                else:
                    data = self.X_train_sample
                
                if instance_index is None:
                    instance_index = 0
                
                instance = data.iloc[instance_index].values
                self._print_info(f"🔍 Explaining {data_source} instance {instance_index} with LIME")
            
            # Get prediction function
            predict_fn = self._get_prediction_function()
            
            # Generate explanation
            self._print_info(f"⚡ Generating LIME explanation...")
            explanation = self.lime_explainer.explain_instance(
                instance, 
                predict_fn, 
                num_features=num_features
            )
            
            # Create plots if requested
            if plot:
                self._print_info(f"🎨 Creating LIME plots...")
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(f'LIME Explanation - Instance {instance_index if instance_index is not None else "Custom"}', 
                           fontsize=16, fontweight='bold')
                
                # LIME plot in notebook
                try:
                    explanation.as_pyplot_figure(label=1 if self.task_type == 'classification' else 0)
                    plt.subplot(1, 2, 1)
                    plt.title('LIME Feature Importance')
                except:
                    plt.subplot(1, 2, 1)
                    plt.text(0.5, 0.5, 'LIME plot not available', ha='center', va='center')
                    plt.title('LIME Feature Importance')
                
                # Custom bar plot
                plt.subplot(1, 2, 2)
                lime_values = explanation.as_list()
                features, values = zip(*lime_values)
                colors = ['red' if v < 0 else 'blue' for v in values]
                
                plt.barh(range(len(features)), values, color=colors, alpha=0.7)
                plt.yticks(range(len(features)), features)
                plt.xlabel('LIME Importance')
                plt.title('LIME Feature Contributions')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
            
            # Prepare results
            lime_list = explanation.as_list()
            results = {
                'explanation': explanation,
                'feature_importance': dict(lime_list),
                'top_features': lime_list,
                'instance_data': instance,
                'prediction': predict_fn(instance.reshape(1, -1))[0] if self.task_type == 'regression' else predict_fn(instance.reshape(1, -1))[0][1],
                'feature_names': self.feature_names
            }
            
            self._print_info(f"✅ LIME explanation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in LIME explanation: {str(e)}")
            self._print_info(f"❌ Error in LIME explanation: {str(e)}")
            return {}
    
    def compare_explanations(self, 
                             instance_index: int = None,
                             instance_data: np.ndarray = None,
                             data_source: str = 'test',
                             num_features: int = 10) -> Dict[str, Any]:
        """
        Compare SHAP and LIME explanations for the same instance.
        
        Parameters:
        -----------
        instance_index : int, optional
            Index of instance to explain
        instance_data : np.ndarray, optional
            Direct instance data to explain
        data_source : str, default='test'
            Source of data: 'train' or 'test'
        num_features : int, default=10
            Number of top features to compare
        
        Returns:
        --------
        dict : Comparison results
        """
        self._print_info(f"\n🔄 Comparing SHAP and LIME explanations...")
        
        # Get SHAP explanation
        shap_results = self.explain_instance_shap(
            instance_index=instance_index,
            instance_data=instance_data,
            data_source=data_source,
            plot=False
        )
        
        # Get LIME explanation
        lime_results = self.explain_instance_lime(
            instance_index=instance_index,
            instance_data=instance_data,
            data_source=data_source,
            num_features=num_features,
            plot=False
        )
        
        if not shap_results or not lime_results:
            self._print_info("❌ Could not generate both explanations for comparison")
            return {}
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'SHAP vs LIME Comparison - Instance {instance_index if instance_index is not None else "Custom"}', 
                     fontsize=16, fontweight='bold')
        
        # SHAP top features
        shap_top = shap_results['top_features'][:num_features]
        shap_features, shap_values = zip(*shap_top)
        
        plt.subplot(2, 2, 1)
        colors_shap = ['red' if v < 0 else 'blue' for v in shap_values]
        plt.barh(range(len(shap_features)), shap_values, color=colors_shap, alpha=0.7)
        plt.yticks(range(len(shap_features)), shap_features)
        plt.xlabel('SHAP Value')
        plt.title('SHAP Top Features')
        plt.grid(True, alpha=0.3)
        
        # LIME top features
        lime_top = lime_results['top_features'][:num_features]
        lime_features, lime_values = zip(*lime_top)
        
        plt.subplot(2, 2, 2)
        colors_lime = ['red' if v < 0 else 'blue' for v in lime_values]
        plt.barh(range(len(lime_features)), lime_values, color=colors_lime, alpha=0.7)
        plt.yticks(range(len(lime_features)), lime_features)
        plt.xlabel('LIME Value')
        plt.title('LIME Top Features')
        plt.grid(True, alpha=0.3)
        
        # Feature ranking comparison
        plt.subplot(2, 2, 3)
        shap_ranking = {feat: i for i, (feat, _) in enumerate(shap_top)}
        lime_ranking = {feat: i for i, (feat, _) in enumerate(lime_top)}
        
        common_features = set(shap_ranking.keys()) & set(lime_ranking.keys())
        if common_features:
            shap_ranks = [shap_ranking[feat] for feat in common_features]
            lime_ranks = [lime_ranking[feat] for feat in common_features]
            
            plt.scatter(shap_ranks, lime_ranks, alpha=0.7, s=100)
            plt.plot([0, max(max(shap_ranks), max(lime_ranks))], 
                    [0, max(max(shap_ranks), max(lime_ranks))], 'r--', alpha=0.5)
            plt.xlabel('SHAP Ranking')
            plt.ylabel('LIME Ranking')
            plt.title('Feature Ranking Comparison')
            plt.grid(True, alpha=0.3)
            
            # Add feature labels
            for feat in common_features:
                plt.annotate(feat[:10], (shap_ranking[feat], lime_ranking[feat]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Correlation plot
        plt.subplot(2, 2, 4)
        shap_importance = shap_results['feature_importance']
        lime_importance = lime_results['feature_importance']
        
        common_features = set(shap_importance.keys()) & set(lime_importance.keys())
        if common_features:
            shap_vals = [abs(shap_importance[feat]) for feat in common_features]
            lime_vals = [abs(lime_importance[feat]) for feat in common_features]
            
            plt.scatter(shap_vals, lime_vals, alpha=0.7, s=100)
            plt.xlabel('|SHAP Importance|')
            plt.ylabel('|LIME Importance|')
            plt.title('Importance Magnitude Correlation')
            plt.grid(True, alpha=0.3)
            
            # Calculate correlation
            correlation = np.corrcoef(shap_vals, lime_vals)[0, 1] if len(shap_vals) > 1 else 0
            plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
        
        plt.tight_layout()
        plt.show()
        
        # Prepare comparison results
        results = {
            'shap_results': shap_results,
            'lime_results': lime_results,
            'common_features': list(common_features) if 'common_features' in locals() else [],
            'correlation': correlation if 'correlation' in locals() else None,
            'agreement_score': len(common_features) / max(len(shap_top), len(lime_top)) if 'common_features' in locals() else 0
        }
        
        self._print_info(f"✅ Explanation comparison completed")
        self._print_info(f"📊 Agreement score: {results['agreement_score']:.3f}")
        if results['correlation'] is not None:
            self._print_info(f"📊 Importance correlation: {results['correlation']:.3f}")
        
        return results
    
    def global_feature_importance(self, plot: bool = True) -> Dict[str, Any]:
        """
        Calculate global feature importance using SHAP.
        
        Parameters:
        -----------
        plot : bool, default=True
            Whether to create plots
        
        Returns:
        --------
        dict : Global importance results
        """
        if self.shap_explainer is None:
            self._print_info("❌ SHAP explainer not initialized. Call setup_shap_explainer() first.")
            return {}
        
        try:
            import shap
            
            self._print_info(f"\n🔄 Calculating global feature importance...")
            
            # Calculate SHAP values for a sample of data
            sample_size = min(500, len(self.X_test_sample))
            sample_data = self.X_test_sample.iloc[:sample_size]
            
            self._print_info(f"⚡ Calculating SHAP values for {sample_size} samples...")
            shap_values = self.shap_explainer.shap_values(sample_data.values)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class classification - use positive class
                shap_values_to_use = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_values_to_use = shap_values
            
            # Calculate global importance
            global_importance = np.mean(np.abs(shap_values_to_use), axis=0)
            feature_importance_dict = dict(zip(self.feature_names, global_importance))
            
            # Sort features by importance
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            if plot:
                self._print_info(f"🎨 Creating global importance plots...")
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle('Global Feature Importance Analysis', fontsize=16, fontweight='bold')
                
                # Bar plot of top features
                plt.subplot(2, 2, 1)
                top_features = sorted_features[:15]
                features, importances = zip(*top_features)
                
                plt.barh(range(len(features)), importances, alpha=0.7)
                plt.yticks(range(len(features)), features)
                plt.xlabel('Mean |SHAP Value|')
                plt.title('Top 15 Features - Global Importance')
                plt.grid(True, alpha=0.3)
                
                # SHAP summary plot
                plt.subplot(2, 2, 2)
                try:
                    shap.summary_plot(shap_values_to_use, sample_data, 
                                    feature_names=self.feature_names, show=False, max_display=15)
                    plt.title('SHAP Summary Plot')
                except:
                    plt.text(0.5, 0.5, 'SHAP summary plot not available', ha='center', va='center')
                    plt.title('SHAP Summary Plot')
                
                # SHAP bar plot
                plt.subplot(2, 2, 3)
                try:
                    shap.summary_plot(shap_values_to_use, sample_data, 
                                    feature_names=self.feature_names, plot_type="bar", show=False, max_display=15)
                    plt.title('SHAP Bar Plot')
                except:
                    plt.text(0.5, 0.5, 'SHAP bar plot not available', ha='center', va='center')
                    plt.title('SHAP Bar Plot')
                
                # Feature importance distribution
                plt.subplot(2, 2, 4)
                plt.hist(global_importance, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Mean |SHAP Value|')
                plt.ylabel('Number of Features')
                plt.title('Feature Importance Distribution')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
            
            results = {
                'feature_importance': feature_importance_dict,
                'sorted_features': sorted_features,
                'shap_values': shap_values_to_use,
                'sample_data': sample_data,
                'top_10_features': sorted_features[:10]
            }
            
            self._print_info(f"✅ Global feature importance calculated")
            self._print_info(f"📊 Top 5 features: {[feat for feat, _ in sorted_features[:5]]}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating global feature importance: {str(e)}")
            self._print_info(f"❌ Error calculating global feature importance: {str(e)}")
            return {}
    
    def model_performance_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of model performance and explanation insights.
        
        Returns:
        --------
        dict : Performance summary
        """
        self._print_info(f"\n📊 Model Performance Summary")
        self._print_info(f"{'='*50}")
        
        summary = {
            'model_type': self.model_type,
            'task_type': self.task_type,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'sample_fraction': self.sample_fraction,
            'train_samples': len(self.X_train_sample) if self.X_train_sample is not None else 0,
            'test_samples': len(self.X_test_sample) if self.X_test_sample is not None else 0,
            'memory_usage': self._get_memory_usage()
        }
        
        # Calculate basic performance metrics if we have predictions
        if self.y_test_sample is not None and self.predictions_sample is not None:
            if self.task_type == 'regression':
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                mse = mean_squared_error(self.y_test_sample, self.predictions_sample)
                mae = mean_absolute_error(self.y_test_sample, self.predictions_sample)
                r2 = r2_score(self.y_test_sample, self.predictions_sample)
                
                summary.update({
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'mae': mae,
                    'r2_score': r2
                })
                
                self._print_info(f"🎯 RMSE: {np.sqrt(mse):.4f}")
                self._print_info(f"🎯 MAE: {mae:.4f}")
                self._print_info(f"🎯 R² Score: {r2:.4f}")
                
            else:
                from sklearn.metrics import accuracy_score, classification_report
                
                # Convert probabilities to predictions if needed
                if len(self.predictions_sample.shape) > 1:
                    y_pred = np.argmax(self.predictions_sample, axis=1)
                else:
                    y_pred = (self.predictions_sample > 0.5).astype(int)
                
                accuracy = accuracy_score(self.y_test_sample, y_pred)
                summary['accuracy'] = accuracy
                
                self._print_info(f"🎯 Accuracy: {accuracy:.4f}")
        
        # Add explainer status
        summary.update({
            'shap_available': self.shap_explainer is not None,
            'lime_available': self.lime_explainer is not None
        })
        
        self._print_info(f"🤖 Model Type: {self.model_type.title()}")
        self._print_info(f"🎯 Task Type: {self.task_type.title()}")
        self._print_info(f"📊 Features: {summary['feature_count']}")
        self._print_info(f"📊 Sample Fraction: {self.sample_fraction:.1%}")
        self._print_info(f"💾 Memory Usage: {summary['memory_usage']}")
        self._print_info(f"🔧 SHAP Available: {summary['shap_available']}")
        self._print_info(f"🔧 LIME Available: {summary['lime_available']}")
        
        return summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """
    Example usage of the ModelExplainer class.
    """
    print("="*80)
    print("MODEL EXPLAINER - EXAMPLE USAGE")
    print("="*80)
    
    # Example 1: Load model from file and explain
    """
    # Initialize explainer with model file
    explainer = ModelExplainer(
        model_path="path/to/your/model.joblib",  # or .yaml for CatBoost
        sample_fraction=0.3,  # Use 30% of data for memory efficiency
        verbose=True
    )
    
    # Prepare data
    explainer.prepare_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        fitted_values=fitted_values,
        predictions=predictions
    )
    
    # Setup explainers
    explainer.setup_shap_explainer()
    explainer.setup_lime_explainer()
    
    # Explain specific instances
    shap_explanation = explainer.explain_instance_shap(instance_index=0)
    lime_explanation = explainer.explain_instance_lime(instance_index=0)
    
    # Compare explanations
    comparison = explainer.compare_explanations(instance_index=0)
    
    # Global feature importance
    global_importance = explainer.global_feature_importance()
    
    # Performance summary
    summary = explainer.model_performance_summary()
    """
    
    # Example 2: Use with pre-loaded model
    """
    # If you already have a loaded model
    explainer = ModelExplainer(
        model=your_loaded_model,
        sample_fraction=0.5,
        verbose=True
    )
    
    # Continue with data preparation and explanations...
    """
    
    print("See the docstrings and comments above for detailed usage examples.")

if __name__ == "__main__":
    example_usage() 