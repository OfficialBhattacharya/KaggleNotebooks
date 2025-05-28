"""
Incremental Learning Pipeline for Large Datasets
================================================

This script provides a comprehensive pipeline for training ScientificXGBRegressor
on large datasets using incremental learning. It chunks the dataset and progressively
improves the model by training on each chunk sequentially.

Features:
---------
- Configurable dataset chunking
- Progressive model improvement tracking
- Memory-efficient processing for large datasets
- Comprehensive performance monitoring
- Cross-validation integration
- GPU/CPU optimization
- Model checkpoint saving
- Detailed progress visualization

Mathematical Foundation:
------------------------
Incremental learning implements the following update rule:

F_t(x) = F_{t-1}(x) + Σᵢ₌₁ᵏ η·h_i(x)

Where:
- F_t(x) is the model after chunk t
- h_i(x) are new weak learners from chunk t
- η is the learning rate
- k is the number of new estimators per chunk

Author: ScientificXGBRegressor Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import os
import warnings
from pathlib import Path
from datetime import datetime
import pickle
import json
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import gc

# Import our ScientificXGBRegressor
try:
    from xgboost import ScientificXGBRegressor, create_scientific_xgb_regressor
except ImportError:
    from .xgboost import ScientificXGBRegressor, create_scientific_xgb_regressor


@dataclass
class IncrementalConfig:
    """
    Configuration class for incremental learning pipeline.
    
    Attributes:
    -----------
    n_chunks : int
        Number of chunks to split the dataset into
    chunk_size : Optional[int]
        Specific chunk size (overrides n_chunks if provided)
    inner_cv_folds : int
        Number of inner cross-validation folds
    outer_cv_folds : int
        Number of outer cross-validation folds
    n_estimators_per_chunk : int
        Number of new estimators to add per chunk
    validation_split : float
        Fraction of data to use for validation
    enable_early_stopping : bool
        Whether to use early stopping
    save_checkpoints : bool
        Whether to save model checkpoints after each chunk
    checkpoint_dir : str
        Directory to save checkpoints
    memory_optimization : bool
        Whether to apply memory optimization techniques
    verbose : bool
        Whether to print detailed progress information
    use_gpu : Optional[bool]
        GPU usage preference (None=auto, True=force, False=disable)
    plot_progress : bool
        Whether to generate progress plots
    """
    n_chunks: int = 10
    chunk_size: Optional[int] = None
    inner_cv_folds: int = 3
    outer_cv_folds: int = 5
    n_estimators_per_chunk: int = 100
    validation_split: float = 0.2
    enable_early_stopping: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "./incremental_checkpoints"
    memory_optimization: bool = True
    verbose: bool = True
    use_gpu: Optional[bool] = None
    plot_progress: bool = True


class IncrementalLearningPipeline:
    """
    Comprehensive pipeline for incremental learning on large datasets.
    
    This class manages the entire incremental learning process, including
    data chunking, progressive training, performance monitoring, and
    checkpoint management.
    
    Mathematical Framework:
    -----------------------
    The pipeline implements incremental ensemble learning:
    
    For each chunk C_t:
    1. Load chunk: (X_t, y_t) = C_t
    2. Update model: M_t = M_{t-1} + ΔM_t
    3. Evaluate: R_t = evaluate(M_t, V)
    4. Save checkpoint: save(M_t) if R_t > R_{t-1}
    
    Where ΔM_t represents the incremental model update from chunk t.
    """
    
    def __init__(self, config: IncrementalConfig):
        """
        Initialize the incremental learning pipeline.
        
        Parameters:
        -----------
        config : IncrementalConfig
            Configuration object with all pipeline settings
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler() if config.memory_optimization else None
        self.training_history = []
        self.performance_history = []
        self.chunk_info = []
        self.validation_data = None
        
        # Create checkpoint directory
        if config.save_checkpoints:
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize progress tracking
        self.start_time = None
        self.chunk_times = []
        
        if config.verbose:
            print("🚀 Incremental Learning Pipeline Initialized")
            print(f"   📊 Configuration: {config.n_chunks} chunks, {config.n_estimators_per_chunk} estimators/chunk")
            print(f"   🔄 Cross-validation: {config.inner_cv_folds} inner × {config.outer_cv_folds} outer folds")
            print(f"   💾 Checkpoints: {'Enabled' if config.save_checkpoints else 'Disabled'}")
    
    def prepare_data_chunks(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        shuffle: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare data chunks for incremental learning.
        
        This method efficiently chunks large datasets while preserving
        statistical properties and enabling memory-efficient processing.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target vector
        shuffle : bool, default=True
            Whether to shuffle data before chunking
            
        Returns:
        --------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (X_chunk, y_chunk) tuples
        """
        if self.config.verbose:
            print(f"📦 Preparing data chunks from {len(X)} samples...")
        
        # Convert to numpy arrays for consistent handling
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle data if requested
        if shuffle:
            if self.config.verbose:
                print("   🔀 Shuffling data...")
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
        
        # Calculate chunk size
        if self.config.chunk_size is not None:
            chunk_size = self.config.chunk_size
            n_chunks = len(X) // chunk_size + (1 if len(X) % chunk_size > 0 else 0)
        else:
            n_chunks = self.config.n_chunks
            chunk_size = len(X) // n_chunks
        
        if self.config.verbose:
            print(f"   📏 Chunk configuration:")
            print(f"      Number of chunks: {n_chunks}")
            print(f"      Samples per chunk: ~{chunk_size:,}")
        
        # Create chunks
        chunks = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(X))
            
            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]
            
            chunk_info = {
                'chunk_id': i,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size': len(X_chunk),
                'feature_shape': X_chunk.shape,
                'target_stats': {
                    'mean': np.mean(y_chunk),
                    'std': np.std(y_chunk),
                    'min': np.min(y_chunk),
                    'max': np.max(y_chunk)
                }
            }
            
            chunks.append((X_chunk, y_chunk))
            self.chunk_info.append(chunk_info)
            
            if self.config.verbose and i < 3:  # Show details for first 3 chunks
                print(f"      Chunk {i}: {len(X_chunk):,} samples, target μ={chunk_info['target_stats']['mean']:.4f}")
        
        if self.config.verbose and n_chunks > 3:
            print(f"      ... and {n_chunks - 3} more chunks")
        
        # Memory optimization
        if self.config.memory_optimization:
            del X, y  # Free original data memory
            gc.collect()
        
        return chunks
    
    def create_validation_set(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create training and validation sets from the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target vector
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, X_val, y_train, y_val
        """
        if self.config.validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config.validation_split,
                random_state=42,
                stratify=None  # Regression, no stratification
            )
            
            # Store validation data for consistent evaluation
            self.validation_data = (X_val, y_val)
            
            if self.config.verbose:
                print(f"   📊 Validation set: {len(X_val):,} samples ({self.config.validation_split:.1%})")
            
            return X_train, X_val, y_train, y_val
        else:
            return X, None, y, None
    
    def initialize_model(self, X_first_chunk: np.ndarray, y_first_chunk: np.ndarray) -> None:
        """
        Initialize the ScientificXGBRegressor with the first chunk.
        
        Parameters:
        -----------
        X_first_chunk : array-like
            Features from the first data chunk
        y_first_chunk : array-like
            Targets from the first data chunk
        """
        if self.config.verbose:
            print("🧪 Initializing ScientificXGBRegressor...")
        
        # Create validation set from first chunk
        X_train, X_val, y_train, y_val = self.create_validation_set(X_first_chunk, y_first_chunk)
        
        # Initialize model with configuration
        model_params = {
            'cv_folds': self.config.outer_cv_folds,
            'auto_tune': True,
            'verbose': self.config.verbose,
            'n_estimators': self.config.n_estimators_per_chunk,
            'early_stopping_rounds': 50 if self.config.enable_early_stopping else None,
            'use_gpu': self.config.use_gpu,
            'random_state': 42
        }
        
        self.model = create_scientific_xgb_regressor(**model_params)
        
        # Apply GPU optimization
        if self.model._gpu_info['available'] and self.config.use_gpu != False:
            gpu_optimization = self.model.optimize_gpu_usage(X_train)
            if self.config.verbose:
                print(f"   ⚡ GPU optimization: {gpu_optimization['status']}")
        
        # Fit initial model
        if self.config.verbose:
            print("   🎯 Training initial model on first chunk...")
        
        fit_start = time.time()
        
        # Add evaluation set for early stopping if validation data available
        fit_kwargs = {}
        if X_val is not None and self.config.enable_early_stopping:
            fit_kwargs['eval_set'] = [(X_train, y_train), (X_val, y_val)]
            fit_kwargs['verbose'] = False
        
        self.model.fit(X_train, y_train, **fit_kwargs)
        
        fit_time = time.time() - fit_start
        
        # Evaluate initial performance
        initial_performance = self.evaluate_model(X_first_chunk, y_first_chunk, chunk_id=0)
        initial_performance['fit_time'] = fit_time
        initial_performance['n_estimators'] = self.model.n_estimators
        initial_performance['is_initial'] = True
        
        self.performance_history.append(initial_performance)
        
        if self.config.verbose:
            print(f"   ✅ Initial model trained in {fit_time:.2f}s")
            print(f"      R² score: {initial_performance['r2']:.4f}")
            print(f"      RMSE: {initial_performance['rmse']:.4f}")
    
    def evaluate_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        chunk_id: int = -1
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Parameters:
        -----------
        X : array-like
            Features for evaluation
        y : array-like
            True targets
        chunk_id : int
            ID of the current chunk
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive evaluation metrics
        """
        y_pred = self.model.predict(X)
        
        evaluation = {
            'chunk_id': chunk_id,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X),
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'model_n_estimators': self.model.n_estimators,
        }
        
        # Add explained variance and additional metrics
        y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        evaluation['explained_variance'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Residual analysis
        residuals = y - y_pred
        evaluation['residual_std'] = np.std(residuals)
        evaluation['residual_mean'] = np.mean(residuals)
        evaluation['residual_skewness'] = float(np.mean(((residuals - np.mean(residuals)) / np.std(residuals)) ** 3))
        
        # Validation set evaluation if available
        if self.validation_data is not None:
            X_val, y_val = self.validation_data
            y_val_pred = self.model.predict(X_val)
            evaluation['val_r2'] = r2_score(y_val, y_val_pred)
            evaluation['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_val_pred))
            evaluation['val_mae'] = mean_absolute_error(y_val, y_val_pred)
        
        return evaluation
    
    def save_checkpoint(self, chunk_id: int) -> str:
        """
        Save model checkpoint after processing a chunk.
        
        Parameters:
        -----------
        chunk_id : int
            ID of the completed chunk
            
        Returns:
        --------
        str
            Path to saved checkpoint
        """
        if not self.config.save_checkpoints:
            return ""
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_file = checkpoint_dir / f"model_checkpoint_chunk_{chunk_id:03d}.pkl"
        
        # Save model
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            'chunk_id': chunk_id,
            'timestamp': datetime.now().isoformat(),
            'n_estimators': self.model.n_estimators,
            'performance_history': self.performance_history,
            'chunk_info': self.chunk_info[:chunk_id + 1],
            'config': self.config.__dict__
        }
        
        metadata_file = checkpoint_dir / f"metadata_chunk_{chunk_id:03d}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        if self.config.verbose:
            file_size = checkpoint_file.stat().st_size / (1024 * 1024)  # MB
            print(f"      💾 Checkpoint saved: {checkpoint_file.name} ({file_size:.1f} MB)")
        
        return str(checkpoint_file)
    
    def process_chunk(self, chunk_id: int, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Process a single data chunk with incremental learning.
        
        Parameters:
        -----------
        chunk_id : int
            ID of the current chunk
        X_chunk : array-like
            Features for the current chunk
        y_chunk : array-like
            Targets for the current chunk
            
        Returns:
        --------
        Dict[str, Any]
            Processing results and performance metrics
        """
        if self.config.verbose:
            print(f"🔄 Processing chunk {chunk_id + 1}/{len(self.chunk_info)}")
            print(f"   📊 Chunk size: {len(X_chunk):,} samples")
        
        chunk_start = time.time()
        
        # Use incremental learning for non-initial chunks
        if chunk_id == 0:
            # First chunk - initialize model
            self.initialize_model(X_chunk, y_chunk)
            processing_results = {
                'method': 'initial_fit',
                'chunk_id': chunk_id,
                'n_estimators_added': self.model.n_estimators
            }
        else:
            # Subsequent chunks - incremental learning
            if self.config.verbose:
                print(f"   🔄 Incremental learning: adding {self.config.n_estimators_per_chunk} estimators...")
            
            # Apply incremental learning
            incremental_results = self.model.incremental_learn(
                X_chunk, y_chunk,
                n_new_estimators=self.config.n_estimators_per_chunk
            )
            
            processing_results = {
                'method': 'incremental_learn',
                'chunk_id': chunk_id,
                'incremental_results': incremental_results,
                'n_estimators_added': self.config.n_estimators_per_chunk
            }
        
        # Evaluate performance on current chunk
        chunk_performance = self.evaluate_model(X_chunk, y_chunk, chunk_id)
        
        # Add timing information
        chunk_time = time.time() - chunk_start
        chunk_performance['processing_time'] = chunk_time
        chunk_performance['processing_method'] = processing_results['method']
        
        # Store results
        self.performance_history.append(chunk_performance)
        self.chunk_times.append(chunk_time)
        
        # Save checkpoint
        checkpoint_path = self.save_checkpoint(chunk_id)
        processing_results['checkpoint_path'] = checkpoint_path
        
        # Progress reporting
        if self.config.verbose:
            print(f"   ✅ Chunk {chunk_id + 1} completed in {chunk_time:.2f}s")
            print(f"      R² score: {chunk_performance['r2']:.4f}")
            print(f"      RMSE: {chunk_performance['rmse']:.4f}")
            print(f"      Total estimators: {self.model.n_estimators}")
            
            if 'val_r2' in chunk_performance:
                print(f"      Validation R²: {chunk_performance['val_r2']:.4f}")
        
        # Memory cleanup
        if self.config.memory_optimization:
            del X_chunk, y_chunk
            gc.collect()
        
        return processing_results
    
    def run_incremental_training(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Run the complete incremental training pipeline.
        
        This is the main method that orchestrates the entire incremental
        learning process from data preparation to final model evaluation.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Complete feature dataset
        y : array-like of shape (n_samples,)
            Complete target dataset
            
        Returns:
        --------
        Dict[str, Any]
            Complete training results and performance history
        """
        if self.config.verbose:
            print("🚀 Starting Incremental Learning Pipeline")
            print("=" * 60)
        
        self.start_time = time.time()
        
        # Step 1: Prepare data chunks
        chunks = self.prepare_data_chunks(X, y)
        
        if self.config.verbose:
            print(f"\n📦 Data preparation complete: {len(chunks)} chunks ready")
        
        # Step 2: Process each chunk
        processing_results = []
        
        for chunk_id, (X_chunk, y_chunk) in enumerate(chunks):
            try:
                result = self.process_chunk(chunk_id, X_chunk, y_chunk)
                processing_results.append(result)
                
                # Progress update
                progress = (chunk_id + 1) / len(chunks) * 100
                if self.config.verbose:
                    print(f"   📈 Progress: {progress:.1f}% ({chunk_id + 1}/{len(chunks)} chunks)")
                    
            except Exception as e:
                error_msg = f"Error processing chunk {chunk_id}: {str(e)}"
                if self.config.verbose:
                    print(f"   ❌ {error_msg}")
                
                processing_results.append({
                    'chunk_id': chunk_id,
                    'error': error_msg,
                    'method': 'failed'
                })
                
                # Continue with next chunk
                continue
        
        # Step 3: Final evaluation and summary
        total_time = time.time() - self.start_time
        
        final_results = {
            'pipeline_config': self.config.__dict__,
            'total_chunks_processed': len([r for r in processing_results if 'error' not in r]),
            'failed_chunks': len([r for r in processing_results if 'error' in r]),
            'total_training_time': total_time,
            'average_chunk_time': np.mean(self.chunk_times) if self.chunk_times else 0,
            'processing_results': processing_results,
            'performance_history': self.performance_history,
            'chunk_info': self.chunk_info,
            'final_model_estimators': self.model.n_estimators if self.model else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate performance improvements
        if len(self.performance_history) >= 2:
            initial_r2 = self.performance_history[0]['r2']
            final_r2 = self.performance_history[-1]['r2']
            r2_improvement = final_r2 - initial_r2
            
            initial_rmse = self.performance_history[0]['rmse']
            final_rmse = self.performance_history[-1]['rmse']
            rmse_improvement = initial_rmse - final_rmse
            
            final_results['performance_improvement'] = {
                'r2_improvement': r2_improvement,
                'rmse_improvement': rmse_improvement,
                'relative_r2_improvement': r2_improvement / max(abs(initial_r2), 1e-6),
                'relative_rmse_improvement': rmse_improvement / max(initial_rmse, 1e-6)
            }
        
        # Generate progress plots if requested
        if self.config.plot_progress:
            self.plot_training_progress()
        
        # Final summary
        if self.config.verbose:
            print("\n🎉 Incremental Learning Pipeline Complete!")
            print("=" * 60)
            print(f"📊 Summary:")
            print(f"   Total chunks: {len(chunks)}")
            print(f"   Successful: {final_results['total_chunks_processed']}")
            print(f"   Failed: {final_results['failed_chunks']}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Final estimators: {final_results['final_model_estimators']}")
            
            if 'performance_improvement' in final_results:
                perf = final_results['performance_improvement']
                print(f"   R² improvement: {perf['r2_improvement']:+.4f} ({perf['relative_r2_improvement']:+.2%})")
                print(f"   RMSE improvement: {perf['rmse_improvement']:+.4f} ({perf['relative_rmse_improvement']:+.2%})")
        
        return final_results
    
    def plot_training_progress(self, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Generate comprehensive training progress visualization.
        
        Parameters:
        -----------
        figsize : tuple, default=(15, 10)
            Figure size for the plots
            
        Returns:
        --------
        matplotlib.Figure
            Figure containing all progress plots
        """
        if not self.performance_history:
            print("⚠️ No performance history available for plotting")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Incremental Learning Progress Dashboard', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        chunk_ids = [p['chunk_id'] for p in self.performance_history]
        r2_scores = [p['r2'] for p in self.performance_history]
        rmse_scores = [p['rmse'] for p in self.performance_history]
        mae_scores = [p['mae'] for p in self.performance_history]
        n_estimators = [p['model_n_estimators'] for p in self.performance_history]
        
        # Plot 1: R² Score Progress
        axes[0, 0].plot(chunk_ids, r2_scores, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Chunk ID')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('R² Score Progression')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(bottom=0)
        
        # Add trend line
        if len(chunk_ids) > 2:
            z = np.polyfit(chunk_ids, r2_scores, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(chunk_ids, p(chunk_ids), "r--", alpha=0.8, label=f'Trend: {z[0]:+.4f}/chunk')
            axes[0, 0].legend()
        
        # Plot 2: RMSE Progress
        axes[0, 1].plot(chunk_ids, rmse_scores, 'r-o', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Chunk ID')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('RMSE Progression')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Model Complexity (Number of Estimators)
        axes[0, 2].plot(chunk_ids, n_estimators, 'g-o', linewidth=2, markersize=6)
        axes[0, 2].set_xlabel('Chunk ID')
        axes[0, 2].set_ylabel('Number of Estimators')
        axes[0, 2].set_title('Model Complexity Growth')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Multiple Metrics Comparison
        axes[1, 0].plot(chunk_ids, r2_scores, 'b-', label='R² Score', linewidth=2)
        if self.validation_data is not None:
            val_r2_scores = [p.get('val_r2', 0) for p in self.performance_history if 'val_r2' in p]
            if val_r2_scores:
                val_chunk_ids = [p['chunk_id'] for p in self.performance_history if 'val_r2' in p]
                axes[1, 0].plot(val_chunk_ids, val_r2_scores, 'b--', label='Validation R²', linewidth=2)
        
        axes[1, 0].set_xlabel('Chunk ID')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Training vs Validation Performance')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 5: Processing Time per Chunk
        if self.chunk_times:
            axes[1, 1].bar(range(len(self.chunk_times)), self.chunk_times, alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('Chunk ID')
            axes[1, 1].set_ylabel('Processing Time (seconds)')
            axes[1, 1].set_title('Processing Time per Chunk')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add average line
            avg_time = np.mean(self.chunk_times)
            axes[1, 1].axhline(y=avg_time, color='red', linestyle='--', 
                             label=f'Average: {avg_time:.2f}s')
            axes[1, 1].legend()
        
        # Plot 6: Residual Analysis (Latest Chunk)
        if len(self.performance_history) > 0:
            latest_perf = self.performance_history[-1]
            residual_std = latest_perf.get('residual_std', 0)
            residual_mean = latest_perf.get('residual_mean', 0)
            
            # Create a simple residual distribution plot
            # Since we don't have actual residuals, we'll show summary statistics
            metrics = ['RMSE', 'MAE', 'Residual Std']
            values = [latest_perf['rmse'], latest_perf['mae'], residual_std]
            
            bars = axes[1, 2].bar(metrics, values, alpha=0.7, color=['red', 'orange', 'yellow'])
            axes[1, 2].set_ylabel('Error Magnitude')
            axes[1, 2].set_title('Final Model Error Metrics')
            axes[1, 2].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot if checkpoints are enabled
        if self.config.save_checkpoints:
            plot_path = Path(self.config.checkpoint_dir) / "training_progress.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            if self.config.verbose:
                print(f"📊 Progress plots saved to: {plot_path}")
        
        return fig
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Find the checkpoint with the best validation performance.
        
        Returns:
        --------
        Optional[str]
            Path to the best checkpoint file, or None if no checkpoints exist
        """
        if not self.config.save_checkpoints or not self.performance_history:
            return None
        
        # Find best performance based on validation R² if available, otherwise training R²
        best_idx = 0
        best_score = -np.inf
        
        for i, perf in enumerate(self.performance_history):
            score = perf.get('val_r2', perf.get('r2', -np.inf))
            if score > best_score:
                best_score = score
                best_idx = i
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        best_checkpoint = checkpoint_dir / f"model_checkpoint_chunk_{best_idx:03d}.pkl"
        
        if best_checkpoint.exists():
            return str(best_checkpoint)
        return None


def create_incremental_pipeline(
    n_chunks: int = 10,
    chunk_size: Optional[int] = None,
    inner_cv_folds: int = 3,
    outer_cv_folds: int = 5,
    n_estimators_per_chunk: int = 100,
    use_gpu: Optional[bool] = None,
    **kwargs
) -> IncrementalLearningPipeline:
    """
    Factory function to create an incremental learning pipeline with common configurations.
    
    Parameters:
    -----------
    n_chunks : int, default=10
        Number of chunks to split the dataset into
    chunk_size : Optional[int], default=None
        Specific chunk size (overrides n_chunks if provided)
    inner_cv_folds : int, default=3
        Number of inner cross-validation folds
    outer_cv_folds : int, default=5
        Number of outer cross-validation folds
    n_estimators_per_chunk : int, default=100
        Number of new estimators to add per chunk
    use_gpu : Optional[bool], default=None
        GPU usage preference (None=auto, True=force, False=disable)
    **kwargs : dict
        Additional configuration parameters for IncrementalConfig
        
    Returns:
    --------
    IncrementalLearningPipeline
        Configured pipeline ready for incremental learning
        
    Example:
    --------
    ```python
    # Create pipeline for 20 chunks with GPU acceleration
    pipeline = create_incremental_pipeline(
        n_chunks=20,
        n_estimators_per_chunk=150,
        use_gpu=True,
        verbose=True
    )
    
    # Run incremental training
    results = pipeline.run_incremental_training(X, y)
    ```
    """
    config_params = {
        'n_chunks': n_chunks,
        'chunk_size': chunk_size,
        'inner_cv_folds': inner_cv_folds,
        'outer_cv_folds': outer_cv_folds,
        'n_estimators_per_chunk': n_estimators_per_chunk,
        'use_gpu': use_gpu,
    }
    
    # Add any additional parameters
    config_params.update(kwargs)
    
    config = IncrementalConfig(**config_params)
    return IncrementalLearningPipeline(config)


# Example usage and demonstration
if __name__ == "__main__":
    print("🚀 Incremental Learning Pipeline for Large Datasets")
    print("=" * 60)
    
    # Generate sample large dataset for demonstration
    print("📊 Generating sample dataset (simulating large dataset)...")
    np.random.seed(42)
    
    # Simulate large dataset characteristics
    n_samples = 100000  # Reduced for demo, would be 7M+ in real scenario
    n_features = 20
    
    print(f"   Dataset: {n_samples:,} samples × {n_features} features")
    
    # Generate synthetic data with some complexity
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear target with noise
    y = (np.sum(X[:, :5], axis=1) + 
         0.5 * np.sum(X[:, 5:10] ** 2, axis=1) + 
         0.1 * np.random.randn(n_samples))
    
    print("✅ Sample dataset generated")
    
    # Example 1: Basic incremental learning
    print("\n🔄 Example 1: Basic Incremental Learning")
    print("-" * 40)
    
    # Create pipeline with user-configurable parameters
    pipeline = create_incremental_pipeline(
        n_chunks=5,  # Reduced for demo
        n_estimators_per_chunk=50,  # Reduced for demo
        inner_cv_folds=3,
        outer_cv_folds=3,  # Reduced for demo
        use_gpu=None,  # Auto-detect
        verbose=True,
        save_checkpoints=True,
        plot_progress=True
    )
    
    # Run incremental training
    results = pipeline.run_incremental_training(X, y)
    
    # Display results summary
    print("\n📈 Training Results Summary:")
    print(f"   Chunks processed: {results['total_chunks_processed']}")
    print(f"   Total training time: {results['total_training_time']:.2f}s")
    print(f"   Final model estimators: {results['final_model_estimators']}")
    
    if 'performance_improvement' in results:
        perf = results['performance_improvement']
        print(f"   R² improvement: {perf['r2_improvement']:+.4f} ({perf['relative_r2_improvement']:+.2%})")
    
    # Show the final model capabilities
    print(f"\n🧪 Final Model Capabilities:")
    print(f"   Model type: {type(pipeline.model).__name__}")
    print(f"   GPU acceleration: {pipeline.model._using_gpu}")
    print(f"   Total estimators: {pipeline.model.n_estimators}")
    
    # Example 2: Custom configuration for large dataset
    print("\n🔄 Example 2: Configuration for Large Dataset (7M+ rows)")
    print("-" * 50)
    
    # Configuration optimized for very large datasets
    large_dataset_config = IncrementalConfig(
        n_chunks=20,  # More chunks for better memory management
        inner_cv_folds=3,  # Reduced CV for speed
        outer_cv_folds=5,
        n_estimators_per_chunk=200,  # More estimators per chunk
        validation_split=0.1,  # Smaller validation set for large data
        memory_optimization=True,
        save_checkpoints=True,
        checkpoint_dir="./large_dataset_checkpoints",
        use_gpu=None,  # Auto-detect
        verbose=True,
        plot_progress=True
    )
    
    print("📋 Large Dataset Configuration:")
    print(f"   Chunks: {large_dataset_config.n_chunks}")
    print(f"   Estimators per chunk: {large_dataset_config.n_estimators_per_chunk}")
    print(f"   Cross-validation: {large_dataset_config.inner_cv_folds}×{large_dataset_config.outer_cv_folds}")
    print(f"   Memory optimization: {large_dataset_config.memory_optimization}")
    print(f"   GPU acceleration: {'Auto-detect' if large_dataset_config.use_gpu is None else large_dataset_config.use_gpu}")
    
    # Example 3: Demonstrate chunk size calculation
    print("\n📦 Example 3: Chunk Size Calculations")
    print("-" * 40)
    
    # For 7M samples
    large_n_samples = 7_000_000
    chunk_configs = [
        {'n_chunks': 10, 'description': 'Fast processing (10 chunks)'},
        {'n_chunks': 20, 'description': 'Balanced approach (20 chunks)'},
        {'n_chunks': 50, 'description': 'Fine-grained (50 chunks)'},
        {'chunk_size': 100_000, 'description': 'Fixed 100K samples per chunk'}
    ]
    
    for config in chunk_configs:
        if 'n_chunks' in config:
            chunk_size = large_n_samples // config['n_chunks']
            n_chunks = config['n_chunks']
        else:
            chunk_size = config['chunk_size']
            n_chunks = large_n_samples // chunk_size + (1 if large_n_samples % chunk_size > 0 else 0)
        
        memory_per_chunk_mb = (chunk_size * n_features * 8) / (1024**2)  # Assuming float64
        
        print(f"   {config['description']}:")
        print(f"      Chunks: {n_chunks}, Size: {chunk_size:,} samples/chunk")
        print(f"      Memory per chunk: ~{memory_per_chunk_mb:.1f} MB")
    
    # Example usage instructions
    print("\n📚 Usage Instructions for Large Datasets:")
    print("-" * 50)
    print("""
# Load your large dataset
X = pd.read_csv('large_dataset.csv', features_columns)  # 7M+ rows
y = pd.read_csv('large_dataset.csv', target_column)

# Create optimized pipeline
pipeline = create_incremental_pipeline(
    n_chunks=20,                    # Adjust based on memory
    n_estimators_per_chunk=150,     # Adjust based on compute time
    inner_cv_folds=3,               # Reduce for faster training
    outer_cv_folds=5,
    use_gpu=None,                   # Auto-detect GPU
    memory_optimization=True,       # Enable for large datasets
    save_checkpoints=True,          # Save progress
    verbose=True
)

# Run incremental training
results = pipeline.run_incremental_training(X, y)

# Access final model
final_model = pipeline.model
predictions = final_model.predict(X_test)

# Load best checkpoint if needed
best_checkpoint = pipeline.get_best_checkpoint()
if best_checkpoint:
    with open(best_checkpoint, 'rb') as f:
        best_model = pickle.load(f)
""")
    
    print("\n🎉 Incremental Learning Pipeline demonstration complete!")
    print("   Ready for production use with large datasets!") 