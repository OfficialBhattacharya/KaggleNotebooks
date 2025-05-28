import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda, Multiply, Dot, Dropout, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def constrained_softmax_activation(max_weight=0.5):
    """
    Creates a constrained softmax activation function.
    This is an alternative approach that might be more stable.
    """
    def activation(x):
        # Apply softmax
        softmax_weights = tf.nn.softmax(x)
        
        # Apply constraint in a simpler way
        # Clip all weights to max_weight
        clipped = tf.clip_by_value(softmax_weights, 0.0, max_weight)
        
        # Renormalize to sum to 1
        normalized = clipped / tf.reduce_sum(clipped, axis=-1, keepdims=True)
        
        return normalized
    
    return activation

class ConstrainedSoftmax(Layer):
    """
    Custom layer that applies softmax with maximum weight constraint.
    Ensures no single model gets more than max_weight (default 0.5).
    """
    def __init__(self, max_weight=0.5, **kwargs):
        super(ConstrainedSoftmax, self).__init__(**kwargs)
        self.max_weight = max_weight
        
    def call(self, inputs):
        # Apply standard softmax first
        softmax_weights = tf.nn.softmax(inputs)
        
        # Method 1: Iterative redistribution (more robust)
        # This ensures the constraint is strictly enforced
        max_w = tf.constant(self.max_weight, dtype=tf.float32)
        
        # Find weights that exceed max_weight
        exceed_mask = tf.greater(softmax_weights, max_w)
        
        # Clip weights to max_weight
        clipped_weights = tf.where(exceed_mask, max_w, softmax_weights)
        
        # Calculate excess weight to redistribute
        excess = tf.reduce_sum(softmax_weights - clipped_weights, axis=1, keepdims=True)
        
        # Count how many weights are below max_weight for redistribution
        below_max = tf.logical_not(exceed_mask)
        num_below = tf.reduce_sum(tf.cast(below_max, tf.float32), axis=1, keepdims=True)
        
        # Redistribute excess weight evenly among weights below max_weight
        redistribution = tf.where(
            tf.greater(num_below, 0),
            excess / tf.maximum(num_below, 1.0),
            0.0
        )
        
        # Add redistribution only to weights below max_weight
        final_weights = tf.where(
            below_max,
            clipped_weights + redistribution,
            clipped_weights
        )
        
        # Final normalization to ensure sum = 1
        weight_sum = tf.reduce_sum(final_weights, axis=1, keepdims=True)
        normalized_weights = final_weights / tf.maximum(weight_sum, 1e-8)
        
        return normalized_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({'max_weight': self.max_weight})
        return config

def test_constraint_layer(max_weight=0.5, n_models=7):
    """
    Test function to verify the ConstrainedSoftmax layer is working correctly.
    """
    print(f"\nTesting ConstrainedSoftmax with max_weight={max_weight}, n_models={n_models}")
    
    # Create test inputs with various scenarios
    test_scenarios = [
        # Scenario 1: One model strongly preferred
        tf.constant([[10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tf.float32),
        # Scenario 2: Two models strongly preferred
        tf.constant([[10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tf.float32),
        # Scenario 3: All equal
        tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=tf.float32),
    ]
    
    constraint_layer = ConstrainedSoftmax(max_weight=max_weight)
    
    for i, test_input in enumerate(test_scenarios):
        print(f"\nScenario {i+1}:")
        
        # Test regular softmax
        regular_softmax = tf.nn.softmax(test_input)
        print(f"Regular softmax: {regular_softmax.numpy()[0]}")
        print(f"Max weight (regular): {tf.reduce_max(regular_softmax).numpy():.4f}")
        
        # Test constrained softmax
        constrained_result = constraint_layer(test_input)
        print(f"Constrained softmax: {constrained_result.numpy()[0]}")
        print(f"Max weight (constrained): {tf.reduce_max(constrained_result).numpy():.4f}")
        print(f"Sum of weights: {tf.reduce_sum(constrained_result).numpy():.4f}")
        
        # Verify constraint
        max_achieved = tf.reduce_max(constrained_result).numpy()
        if max_achieved <= max_weight + 1e-6:  # Small tolerance for floating point
            print("✓ Constraint working correctly!")
        else:
            print(f"✗ Constraint FAILED! Max weight {max_achieved:.4f} > {max_weight}")
    
    return True

def stacked_ensemble_meta_learner(X_train, y_train, y_trainmodelpreds, y_testmodelpreds, X_test,
                                 validation_split=0.2, epochs=100, batch_size=1024, 
                                 hidden_units=[128, 64, 32], dropout_rate=0.3,
                                 learning_rate=0.001, l2_reg=0.001, max_weight=0.5, verbose=True):
    """
    Implements a stacking ensemble with neural network weighting and provides detailed diagnostics.
    Optimized for large datasets (750K+ samples) with gradual learning per epoch.
    
    Parameters:
    -----------
    X_train : array-like, shape (n_samples, n_features)
        Training features
    y_train : array-like, shape (n_samples,)
        Training targets
    y_trainmodelpreds : array-like, shape (n_samples, n_models)
        Base models' predictions on training data
    y_testmodelpreds : array-like, shape (n_test_samples, n_models)
        Base models' predictions on test data
    X_test : array-like, shape (n_test_samples, n_features)
        Test features
    validation_split : float, default=0.2
        Fraction of training data to use for validation
    epochs : int, default=100
        Number of training epochs (increased for large datasets)
    batch_size : int, default=1024
        Training batch size (optimized for large datasets)
    hidden_units : list, default=[128, 64, 32]
        List of hidden units for each layer (deeper network for complex patterns)
    dropout_rate : float, default=0.3
        Dropout rate for regularization
    learning_rate : float, default=0.001
        Learning rate for the optimizer
    l2_reg : float, default=0.001
        L2 regularization strength
    max_weight : float, default=0.5
        Maximum weight any single model can receive (prevents dominance)
    verbose : bool, default=True
        Whether to print detailed progress information
        
    Returns:
    --------
    train_df : pandas.DataFrame
        Training results with actual values, weighted predictions, and individual model predictions/weights
    test_df : pandas.DataFrame
        Test results with weighted predictions and individual model predictions/weights
        
    Weighting Mechanism:
    -------------------
    The neural meta-learner uses a deeper architecture with multiple hidden layers to learn 
    complex feature-dependent weighting patterns. Each layer progressively learns more abstract
    representations for optimal model weighting.
    
    Weight Constraint:
    -----------------
    The maximum weight any single model can receive is constrained to max_weight (default 0.5).
    This prevents any single model from dominating the ensemble and ensures more balanced
    model combination. The remaining weight is distributed among all models proportionally.
    
    Architecture Optimizations for Large Datasets:
    ---------------------------------------------
    - Larger batch sizes (1024) for stable gradients with large datasets
    - Deeper network (3 hidden layers) to capture complex feature interactions
    - Dropout regularization to prevent overfitting
    - L2 regularization for weight decay
    - Learning rate scheduling for gradual convergence
    - More epochs with early stopping for optimal training
    - Constrained softmax to prevent model dominance
    """
    
    if verbose:
        print("="*80)
        print("STACKED ENSEMBLE META-LEARNER")
        print("="*80)
        print("Step 1: Input Validation and Preprocessing")
    
    # Input validation
    X_train = np.array(X_train)
    y_train = np.array(y_train).flatten()  # Ensure 1D
    y_trainmodelpreds = np.array(y_trainmodelpreds)
    y_testmodelpreds = np.array(y_testmodelpreds)
    X_test = np.array(X_test)
    
    # Shape validation
    assert X_train.shape[0] == y_train.shape[0] == y_trainmodelpreds.shape[0], \
        "Training data dimensions don't match"
    assert X_test.shape[0] == y_testmodelpreds.shape[0], \
        "Test data dimensions don't match"
    assert y_trainmodelpreds.shape[1] == y_testmodelpreds.shape[1], \
        "Number of base models must be consistent between train and test"
    assert X_train.shape[1] == X_test.shape[1], \
        "Feature dimensions must match between train and test"
    
    n_samples, n_features = X_train.shape
    n_models = y_trainmodelpreds.shape[1]
    
    if verbose:
        print(f"✓ Training samples: {n_samples}")
        print(f"✓ Features: {n_features}")
        print(f"✓ Base models: {n_models}")
        print(f"✓ Test samples: {X_test.shape[0]}")
        
        # Test the constraint layer before proceeding
        print("\nVerifying weight constraint implementation...")
        test_constraint_layer(max_weight=max_weight, n_models=n_models)
    
    # Split training data for validation if requested
    if validation_split > 0:
        val_size = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_idx = indices[val_size:]
        val_idx = indices[:val_size]
        
        X_train_split = X_train[train_idx]
        y_train_split = y_train[train_idx]
        y_trainmodelpreds_split = y_trainmodelpreds[train_idx]
        
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        y_valmodelpreds = y_trainmodelpreds[val_idx]
        
        if verbose:
            print(f"✓ Training split: {len(train_idx)} samples")
            print(f"✓ Validation split: {len(val_idx)} samples")
    else:
        X_train_split = X_train
        y_train_split = y_train
        y_trainmodelpreds_split = y_trainmodelpreds
        X_val = y_val = y_valmodelpreds = None
    
    if verbose:
        print("\nStep 2: Building Neural Meta-Learner Architecture")
        print(f"✓ Architecture: {len(hidden_units)} hidden layers with {hidden_units} units")
        print(f"✓ Regularization: Dropout={dropout_rate}, L2={l2_reg}")
        print(f"✓ Weight constraint: Max weight per model = {max_weight}")
        print(f"✓ Output: {n_models} model weights (constrained softmax)")
    
    # Define meta-model architecture that takes both features and base predictions
    feature_input = Input(shape=(n_features,), name='features')
    base_preds_input = Input(shape=(n_models,), name='base_predictions')
    
    # Neural network to learn weights based on features with regularization
    x = feature_input
    
    # Build dynamic architecture based on hidden_units list
    for i, units in enumerate(hidden_units):
        x = Dense(units, activation='relu', 
                  kernel_regularizer=regularizers.l2(l2_reg),
                  name=f'hidden_layer_{i+1}')(x)
        x = Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
    
    # Output layer for raw logits
    logits = Dense(n_models, 
                   kernel_regularizer=regularizers.l2(l2_reg),
                   name='model_logits')(x)
    
    # Apply constrained softmax to get weights
    # Method 1: Using custom layer (recommended)
    weights = ConstrainedSoftmax(max_weight=max_weight, name='model_weights')(logits)
    
    # Method 2: Alternative using activation function (uncomment if Method 1 fails)
    # weights = Dense(n_models, 
    #                activation=constrained_softmax_activation(max_weight),
    #                kernel_regularizer=regularizers.l2(l2_reg),
    #                name='model_weights')(x)
    
    # Calculate weighted prediction using Keras operations
    # Method 1: Using Multiply and Lambda layers
    weighted_preds = Multiply(name='weighted_predictions')([weights, base_preds_input])
    weighted_pred = Lambda(lambda x: tf.keras.backend.sum(x, axis=1, keepdims=True), 
                          name='final_prediction')(weighted_preds)
    
    # Create the model
    meta_model = Model(inputs=[feature_input, base_preds_input], outputs=weighted_pred, name='meta_learner')
    
    if verbose:
        print("✓ Meta-learner architecture:")
        meta_model.summary()
    
    if verbose:
        print("\nStep 3: Compiling and Training Meta-Learner")
    
    # Compile model with standard MSE loss since weighted prediction is calculated in the model
    meta_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # Create a custom callback for RMSE tracking
    class CustomRMSECallback(Callback):
        def __init__(self, X_train, y_train, y_trainmodelpreds, X_val=None, y_val=None, y_valmodelpreds=None, max_weight=0.5):
            super().__init__()
            self.X_train = X_train
            self.y_train = y_train
            self.y_trainmodelpreds = y_trainmodelpreds
            self.X_val = X_val
            self.y_val = y_val
            self.y_valmodelpreds = y_valmodelpreds
            self.train_rmse = []
            self.val_rmse = []
            self.max_weight = max_weight
            self.weight_violations = []
            
        def on_epoch_end(self, epoch, logs=None):
            # Calculate training RMSE
            train_pred = self.model.predict([self.X_train, self.y_trainmodelpreds], verbose=0)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred.flatten()))
            self.train_rmse.append(train_rmse)
            
            # Check weight constraints
            # Extract the constraint layer to verify weights
            for layer in self.model.layers:
                if isinstance(layer, ConstrainedSoftmax) or layer.name == 'model_weights':
                    # Get a sample of inputs to check weights
                    sample_features = self.X_train[:100]  # Check first 100 samples
                    
                    # Get intermediate model to extract weights
                    weight_extractor = None
                    for i, l in enumerate(self.model.layers):
                        if l.name == 'features':
                            input_layer = l
                        elif isinstance(l, ConstrainedSoftmax) or l.name == 'model_weights':
                            output_layer = l
                            # Create intermediate model
                            intermediate_model = Model(inputs=self.model.input[0], outputs=output_layer.output)
                            weights = intermediate_model.predict(sample_features, verbose=0)
                            max_weight_found = np.max(weights)
                            
                            if max_weight_found > self.max_weight + 1e-6:
                                self.weight_violations.append((epoch, max_weight_found))
                                if epoch % 10 == 0:
                                    print(f"\n⚠️  Weight constraint violation at epoch {epoch+1}: max_weight={max_weight_found:.4f} > {self.max_weight}")
                            break
            
            # Calculate validation RMSE if validation data is provided
            if self.X_val is not None:
                val_pred = self.model.predict([self.X_val, self.y_valmodelpreds], verbose=0)
                val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred.flatten()))
                self.val_rmse.append(val_rmse)
                if epoch % 5 == 0 or epoch == 0:  # Print every 5 epochs
                    print(f"Epoch {epoch+1}: Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
            else:
                if epoch % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1}: Train RMSE: {train_rmse:.4f}")
    
    # Setup callbacks for optimal training
    rmse_callback = CustomRMSECallback(
        X_train_split, y_train_split, y_trainmodelpreds_split,
        X_val, y_val, y_valmodelpreds,
        max_weight=max_weight
    )
    
    callbacks = [rmse_callback]
    
    # Add early stopping and learning rate reduction if validation data is available
    if validation_split > 0:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1 if verbose else 0
        )
        
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1 if verbose else 0
        )
        
        callbacks.extend([early_stopping, lr_scheduler])
    
    # Train model
    if verbose:
        print("✓ Starting training...")
        print(f"✓ Batch size: {batch_size} (optimized for {n_samples:,} samples)")
        print(f"✓ Steps per epoch: {n_samples // batch_size}")
    
    history = meta_model.fit(
        [X_train_split, y_trainmodelpreds_split], y_train_split,
        validation_data=([X_val, y_valmodelpreds], y_val) if validation_split > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    
    if verbose:
        final_train_rmse = rmse_callback.train_rmse[-1]
        print(f"✓ Training completed!")
        print(f"✓ Final Training RMSE: {final_train_rmse:.4f}")
        if validation_split > 0:
            final_val_rmse = rmse_callback.val_rmse[-1]
            print(f"✓ Final Validation RMSE: {final_val_rmse:.4f}")
        
        # Report on weight constraint violations during training
        if rmse_callback.weight_violations:
            print(f"\n⚠️  WARNING: Weight constraint violations detected during training!")
            print(f"   Total violations: {len(rmse_callback.weight_violations)}")
            print(f"   Max violation: {max(v[1] for v in rmse_callback.weight_violations):.4f}")
        else:
            print(f"✓ Weight constraints maintained throughout training (max_weight={max_weight})")
    
    if verbose:
        print("\nStep 4: Generating Predictions and Weights")
    
    # Create a separate model to extract weights for analysis
    # We need to rebuild the weight generation path to ensure constraint is applied
    weight_input = Input(shape=(n_features,), name='weight_features')
    
    # Rebuild the same architecture for weight extraction
    x_weight = weight_input
    for i, units in enumerate(hidden_units):
        x_weight = Dense(units, activation='relu', 
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name=f'weight_hidden_{i+1}')(x_weight)
        x_weight = Dropout(dropout_rate, name=f'weight_dropout_{i+1}')(x_weight)
    
    # Generate logits
    weight_logits = Dense(n_models, 
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name='weight_logits')(x_weight)
    
    # Apply the same constrained softmax
    constrained_weights = ConstrainedSoftmax(max_weight=max_weight, name='constrained_weights')(weight_logits)
    
    weight_model = Model(inputs=weight_input, outputs=constrained_weights, name='weight_extractor')
    
    # Copy weights from the trained meta-model to the weight extraction model
    # Match layers by name pattern
    for layer in meta_model.layers:
        if 'hidden_layer_' in layer.name or 'model_logits' in layer.name:
            # Find corresponding layer in weight_model
            layer_num = layer.name.split('_')[-1]
            if 'hidden_layer_' in layer.name:
                weight_layer_name = f'weight_hidden_{layer_num}'
            else:  # model_logits
                weight_layer_name = 'weight_logits'
            
            try:
                weight_layer = weight_model.get_layer(weight_layer_name)
                weight_layer.set_weights(layer.get_weights())
            except:
                pass  # Skip if layer not found
    
    # Generate weights and predictions
    train_weights = weight_model.predict(X_train, verbose=0)
    test_weights = weight_model.predict(X_test, verbose=0)
    
    # Generate final predictions
    train_weighted_pred = meta_model.predict([X_train, y_trainmodelpreds], verbose=0).flatten()
    test_weighted_pred = meta_model.predict([X_test, y_testmodelpreds], verbose=0).flatten()
    
    if verbose:
        print("✓ Generated predictions for all data")
        print(f"✓ Average base model weights:")
        avg_weights = np.mean(train_weights, axis=0)
        for i, weight in enumerate(avg_weights):
            print(f"   Model {i+1}: {weight:.4f}")
        
        # Verify constraint enforcement on extracted weights
        max_weight_found = np.max(train_weights)
        min_weight_found = np.min(train_weights)
        print(f"\n✓ Weight statistics:")
        print(f"   Max weight across all samples: {max_weight_found:.4f}")
        print(f"   Min weight across all samples: {min_weight_found:.4f}")
        print(f"   Constraint limit: {max_weight:.4f}")
        
        if max_weight_found > max_weight + 1e-6:
            print(f"\n⚠️  WARNING: Weight constraint violated in extracted weights!")
            print(f"   This suggests an issue with weight extraction model.")
            
            # Debug: Check if the constraint layer is in the weight model
            print("\n   Debugging weight extraction model:")
            for layer in weight_model.layers:
                print(f"   - {layer.name}: {type(layer).__name__}")
                if isinstance(layer, ConstrainedSoftmax):
                    print(f"     ✓ ConstrainedSoftmax found with max_weight={layer.max_weight}")
        else:
            print(f"✓ Weight constraint properly enforced in extracted weights")
    
    if verbose:
        print("\nStep 5: Creating Visualizations")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Stacked Ensemble Meta-Learner Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Training and validation RMSE curves
    axes[0, 0].plot(rmse_callback.train_rmse, label='Training RMSE', linewidth=2)
    if validation_split > 0:
        axes[0, 0].plot(rmse_callback.val_rmse, label='Validation RMSE', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average base model weights
    model_names = [f'Model {i+1}' for i in range(n_models)]
    avg_weights = np.mean(train_weights, axis=0)
    bars = axes[0, 1].bar(model_names, avg_weights, color='skyblue', edgecolor='navy', alpha=0.7)
    axes[0, 1].set_xlabel('Base Models')
    axes[0, 1].set_ylabel('Average Weight')
    axes[0, 1].set_title('Average Base Model Weights')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, weight in zip(bars, avg_weights):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{weight:.3f}', ha='center', va='bottom')
    
    # Plot 3: Weighted predictions vs actuals (training data)
    axes[1, 0].scatter(y_train, train_weighted_pred, alpha=0.6, color='coral', s=10)
    min_val = min(y_train.min(), train_weighted_pred.min())
    max_val = max(y_train.max(), train_weighted_pred.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Weighted Predictions')
    axes[1, 0].set_title('Predictions vs Actuals (Training)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Calculate and display R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_train, train_weighted_pred)
    axes[1, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[1, 0].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: Weight distribution across samples
    axes[1, 1].boxplot([train_weights[:, i] for i in range(n_models)], labels=model_names)
    axes[1, 1].set_xlabel('Base Models')
    axes[1, 1].set_ylabel('Weight Distribution')
    axes[1, 1].set_title('Weight Distribution Across Samples')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    if verbose:
        print("✓ Visualizations generated")
        print("\nStep 6: Creating Output DataFrames")
    
    # Create training DataFrame
    train_data = {
        'Actual': y_train,
        'Weighted_Prediction': train_weighted_pred
    }
    
    # Add individual model predictions and weights
    for i in range(n_models):
        train_data[f'Model{i+1}_Pred'] = y_trainmodelpreds[:, i]
        train_data[f'Model{i+1}_Weight'] = train_weights[:, i]
    
    train_df = pd.DataFrame(train_data)
    
    # Create test DataFrame
    test_data = {
        'Weighted_Prediction': test_weighted_pred
    }
    
    # Add individual model predictions and weights
    for i in range(n_models):
        test_data[f'Model{i+1}_Pred'] = y_testmodelpreds[:, i]
        test_data[f'Model{i+1}_Weight'] = test_weights[:, i]
    
    test_df = pd.DataFrame(test_data)
    
    if verbose:
        print("✓ DataFrames created successfully")
        print(f"✓ Training DataFrame shape: {train_df.shape}")
        print(f"✓ Test DataFrame shape: {test_df.shape}")
        print("\nStep 7: Final Summary")
        print("="*80)
        print("ENSEMBLE LEARNING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Final Training RMSE: {np.sqrt(mean_squared_error(y_train, train_weighted_pred)):.4f}")
        print("Use the returned DataFrames to analyze results and make predictions.")
        print("="*80)
    
    return train_df, test_df

def get_optimal_hyperparameters(n_samples, n_features, n_models, max_weight=0.5):
    """
    Returns optimal hyperparameters based on dataset size for the stacked ensemble.
    
    Parameters:
    -----------
    n_samples : int
        Number of training samples
    n_features : int
        Number of input features
    n_models : int
        Number of base models
    max_weight : float, default=0.5
        Maximum weight any single model can receive
        
    Returns:
    --------
    dict : Dictionary containing optimal hyperparameters
    """
    
    # Base configuration for large datasets (500K+ samples)
    if n_samples >= 500000:
        config = {
            'batch_size': min(2048, n_samples // 200),  # Larger batches for stability
            'hidden_units': [256, 128, 64],  # Deeper network for complex patterns
            'dropout_rate': 0.4,  # Higher dropout for regularization
            'learning_rate': 0.0005,  # Lower LR for stable convergence
            'l2_reg': 0.001,
            'epochs': 150,
            'validation_split': 0.15,  # Smaller validation split for more training data
            'max_weight': max_weight,
        }
    # Medium datasets (100K-500K samples)
    elif n_samples >= 100000:
        config = {
            'batch_size': min(1024, n_samples // 100),
            'hidden_units': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'l2_reg': 0.0005,
            'epochs': 100,
            'validation_split': 0.2,
            'max_weight': max_weight,
        }
    # Small datasets (<100K samples)
    else:
        config = {
            'batch_size': min(512, n_samples // 50),
            'hidden_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'l2_reg': 0.0001,
            'epochs': 80,
            'validation_split': 0.25,
            'max_weight': max_weight,
        }
    
    # Adjust based on number of features
    if n_features > 20:
        config['hidden_units'][0] = max(config['hidden_units'][0], n_features * 4)
    
    # Adjust based on number of models
    if n_models > 10:
        config['hidden_units'][-1] = max(config['hidden_units'][-1], n_models * 2)
    
    print(f"Optimal hyperparameters for {n_samples:,} samples, {n_features} features, {n_models} models:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

# Example usage (commented out):
"""
# For your specific case (750K samples, 13-14 features, 7 models):
# Default: max weight = 0.5 (no single model can dominate)
optimal_params = get_optimal_hyperparameters(750000, 14, 7)

# Or customize the maximum weight constraint:
# optimal_params = get_optimal_hyperparameters(750000, 14, 7, max_weight=0.4)  # Even more balanced
# optimal_params = get_optimal_hyperparameters(750000, 14, 7, max_weight=0.6)  # Allow slight dominance

train_df, test_df = stacked_ensemble_meta_learner(
    X_train=X_train,
    y_train=y_train,
    y_trainmodelpreds=base_model_train_predictions,
    y_testmodelpreds=base_model_test_predictions,
    X_test=X_test,
    **optimal_params,  # Use optimal parameters
    verbose=True
)

# Analyze weight distribution to see the constraint in action:
weight_columns = [col for col in train_df.columns if 'Weight' in col]
print("Weight statistics:")
print(train_df[weight_columns].describe())
print(f"Maximum weight per sample: {train_df[weight_columns].max(axis=1).max():.4f}")
"""