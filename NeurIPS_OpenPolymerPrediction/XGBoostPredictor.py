import pandas as pd
import numpy as np
from typing import List
import warnings
import logging
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('polymer_prediction.log')
    ]
)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check if GPU is available and return appropriate XGBoost parameters."""
    try:
        # Try to create a simple XGBoost model to check GPU availability
        xgb.XGBRegressor(tree_method='gpu_hist').get_params()
        logger.info("GPU detected! Using GPU acceleration for XGBoost")
        return {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'predictor': 'gpu_predictor'
        }
    except Exception as e:
        logger.warning(f"GPU not available: {str(e)}. Falling back to CPU")
        return {
            'tree_method': 'hist',
            'predictor': 'cpu_predictor'
        }

class PolymerPredictor:
    def __init__(self, n_estimators: int = 1000, learning_rate: float = 0.01):
        """Initialize the polymer property predictor.
        
        Args:
            n_estimators: Number of trees in XGBoost model
            learning_rate: Learning rate for XGBoost
        """
        logger.info("Initializing PolymerPredictor model...")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = {}
        self.scalers = {}
        
        # Check GPU availability and set XGBoost parameters
        self.xgb_params = check_gpu_availability()
        
        # Define reasonable ranges for each property
        # These ranges help in normalizing the weighted MAE metric
        self.property_ranges = {
            'Tg': 500,      # Glass transition temperature range in K
            'FFV': 0.5,     # Fractional free volume (0-1 range)
            'Tc': 1000,     # Critical temperature range in K
            'Density': 2.0, # Density range in g/cm³
            'Rg': 100       # Radius of gyration range in Å
        }
        logger.info(f"Model initialized with {n_estimators} trees and learning rate {learning_rate}")
        logger.info(f"Using XGBoost parameters: {self.xgb_params}")
        
    def _extract_features(self, smiles_list: List[str]) -> np.ndarray:
        """Extract chemical features from SMILES string using RDKit.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            np.ndarray: Feature vector containing molecular descriptors
        """
        logger.info("Starting feature extraction from SMILES strings...")
        features = []
        # Use the new MorganGenerator API to avoid deprecation warnings
        morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
        
        for smiles in tqdm(smiles_list, desc="Extracting features"):
            try:
                # Convert SMILES to RDKit molecule
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES string: {smiles}")
                    features.append(np.zeros(200))  # Default feature vector
                    continue
                
                # Calculate molecular descriptors
                feature_vector = []
                
                # Basic descriptors
                feature_vector.extend([
                    Descriptors.MolWt(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumAliphaticRings(mol),
                    Descriptors.NumSaturatedRings(mol),
                    Descriptors.NumHeteroatoms(mol)
                ])
                
                # Morgan fingerprints (ECFP4) using the new generator API
                fp = morgan_gen.GetFingerprint(mol).ToBitString()
                fp_bits = [int(x) for x in fp]
                feature_vector.extend(fp_bits)
                
                # Additional descriptors
                feature_vector.extend([
                    rdMolDescriptors.CalcNumRings(mol),
                    rdMolDescriptors.CalcNumAromaticRings(mol),
                    rdMolDescriptors.CalcNumAliphaticRings(mol),
                    rdMolDescriptors.CalcNumSaturatedRings(mol),
                    rdMolDescriptors.CalcNumHeterocycles(mol),
                    rdMolDescriptors.CalcNumSpiroAtoms(mol),
                    rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                    rdMolDescriptors.CalcNumAtomStereoCenters(mol),
                    rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
                ])
                
                # Pad or truncate to ensure consistent feature vector size
                feature_vector = feature_vector[:200] + [0] * (200 - len(feature_vector))
                features.append(feature_vector)
                
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {str(e)}")
                features.append(np.zeros(200))
        
        logger.info(f"Successfully extracted features for {len(features)} molecules")
        return np.array(features)
        
    def _weighted_mae(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """Calculate weighted MAE as per competition metric.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            float: Weighted MAE score
        """
        logger.info("Calculating weighted MAE score...")
        
        # Count number of samples for each property
        n_p = {col: len(y_true[col].dropna()) for col in y_true.columns}
        logger.info(f"Number of samples per property: {n_p}")
        
        # Calculate weights based on sample count and property range
        weights = {p: (n_p[p]**-0.5)/self.property_ranges[p] for p in self.property_ranges}
        total_weight = sum(weights.values())
        norm_weights = {p: weights[p] * len(weights)/total_weight for p in weights}
        logger.info(f"Normalized weights: {norm_weights}")
        
        # Calculate weighted errors
        errors = np.abs(y_true - y_pred)
        weighted_score = np.mean([errors[p].mean() * norm_weights[p] for p in y_true.columns])
        
        logger.info(f"Final weighted MAE score: {weighted_score:.4f}")
        return weighted_score
        
    def optimize_xgb(self, X, y, param_space, n_trials=20):
        """
        Use Optuna to find the best XGBoost hyperparameters for a given property.
        Args:
            X: Feature matrix
            y: Target vector
            param_space: Dictionary specifying the search space for each hyperparameter
            n_trials: Number of Optuna trials
        Returns:
            dict: Best hyperparameters found
        """
        logger.info(f"Starting Optuna optimization for property with {n_trials} trials...")
        def objective(trial):
            params = {}
            for k, v in param_space.items():
                if isinstance(v, tuple) and len(v) == 2:
                    # Float or int range
                    if all(isinstance(x, int) for x in v):
                        params[k] = trial.suggest_int(k, v[0], v[1])
                    else:
                        params[k] = trial.suggest_float(k, v[0], v[1], log=True if k == 'learning_rate' else False)
                elif isinstance(v, list):
                    params[k] = trial.suggest_categorical(k, v)
            params.update(self.xgb_params)
            params['objective'] = 'reg:absoluteerror'  # Changed to MAE objective
            params['random_state'] = 42

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=20, verbose=False)
            preds = model.predict(X_valid)
            return mean_absolute_error(y_valid, preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"Best params found: {study.best_params}")
        return study.best_params

    def fit(self, train_smiles: list, train_targets: pd.DataFrame, optimize=False, param_space=None, n_trials=20) -> None:
        """Train the model on the provided data. Optionally use Optuna for hyperparameter optimization.
        
        Args:
            train_smiles: List of SMILES strings
            train_targets: DataFrame with target properties
            optimize: Whether to use Optuna for hyperparameter search
            param_space: Dictionary specifying the search space for each hyperparameter
            n_trials: Number of Optuna trials
        """
        logger.info("Starting model training process...")
        
        # Extract features from SMILES strings
        logger.info("Extracting molecular features from SMILES strings using RDKit...")
        X = self._extract_features(train_smiles)
        logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} molecules")
        
        # Train separate models for each property
        for property_name in train_targets.columns:
            logger.info(f"\nTraining model for {property_name}...")
            
            # Get valid indices for this property
            valid_idx = ~train_targets[property_name].isna()
            if not valid_idx.any():
                logger.warning(f"No valid data for {property_name}, skipping...")
                continue
                
            X_prop = X[valid_idx]
            y_prop = train_targets[property_name][valid_idx]
            logger.info(f"Training data size for {property_name}: {len(y_prop)} samples")
            
            # Scale features
            logger.info("Scaling features using StandardScaler...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_prop)
            self.scalers[property_name] = scaler
            
            # Hyperparameter optimization with Optuna if requested
            if optimize and param_space is not None:
                logger.info(f"Optimizing hyperparameters for {property_name} using Optuna...")
                best_params = self.optimize_xgb(X_scaled, y_prop, param_space, n_trials)
                model_params = best_params.copy()
                model_params.update(self.xgb_params)
                model_params['objective'] = 'reg:absoluteerror'
                model_params['random_state'] = 42
            else:
                model_params = dict(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    objective='reg:absoluteerror',
                    random_state=42,
                    **self.xgb_params,
                    max_depth=7,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0
                )
            
            # Train XGBoost model
            logger.info("Training XGBoost model with parameters: {}".format(model_params))
            model = xgb.XGBRegressor(**model_params)
            logger.info("Starting model training with early stopping...")
            model.fit(
                X_scaled, y_prop,
                eval_set=[(X_scaled, y_prop)],
                early_stopping_rounds=50,
                verbose=100
            )
            
            self.models[property_name] = model
            logger.info(f"Completed training for {property_name}")
            
    def predict(self, test_smiles: list) -> pd.DataFrame:
        """Make predictions for new SMILES strings.
        
        Args:
            test_smiles: List of SMILES strings to predict
            
        Returns:
            pd.DataFrame: Predictions for all properties
        """
        logger.info("Starting prediction process...")
        
        # Extract features from test SMILES
        logger.info("Extracting features from test molecules using RDKit...")
        X = self._extract_features(test_smiles)
        logger.info(f"Extracted features for {len(test_smiles)} test molecules")
        
        # Make predictions for each property
        predictions = {}
        for property_name, model in self.models.items():
            logger.info(f"Making predictions for {property_name}...")
            X_scaled = self.scalers[property_name].transform(X)
            
            # Create DMatrix for faster prediction
            dtest = xgb.DMatrix(X_scaled)
            predictions[property_name] = model.predict(X_scaled)
            
        logger.info("Completed all predictions")
        return pd.DataFrame(predictions)

    def fit_predict_best_model(self, train_smiles, train_targets, test_smiles, best_params):
        """
        Fit and predict in one shot using the best parameters for all properties.
        Args:
            train_smiles: List of SMILES strings for training
            train_targets: DataFrame with target properties
            test_smiles: List of SMILES strings for prediction
            best_params: Dict of best hyperparameters (from Optuna or manual)
        Returns:
            pd.DataFrame: Predictions for all properties
        """
        logger.info("Fitting and predicting with best parameters in one shot...")
        # Extract features
        X_train = self._extract_features(train_smiles)
        X_test = self._extract_features(test_smiles)
        predictions = {}

        # Create directory for saving models if it doesn't exist
        os.makedirs('saved_models', exist_ok=True)

        for property_name in train_targets.columns:
            logger.info(f"Processing property: {property_name}")
            valid_idx = ~train_targets[property_name].isna()
            if not valid_idx.any():
                logger.warning(f"No valid data for {property_name}, skipping...")
                continue

            X_prop = X_train[valid_idx]
            y_prop = train_targets[property_name][valid_idx]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_prop)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[property_name] = scaler

            # Merge best params with default XGBoost params
            model_params = best_params.copy()
            model_params.update(self.xgb_params)
            model_params['objective'] = 'reg:absoluteerror'  # Changed to MAE objective
            model_params['random_state'] = 42

            logger.info(f"Training XGBoost model for {property_name} with best parameters: {model_params}")
            model = xgb.XGBRegressor(**model_params)
            model.fit(X_scaled, y_prop, eval_set=[(X_scaled, y_prop)], early_stopping_rounds=50, verbose=100)
            self.models[property_name] = model

            # Save model and scaler
            model_path = os.path.join('saved_models', f'{property_name}_model.joblib')
            scaler_path = os.path.join('saved_models', f'{property_name}_scaler.joblib')
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved model and scaler for {property_name}")

            # Predict
            predictions[property_name] = model.predict(X_test_scaled)

        logger.info("Completed fit and predict for all properties.")
        return pd.DataFrame(predictions)

def load_data():
    """Load competition data.
    
    Returns:
        Tuple: (train_smiles, train_targets, test_df)
    """
    logger.info("Loading competition data...")
    
    # Load training and test data
    train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    logger.info(f"Loaded {len(train)} training samples and {len(test)} test samples")
    
    # Extract SMILES and target properties
    train_smiles = train['SMILES'].values
    train_targets = train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]
    
    # Log data statistics
    for col in train_targets.columns:
        valid_count = train_targets[col].notna().sum()
        logger.info(f"Property {col}: {valid_count} valid samples")
    
    return train_smiles, train_targets, test

if __name__ == "__main__":
    # Load data
    train_smiles, train_targets, test = load_data()
    
    #--- Optuna Hyperparameter Optimization Usage ---
    param_space = {
        'n_estimators': (1000, 4500),                # Integer range
        'learning_rate': (0.01, 0.4),              # Float range (log scale for learning_rate)
        'max_depth': (7, 10),                       # Integer range
        'subsample': (0.6, 0.8),                    # Float range
        'colsample_bytree': (0.6, 0.8),             # Float range
        'gamma': (0, 1.0),                          # Float range
        'reg_alpha': (0, 1.0),                      # Float range
        'reg_lambda': (0, 2.0),                     # Float range
    }
    n_trials = 30
    logger.info("Initializing and training model with Optuna hyperparameter optimization...")
    model = PolymerPredictor()
    
    # Dictionary to store best parameters for each property
    best_params_per_property = {}
    
    # Train models for each property
    for property_name in train_targets.columns:
        logger.info(f"\nOptimizing hyperparameters for {property_name}...")
        valid_idx = ~train_targets[property_name].isna()
        if not valid_idx.any():
            logger.warning(f"No valid data for {property_name}, skipping...")
            continue
            
        X = model._extract_features(train_smiles[valid_idx])
        y = train_targets[property_name][valid_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Optimize hyperparameters
        best_params = model.optimize_xgb(X_scaled, y, param_space, n_trials)
        best_params_per_property[property_name] = best_params
        
        # Save scaler
        scaler_path = f'/kaggle/working/{property_name}_scaler.joblib'
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler for {property_name} to {scaler_path}")
    
    # Print best parameters for each property
    logger.info("\nBest hyperparameters for each property:")
    best_params_df = pd.DataFrame(best_params_per_property).T
    print("\nBest Hyperparameters per Property:")
    print(best_params_df)
    
    # Train final models with best parameters
    model.fit(
        train_smiles,
        train_targets,
        optimize=False  # We already have the best parameters
    )
    
    # Save models
    for property_name, model_obj in model.models.items():
        model_path = f'/kaggle/working/{property_name}_model.joblib'
        joblib.dump(model_obj, model_path)
        logger.info(f"Saved model for {property_name} to {model_path}")
    
    logger.info("Making predictions on test set...")
    predictions = model.predict(test['SMILES'].values)
    logger.info("Creating submission file...")
    submission = pd.DataFrame({
        'id': test['id'],
        **predictions
    })
    logger.info("Saving submission file...")
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    logger.info("Process completed successfully!") 

    # === Alternative: Load saved models, refit with best parameters, and predict ===
    """
    # Load data
    train_smiles, train_targets, test = load_data()
    
    # Initialize predictor
    model = PolymerPredictor()
    
    # Dictionary to store best parameters for each property
    best_params_per_property = {}
    
    # Define the base path for saved models
    base_path = '/kaggle/working'
    
    # Load saved models and extract their parameters
    for property_name in train_targets.columns:
        model_path = os.path.join(base_path, f'{property_name}_model.joblib')
        scaler_path = os.path.join(base_path, f'{property_name}_scaler.joblib')
        
        logger.info(f"\nChecking for saved model at: {model_path}")
        logger.info(f"Checking for saved scaler at: {scaler_path}")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            logger.info(f"Found saved files for {property_name}")
            try:
                # Load and verify the model
                saved_model = joblib.load(model_path)
                logger.info(f"Successfully loaded model for {property_name}")
                
                # Print raw model parameters for verification
                logger.info(f"\nRaw parameters from saved model for {property_name}:")
                raw_params = saved_model.get_params()
                for param, value in raw_params.items():
                    logger.info(f"{param}: {value}")
                
                # Extract specific parameters we need
                best_params = {
                    'n_estimators': raw_params.get('n_estimators'),
                    'learning_rate': raw_params.get('learning_rate'),
                    'max_depth': raw_params.get('max_depth'),
                    'subsample': raw_params.get('subsample'),
                    'colsample_bytree': raw_params.get('colsample_bytree'),
                    'gamma': raw_params.get('gamma'),
                    'reg_alpha': raw_params.get('reg_alpha'),
                    'reg_lambda': raw_params.get('reg_lambda')
                }
                
                # Verify the extracted parameters
                logger.info(f"\nExtracted parameters for {property_name}:")
                for param, value in best_params.items():
                    logger.info(f"{param}: {value}")
                
                best_params_per_property[property_name] = best_params
                
                # Load the scaler
                model.scalers[property_name] = joblib.load(scaler_path)
                logger.info(f"Successfully loaded scaler for {property_name}")
                
            except Exception as e:
                logger.error(f"Error loading model or scaler for {property_name}: {str(e)}")
                continue
        else:
            logger.warning(f"Saved files not found for {property_name}")
            if not os.path.exists(model_path):
                logger.warning(f"Model file missing: {model_path}")
            if not os.path.exists(scaler_path):
                logger.warning(f"Scaler file missing: {scaler_path}")
    
    # Print best parameters for each property
    logger.info("\nBest hyperparameters for each property:")
    best_params_df = pd.DataFrame(best_params_per_property).T
    print("\nBest Hyperparameters per Property:")
    print(best_params_df)
    
    # Verify we have parameters for all properties
    missing_properties = set(train_targets.columns) - set(best_params_per_property.keys())
    if missing_properties:
        logger.warning(f"Missing parameters for properties: {missing_properties}")
        raise ValueError("Not all properties have parameters loaded")
    
    # Extract features from training data
    logger.info("Extracting features from training molecules...")
    X_train = model._extract_features(train_smiles)
    
    # Train new models with best parameters
    for property_name in train_targets.columns:
        if property_name not in best_params_per_property:
            continue
            
        logger.info(f"\nTraining new model for {property_name} with best parameters...")
        valid_idx = ~train_targets[property_name].isna()
        if not valid_idx.any():
            logger.warning(f"No valid data for {property_name}, skipping...")
            continue
            
        X_prop = X_train[valid_idx]
        y_prop = train_targets[property_name][valid_idx]
        
        # Scale features
        X_scaled = model.scalers[property_name].transform(X_prop)
        
        # Train new model with best parameters
        model_params = best_params_per_property[property_name].copy()
        model_params.update(model.xgb_params)  # Add GPU/CPU parameters
        model_params['objective'] = 'reg:absoluteerror'
        model_params['random_state'] = 42
        
        logger.info(f"Training XGBoost model for {property_name} with parameters: {model_params}")
        new_model = xgb.XGBRegressor(**model_params)
        new_model.fit(
            X_scaled, y_prop,
            eval_set=[(X_scaled, y_prop)],
            early_stopping_rounds=50,
            verbose=100
        )
        model.models[property_name] = new_model
        
        # Save the newly trained model
        model_path = os.path.join(base_path, f'{property_name}_model_refit.joblib')
        joblib.dump(new_model, model_path)
        logger.info(f"Saved refit model for {property_name} to {model_path}")
    
    # Extract features from test SMILES
    logger.info("Extracting features from test molecules...")
    X_test = model._extract_features(test['SMILES'].values)
    
    # Make predictions
    logger.info("Making predictions using refit models...")
    predictions = {}
    for property_name, model_obj in model.models.items():
        logger.info(f"Predicting {property_name}...")
        X_test_scaled = model.scalers[property_name].transform(X_test)
        preds = model_obj.predict(X_test_scaled)
        predictions[property_name] = preds
        
        # Validate predictions
        logger.info(f"Property {property_name} predictions - Shape: {preds.shape}, Min: {preds.min():.2f}, Max: {preds.max():.2f}")
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test['id'].values,  # Ensure we're using values
        **predictions
    })
    
    # Validate submission format
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Submission columns: {submission.columns.tolist()}")
    logger.info(f"Submission dtypes:\n{submission.dtypes}")
    
    # Check for any missing values
    missing_values = submission.isnull().sum()
    if missing_values.any():
        logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
        # Fill missing values with median of each column
        for col in submission.columns:
            if submission[col].isnull().any():
                median_val = submission[col].median()
                submission[col] = submission[col].fillna(median_val)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
    
    # Ensure all columns are in the correct order
    expected_columns = ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
    submission = submission[expected_columns]
    
    # Save submission
    submission.to_csv(os.path.join(base_path, 'submission_refit_models.csv'), index=False)
    logger.info("Predictions completed using refit models!")
    """ 