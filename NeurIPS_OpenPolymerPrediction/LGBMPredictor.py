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
import lightgbm as lgb
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
    """Check if GPU is available and return appropriate LightGBM parameters."""
    try:
        # Try to create a simple LightGBM model to check GPU availability
        lgb.LGBMRegressor(device='gpu').get_params()
        logger.info("GPU detected! Using GPU acceleration for LightGBM")
        return {'device': 'gpu'}
    except Exception as e:
        logger.warning(f"GPU not available: {str(e)}. Falling back to CPU")
        return {'device': 'cpu'}

class PolymerPredictor:
    def __init__(self, num_boost_round: int = 20000, learning_rate: float = 0.001):
        """Initialize the polymer property predictor.
        
        Args:
            num_boost_round: Number of boosting iterations
            learning_rate: Learning rate for LightGBM
        """
        logger.info("Initializing PolymerPredictor model...")
        self.num_boost_round = num_boost_round
        self.learning_rate = learning_rate
        self.models = {}
        self.scalers = {}
        
        # Check GPU availability and set LightGBM parameters
        self.lgb_params = check_gpu_availability()
        
        # Define reasonable ranges for each property
        self.property_ranges = {
            'Tg': 500,      # Glass transition temperature range in K
            'FFV': 0.5,     # Fractional free volume (0-1 range)
            'Tc': 1000,     # Critical temperature range in K
            'Density': 2.0, # Density range in g/cm³
            'Rg': 100       # Radius of gyration range in Å
        }
        logger.info(f"Model initialized with {num_boost_round} boosting rounds and learning rate {learning_rate}")
        logger.info(f"Using LightGBM parameters: {self.lgb_params}")
        
    def _extract_features(self, smiles_list: List[str]) -> np.ndarray:
        """Extract chemical features from SMILES string using RDKit.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            np.ndarray: Feature vector containing molecular descriptors
        """
        logger.info("Starting feature extraction from SMILES strings...")
        features = []
        morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
        
        for smiles in tqdm(smiles_list, desc="Extracting features"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    logger.warning(f"Invalid SMILES string: {smiles}")
                    features.append(np.zeros(200))
                    continue
                
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
                
                # Morgan fingerprints
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
                
                feature_vector = feature_vector[:200] + [0] * (200 - len(feature_vector))
                features.append(feature_vector)
                
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {str(e)}")
                features.append(np.zeros(200))
        
        logger.info(f"Successfully extracted features for {len(features)} molecules")
        return np.array(features)
        
    def _weighted_mae(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """Calculate weighted MAE as per competition metric."""
        logger.info("Calculating weighted MAE score...")
        
        n_p = {col: len(y_true[col].dropna()) for col in y_true.columns}
        logger.info(f"Number of samples per property: {n_p}")
        
        weights = {p: (n_p[p]**-0.5)/self.property_ranges[p] for p in self.property_ranges}
        total_weight = sum(weights.values())
        norm_weights = {p: weights[p] * len(weights)/total_weight for p in weights}
        logger.info(f"Normalized weights: {norm_weights}")
        
        errors = np.abs(y_true - y_pred)
        weighted_score = np.mean([errors[p].mean() * norm_weights[p] for p in y_true.columns])
        
        logger.info(f"Final weighted MAE score: {weighted_score:.4f}")
        return weighted_score
        
    def optimize_lgb(self, X, y, param_space, n_trials=20):
        """Use Optuna to find the best LightGBM hyperparameters."""
        logger.info(f"Starting Optuna optimization for property with {n_trials} trials...")
        def objective(trial):
            params = {
                'objective': 'mae',
                'metric': 'mae',
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
                'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
                'verbosity': -1,
                **self.lgb_params
            }

            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            preds = model.predict(X_valid)
            return mean_absolute_error(y_valid, preds)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        logger.info(f"Best params found: {study.best_params}")
        return study.best_params

    def fit(self, train_smiles: list, train_targets: pd.DataFrame, optimize=False, param_space=None, n_trials=20) -> None:
        """Train the model on the provided data."""
        logger.info("Starting model training process...")
        
        X = self._extract_features(train_smiles)
        logger.info(f"Extracted {X.shape[1]} features from {X.shape[0]} molecules")
        
        for property_name in train_targets.columns:
            logger.info(f"\nTraining model for {property_name}...")
            
            valid_idx = ~train_targets[property_name].isna()
            if not valid_idx.any():
                logger.warning(f"No valid data for {property_name}, skipping...")
                continue
                
            X_prop = X[valid_idx]
            y_prop = train_targets[property_name][valid_idx]
            logger.info(f"Training data size for {property_name}: {len(y_prop)} samples")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_prop)
            self.scalers[property_name] = scaler
            
            if optimize and param_space is not None:
                logger.info(f"Optimizing hyperparameters for {property_name} using Optuna...")
                best_params = self.optimize_lgb(X_scaled, y_prop, param_space, n_trials)
                model_params = best_params.copy()
            else:
                model_params = {
                    'objective': 'mae',
                    'metric': 'mae',
                    'learning_rate': self.learning_rate,
                    'num_leaves': 100,
                    'min_data_in_leaf': 30,
                    'max_depth': -1,
                    'max_bin': 256,
                    'boosting': 'gbdt',
                    'feature_fraction': 0.7,
                    'bagging_freq': 1,
                    'bagging_fraction': 0.7,
                    'bagging_seed': 42,
                    'lambda_l1': 1,
                    'lambda_l2': 1,
                    'verbosity': -1,
                    **self.lgb_params
                }
            
            logger.info("Training LightGBM model with parameters: {}".format(model_params))
            train_data = lgb.Dataset(X_scaled, label=y_prop)
            
            model = lgb.train(
                model_params,
                train_data,
                num_boost_round=self.num_boost_round,
                valid_sets=[train_data],
                early_stopping_rounds=50,
                verbose_eval=100
            )
            
            self.models[property_name] = model
            logger.info(f"Completed training for {property_name}")
            
    def predict(self, test_smiles: list) -> pd.DataFrame:
        """Make predictions for new SMILES strings."""
        logger.info("Starting prediction process...")
        
        X = self._extract_features(test_smiles)
        logger.info(f"Extracted features for {len(test_smiles)} test molecules")
        
        predictions = {}
        for property_name, model in self.models.items():
            logger.info(f"Making predictions for {property_name}...")
            X_scaled = self.scalers[property_name].transform(X)
            predictions[property_name] = model.predict(X_scaled)
            
        logger.info("Completed all predictions")
        return pd.DataFrame(predictions)

    def fit_predict_best_model(self, train_smiles, train_targets, test_smiles, best_params):
        """Fit and predict in one shot using the best parameters."""
        logger.info("Fitting and predicting with best parameters in one shot...")
        X_train = self._extract_features(train_smiles)
        X_test = self._extract_features(test_smiles)
        predictions = {}

        for property_name in train_targets.columns:
            logger.info(f"Processing property: {property_name}")
            valid_idx = ~train_targets[property_name].isna()
            if not valid_idx.any():
                logger.warning(f"No valid data for {property_name}, skipping...")
                continue

            X_prop = X_train[valid_idx]
            y_prop = train_targets[property_name][valid_idx]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_prop)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[property_name] = scaler

            model_params = best_params.copy()
            model_params.update(self.lgb_params)
            model_params['objective'] = 'mae'
            model_params['metric'] = 'mae'

            logger.info(f"Training LightGBM model for {property_name} with best parameters: {model_params}")
            train_data = lgb.Dataset(X_scaled, label=y_prop)
            
            model = lgb.train(
                model_params,
                train_data,
                num_boost_round=self.num_boost_round,
                valid_sets=[train_data],
                early_stopping_rounds=50,
                verbose_eval=100
            )
            
            self.models[property_name] = model
            predictions[property_name] = model.predict(X_test_scaled)

        logger.info("Completed fit and predict for all properties.")
        return pd.DataFrame(predictions)

    def save_models(self, path):
        """Save all models and scalers using joblib."""
        logger.info(f"Saving models and scalers to {path}")
        save_dict = {
            'models': self.models,
            'scalers': self.scalers
        }
        joblib.dump(save_dict, path)
        logger.info("Models and scalers saved successfully")

def load_data():
    """Load competition data."""
    logger.info("Loading competition data...")
    
    train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
    test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    logger.info(f"Loaded {len(train)} training samples and {len(test)} test samples")
    
    train_smiles = train['SMILES'].values
    train_targets = train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]
    
    for col in train_targets.columns:
        valid_count = train_targets[col].notna().sum()
        logger.info(f"Property {col}: {valid_count} valid samples")
    
    return train_smiles, train_targets, test

if __name__ == "__main__":
    # --- Optuna Hyperparameter Optimization Usage ---
    # Uncomment to run Optuna hyperparameter search and fit
    # param_space = {
    #     'learning_rate': (0.001, 0.1),              # Float range (log scale)
    #     'num_leaves': (20, 150),                    # Integer range
    #     'min_data_in_leaf': (10, 100),              # Integer range
    #     'max_depth': (3, 12),                       # Integer range
    #     'feature_fraction': (0.5, 1.0),             # Float range
    #     'bagging_fraction': (0.5, 1.0),             # Float range
    #     'lambda_l1': (0, 5),                        # Float range
    #     'lambda_l2': (0, 5),                        # Float range
    #     'boosting': ['gbdt', 'dart'],               # Categorical
    #     'device': ['gpu', 'cpu']                    # Categorical
    # }
    # n_trials = 30
    # logger.info("Initializing and training model with Optuna hyperparameter optimization...")
    # model = PolymerPredictor()
    # model.fit(
    #     train_smiles,
    #     train_targets,
    #     optimize=True,
    #     param_space=param_space,
    #     n_trials=n_trials
    # )
    # logger.info("Making predictions on test set...")
    # predictions = model.predict(test['SMILES'].values)
    # logger.info("Creating submission file...")
    # submission = pd.DataFrame({
    #     'id': test['id'],
    #     **predictions
    # })
    # logger.info("Saving submission file...")
    # submission.to_csv('/kaggle/working/submission.csv', index=False)
    # logger.info("Process completed successfully!")

    # --- Fit and Predict with Best Parameters Usage ---
    # Uncomment to run fit and predict in one shot with best parameters
    # best_params = {
    #     'learning_rate': 0.012283669018343166,
    #     'num_leaves': 100,
    #     'min_data_in_leaf': 30,
    #     'max_depth': 6,
    #     'feature_fraction': 0.7,
    #     'bagging_fraction': 0.7,
    #     'lambda_l1': 1,
    #     'lambda_l2': 1
    # }
    # logger.info("Running fit and predict with best parameters...")
    # model = PolymerPredictor()
    # predictions = model.fit_predict_best_model(
    #     train_smiles,
    #     train_targets,
    #     test['SMILES'].values,
    #     best_params
    # )
    # submission = pd.DataFrame({
    #     'id': test['id'],
    #     **predictions
    # })
    # submission.to_csv('/kaggle/working/submission.csv', index=False)
    # logger.info("Process completed successfully!")

    # === Default: One-shot fit and predict with best parameters ===
    # Load data
    train_smiles, train_targets, test = load_data()

    # Your best parameters from Optuna
    best_params = {
        'learning_rate': 1e-3,
        'num_leaves': 100,
        'min_data_in_leaf': 30,
        'max_depth': -1,
        'max_bin': 256,
        'boosting': 'gbdt',
        'feature_fraction': 0.7,
        'bagging_freq': 1,
        'bagging_fraction': 0.7,
        'bagging_seed': 42,
        'lambda_l1': 1,
        'lambda_l2': 1,
        'verbosity': -1,
        'num_boost_round': 20000
    }

    logger.info("Running fit and predict with best parameters...")
    model = PolymerPredictor()
    predictions = model.fit_predict_best_model(
        train_smiles,
        train_targets,
        test['SMILES'].values,
        best_params
    )

    # Create submission
    submission = pd.DataFrame({
        'id': test['id'],
        **predictions
    })
    submission.to_csv('/kaggle/working/submission.csv', index=False)
    
    # Save models
    model.save_models('/kaggle/working/polymer_models.joblib')
    
    logger.info("Process completed successfully!") 