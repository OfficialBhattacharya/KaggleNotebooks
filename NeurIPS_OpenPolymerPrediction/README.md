# Polymer Property Prediction Model

This repository contains a sophisticated model for predicting polymer properties from SMILES strings, developed for the NeurIPS Open Polymer Prediction 2025 competition.

## Features

- **Chemical Validity**: Uses RDKit for SMILES parsing and validation
- **Comprehensive Feature Engineering**:
  - Constitutional descriptors (atom counts, molecular weight)
  - Topological descriptors (Morgan fingerprints)
  - Polar surface area and rotatable bond counts
- **Multi-output XGBoost Model**:
  - Separate regression heads for each property
  - Quantile regression for uncertainty estimation
  - Feature importance analysis
- **Competition Metric**: Implements weighted MAE with property-specific weights
- **User-friendly Interface**: Simple API for training and prediction
- **Visualization Tools**: RDKit-based molecular visualization

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For RDKit installation on Windows, you may need to use conda:
```bash
conda install -c conda-forge rdkit
```

## Usage

```python
from polymer_predictor import PolymerPredictor, load_data

# Load data
train_smiles, train_targets, test = load_data()

# Initialize and train model
model = PolymerPredictor(n_estimators=1000, learning_rate=0.01)
model.fit(train_smiles, train_targets)

# Make predictions
predictions = model.predict(test['SMILES'].values)

# Visualize a monomer
model.visualize_monomer("C=CC(=O)OC")
```

## Model Architecture

### Feature Engineering
1. **Constitutional Descriptors**:
   - Number of atoms
   - Molecular weight
   - Heavy atom count
   - Number of rotatable bonds
   - Topological polar surface area
   - LogP

2. **Morgan Fingerprints**:
   - Radius: 3
   - 2048 bits
   - Captures local structural information

3. **Topological Features**:
   - Number of rings
   - Number of aromatic rings
   - Number of heteroatoms

### Model Details
- XGBoost with quantile regression
- Property-specific feature scaling
- Separate models for each property
- Memory-efficient implementation

## Chemical Validation

The model includes several validation steps:
1. SMILES parsing validation
2. Polymer-specific pattern checking
3. Size validation (minimum 4 atoms)
4. Physical property range checks

## Performance Optimization

- Progress bars for long operations
- Memory-efficient feature extraction
- Parallel processing where applicable
- Early stopping for model training

## Contributing

Feel free to submit issues and enhancement requests! 