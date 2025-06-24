"""
Compact Dataset Creator for NeurIPS Open Polymer Prediction 2025

This script creates a focused, high-quality dataset by:
1. Using only Tg and Tc data (which are polymer-relevant)
2. Excluding PI1M data (which contains many invalid SMILES)
3. Including gas permeability data where available
4. Filtering for valid SMILES only
5. Prioritizing samples with actual property data

Output:
- compact_additional_dataset.csv: Clean, focused dataset
- compact_additional_dataset_summary.txt: Summary statistics
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import warnings
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompactDatasetCreator:
    def __init__(self, raw_data_dir='RawData', output_dir='.'):
        """
        Initialize the compact dataset creator.
        
        Args:
            raw_data_dir: Directory containing raw data files
            output_dir: Directory to save output files
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.compact_df = None
        self.stats = {}
        
    def is_valid_smiles(self, smiles):
        """Check if SMILES string is valid using basic pattern matching."""
        if pd.isna(smiles) or not isinstance(smiles, str):
            return False
        
        # Basic SMILES validation patterns
        # Check for common invalid patterns
        invalid_patterns = [
            r'[^A-Za-z0-9\[\]\(\)=#@\+\-\.\*%]',  # Invalid characters
            r'[A-Z]{3,}',  # Too many consecutive uppercase letters
            r'[0-9]{3,}',  # Too many consecutive numbers
            r'[A-Z][A-Z][A-Z]',  # Three consecutive uppercase letters
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, smiles):
                return False
        
        # Check for balanced parentheses and brackets
        if smiles.count('(') != smiles.count(')') or smiles.count('[') != smiles.count(']'):
            return False
        
        # Check for reasonable length
        if len(smiles) < 3 or len(smiles) > 500:
            return False
        
        # Check for common polymer patterns (should start with *)
        if not smiles.startswith('*'):
            return False
        
        return True
    
    def load_tg_data(self):
        """Load and clean Tg data."""
        logger.info("Loading Tg data...")
        
        tg_path = self.raw_data_dir / 'Tg_SMILES_class_pid_polyinfo_median.csv'
        if tg_path.exists():
            tg_df = pd.read_csv(tg_path)
            logger.info(f"Loaded {len(tg_df)} Tg samples")
            
            # Filter for valid SMILES
            tg_df['valid_smiles'] = tg_df['SMILES'].apply(self.is_valid_smiles)
            tg_df = tg_df[tg_df['valid_smiles']].drop('valid_smiles', axis=1)
            logger.info(f"Valid Tg samples: {len(tg_df)}")
            
            # Clean and prepare Tg data
            tg_clean = tg_df[['SMILES', 'PID', 'Tg']].copy()
            tg_clean[['FFV', 'Tc', 'Density', 'Rg']] = np.nan
            
            # Add gas columns
            for gas in ['CH4', 'CO2', 'H2', 'N2', 'O2']:
                tg_clean[gas] = np.nan
            
            self.stats['tg_samples'] = len(tg_clean)
            return tg_clean
        else:
            logger.warning("Tg_SMILES_class_pid_polyinfo_median.csv not found")
            return pd.DataFrame()
    
    def load_tc_data(self):
        """Load and clean Tc data."""
        logger.info("Loading Tc data...")
        
        tc_path = self.raw_data_dir / 'TC_MD_20240306.xlsx'
        if tc_path.exists():
            tc_df = pd.read_excel(tc_path)
            logger.info(f"Loaded {len(tc_df)} Tc samples")
            
            # Note: Tc data doesn't have SMILES, so we can't validate it
            # We'll merge it with Tg data later
            tc_clean = tc_df[['PID', 'TC_mean']].copy()
            tc_clean = tc_clean.rename(columns={'TC_mean': 'Tc'})
            tc_clean[['SMILES', 'Tg', 'FFV', 'Density', 'Rg']] = np.nan
            
            # Add gas columns
            for gas in ['CH4', 'CO2', 'H2', 'N2', 'O2']:
                tc_clean[gas] = np.nan
            
            self.stats['tc_samples'] = len(tc_clean)
            return tc_clean
        else:
            logger.warning("TC_MD_20240306.xlsx not found")
            return pd.DataFrame()
    
    def load_gas_data(self):
        """Load gas permeability data."""
        logger.info("Loading gas permeability data...")
        
        gas_files = ['CH4_raw.csv', 'CO2_raw.csv', 'H2_raw.csv', 'N2_raw.csv', 'O2_raw.csv']
        gas_data = {}
        
        for gas_file in gas_files:
            gas_path = self.raw_data_dir / gas_file
            if gas_path.exists():
                gas_name = gas_file.replace('_raw.csv', '')
                gas_df = pd.read_csv(gas_path)
                
                # Filter for valid SMILES
                gas_df['valid_smiles'] = gas_df['SMILES'].apply(self.is_valid_smiles)
                gas_df = gas_df[gas_df['valid_smiles']].drop('valid_smiles', axis=1)
                
                gas_data[gas_name] = gas_df
                logger.info(f"Loaded {len(gas_df)} valid {gas_name} samples")
                self.stats[f'{gas_name.lower()}_samples'] = len(gas_df)
            else:
                logger.warning(f"{gas_file} not found")
        
        return gas_data
    
    def merge_tg_tc_data(self, tg_df, tc_df):
        """Merge Tg and Tc data based on PID."""
        logger.info("Merging Tg and Tc data...")
        
        if tg_df.empty and tc_df.empty:
            return pd.DataFrame()
        
        if tg_df.empty:
            return tc_df
        elif tc_df.empty:
            return tg_df
        
        # Merge on PID
        merged_df = pd.merge(tg_df, tc_df[['PID', 'Tc']], on='PID', how='outer')
        
        # Update Tc column in merged data
        merged_df['Tc'] = merged_df['Tc_y'].fillna(merged_df['Tc_x'])
        merged_df = merged_df.drop(['Tc_x', 'Tc_y'], axis=1)
        
        # Ensure all required columns exist
        required_cols = ['PID', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg', 'CH4', 'CO2', 'H2', 'N2', 'O2']
        for col in required_cols:
            if col not in merged_df.columns:
                merged_df[col] = np.nan
        
        # Reorder columns
        merged_df = merged_df[required_cols]
        
        logger.info(f"Merged dataset has {len(merged_df)} samples")
        return merged_df
    
    def merge_gas_data(self, master_df, gas_data):
        """Merge gas permeability data into master_df by SMILES."""
        logger.info("Merging gas permeability data by SMILES...")
        for gas, df in gas_data.items():
            if 'SMILES' in df.columns and df.shape[1] == 2:
                df = df.rename(columns={df.columns[1]: gas})
                master_df = master_df.merge(df, on='SMILES', how='left')
            else:
                logger.warning(f"Gas file for {gas} does not have expected columns.")
        return master_df
    
    def create_compact_dataset(self):
        """Create the compact additional dataset."""
        logger.info("Creating compact additional dataset...")
        
        # Load only relevant data (no PI1M)
        tg_df = self.load_tg_data()
        tc_df = self.load_tc_data()
        gas_data = self.load_gas_data()
        
        # Merge Tg and Tc data
        additional_df = self.merge_tg_tc_data(tg_df, tc_df)
        
        if additional_df.empty:
            logger.warning("No additional data found!")
            return None
        
        # Merge gas data by SMILES
        if gas_data:
            additional_df = self.merge_gas_data(additional_df, gas_data)
            self.stats['gas_data_merged'] = True
        
        # Remove rows with no SMILES (from Tc-only data)
        additional_df = additional_df.dropna(subset=['SMILES'])
        
        # Ensure all required columns exist before reordering
        required_cols = ['id', 'PID', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg', 'CH4', 'CO2', 'H2', 'N2', 'O2']
        for col in required_cols:
            if col not in additional_df.columns:
                additional_df[col] = np.nan
        
        # Generate unique IDs
        additional_df = additional_df.reset_index(drop=True)
        additional_df['id'] = range(len(additional_df))
        
        # Move id to first column
        additional_df = additional_df[required_cols]
        
        self.compact_df = additional_df.copy()
        logger.info(f"Compact additional dataset created with {len(self.compact_df)} samples")
        return self.compact_df
    
    def save_compact_dataset(self):
        """Save the compact dataset and summary."""
        if self.compact_df is None:
            logger.error("No compact dataset to save!")
            return
        
        # Save compact dataset
        output_path = self.output_dir / 'compact_additional_dataset.csv'
        self.compact_df.to_csv(output_path, index=False)
        logger.info(f"Compact dataset saved to {output_path}")
        
        # Create and save summary
        self.create_summary()
        
    def create_summary(self):
        """Create a summary of the compact dataset."""
        if self.compact_df is None:
            return
        
        summary_lines = []
        summary_lines.append("COMPACT ADDITIONAL DATASET SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Total samples: {len(self.compact_df)}")
        summary_lines.append("")
        
        # Property statistics
        properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg', 'CH4', 'CO2', 'H2', 'N2', 'O2']
        summary_lines.append("PROPERTY STATISTICS:")
        summary_lines.append("-" * 30)
        
        for prop in properties:
            if prop in self.compact_df.columns:
                valid_count = self.compact_df[prop].notna().sum()
                total_count = len(self.compact_df)
                percentage = (valid_count / total_count) * 100 if total_count > 0 else 0
                
                if valid_count > 0:
                    mean_val = self.compact_df[prop].mean()
                    std_val = self.compact_df[prop].std()
                    min_val = self.compact_df[prop].min()
                    max_val = self.compact_df[prop].max()
                    
                    summary_lines.append(f"{prop}:")
                    summary_lines.append(f"  Valid samples: {valid_count}/{total_count} ({percentage:.1f}%)")
                    summary_lines.append(f"  Mean: {mean_val:.4f}")
                    summary_lines.append(f"  Std: {std_val:.4f}")
                    summary_lines.append(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
                    summary_lines.append("")
                else:
                    summary_lines.append(f"{prop}: No valid data")
                    summary_lines.append("")
        
        # Dataset statistics
        summary_lines.append("DATASET STATISTICS:")
        summary_lines.append("-" * 30)
        for key, value in self.stats.items():
            summary_lines.append(f"{key}: {value}")
        
        # Save summary
        summary_path = self.output_dir / 'compact_additional_dataset_summary.txt'
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Summary saved to {summary_path}")
        
        # Print summary to console
        print('\n'.join(summary_lines))
    
    def run(self):
        """Run the complete dataset creation process."""
        logger.info("Starting compact dataset creation...")
        
        try:
            # Create compact dataset
            compact_df = self.create_compact_dataset()
            
            if compact_df is not None:
                # Save results
                self.save_compact_dataset()
                logger.info("Compact dataset creation completed successfully!")
                return True
            else:
                logger.error("Failed to create compact dataset!")
                return False
                
        except Exception as e:
            logger.error(f"Error during dataset creation: {str(e)}")
            return False

def main():
    """Main function to run the dataset creator."""
    # Create dataset creator
    creator = CompactDatasetCreator()
    
    # Run the creation process
    success = creator.run()
    
    if success:
        print("\n✅ Compact dataset creation completed successfully!")
        print("📁 Check the output files:")
        print("   - compact_additional_dataset.csv")
        print("   - compact_additional_dataset_summary.txt")
    else:
        print("\n❌ Compact dataset creation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 