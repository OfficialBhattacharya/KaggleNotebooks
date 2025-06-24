"""
Master Dataset Creator for NeurIPS Open Polymer Prediction 2025

This script processes all raw data files and creates a master additional dataset
that can be used by both LGBMPredictor.py and XGBoostPredictor.py.

The script handles:
1. Tg_SMILES_class_pid_polyinfo_median.csv - Contains Tg values with PID mapping
2. TC_MD_20240306.xlsx - Contains Tc values with PID mapping
3. Gas data files (CH4_raw.csv, CO2_raw.csv, H2_raw.csv, N2_raw.csv, O2_raw.csv) - Contains gas permeability data
4. PI1M.csv and PI1M_v2.csv - Contains additional SMILES strings
5. train.csv and test.csv - Competition data

Output:
- master_additional_dataset.csv: Combined dataset with all available properties
- master_additional_dataset_summary.txt: Summary statistics of the combined dataset
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MasterDatasetCreator:
    def __init__(self, raw_data_dir='RawData', output_dir='.'):
        """
        Initialize the master dataset creator.
        
        Args:
            raw_data_dir: Directory containing raw data files
            output_dir: Directory to save output files
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.master_df = None
        self.stats = {}
        
    def load_competition_data(self):
        """Load competition training and test data."""
        logger.info("Loading competition data...")
        
        # Load training data
        train_path = self.raw_data_dir / 'train.csv'
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            logger.info(f"Loaded {len(train_df)} training samples")
            self.stats['competition_train'] = len(train_df)
        else:
            logger.warning("train.csv not found in raw data directory")
            train_df = pd.DataFrame()
            
        # Load test data
        test_path = self.raw_data_dir / 'test.csv'
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            logger.info(f"Loaded {len(test_df)} test samples")
            self.stats['competition_test'] = len(test_df)
        else:
            logger.warning("test.csv not found in raw data directory")
            test_df = pd.DataFrame()
            
        return train_df, test_df
    
    def load_tg_data(self):
        """Load Tg data from Tg_SMILES_class_pid_polyinfo_median.csv."""
        logger.info("Loading Tg data...")
        
        tg_path = self.raw_data_dir / 'Tg_SMILES_class_pid_polyinfo_median.csv'
        if tg_path.exists():
            tg_df = pd.read_csv(tg_path)
            logger.info(f"Loaded {len(tg_df)} Tg samples")
            self.stats['tg_samples'] = len(tg_df)
            
            # Clean and prepare Tg data
            tg_clean = tg_df[['SMILES', 'PID', 'Tg']].copy()
            tg_clean[['FFV', 'Tc', 'Density', 'Rg']] = np.nan
            
            # Reorder columns to match competition format
            tg_clean = tg_clean[['PID', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
            
            return tg_clean
        else:
            logger.warning("Tg_SMILES_class_pid_polyinfo_median.csv not found")
            return pd.DataFrame()
    
    def load_tc_data(self):
        """Load Tc data from TC_MD_20240306.xlsx."""
        logger.info("Loading Tc data...")
        
        tc_path = self.raw_data_dir / 'TC_MD_20240306.xlsx'
        if tc_path.exists():
            tc_df = pd.read_excel(tc_path)
            logger.info(f"Loaded {len(tc_df)} Tc samples")
            self.stats['tc_samples'] = len(tc_df)
            
            # Clean and prepare Tc data
            tc_clean = tc_df[['PID', 'TC_mean']].copy()
            tc_clean = tc_clean.rename(columns={'TC_mean': 'Tc'})
            tc_clean[['SMILES', 'Tg', 'FFV', 'Density', 'Rg']] = np.nan
            
            # Reorder columns to match competition format
            tc_clean = tc_clean[['PID', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
            
            return tc_clean
        else:
            logger.warning("TC_MD_20240306.xlsx not found")
            return pd.DataFrame()
    
    def load_gas_data(self):
        """Load gas permeability data from all gas files."""
        logger.info("Loading gas permeability data...")
        
        gas_files = ['CH4_raw.csv', 'CO2_raw.csv', 'H2_raw.csv', 'N2_raw.csv', 'O2_raw.csv']
        gas_data = {}
        
        for gas_file in gas_files:
            gas_path = self.raw_data_dir / gas_file
            if gas_path.exists():
                gas_name = gas_file.replace('_raw.csv', '')
                gas_df = pd.read_csv(gas_path)
                gas_data[gas_name] = gas_df
                logger.info(f"Loaded {len(gas_df)} {gas_name} samples")
                self.stats[f'{gas_name.lower()}_samples'] = len(gas_df)
            else:
                logger.warning(f"{gas_file} not found")
        
        return gas_data
    
    def load_pi1m_data(self):
        """Load PI1M data."""
        logger.info("Loading PI1M data...")
        
        pi1m_data = {}
        
        # Load PI1M.csv
        pi1m_path = self.raw_data_dir / 'PI1M.csv'
        if pi1m_path.exists():
            pi1m_df = pd.read_csv(pi1m_path)
            pi1m_data['PI1M'] = pi1m_df
            logger.info(f"Loaded {len(pi1m_df)} PI1M samples")
            self.stats['pi1m_samples'] = len(pi1m_df)
        
        # Load PI1M_v2.csv
        pi1m_v2_path = self.raw_data_dir / 'PI1M_v2.csv'
        if pi1m_v2_path.exists():
            pi1m_v2_df = pd.read_csv(pi1m_v2_path)
            pi1m_data['PI1M_v2'] = pi1m_v2_df
            logger.info(f"Loaded {len(pi1m_v2_df)} PI1M_v2 samples")
            self.stats['pi1m_v2_samples'] = len(pi1m_v2_df)
        
        return pi1m_data
    
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
        required_cols = ['PID', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
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

    def add_unique_pi1m_smiles(self, master_df, pi1m_data):
        """Add unique PI1M SMILES not already present in master_df."""
        logger.info("Adding unique PI1M SMILES not already present in master dataset...")
        all_pi1m_smiles = set()
        for key, df in pi1m_data.items():
            if 'SMILES' in df.columns:
                all_pi1m_smiles.update(df['SMILES'].unique())
        existing_smiles = set(master_df['SMILES'].unique())
        new_smiles = all_pi1m_smiles - existing_smiles
        logger.info(f"Found {len(new_smiles)} unique PI1M SMILES to add.")
        if not new_smiles:
            return master_df
        # Create new rows with NaN for all properties except SMILES
        new_rows = pd.DataFrame({
            'SMILES': list(new_smiles),
            'Tg': np.nan,
            'FFV': np.nan,
            'Tc': np.nan,
            'Density': np.nan,
            'Rg': np.nan,
            'CH4': np.nan,
            'CO2': np.nan,
            'H2': np.nan,
            'N2': np.nan,
            'O2': np.nan,
        })
        # Add dummy PID (can be blank or unique string)
        new_rows['PID'] = ''
        # Add id (will be reassigned later)
        new_rows['id'] = np.nan
        # Reorder columns
        cols = ['id', 'PID', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg', 'CH4', 'CO2', 'H2', 'N2', 'O2']
        new_rows = new_rows[cols]
        # Append
        master_df = pd.concat([master_df, new_rows], ignore_index=True)
        return master_df

    def create_master_dataset(self):
        """Create the master additional dataset."""
        logger.info("Creating master additional dataset...")
        # Load all data
        train_df, test_df = self.load_competition_data()
        tg_df = self.load_tg_data()
        tc_df = self.load_tc_data()
        gas_data = self.load_gas_data()
        pi1m_data = self.load_pi1m_data()
        # Merge Tg and Tc data
        additional_df = self.merge_tg_tc_data(tg_df, tc_df)
        if additional_df.empty:
            logger.warning("No additional data found!")
            return None
        # Add gas columns if not present
        for gas in ['CH4', 'CO2', 'H2', 'N2', 'O2']:
            if gas not in additional_df.columns:
                additional_df[gas] = np.nan
        # Merge gas data by SMILES
        if gas_data:
            additional_df = self.merge_gas_data(additional_df, gas_data)
            self.stats['gas_data_merged'] = True
        # Add PI1M unique SMILES
        if pi1m_data:
            additional_df = self.add_unique_pi1m_smiles(additional_df, pi1m_data)
            self.stats['pi1m_unique_added'] = True
        # Generate unique IDs for all rows
        additional_df = additional_df.reset_index(drop=True)
        additional_df['id'] = range(len(additional_df))
        # Move id to first column
        cols = ['id', 'PID', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg', 'CH4', 'CO2', 'H2', 'N2', 'O2']
        additional_df = additional_df[cols]
        self.master_df = additional_df.copy()
        logger.info(f"Master additional dataset created with {len(self.master_df)} samples (including PI1M and gas data)")
        return self.master_df
    
    def save_master_dataset(self):
        """Save the master dataset and summary."""
        if self.master_df is None:
            logger.error("No master dataset to save!")
            return
        
        # Save master dataset
        output_path = self.output_dir / 'master_additional_dataset.csv'
        self.master_df.to_csv(output_path, index=False)
        logger.info(f"Master dataset saved to {output_path}")
        
        # Create and save summary
        self.create_summary()
        
    def create_summary(self):
        """Create a summary of the master dataset."""
        if self.master_df is None:
            return
        summary_lines = []
        summary_lines.append("MASTER ADDITIONAL DATASET SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Total samples: {len(self.master_df)}")
        summary_lines.append("")
        # Property statistics
        properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg', 'CH4', 'CO2', 'H2', 'N2', 'O2']
        summary_lines.append("PROPERTY STATISTICS:")
        summary_lines.append("-" * 30)
        for prop in properties:
            if prop in self.master_df.columns:
                valid_count = self.master_df[prop].notna().sum()
                total_count = len(self.master_df)
                percentage = (valid_count / total_count) * 100 if total_count > 0 else 0
                if valid_count > 0:
                    mean_val = self.master_df[prop].mean()
                    std_val = self.master_df[prop].std()
                    min_val = self.master_df[prop].min()
                    max_val = self.master_df[prop].max()
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
        summary_path = self.output_dir / 'master_additional_dataset_summary.txt'
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        logger.info(f"Summary saved to {summary_path}")
        # Print summary to console
        print('\n'.join(summary_lines))
    
    def run(self):
        """Run the complete dataset creation process."""
        logger.info("Starting master dataset creation...")
        
        try:
            # Create master dataset
            master_df = self.create_master_dataset()
            
            if master_df is not None:
                # Save results
                self.save_master_dataset()
                logger.info("Master dataset creation completed successfully!")
                return True
            else:
                logger.error("Failed to create master dataset!")
                return False
                
        except Exception as e:
            logger.error(f"Error during dataset creation: {str(e)}")
            return False

def main():
    """Main function to run the dataset creator."""
    # Create dataset creator
    creator = MasterDatasetCreator()
    
    # Run the creation process
    success = creator.run()
    
    if success:
        print("\n✅ Master dataset creation completed successfully!")
        print("📁 Check the output files:")
        print("   - master_additional_dataset.csv")
        print("   - master_additional_dataset_summary.txt")
    else:
        print("\n❌ Master dataset creation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 