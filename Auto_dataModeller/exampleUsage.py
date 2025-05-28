import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import logging
from datetime import datetime, timedelta
import random

# Import the EDA functions
from Auto_dataModeller.edaExplorer import (
    remove_single_unique_or_all_nans,
    columns_with_missing_values,
    columns_with_more_than_X_percent_unique,
    get_numeric_and_non_numeric_columns,
    plot_numeric_features,
    plot_categorical_features,
    plot_correlation_heatmap,
    plot_categorical_boxplots,
    plot_numeric_analysis_with_sampling,
    generate_pair_plot_for_numeric_columns,
    plot_numeric_vs_target_density,
    generate_categorical_numeric_plot,
    stacked_bar_plot,
    plot_countplot_by_target,
    plot_cdf_numerical_by_target,
    plot_average_numerical_by_categorical_and_target,
    plot_pairplot_with_target_hue
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_synthetic_dataset(n_samples=1000):
    """
    Create a synthetic dataset with both numerical and categorical features
    for binary classification.
    """
    # Create numerical features using make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=2,
        random_state=42
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f'numeric_feature_{i+1}' for i in range(5)])
    df['target'] = y
    
    # Add categorical features
    categories = {
        'category_A': ['A1', 'A2', 'A3'],
        'category_B': ['B1', 'B2', 'B3', 'B4'],
        'category_C': ['C1', 'C2']
    }
    
    for cat_name, cat_values in categories.items():
        df[cat_name] = np.random.choice(cat_values, size=n_samples)
    
    # Add datetime feature
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(n_samples)]
    random.shuffle(dates)
    df['date'] = dates
    
    # Add some missing values
    for col in df.columns:
        if col != 'target':
            mask = np.random.random(n_samples) < 0.05
            df.loc[mask, col] = np.nan
    
    return df

def main():
    # Create synthetic dataset
    logging.info("Creating synthetic dataset...")
    df = create_synthetic_dataset()
    
    # Basic data exploration
    logging.info("\nDataset Shape:")
    print(df.shape)
    
    logging.info("\nDataset Info:")
    print(df.info())
    
    logging.info("\nBasic Statistics:")
    print(df.describe())
    
    # Remove columns with single unique value or all NaNs
    df = remove_single_unique_or_all_nans(df)
    
    # Get columns with missing values
    missing_cols = columns_with_missing_values(df)
    
    # Get numeric and non-numeric columns
    numeric_cols, non_numeric_cols = get_numeric_and_non_numeric_columns(df)
    
    # Plot numeric features
    logging.info("\nPlotting numeric features...")
    plot_numeric_features(df, numeric_cols, apply_box_cox=True)
    
    # Plot categorical features
    logging.info("\nPlotting categorical features...")
    plot_categorical_features(df, non_numeric_cols)
    
    # Plot correlation heatmap
    logging.info("\nPlotting correlation heatmap...")
    plot_correlation_heatmap(df, numeric_cols, corr_type="spearman")
    
    # Plot categorical boxplots
    logging.info("\nPlotting categorical boxplots...")
    plot_categorical_boxplots(df, non_numeric_cols, 'numeric_feature_1')
    
    # Plot numeric analysis with sampling
    logging.info("\nPlotting numeric analysis with sampling...")
    plot_numeric_analysis_with_sampling(df, numeric_cols, 'target', sample=0.5)
    
    # Generate pair plot for numeric columns
    logging.info("\nGenerating pair plot...")
    generate_pair_plot_for_numeric_columns(df, numeric_cols, hue='target', sample_size=500)
    
    # Plot numeric vs target density
    logging.info("\nPlotting numeric vs target density...")
    plot_numeric_vs_target_density(df, numeric_cols, 'target')
    
    # Generate categorical numeric plot
    logging.info("\nGenerating categorical numeric plot...")
    generate_categorical_numeric_plot(
        df, 
        'category_A', 
        'category_B', 
        'numeric_feature_1',
        cat1_order=['A1', 'A2', 'A3'],
        cat2_order=['B1', 'B2', 'B3', 'B4']
    )
    
    # Plot stacked bar plot
    logging.info("\nPlotting stacked bar plot...")
    stacked_bar_plot(df, 'category_A', 'target')
    
    # Plot countplot by target
    logging.info("\nPlotting countplot by target...")
    plot_countplot_by_target(df, 'category_B', 'target')
    
    # Plot CDF numerical by target
    logging.info("\nPlotting CDF numerical by target...")
    plot_cdf_numerical_by_target(df, numeric_cols, 'target')
    
    # Plot average numerical by categorical and target
    logging.info("\nPlotting average numerical by categorical and target...")
    plot_average_numerical_by_categorical_and_target(
        df, 
        'category_A', 
        'numeric_feature_1', 
        'target'
    )
    
    # Plot pairplot with target hue
    logging.info("\nPlotting pairplot with target hue...")
    plot_pairplot_with_target_hue(
        df,
        numeric_cols,
        'target',
        sample_size=500
    )

if __name__ == "__main__":
    main()
