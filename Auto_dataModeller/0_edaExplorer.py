import os
import random
import warnings
from typing import List, Dict, Tuple, Optional, Callable
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, roc_curve
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2


import seaborn as sns
import matplotlib.pyplot as plt


from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from minepy import MINE  # For MIC (Maximum Information Coefficient)
import ppscore as pps  # For Predictive Power Score (PPS)

import re
from itertools import combinations
import Levenshtein
from collections import defaultdict
from scipy.interpolate import LSQUnivariateSpline
from scipy.io import arff
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.stats import boxcox
from scipy.stats import spearmanr, pearsonr

import pgmpy.estimators as ests
from pgmpy.estimators import TreeSearch
from pgmpy.models import BayesianNetwork
from pgmpy.metrics import structure_score 
from pgmpy.inference import BeliefPropagation
from pgmpy.inference import VariableElimination


import logging
import warnings
warnings.filterwarnings('ignore')



def intersection_of_lists(list1, list2):
    return list(set(list1) & set(list2))


def difference_of_lists(list1, list2):
    return [item for item in list1 if item not in list2]


def remove_single_unique_or_all_nans(df):
    removed_columns = []
    for column in df.columns:
        if df[column].nunique() <= 1 or df[column].isna().all():
            removed_columns.append(column)
            df = df.drop(columns=[column])
    print(f"Removed columns due to all NaN or only 1 unique value: {removed_columns}")
    return df


def columns_with_missing_values(df):
    missing_cols = [col for col in df.columns if df[col].isna().values.any()]
    print(f"Missing data columns: {missing_cols}")
    return missing_cols


def columns_with_more_than_X_percent_unique(df, colNames, perc):
    total_rows = len(df)
    threshold = total_rows * 0.01 * perc  
    cols_with_high_uniques = [col for col in colNames if df[col].nunique() > threshold]
    print(f"Columns with high uniques , >= {perc} %  of number of rows in the data: {cols_with_high_uniques}")
    return cols_with_high_uniques


def get_numeric_and_non_numeric_columns(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        print(f"Numeric columns: {numeric_cols}")
        print(f"Non-numeric columns: {non_numeric_cols}")
        return numeric_cols, non_numeric_cols





def plot_numeric_features(df, numerical_features, apply_box_cox=False):
    """
    Function to plot density plots for features with absolute skewness > 10, histograms otherwise,
    and box plots for all numerical features. Applies Box-Cox transformation if specified.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numerical_features (list): List of numeric column names.
    - apply_box_cox (bool): If True, applies Box-Cox transformation to features with high skewness.
    """
    for feature in numerical_features:
        # Drop rows with missing values for the current feature
        valid_data = df[feature].dropna()

        if valid_data.empty:
            print(f"No valid data available for feature: {feature}")
            continue
        
        # Calculate skewness
        skewness = valid_data.skew()

        plt.figure(figsize=(12, 6))

        # Conditional plotting based on skewness
        if abs(skewness) > 10:
            if apply_box_cox:
                # Apply Box-Cox transformation (only for positive values)
                valid_data = valid_data[valid_data > 0]  # Box-Cox requires positive values
                if valid_data.empty:
                    print(f"No valid positive data available for Box-Cox transformation for {feature}")
                    continue
                transformed_data, _ = boxcox(valid_data)
                plt.subplot(1, 2, 1)
                sns.kdeplot(transformed_data, fill=True)
                plt.title(f"Density Plot of {feature} (Box-Cox Transformed)")
                plt.xlabel(f"{feature} (Box-Cox Transformed)")
                plt.ylabel("Density")
            else:
                # Density plot for features with high skewness (without transformation)
                plt.subplot(1, 2, 1)
                sns.kdeplot(valid_data, fill=True)
                plt.title(f"Density Plot of {feature} (Skewness: {skewness:.2f})")
                plt.xlabel(feature)
                plt.ylabel("Density")
        else:
            # Histogram for features with lower skewness
            plt.subplot(1, 2, 1)
            sns.histplot(valid_data, kde=True, bins=30)
            plt.title(f"Histogram of {feature} (Skewness: {skewness:.2f})")
            plt.xlabel(feature)
            plt.ylabel("Frequency")

        # Box plot for all features
        plt.subplot(1, 2, 2)
        sns.boxplot(x=valid_data)
        plt.title(f"Box Plot of {feature}")
        
        plt.tight_layout()
        plt.show()

        # Print additional statistics
        print(f"\nStatistics for {feature}:")
        print(f"Skewness: {skewness:.2f}")
        print(f"Number of Missing Values: {df[feature].isnull().sum()}")


def plot_categorical_features(df, categorical_features):
    """
    Function to plot pie charts for categorical features with fewer than 10 unique values,
    or bar graphs otherwise.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - categorical_features (list): List of categorical column names.
    """
    for feature in categorical_features:
        # Drop rows with missing values for the current feature
        valid_data = df[feature].dropna()
        
        if valid_data.empty:
            print(f"No valid data available for feature: {feature}")
            continue
        
        # Calculate value counts
        value_counts = valid_data.value_counts()
        
        # Decide chart type based on the number of unique values
        if value_counts.size < 11:
            # Pie chart for features with fewer than 11 unique values
            percentages = (value_counts / value_counts.sum()) * 100
            plt.figure(figsize=(8, 8))
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            plt.title(f"Distribution of {feature} (Pie Chart)")
            plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        else:
            # Bar graph for features with 11 or more unique values
            plt.figure(figsize=(10, 6))
            plt.bar(value_counts.index, value_counts.values, color=plt.cm.Paired.colors[:len(value_counts)])
            plt.title(f"Distribution of {feature} (Bar Graph)")
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

        # Print additional statistics
        print(f"Statistics for {feature}:")
        print(f"Number of Unique Values: {df[feature].nunique()}")
        print(f"Missing Values in {feature}: {df[feature].isnull().sum()}")


def plot_correlation_heatmap(df, numerical_features, corr_type="spearman"):
    """
    Function to plot a correlation heatmap for numerical features using specified correlation type.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numerical_features (list): List of numeric column names.
    - corr_type (str): Type of correlation ('spearman', 'pearson', 'kendall', 'mic', 'pps').
    """
    # Filter valid numerical columns
    valid_data = df[numerical_features].dropna()
    
    if corr_type == "spearman":
        correlation_matrix = valid_data.corr(method="spearman")
    elif corr_type == "pearson":
        correlation_matrix = valid_data.corr(method="pearson")
    elif corr_type == "kendall":
        correlation_matrix = valid_data.corr(method="kendall")
    else:
        raise ValueError(f"Unsupported correlation type: {corr_type}. Use 'spearman', 'pearson', 'kendall'.")
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Matrix of Numerical Features ({corr_type.title()} Correlation)")
    plt.show()


def plot_categorical_boxplots(df, categorical_features, label_col):
    """
    Function to plot box plots for categorical features against a label column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - categorical_features (list): List of categorical column names.
    - label_col (str): Name of the label column for the y-axis.
    """
    for feature in categorical_features:
        # Skip high-cardinality features
        if feature not in ["Podcast_Name", "Episode_Title"]:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature], y=df[label_col])
            plt.title(f"{feature} vs. {label_col}")
            plt.xlabel(feature)
            plt.ylabel(label_col)
            plt.xticks(rotation=45)
            plt.tight_layout()  # Ensure plots fit within the figure area
            plt.show()


def plot_numeric_analysis_with_sampling(df, numeric_cols, label_col, sample=0.1):
    """
    Function to compute MIC and PPS scores and plot univariate relationships between numeric columns
    and a label column, using sampled data for faster processing.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numeric_cols (list): List of numeric column names.
    - label_col (str): Name of the label column (y-variable).
    - sample (float): Fraction of valid data to sample (between 0 and 1).
    """
    logging.info("Starting numeric analysis function.")
    mine = MINE()

    for numeric_col in numeric_cols:
        logging.info(f"Processing column: {numeric_col}")
        
        # Filter rows with non-missing values for both the numeric column and the label column
        valid_data = df[[numeric_col, label_col]].dropna()

        if valid_data.empty:
            logging.warning(f"No valid data available for {numeric_col} vs. {label_col}. Skipping...")
            continue

        logging.info(f"Valid data fetched for {numeric_col} vs. {label_col}. Rows: {len(valid_data)}")
        
        # Sample the data if the fraction is specified
        if 0 < sample < 1:
            valid_data = valid_data.sample(frac=sample, random_state=42)
            logging.info(f"Data sampled. Using {len(valid_data)} rows for analysis.")

        # MIC calculation
        logging.info(f"Calculating MIC for {numeric_col} vs. {label_col}.")
        mine.compute_score(valid_data[numeric_col].values, valid_data[label_col].values)
        mic_score = mine.mic()
        logging.info(f"MIC calculated: {mic_score:.2f}")

        # PPS calculation
        logging.info(f"Calculating PPS for {numeric_col} vs. {label_col}.")
        pps_score = pps.score(valid_data, x=numeric_col, y=label_col).get("ppscore", 0)
        logging.info(f"PPS calculated: {pps_score:.2f}")

        # Plot univariate relationship
        logging.info(f"Generating scatter plot for {numeric_col} vs. {label_col}.")
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_data[numeric_col], valid_data[label_col], alpha=0.6, color="blue")
        plt.title(f"{numeric_col} vs. {label_col} (MIC: {mic_score:.2f}, PPS: {pps_score:.2f})")
        plt.xlabel(numeric_col)
        plt.ylabel(label_col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        logging.info(f"Scatter plot displayed for {numeric_col} vs. {label_col}.")

        # Log the results
        logging.info(f"MIC: {mic_score:.2f}, PPS: {pps_score:.2f} for {numeric_col} vs. {label_col}.")

    logging.info("Numeric analysis function completed.")


def generate_pair_plot_for_numeric_columns(df, numeric_cols, hue=None, sample_size=1000):
    """
    Function to generate a pair plot for a given list of numeric columns, with optional hue and sampling.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numeric_cols (list): List of numeric columns to include in the pair plot.
    - hue (str, optional): Name of the categorical column for hue. If None, no hue is used.
    - sample_size (int): The number of rows to sample for the pair plot (default=5000).
    """
    logging.info("Starting pair plot generation.")

    # Check if all required numeric columns exist in the DataFrame
    if all(col in df.columns for col in numeric_cols):
        logging.info("All numeric columns found in the DataFrame.")
        
        # Sample the data if the dataset exceeds the sample size
        if len(df) > sample_size:
            logging.info(f"Sampling {sample_size} data points for the pair plot.")
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            logging.info("Using the entire dataset for the pair plot.")
            df_sample = df

        # Generate the pair plot
        logging.info("Generating the pair plot (this may take some time).")
        sns.pairplot(
            df_sample[numeric_cols].dropna(),
            hue=hue if hue and hue in df.columns else None,  # Use hue only if it's provided and valid
            palette='dark' if hue else None,
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 10},
        )
        title = f"Pairwise Relationships" + (f" by {hue}" if hue else "")
        plt.suptitle(title, y=1.02)
        plt.show()
        logging.info("Pair plot generated successfully.")
    else:
        missing_cols = [col for col in numeric_cols if col not in df.columns]
        logging.warning(f"Missing numeric columns for pair plot: {missing_cols}")


def plot_numeric_vs_target_density(df, numeric_cols, target_col):
    """
    Function to generate density plots (hexbin) for numeric columns against a target column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - numeric_cols (list): List of numeric column names to compare against the target column.
    - target_col (str): The target column (y-axis variable) for the density plot.
    """
    logging.info("Starting density plot generation.")
    
    # Check if the target column exists
    if target_col not in df.columns:
        logging.error(f"Target column '{target_col}' not found in the DataFrame.")
        return
    
    for numeric_col in numeric_cols:
        # Check if the numeric column exists in the DataFrame
        if numeric_col not in df.columns:
            logging.warning(f"Numeric column '{numeric_col}' not found in the DataFrame. Skipping...")
            continue
        
        logging.info(f"Generating density plot for {numeric_col} vs. {target_col}.")
        
        # Create the jointplot
        sns.jointplot(
            data=df,
            x=numeric_col,
            y=target_col,
            kind='hex',
            cmap='viridis',
            gridsize=40
        )
        plt.suptitle(f'Density of {numeric_col} vs. {target_col}', y=1.02)
        plt.tight_layout()
        plt.show()
        logging.info(f"Density plot for {numeric_col} vs. {target_col} generated successfully.")
    
    logging.info("Density plot generation completed.")


def generate_categorical_numeric_plot(
    df, 
    cat_col1, 
    cat_col2, 
    numeric_col, 
    cat1_order=None, 
    cat2_order=None, 
    figsize=(12, 6), 
    errorbar_ci=99
):
    """
    Function to generate a plot for a numeric column aggregated by two categorical columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - cat_col1 (str): Name of the first categorical column (e.g., 'Day_of_Week').
    - cat_col2 (str): Name of the second categorical column (e.g., 'Time_of_Day').
    - numeric_col (str): Name of the numeric column (e.g., 'Listening_Time_minutes').
    - cat1_order (list): Desired order of the first categorical column (optional).
    - cat2_order (list): Desired order of the second categorical column (optional).
    - figsize (tuple): Size of the plot (default=(12, 6)).
    - errorbar_ci (int): Confidence interval for error bars (default=99).

    Returns:
    None
    """
    try:
        logging.info("Starting categorical-numeric plot generation...")
        
        # Process the first categorical column
        if cat1_order:
            if cat_col1 in df.columns:
                logging.info(f"Processing {cat_col1} with specified order...")
                df[cat_col1] = pd.Categorical(df[cat_col1], categories=cat1_order, ordered=True)
            else:
                raise ValueError(f"Column '{cat_col1}' not found in the DataFrame.")
        
        # Process the second categorical column
        if cat2_order:
            if cat_col2 in df.columns:
                logging.info(f"Processing {cat_col2} with specified order...")
                df[cat_col2] = pd.Categorical(df[cat_col2], categories=cat2_order, ordered=True)
            else:
                raise ValueError(f"Column '{cat_col2}' not found in the DataFrame.")
        
        # Check if numeric column exists
        if numeric_col not in df.columns:
            raise ValueError(f"Numeric column '{numeric_col}' not found in the DataFrame.")
        
        # Prepare the data for plotting
        logging.info("Filtering data for valid rows...")
        plot_data = df.dropna(subset=[cat_col1, cat_col2, numeric_col])

        if plot_data.empty:
            logging.warning("No valid data available for plotting after filtering NaNs.")
            return
        
        # Generate the plot
        logging.info("Generating the plot...")
        plt.figure(figsize=figsize)
        palette = sns.color_palette("tab10", n_colors=len(cat2_order) if cat2_order else 10)
        sns.lineplot(
            data=plot_data,
            x=cat_col1,
            y=numeric_col,
            hue=cat_col2,
            hue_order=cat2_order,
            palette=palette,
            marker='o',
            errorbar=('ci', errorbar_ci)
        )
        plt.title(f'Average {numeric_col} by {cat_col1} and {cat_col2}')
        plt.xlabel(cat_col1)
        plt.ylabel(f'Average {numeric_col}')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title=cat_col2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
        logging.info("Plot generated successfully.")

    except Exception as e:
        logging.error(f"Error in categorical-numeric plot generation: {e}")


def stacked_bar_plot(df, feature, target='loan_status'):
    crosstab = pd.crosstab(df[feature], df[target], normalize='index')
    crosstab.plot(kind='bar', stacked=True, figsize=(12, 6), cmap='coolwarm')
    plt.title(f'Stacked Bar Plot of {feature} vs {target}')
    plt.ylabel('Proportion')
    plt.show()


def plot_countplot_by_target(
    df: pd.DataFrame, 
    categorical_col: str, 
    target_col: str, 
    figsize: tuple = (10, 6),
    title: str = None
) -> None:
    """
    Creates a seaborn countplot between a binary target (0,1) and a categorical column.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data
    - categorical_col (str): Name of the categorical column for x-axis
    - target_col (str): Name of the binary target column for hue
    - figsize (tuple): Figure size as (width, height), default (10, 6)
    - title (str): Custom title for the plot, if None uses default format
    
    Returns:
    None: Displays the plot
    """
    try:
        # Validate input columns exist
        if categorical_col not in df.columns:
            raise ValueError(f"Categorical column '{categorical_col}' not found in DataFrame")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Create the plot
        plt.figure(figsize=figsize)
        sns.countplot(data=df, x=categorical_col, hue=target_col)
        
        # Set title
        if title is None:
            title = f'Count Distribution of {categorical_col} by {target_col}'
        plt.title(title)
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        logging.info(f"Countplot generated successfully for {categorical_col} vs {target_col}")
        
    except Exception as e:
        logging.error(f"Error generating countplot: {e}")


def plot_cdf_numerical_by_target(
    df: pd.DataFrame, 
    numerical_cols: list, 
    target_col: str, 
    figsize: tuple = (10, 6),
    target_labels: dict = None
) -> None:
    """
    Creates CDF (Cumulative Distribution Function) plots of numerical columns by binary target variable.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data
    - numerical_cols (list): List of numerical column names to plot
    - target_col (str): Name of the binary target column (0,1)
    - figsize (tuple): Figure size as (width, height), default (10, 6)
    - target_labels (dict): Optional mapping for target values {0: 'label0', 1: 'label1'}
    
    Returns:
    None: Displays the plots for each numerical column
    """
    try:
        # Validate target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Default labels if not provided
        if target_labels is None:
            target_labels = {0: 'Class 0', 1: 'Class 1'}
        
        # Get unique target values
        target_values = sorted(df[target_col].dropna().unique())
        
        for numerical_col in numerical_cols:
            # Validate numerical column exists
            if numerical_col not in df.columns:
                logging.warning(f"Numerical column '{numerical_col}' not found in DataFrame. Skipping...")
                continue
            
            # Filter valid data
            valid_data = df[[numerical_col, target_col]].dropna()
            
            if valid_data.empty:
                logging.warning(f"No valid data for {numerical_col} vs {target_col}. Skipping...")
                continue
            
            # Create the plot
            plt.figure(figsize=figsize)
            
            for target_val in target_values:
                subset_data = valid_data[valid_data[target_col] == target_val][numerical_col]
                label = target_labels.get(target_val, f'Target {target_val}')
                sns.kdeplot(subset_data, label=label, fill=True, alpha=0.7)
            
            plt.title(f'CDF of {numerical_col} by {target_col}')
            plt.xlabel(numerical_col)
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            logging.info(f"CDF plot generated successfully for {numerical_col} vs {target_col}")
            
    except Exception as e:
        logging.error(f"Error generating CDF plots: {e}")


def plot_average_numerical_by_categorical_and_target(
    df: pd.DataFrame, 
    categorical_col: str, 
    numerical_col: str, 
    target_col: str, 
    figsize: tuple = (12, 6),
    estimator: callable = np.mean,
    title: str = None
) -> None:
    """
    Creates a barplot showing the average of a numerical column by a categorical column and target,
    with the target as hue.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data
    - categorical_col (str): Name of the categorical column for x-axis
    - numerical_col (str): Name of the numerical column for y-axis (to be averaged)
    - target_col (str): Name of the binary target column for hue
    - figsize (tuple): Figure size as (width, height), default (12, 6)
    - estimator (callable): Function to compute the estimate, default np.mean
    - title (str): Custom title for the plot, if None uses default format
    
    Returns:
    None: Displays the plot
    """
    try:
        # Validate input columns exist
        required_cols = [categorical_col, numerical_col, target_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
        # Filter valid data
        valid_data = df[required_cols].dropna()
        
        if valid_data.empty:
            logging.warning("No valid data available after removing NaN values")
            return
        
        # Create the plot
        plt.figure(figsize=figsize)
        sns.barplot(
            x=categorical_col, 
            y=numerical_col, 
            hue=target_col, 
            data=valid_data, 
            estimator=estimator
        )
        
        # Set title
        if title is None:
            estimator_name = getattr(estimator, '__name__', 'Estimated')
            title = f'{estimator_name.title()} {numerical_col} by {categorical_col} and {target_col}'
        plt.title(title)
        
        plt.xlabel(categorical_col)
        plt.ylabel(f'{estimator.__name__.title()} {numerical_col}')
        plt.legend(title=target_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        logging.info(f"Average barplot generated successfully for {numerical_col} by {categorical_col} and {target_col}")
        
    except Exception as e:
        logging.error(f"Error generating average barplot: {e}")


def plot_pairplot_with_target_hue(
    df: pd.DataFrame, 
    numerical_features: list, 
    target_col: str, 
    sample_size: int = 5000,
    figsize: tuple = None,
    title: str = None
) -> None:
    """
    Creates a seaborn pairplot for a subset of numerical features with target column as hue.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data
    - numerical_features (list): List of numerical column names to include in pairplot
    - target_col (str): Name of the target column to use as hue
    - sample_size (int): Maximum number of rows to sample for performance, default 5000
    - figsize (tuple): Figure size, if None uses seaborn default
    - title (str): Custom title for the plot, if None uses default format
    
    Returns:
    None: Displays the pairplot
    """
    try:
        # Validate that target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Validate numerical features exist
        missing_features = [col for col in numerical_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing numerical features in DataFrame: {missing_features}")
        
        # Prepare feature list including target
        features_with_target = numerical_features + [target_col] if target_col not in numerical_features else numerical_features
        
        # Filter valid data
        plot_data = df[features_with_target].dropna()
        
        if plot_data.empty:
            logging.warning("No valid data available after removing NaN values")
            return
        
        # Sample data if necessary for performance
        if len(plot_data) > sample_size:
            logging.info(f"Sampling {sample_size} rows from {len(plot_data)} total rows for performance")
            plot_data = plot_data.sample(n=sample_size, random_state=42)
        
        # Create the pairplot
        logging.info("Generating pairplot (this may take some time)...")
        pairplot = sns.pairplot(
            plot_data, 
            hue=target_col, 
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 20}
        )
        
        # Set title
        if title is None:
            title = f'Pair Plot of Selected Features by {target_col}'
        pairplot.fig.suptitle(title, y=1.02)
        
        # Adjust figure size if specified
        if figsize is not None:
            pairplot.fig.set_size_inches(figsize)
        
        plt.tight_layout()
        plt.show()
        
        logging.info(f"Pairplot generated successfully for features: {numerical_features}")
        
    except Exception as e:
        logging.error(f"Error generating pairplot: {e}")

