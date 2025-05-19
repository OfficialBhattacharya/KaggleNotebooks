import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Suppress specific pandas warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas.io.formats.format')

# Define input file path
usaBaseline_Input = "/kaggle/input/us-home-price-index/US_home_price_Index.csv"

def process_hpi_data(file_path, date_format=None):
    """
    Process the US Home Price Index data:
    1. Read the CSV file
    2. Convert date column to proper datetime format
    3. Check for missing months
    4. Check for missing values
    5. Return clean, transformed DataFrame
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    date_format : str, optional
        Format of the date column (e.g., '%Y-%d-%m' for yyyy-dd-mm)
        Default is None, which lets pandas auto-detect the format
    """
    # Read the CSV file
    print(f"Reading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Remove 'Unnamed' columns
    print("\nRemoving unnamed columns...")
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        print(f"Dropping columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)
    
    # Display initial information
    print("\nInitial DataFrame info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Convert date column to datetime format
    # Assuming the date column is named 'date' - will adjust if different
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        date_col = date_cols[0]
        print(f"\nConverting {date_col} to datetime format")
        try:
            if date_format:
                print(f"Using specified date format: {date_format}")
                df[date_col] = pd.to_datetime(df[date_col], format=date_format)
            else:
                df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            print(f"Error converting date column: {e}")
            # Try different formats if specified format fails
            common_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%d-%m', '%m-%d-%Y', '%d-%m-%Y']
            for fmt in common_formats:
                try:
                    print(f"Trying format: {fmt}")
                    df[date_col] = pd.to_datetime(df[date_col], format=fmt)
                    print(f"Successfully converted using format: {fmt}")
                    break
                except:
                    continue
            else:
                print("Could not convert date column after trying multiple formats")
                return None
    else:
        print("No date column found. Please check column names.")
        return None
    
    # Sort by date
    df = df.sort_values(by=date_col)
    
    # Check for missing months
    if len(df) > 1:
        print("\nChecking for missing months...")
        date_series = df[date_col]
        min_date = date_series.min()
        max_date = date_series.max()
        
        # Create a complete date range with monthly frequency
        complete_date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
        
        # Get the month start date for each date in the dataset
        month_starts = df[date_col].dt.to_period('M').dt.to_timestamp()
        
        # Check if all months are present
        missing_months = set(complete_date_range) - set(month_starts)
        if missing_months:
            print(f"Missing {len(missing_months)} months between {min_date} and {max_date}:")
            for missing in sorted(missing_months):
                print(f"  - {missing.strftime('%Y-%m')}")
        else:
            print("All months are present in the dataset.")
    
    # Check for missing values
    print("\nChecking for missing values...")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Found missing values:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"  - {col}: {count} missing values")
    else:
        print("No missing values found.")
    
    # Fix data types for numeric columns
    print("\nFixing data types for numeric columns...")
    for col in df.columns:
        if col != date_col:
            # Attempt to convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values if any
    if df.isnull().sum().sum() > 0:
        print("\nHandling missing values...")
        # For numeric columns, interpolate
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                print(f"  - Interpolating {col}")
                df[col] = df[col].interpolate(method='linear')
        
        # Check if any missing values remain
        remaining_missing = df.isnull().sum().sum()
        if remaining_missing > 0:
            print(f"WARNING: {remaining_missing} missing values remain after interpolation.")
            print("Dropping rows with missing values...")
            df = df.dropna()
        else:
            print("All missing values successfully handled.")
    
    # Handle infinite values if any
    print("\nChecking for infinite values...")
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        print(f"Found {inf_count} infinite values. Replacing with NaN and interpolating...")
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        # Interpolate the NaN values
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].interpolate(method='linear')
                # If interpolation fails for any reason, fill with column median
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
    else:
        print("No infinite values found.")
    
    # Final validation and cleaning
    # Ensure no invalid values remain that could cause formatting warnings
    print("\nPerforming final validation...")
    for col in df.select_dtypes(include=[np.number]).columns:
        # Replace any remaining NaN with median
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
        
        # Check for and replace any remaining problematic values
        if (df[col] == np.inf).any() or (df[col] == -np.inf).any():
            df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
    
    # Final check
    if df.isnull().sum().sum() > 0:
        print("\nWARNING: DataFrame still contains missing values.")
        return None
    else:
        print("\nFinal DataFrame is clean with no missing values.")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        return df

def check_missing_months(df, date_col=None):
    """
    Check if there are any missing months in the cleaned data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The cleaned DataFrame to check
    date_col : str, optional
        The name of the date column. If None, will try to detect it automatically.
        
    Returns:
    --------
    missing_months : list or None
        List of missing months if any, None if no months are missing
    """
    if date_col is None:
        # Try to identify date column
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if not date_cols:
            print("Error: No date column found. Cannot check for missing months.")
            return None
        date_col = date_cols[0]
    
    print(f"\nChecking for missing months in cleaned data using column: {date_col}")
    
    # Ensure the column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            print(f"Error: Failed to convert {date_col} to datetime. Cannot check for missing months.")
            return None
    
    # Sort by date
    df_sorted = df.sort_values(by=date_col)
    
    # Get min and max dates
    min_date = df_sorted[date_col].min()
    max_date = df_sorted[date_col].max()
    
    print(f"Checking for missing months between {min_date.strftime('%Y-%m-%d')} and {max_date.strftime('%Y-%m-%d')}")
    
    # Create complete monthly date range
    complete_date_range = pd.date_range(start=min_date.replace(day=1), 
                                        end=max_date.replace(day=1), 
                                        freq='MS')
    
    # Get month start dates from actual data
    month_starts = df_sorted[date_col].dt.to_period('M').dt.to_timestamp()
    
    # Find missing months
    missing_months = set(complete_date_range) - set(month_starts)
    
    if missing_months:
        print(f"Found {len(missing_months)} missing months:")
        for missing in sorted(missing_months):
            print(f"  - {missing.strftime('%Y-%m')}")
        return sorted(list(missing_months))
    else:
        print("No missing months found in the cleaned data.")
        return None

def create_transformed_features(df, date_col=None):
    """
    Create additional transformed features from the cleaned data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The cleaned DataFrame
    date_col : str, optional
        The name of the date column. If None, will try to detect it automatically.
        
    Returns:
    --------
    df_enhanced : pandas.DataFrame
        DataFrame with additional features
    """
    # Make a copy to avoid modifying the original
    df_enhanced = df.copy()
    
    if date_col is None:
        # Try to identify date column
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if not date_cols:
            print("Error: No date column found. Cannot create time-based features.")
            return df_enhanced
        date_col = date_cols[0]
    
    print(f"\nCreating transformed features using date column: {date_col}")
    
    # Ensure the column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_enhanced[date_col]):
        try:
            df_enhanced[date_col] = pd.to_datetime(df_enhanced[date_col])
        except:
            print(f"Error: Failed to convert {date_col} to datetime. Cannot create time-based features.")
            return df_enhanced
    
    # Sort by date for time series features
    df_enhanced = df_enhanced.sort_values(by=date_col).reset_index(drop=True)
    
    # 1. Extract time components
    print("Creating date component features...")
    df_enhanced['year'] = df_enhanced[date_col].dt.year
    df_enhanced['month'] = df_enhanced[date_col].dt.month
    df_enhanced['quarter'] = df_enhanced[date_col].dt.quarter
    
    # 2. Create cyclical features for month and quarter
    print("Creating cyclical features...")
    # Month as cyclical feature (sine and cosine transformation)
    df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['month']/12)
    df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['month']/12)
    
    # Quarter as cyclical feature
    df_enhanced['quarter_sin'] = np.sin(2 * np.pi * df_enhanced['quarter']/4)
    df_enhanced['quarter_cos'] = np.cos(2 * np.pi * df_enhanced['quarter']/4)
    
    # 3. Create lag features for numeric columns
    print("Creating lag features...")
    numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude newly created features
    exclude_cols = ['year', 'month', 'quarter', 'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create lag features of 1, 3, 6, and 12 months for key metrics
    lag_periods = [1, 3, 6, 12]
    for col in numeric_cols:
        for lag in lag_periods:
            lag_col_name = f"{col}_lag_{lag}"
            df_enhanced[lag_col_name] = df_enhanced[col].shift(lag)
    
    # 4. Create rolling window features (3-month, 6-month, 12-month)
    print("Creating rolling window features...")
    windows = [3, 6, 12]
    for col in numeric_cols:
        for window in windows:
            # Rolling mean
            df_enhanced[f"{col}_rolling_mean_{window}"] = df_enhanced[col].rolling(window=window).mean()
            # Rolling standard deviation (for volatility)
            df_enhanced[f"{col}_rolling_std_{window}"] = df_enhanced[col].rolling(window=window).std()
    
    # 5. Create percentage change features
    print("Creating percentage change features...")
    for col in numeric_cols:
        # Month-over-month percentage change
        df_enhanced[f"{col}_pct_change_1m"] = df_enhanced[col].pct_change(periods=1)
        # Year-over-year percentage change
        df_enhanced[f"{col}_pct_change_12m"] = df_enhanced[col].pct_change(periods=12)
    
    # 6. Create ratio features between different metrics if appropriate
    print("Creating ratio features...")
    # Example: If we have economic indicators like GDP and unemployment
    if 'GDP' in df_enhanced.columns and 'Unemployment_rate' in df_enhanced.columns:
        df_enhanced['GDP_per_unemployment'] = df_enhanced['GDP'] / df_enhanced['Unemployment_rate']
    
    # If we have mortgage rates and house price index
    if 'House_price_index' in df_enhanced.columns and 'Mortgage' in df_enhanced.columns:
        df_enhanced['HPI_mortgage_ratio'] = df_enhanced['House_price_index'] / df_enhanced['Mortgage']
    
    # If we have inflation rate
    if 'House_price_index' in df_enhanced.columns and 'Inflation_rate' in df_enhanced.columns:
        df_enhanced['HPI_inflation_ratio'] = df_enhanced['House_price_index'] / df_enhanced['Inflation_rate']
    
    # 7. Create interaction features
    print("Creating interaction features...")
    if 'Interest_rate' in df_enhanced.columns and 'Inflation_rate' in df_enhanced.columns:
        df_enhanced['real_interest_rate'] = df_enhanced['Interest_rate'] - df_enhanced['Inflation_rate']
    
    # 8. Fill NaN values created by lag and rolling features
    print("Filling NaN values in new features...")
    # Count NaNs before filling
    na_count_before = df_enhanced.isna().sum().sum()
    
    # Fill NaN values with appropriate methods
    for col in df_enhanced.columns:
        if col != date_col and df_enhanced[col].isna().any():
            # Use forward fill for lag features to maintain time series integrity
            if 'lag' in col or 'rolling' in col or 'pct_change' in col:
                df_enhanced[col] = df_enhanced[col].ffill()
                # If any NaN values remain at the beginning, fill with the first non-NaN value
                if df_enhanced[col].isna().any():
                    first_valid = df_enhanced[col].first_valid_index()
                    if first_valid is not None:
                        df_enhanced[col] = df_enhanced[col].fillna(df_enhanced[col].iloc[first_valid])
    
    # Count NaNs after filling
    na_count_after = df_enhanced.isna().sum().sum()
    print(f"Filled {na_count_before - na_count_after} NaN values. {na_count_after} NaN values remain.")
    
    # Drop rows with NaN if any remain
    if na_count_after > 0:
        print("Warning: Some NaN values could not be filled. Dropping rows with NaN values.")
        df_enhanced = df_enhanced.dropna()
    
    print(f"\nTransformed DataFrame shape: {df_enhanced.shape}")
    print(f"Added {df_enhanced.shape[1] - df.shape[1]} new features")
    
    return df_enhanced

def main():
    # Process the data - specify '%Y-%d-%m' for yyyy-dd-mm format
    clean_df = process_hpi_data(usaBaseline_Input, date_format='%Y-%d-%m')
    
    if clean_df is not None:
        print("\nProcessing completed successfully!")
        
        # Check for missing months in the cleaned data
        missing_months = check_missing_months(clean_df)
        
        # Create transformed features
        enhanced_df = create_transformed_features(clean_df)
        
        # Return the enhanced DataFrame
        return enhanced_df
    else:
        print("\nProcessing failed. Please check the errors above.")
        return None

if __name__ == "__main__":
    main()


