"""
Data utilities for validation, cleaning, and preprocessing.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def validate_csv(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the uploaded CSV contains required OHLCV columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    if df.empty:
        return False, "CSV file is empty"
    
    # Check for required columns (case-insensitive)
    df_columns_lower = [col.lower().strip() for col in df.columns]
    missing_columns = []
    
    for req_col in required_columns:
        if req_col not in df_columns_lower:
            missing_columns.append(req_col)
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    return True, ""


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the DataFrame.
    
    Args:
        df: Raw DataFrame from CSV
        
    Returns:
        Cleaned DataFrame with standardized column names
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Standardize column names (case-insensitive, strip whitespace)
    column_mapping = {}
    for col in df_clean.columns:
        col_lower = col.lower().strip()
        if col_lower in ['date', 'open', 'high', 'low', 'close', 'volume']:
            column_mapping[col] = col_lower
    
    df_clean = df_clean.rename(columns=column_mapping)
    
    # Convert date column to datetime
    if 'date' in df_clean.columns:
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove rows with missing critical data
    df_clean = df_clean.dropna(subset=['date', 'open', 'high', 'low', 'close'])
    
    # Sort by date
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    
    # Validate OHLC relationships
    invalid_rows = (
        (df_clean['high'] < df_clean['low']) |
        (df_clean['high'] < df_clean['open']) |
        (df_clean['high'] < df_clean['close']) |
        (df_clean['low'] > df_clean['open']) |
        (df_clean['low'] > df_clean['close'])
    )
    
    if invalid_rows.any():
        df_clean = df_clean[~invalid_rows]
    
    # Fill missing volume with 0
    if 'volume' in df_clean.columns:
        df_clean['volume'] = df_clean['volume'].fillna(0)
    
    return df_clean


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to validate and clean data.
    
    Args:
        df: Raw DataFrame from CSV upload
        
    Returns:
        Cleaned and validated DataFrame
    """
    is_valid, error_msg = validate_csv(df)
    if not is_valid:
        raise ValueError(f"Data validation failed: {error_msg}")
    
    df_clean = clean_data(df)
    
    if df_clean.empty:
        raise ValueError("Data cleaning resulted in empty DataFrame")
    
    return df_clean

