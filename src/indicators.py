"""
Technical indicators computation using ta library and custom calculations.
"""
import pandas as pd
import numpy as np
import ta


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for the DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added indicator columns
    """
    df_ind = df.copy()
    
    # Ensure we have the required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df_ind.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Simple Moving Averages
    df_ind['sma_20'] = ta.trend.SMAIndicator(df_ind['close'], window=20).sma_indicator()
    df_ind['sma_50'] = ta.trend.SMAIndicator(df_ind['close'], window=50).sma_indicator()
    
    # Exponential Moving Average
    df_ind['ema_20'] = ta.trend.EMAIndicator(df_ind['close'], window=20).ema_indicator()
    
    # RSI (14 period)
    rsi_indicator = ta.momentum.RSIIndicator(df_ind['close'], window=14)
    df_ind['rsi_14'] = rsi_indicator.rsi()
    
    # MACD
    macd_indicator = ta.trend.MACD(df_ind['close'])
    df_ind['macd'] = macd_indicator.macd()
    df_ind['macd_signal'] = macd_indicator.macd_signal()
    df_ind['macd_diff'] = macd_indicator.macd_diff()
    
    # ATR (14 period)
    atr_indicator = ta.volatility.AverageTrueRange(
        df_ind['high'], df_ind['low'], df_ind['close'], window=14
    )
    df_ind['atr_14'] = atr_indicator.average_true_range()
    
    # Daily returns
    df_ind['daily_returns'] = df_ind['close'].pct_change()
    
    # Additional useful metrics
    df_ind['price_change'] = df_ind['close'].diff()
    df_ind['price_change_pct'] = df_ind['close'].pct_change() * 100
    
    return df_ind


def get_indicator_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of current indicator values.
    
    Args:
        df: DataFrame with indicators calculated
        
    Returns:
        Dictionary with current indicator values
    """
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    
    # Safely get indicator values, checking if columns exist
    summary = {
        'current_price': float(latest['close']) if 'close' in df.columns and pd.notna(latest['close']) else None,
        'sma_20': float(latest['sma_20']) if 'sma_20' in df.columns and pd.notna(latest['sma_20']) else None,
        'sma_50': float(latest['sma_50']) if 'sma_50' in df.columns and pd.notna(latest['sma_50']) else None,
        'ema_20': float(latest['ema_20']) if 'ema_20' in df.columns and pd.notna(latest['ema_20']) else None,
        'rsi_14': float(latest['rsi_14']) if 'rsi_14' in df.columns and pd.notna(latest['rsi_14']) else None,
        'macd': float(latest['macd']) if 'macd' in df.columns and pd.notna(latest['macd']) else None,
        'macd_signal': float(latest['macd_signal']) if 'macd_signal' in df.columns and pd.notna(latest['macd_signal']) else None,
        'atr_14': float(latest['atr_14']) if 'atr_14' in df.columns and pd.notna(latest['atr_14']) else None,
        'daily_return': float(latest['daily_returns']) if 'daily_returns' in df.columns and pd.notna(latest['daily_returns']) else None,
    }
    
    # Add RSI interpretation
    if summary['rsi_14'] is not None:
        if summary['rsi_14'] > 70:
            summary['rsi_signal'] = 'Overbought'
        elif summary['rsi_14'] < 30:
            summary['rsi_signal'] = 'Oversold'
        else:
            summary['rsi_signal'] = 'Neutral'
    
    # Add MACD signal
    if summary['macd'] is not None and summary['macd_signal'] is not None:
        if summary['macd'] > summary['macd_signal']:
            summary['macd_signal_direction'] = 'Bullish'
        else:
            summary['macd_signal_direction'] = 'Bearish'
    
    return summary

