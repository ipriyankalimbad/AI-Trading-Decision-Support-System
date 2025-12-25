"""
Risk management calculations including ATR-based stop-loss and targets.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_atr_based_stop_loss(entry_price: float, atr: float, 
                                  multiplier: float = 2.0, 
                                  position_type: str = 'long') -> float:
    """
    Calculate ATR-based stop-loss.
    
    Args:
        entry_price: Entry price for the position
        atr: Average True Range value
        multiplier: ATR multiplier (default 2.0)
        position_type: 'long' or 'short'
        
    Returns:
        Stop-loss price
    """
    if pd.isna(atr) or atr <= 0:
        return None
    
    atr_distance = atr * multiplier
    
    if position_type == 'long':
        stop_loss = entry_price - atr_distance
    else:
        stop_loss = entry_price + atr_distance
    
    return float(stop_loss)


def calculate_target_price(entry_price: float, stop_loss: float, 
                           risk_reward_ratio: float = 2.0,
                           position_type: str = 'long') -> float:
    """
    Calculate target price based on risk-reward ratio.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop-loss price
        risk_reward_ratio: Desired risk-reward ratio (default 2.0)
        position_type: 'long' or 'short'
        
    Returns:
        Target price
    """
    if stop_loss is None:
        return None
    
    if position_type == 'long':
        risk = entry_price - stop_loss
        if risk <= 0:
            return None
        target = entry_price + (risk * risk_reward_ratio)
    else:
        risk = stop_loss - entry_price
        if risk <= 0:
            return None
        target = entry_price - (risk * risk_reward_ratio)
    
    return float(target)


def calculate_risk_metrics(entry_price: float, stop_loss: Optional[float], 
                          target: Optional[float], position_size: float = 1.0,
                          position_type: str = 'long') -> Dict:
    """
    Calculate comprehensive risk metrics.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop-loss price
        target: Target price
        position_size: Position size (default 1.0)
        position_type: 'long' or 'short'
        
    Returns:
        Dictionary with risk metrics
    """
    metrics = {
        'entry_price': float(entry_price),
        'stop_loss': float(stop_loss) if stop_loss is not None else None,
        'target': float(target) if target is not None else None,
        'position_size': float(position_size),
        'position_type': position_type
    }
    
    if stop_loss is not None:
        if position_type == 'long':
            risk_per_share = entry_price - stop_loss
            risk_pct = (risk_per_share / entry_price) * 100
        else:
            risk_per_share = stop_loss - entry_price
            risk_pct = (risk_per_share / entry_price) * 100
        
        metrics['risk_per_share'] = float(risk_per_share)
        metrics['risk_percentage'] = float(risk_pct)
        metrics['total_risk'] = float(risk_per_share * position_size)
    else:
        metrics['risk_per_share'] = None
        metrics['risk_percentage'] = None
        metrics['total_risk'] = None
    
    if target is not None and stop_loss is not None:
        if position_type == 'long':
            reward_per_share = target - entry_price
            reward_pct = (reward_per_share / entry_price) * 100
        else:
            reward_per_share = entry_price - target
            reward_pct = (reward_per_share / entry_price) * 100
        
        metrics['reward_per_share'] = float(reward_per_share)
        metrics['reward_percentage'] = float(reward_pct)
        metrics['total_reward'] = float(reward_per_share * position_size)
        
        # Risk-reward ratio
        if metrics['risk_per_share'] and metrics['risk_per_share'] > 0:
            metrics['risk_reward_ratio'] = float(reward_per_share / risk_per_share)
        else:
            metrics['risk_reward_ratio'] = None
    else:
        metrics['reward_per_share'] = None
        metrics['reward_percentage'] = None
        metrics['total_reward'] = None
        metrics['risk_reward_ratio'] = None
    
    return metrics


def get_risk_recommendations(df: pd.DataFrame, entry_price: float, 
                            atr_multiplier: float = 2.0,
                            risk_reward_ratio: float = 2.0) -> Dict:
    """
    Get comprehensive risk management recommendations.
    
    Args:
        df: DataFrame with indicators (must have 'atr_14' column)
        entry_price: Assumed entry price
        atr_multiplier: ATR multiplier for stop-loss
        risk_reward_ratio: Desired risk-reward ratio
        
    Returns:
        Dictionary with risk recommendations
    """
    if df.empty or 'atr_14' not in df.columns:
        return {
            'error': 'Insufficient data for risk calculations'
        }
    
    # Get latest ATR
    latest_atr = df['atr_14'].iloc[-1]
    current_price = df['close'].iloc[-1]
    
    # Calculate stop-loss
    stop_loss = calculate_atr_based_stop_loss(
        entry_price, latest_atr, atr_multiplier, 'long'
    )
    
    # Calculate target
    target = calculate_target_price(
        entry_price, stop_loss, risk_reward_ratio, 'long'
    )
    
    # Calculate all metrics
    risk_metrics = calculate_risk_metrics(
        entry_price, stop_loss, target, position_size=1.0, position_type='long'
    )
    
    # Add additional context
    risk_metrics['current_price'] = float(current_price)
    risk_metrics['atr_value'] = float(latest_atr) if pd.notna(latest_atr) else None
    risk_metrics['atr_multiplier'] = atr_multiplier
    risk_metrics['desired_risk_reward'] = risk_reward_ratio
    
    # Risk assessment
    if risk_metrics['risk_percentage'] is not None:
        if risk_metrics['risk_percentage'] < 2:
            risk_metrics['risk_level'] = 'Low'
        elif risk_metrics['risk_percentage'] < 5:
            risk_metrics['risk_level'] = 'Moderate'
        else:
            risk_metrics['risk_level'] = 'High'
    else:
        risk_metrics['risk_level'] = 'Unknown'
    
    return risk_metrics

