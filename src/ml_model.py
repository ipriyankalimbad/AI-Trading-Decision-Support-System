"""
Machine Learning model for price direction prediction using RandomForestClassifier.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for ML model.
    
    Args:
        df: DataFrame with indicators calculated
        
    Returns:
        Tuple of (features_df, target_series)
    """
    df_features = df.copy()
    
    # Create target: next day price direction (1 for up, 0 for down)
    df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
    
    # Select features
    feature_columns = [
        'sma_20', 'sma_50', 'ema_20', 'rsi_14',
        'macd', 'macd_signal', 'macd_diff',
        'atr_14', 'daily_returns',
        'price_change', 'price_change_pct'
    ]
    
    # Add volume if available
    if 'volume' in df_features.columns:
        feature_columns.append('volume')
    
    # Create lagged features for better prediction
    for col in ['close', 'rsi_14', 'macd']:
        if col in df_features.columns:
            df_features[f'{col}_lag1'] = df_features[col].shift(1)
            df_features[f'{col}_lag2'] = df_features[col].shift(2)
            feature_columns.extend([f'{col}_lag1', f'{col}_lag2'])
    
    # Select only available features
    available_features = [f for f in feature_columns if f in df_features.columns]
    
    # Remove rows with NaN values
    df_features = df_features[available_features + ['target']].dropna()
    
    if df_features.empty:
        raise ValueError("No valid data for ML training after feature preparation")
    
    X = df_features[available_features]
    y = df_features['target']
    
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train RandomForestClassifier with time-series split.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for testing (from the end)
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    # Time-series split: use last portion for testing
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    if len(X_train) < 10:
        raise ValueError("Insufficient data for training. Need at least 10 samples.")
    
    # Train RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilities
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)
    
    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    metrics = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_importance': feature_importance,
        'top_features': top_features,
        'model_type': 'RandomForestClassifier'
    }
    
    return model, metrics


def predict_next_direction(model: RandomForestClassifier, X: pd.DataFrame, 
                          feature_columns: list) -> Dict:
    """
    Predict next price direction using the trained model.
    
    Args:
        model: Trained RandomForestClassifier
        X: Feature matrix (should include latest row)
        feature_columns: List of feature column names used in training
        
    Returns:
        Dictionary with prediction results
    """
    if X.empty:
        return {
            'direction': None,
            'probability_up': None,
            'probability_down': None,
            'confidence': None
        }
    
    # Get latest row
    latest_features = X[feature_columns].iloc[[-1]]
    
    # Check for NaN values
    if latest_features.isna().any().any():
        return {
            'direction': None,
            'probability_up': None,
            'probability_down': None,
            'confidence': None,
            'error': 'Missing features for prediction'
        }
    
    # Predict
    prediction = model.predict(latest_features)[0]
    probabilities = model.predict_proba(latest_features)[0]
    
    direction = 'Up' if prediction == 1 else 'Down'
    prob_up = float(probabilities[1]) if len(probabilities) > 1 else 0.0
    prob_down = float(probabilities[0])
    confidence = max(prob_up, prob_down)
    
    return {
        'direction': direction,
        'probability_up': prob_up,
        'probability_down': prob_down,
        'confidence': confidence
    }


def run_ml_analysis(df: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict, Dict]:
    """
    Complete ML analysis pipeline.
    
    Args:
        df: DataFrame with indicators calculated
        
    Returns:
        Tuple of (model, metrics, prediction)
    """
    # Prepare features
    X, y = prepare_features(df)
    
    # Train model
    model, metrics = train_model(X, y)
    
    # Make prediction
    prediction = predict_next_direction(model, X, list(X.columns))
    
    return model, metrics, prediction

