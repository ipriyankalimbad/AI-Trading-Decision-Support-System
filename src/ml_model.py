"""
Advanced Machine Learning model for price direction prediction using Ensemble methods.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from typing import Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare advanced features for ML model with extensive feature engineering.
    
    Args:
        df: DataFrame with indicators calculated
        
    Returns:
        Tuple of (features_df, target_series)
    """
    df_features = df.copy()
    
    # Create target: next day price direction (1 for up, 0 for down)
    df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)
    
    # Base features
    feature_columns = [
        'sma_20', 'sma_50', 'ema_20', 'rsi_14',
        'macd', 'macd_signal', 'macd_diff',
        'atr_14', 'daily_returns',
        'price_change', 'price_change_pct'
    ]
    
    # Add volume if available
    if 'volume' in df_features.columns:
        feature_columns.append('volume')
        # Advanced volume features
        if len(df_features) > 20:
            df_features['volume_sma_20'] = df_features['volume'].rolling(window=20).mean()
            df_features['volume_sma_50'] = df_features['volume'].rolling(window=50).mean() if len(df_features) > 50 else df_features['volume_sma_20']
            df_features['volume_ratio'] = df_features['volume'] / (df_features['volume_sma_20'] + 1e-10)
            df_features['volume_trend'] = df_features['volume'].rolling(window=5).mean() / (df_features['volume'].rolling(window=20).mean() + 1e-10)
            feature_columns.extend(['volume_sma_20', 'volume_sma_50', 'volume_ratio', 'volume_trend'])
    
    # Price position features (where is price relative to SMAs)
    if 'sma_20' in df_features.columns and 'close' in df_features.columns:
        df_features['price_vs_sma20'] = (df_features['close'] - df_features['sma_20']) / (df_features['sma_20'] + 1e-10)
        df_features['price_vs_sma50'] = (df_features['close'] - df_features['sma_50']) / (df_features['sma_50'] + 1e-10) if 'sma_50' in df_features.columns else 0
        df_features['sma20_vs_sma50'] = (df_features['sma_20'] - df_features['sma_50']) / (df_features['sma_50'] + 1e-10) if 'sma_50' in df_features.columns else 0
        feature_columns.extend(['price_vs_sma20', 'price_vs_sma50', 'sma20_vs_sma50'])
    
    # RSI advanced features
    if 'rsi_14' in df_features.columns:
        df_features['rsi_normalized'] = (df_features['rsi_14'] - 50) / 50
        df_features['rsi_overbought'] = (df_features['rsi_14'] > 70).astype(int)
        df_features['rsi_oversold'] = (df_features['rsi_14'] < 30).astype(int)
        df_features['rsi_trend'] = df_features['rsi_14'].rolling(window=5).mean() - df_features['rsi_14'].rolling(window=10).mean()
        feature_columns.extend(['rsi_normalized', 'rsi_overbought', 'rsi_oversold', 'rsi_trend'])
    
    # MACD advanced features
    if 'macd' in df_features.columns and 'macd_signal' in df_features.columns:
        df_features['macd_momentum'] = df_features['macd'] - df_features['macd_signal']
        df_features['macd_cross'] = ((df_features['macd'] > df_features['macd_signal']) & 
                                     (df_features['macd'].shift(1) <= df_features['macd_signal'].shift(1))).astype(int)
        df_features['macd_strength'] = abs(df_features['macd'] - df_features['macd_signal']) / (abs(df_features['macd']) + 1e-10)
        feature_columns.extend(['macd_momentum', 'macd_cross', 'macd_strength'])
    
    # Advanced lagged features (more lags, more features)
    for col in ['close', 'rsi_14', 'macd', 'daily_returns', 'volume']:
        if col in df_features.columns:
            for lag in [1, 2, 3, 5]:  # More lag periods
                df_features[f'{col}_lag{lag}'] = df_features[col].shift(lag)
                feature_columns.append(f'{col}_lag{lag}')
    
    # Rolling statistics (volatility, momentum, trends)
    if 'daily_returns' in df_features.columns and len(df_features) > 10:
        df_features['volatility_5'] = df_features['daily_returns'].rolling(window=5).std()
        df_features['volatility_10'] = df_features['daily_returns'].rolling(window=10).std()
        df_features['volatility_20'] = df_features['daily_returns'].rolling(window=20).std() if len(df_features) > 20 else df_features['volatility_10']
        df_features['momentum_5'] = df_features['close'].pct_change(5)
        df_features['momentum_10'] = df_features['close'].pct_change(10)
        df_features['trend_5'] = df_features['close'].rolling(window=5).mean() / df_features['close'].rolling(window=10).mean() - 1
        feature_columns.extend(['volatility_5', 'volatility_10', 'volatility_20', 'momentum_5', 'momentum_10', 'trend_5'])
    
    # Price action features
    if 'high' in df_features.columns and 'low' in df_features.columns and 'close' in df_features.columns:
        df_features['price_range'] = (df_features['high'] - df_features['low']) / (df_features['close'] + 1e-10)
        df_features['upper_shadow'] = (df_features['high'] - df_features[['open', 'close']].max(axis=1)) / (df_features['close'] + 1e-10)
        df_features['lower_shadow'] = (df_features[['open', 'close']].min(axis=1) - df_features['low']) / (df_features['close'] + 1e-10)
        feature_columns.extend(['price_range', 'upper_shadow', 'lower_shadow'])
    
    # ATR-based features
    if 'atr_14' in df_features.columns and 'close' in df_features.columns:
        df_features['atr_percent'] = df_features['atr_14'] / (df_features['close'] + 1e-10)
        df_features['atr_trend'] = df_features['atr_14'].rolling(window=5).mean() / (df_features['atr_14'].rolling(window=10).mean() + 1e-10)
        feature_columns.extend(['atr_percent', 'atr_trend'])
    
    # Select only available features
    available_features = [f for f in feature_columns if f in df_features.columns]
    
    # Fill NaN values with forward fill, then backward fill, then median
    for col in available_features:
        df_features[col] = df_features[col].ffill().bfill()
        if df_features[col].isna().any():
            df_features[col] = df_features[col].fillna(df_features[col].median())
    
    # Remove rows where target is NaN (last row)
    df_features = df_features[df_features['target'].notna()].copy()
    
    if df_features.empty or len(df_features) < 30:
        raise ValueError("Insufficient data for ML training after feature preparation")
    
    X = df_features[available_features]
    y = df_features['target']
    
    # Replace any remaining NaN with 0 (shouldn't happen, but safety)
    X = X.fillna(0)
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[Any, Any, Any, Dict]:
    """
    Train advanced ensemble model with feature scaling and selection.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for testing (from the end)
        
    Returns:
        Tuple of (trained_model, scaler, feature_selector, metrics_dict)
    """
    # Time-series split: use last portion for testing
    split_idx = int(len(X) * (1 - test_size))
    
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    if len(X_train) < 30:
        raise ValueError("Insufficient data for training. Need at least 30 samples.")
    
    # Feature scaling (RobustScaler is better for financial data with outliers)
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Feature selection (select top K features)
    k_features = min(50, len(X_train.columns))  # Select top 50 features or all if less
    feature_selector = SelectKBest(score_func=f_classif, k=k_features)
    X_train_selected = pd.DataFrame(
        feature_selector.fit_transform(X_train_scaled, y_train),
        columns=[X_train.columns[i] for i in feature_selector.get_support(indices=True)],
        index=X_train.index
    )
    X_test_selected = pd.DataFrame(
        feature_selector.transform(X_test_scaled),
        columns=X_train_selected.columns,
        index=X_test.index
    )
    
    # Create ensemble of models
    # Model 1: Random Forest (good for non-linear patterns)
    rf_model = RandomForestClassifier(
        n_estimators=300,  # More trees
        max_depth=20,  # Deeper trees
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Model 2: Gradient Boosting (better sequential learning)
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    
    # Ensemble: Voting Classifier (combines both models)
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('gb', gb_model)],
        voting='soft',  # Use probability voting
        weights=[2, 1]  # Give more weight to Random Forest
    )
    
    # Train ensemble
    ensemble_model.fit(X_train_selected, y_train)
    
    # Predictions
    y_train_pred = ensemble_model.predict(X_train_selected)
    y_test_pred = ensemble_model.predict(X_test_selected)
    
    # Probabilities
    y_train_proba = ensemble_model.predict_proba(X_train_selected)
    y_test_proba = ensemble_model.predict_proba(X_test_selected)
    
    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    # Feature importance (from Random Forest)
    rf_feature_importance = dict(zip(X_train_selected.columns, rf_model.feature_importances_))
    top_features = sorted(rf_feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    metrics = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'train_precision': float(train_precision),
        'test_precision': float(test_precision),
        'train_recall': float(train_recall),
        'test_recall': float(test_recall),
        'train_f1': float(train_f1),
        'test_f1': float(test_f1),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_importance': rf_feature_importance,
        'top_features': top_features,
        'model_type': 'Ensemble (RandomForest + GradientBoosting)',
        'num_features_used': len(X_train_selected.columns),
        'total_features': len(X_train.columns)
    }
    
    return ensemble_model, scaler, feature_selector, metrics


def predict_next_direction(model, scaler, feature_selector, X: pd.DataFrame, 
                          feature_columns: list) -> Dict:
    """
    Predict next price direction using the trained ensemble model.
    
    Args:
        model: Trained ensemble model
        scaler: Fitted scaler
        feature_selector: Fitted feature selector
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
    latest_features = X[feature_columns].iloc[[-1]].copy()
    
    # Check for NaN values
    if latest_features.isna().any().any():
        latest_features = latest_features.fillna(0)
    
    # Replace infinite values
    latest_features = latest_features.replace([np.inf, -np.inf], 0)
    
    # Scale features
    latest_scaled = pd.DataFrame(
        scaler.transform(latest_features),
        columns=latest_features.columns,
        index=latest_features.index
    )
    
    # Select features
    latest_selected = pd.DataFrame(
        feature_selector.transform(latest_scaled),
        columns=[latest_features.columns[i] for i in feature_selector.get_support(indices=True)],
        index=latest_features.index
    )
    
    # Predict
    prediction = model.predict(latest_selected)[0]
    probabilities = model.predict_proba(latest_selected)[0]
    
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


def run_ml_analysis(df: pd.DataFrame) -> Tuple[Tuple[Any, Any, Any], Dict, Dict]:
    """
    Complete advanced ML analysis pipeline.
    
    Args:
        df: DataFrame with indicators calculated
        
    Returns:
        Tuple of (model, scaler, feature_selector, metrics, prediction)
    """
    # Prepare features
    X, y = prepare_features(df)
    
    # Train model
    model, scaler, feature_selector, metrics = train_model(X, y)
    
    # Make prediction
    prediction = predict_next_direction(model, scaler, feature_selector, X, list(X.columns))
    
    return (model, scaler, feature_selector), metrics, prediction
