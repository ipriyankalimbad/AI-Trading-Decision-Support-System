"""
Advanced Machine Learning model for price direction prediction using Ensemble methods.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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
    
    # Advanced lagged features (strategic lags for better prediction)
    for col in ['close', 'rsi_14', 'macd', 'daily_returns']:
        if col in df_features.columns:
            for lag in [1, 2, 3, 5, 7]:  # More strategic lag periods
                df_features[f'{col}_lag{lag}'] = df_features[col].shift(lag)
                feature_columns.append(f'{col}_lag{lag}')
    
    # Volume lagged features (fewer lags to avoid noise)
    if 'volume' in df_features.columns:
        for lag in [1, 2, 3]:
            df_features[f'volume_lag{lag}'] = df_features['volume'].shift(lag)
            feature_columns.append(f'volume_lag{lag}')
    
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
        df_features['atr_change'] = df_features['atr_14'].pct_change()  # ATR momentum
        feature_columns.extend(['atr_percent', 'atr_trend', 'atr_change'])
    
    # Feature interactions (key indicator combinations)
    if 'rsi_14' in df_features.columns and 'macd' in df_features.columns:
        df_features['rsi_macd_interaction'] = df_features['rsi_14'] * df_features['macd']
        feature_columns.append('rsi_macd_interaction')
    
    if 'sma_20' in df_features.columns and 'rsi_14' in df_features.columns:
        df_features['sma_rsi_interaction'] = (df_features['sma_20'] / df_features['close']) * df_features['rsi_14']
        feature_columns.append('sma_rsi_interaction')
    
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
    
    # Create validation set for early stopping (10% of training data)
    val_size = max(10, int(len(X_train) * 0.1))
    X_train_fit = X_train.iloc[:-val_size]
    X_val = X_train.iloc[-val_size:]
    y_train_fit = y_train.iloc[:-val_size]
    y_val = y_train.iloc[-val_size:]
    
    # Feature scaling (RobustScaler is better for financial data with outliers)
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_fit),
        columns=X_train_fit.columns,
        index=X_train_fit.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Feature selection (select top K features - adaptive based on data size)
    # Use mutual information for better feature selection (captures non-linear relationships)
    if len(X_train) < 200:
        k_features = min(25, len(X_train.columns))
    elif len(X_train) < 500:
        k_features = min(35, len(X_train.columns))
    else:
        k_features = min(40, len(X_train.columns))
    
    # Use mutual information for better feature selection (better for non-linear relationships)
    feature_selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    X_train_selected = pd.DataFrame(
        feature_selector.fit_transform(X_train_scaled, y_train_fit),
        columns=[X_train_fit.columns[i] for i in feature_selector.get_support(indices=True)],
        index=X_train_fit.index
    )
    X_val_selected = pd.DataFrame(
        feature_selector.transform(X_val_scaled),
        columns=X_train_selected.columns,
        index=X_val.index
    )
    X_test_selected = pd.DataFrame(
        feature_selector.transform(X_test_scaled),
        columns=X_train_selected.columns,
        index=X_test.index
    )
    
    # Create ensemble of models with optimal hyperparameters
    # Model 1: Random Forest (good for non-linear patterns)
    # Adaptive parameters based on data size - optimized for better test accuracy
    if len(X_train) < 200:
        rf_max_depth = 6
        rf_min_split = 25
        rf_min_leaf = 10
        rf_n_est = 150
    elif len(X_train) < 500:
        rf_max_depth = 8
        rf_min_split = 20
        rf_min_leaf = 8
        rf_n_est = 200
    else:
        rf_max_depth = 10
        rf_min_split = 15
        rf_min_leaf = 5
        rf_n_est = 250
    
    rf_model = RandomForestClassifier(
        n_estimators=rf_n_est,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_split,
        min_samples_leaf=rf_min_leaf,
        max_features='log2',  # Use log2 for better feature diversity
        max_samples=0.85,  # Use 85% of samples per tree (slightly more data)
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Model 2: Gradient Boosting with early stopping
    # Adaptive parameters - optimized for better test accuracy
    if len(X_train) < 200:
        gb_max_depth = 4
        gb_min_split = 20
        gb_min_leaf = 8
        gb_n_est = 150
        gb_lr = 0.08
    elif len(X_train) < 500:
        gb_max_depth = 5
        gb_min_split = 15
        gb_min_leaf = 6
        gb_n_est = 200
        gb_lr = 0.06
    else:
        gb_max_depth = 6
        gb_min_split = 12
        gb_min_leaf = 4
        gb_n_est = 250
        gb_lr = 0.05
    
    gb_model = GradientBoostingClassifier(
        n_estimators=gb_n_est,
        max_depth=gb_max_depth,
        learning_rate=gb_lr,
        min_samples_split=gb_min_split,
        min_samples_leaf=gb_min_leaf,
        subsample=0.75,  # Increased for more data usage
        validation_fraction=0.1,
        n_iter_no_change=15,  # More patience for early stopping
        random_state=42
    )
    
    # Ensemble: Voting Classifier (combines both models)
    # Optimized weights for better performance
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('gb', gb_model)],
        voting='soft',  # Use probability voting
        weights=[1.5, 1]  # Balanced weights for better ensemble performance
    )
    
    # Train models with validation set
    # Fit rf_model separately for feature importance (use full training set)
    rf_model.fit(X_train_selected, y_train_fit)
    
    # Fit gb_model with early stopping on validation set
    gb_model.fit(X_train_selected, y_train_fit)
    
    # Create and train ensemble
    ensemble_model = VotingClassifier(
        estimators=[('rf', rf_model), ('gb', gb_model)],
        voting='soft',
        weights=[2, 1]
    )
    ensemble_model.fit(X_train_selected, y_train_fit)
    
    # Use full training set for final predictions (combine train_fit + val)
    X_train_full = pd.concat([X_train_selected, X_val_selected])
    y_train_full = pd.concat([y_train_fit, y_val])
    
    # Predictions
    y_train_pred = ensemble_model.predict(X_train_full)
    y_test_pred = ensemble_model.predict(X_test_selected)
    
    # Probabilities
    y_train_proba = ensemble_model.predict_proba(X_train_full)
    y_test_proba = ensemble_model.predict_proba(X_test_selected)
    
    # Metrics
    train_accuracy = accuracy_score(y_train_full, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision = precision_score(y_train_full, y_train_pred, average='weighted', zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    train_recall = recall_score(y_train_full, y_train_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    train_f1 = f1_score(y_train_full, y_train_pred, average='weighted', zero_division=0)
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
        'train_samples': len(X_train_full),
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
