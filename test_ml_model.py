"""
Test script to evaluate ML model performance and display average accuracies.
"""
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import prepare_data
from src.indicators import calculate_indicators
from src.ml_model import run_ml_analysis

def test_model_with_sample_data():
    """Test the ML model with sample data."""
    print("=" * 80)
    print("ML MODEL PERFORMANCE TEST")
    print("=" * 80)
    print()
    
    # Create sample data
    print("Generating sample trading data...")
    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
    
    # Generate realistic price data with trends
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 500)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, 500)),
        'high': prices * (1 + abs(np.random.normal(0, 0.01, 500))),
        'low': prices * (1 - abs(np.random.normal(0, 0.01, 500))),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 500)
    })
    
    print(f"Data shape: {df.shape}")
    print()
    
    try:
        # Prepare data
        df_processed = prepare_data(df)
        
        # Calculate indicators
        print("Calculating technical indicators...")
        df_with_indicators = calculate_indicators(df_processed)
        print(f"Data with indicators shape: {df_with_indicators.shape}")
        print()
        
        # Run ML analysis
        print("Training ML model with walk-forward validation...")
        print("-" * 80)
        model_tuple, metrics, prediction = run_ml_analysis(df_with_indicators)
        print()
        
        # Display results
        print("=" * 80)
        print("MODEL PERFORMANCE RESULTS")
        print("=" * 80)
        print()
        
        print("SINGLE SPLIT METRICS:")
        print(f"  Training Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"  Test Accuracy:     {metrics['test_accuracy']:.2%}")
        print()
        
        if metrics.get('walk_forward_n_splits', 0) > 0:
            print("WALK-FORWARD VALIDATION METRICS:")
            print(f"  Number of Folds:  {metrics['walk_forward_n_splits']}")
            print(f"  Train Accuracy:  {metrics['walk_forward_train_mean']:.2%} +/- {metrics['walk_forward_train_std']:.2%}")
            print(f"  Test Accuracy:   {metrics['walk_forward_test_mean']:.2%} +/- {metrics['walk_forward_test_std']:.2%}")
            print()
            print("  [OK] Average Training Accuracy:", f"{metrics['walk_forward_train_mean']:.2%}")
            print("  [OK] Average Testing Accuracy: ", f"{metrics['walk_forward_test_mean']:.2%}")
        else:
            print("[WARNING] Walk-forward validation not performed (insufficient data)")
            print()
            print("  [OK] Average Training Accuracy:", f"{metrics['train_accuracy']:.2%}")
            print("  [OK] Average Testing Accuracy: ", f"{metrics['test_accuracy']:.2%}")
        
        print()
        print("MODEL CONFIGURATION:")
        print(f"  Model Type:        {metrics['model_type']}")
        print(f"  Features Used:     {metrics['num_features_used']} / {metrics['total_features']}")
        print(f"  Optimal Threshold: {metrics.get('optimal_threshold', 0.5):.3f}")
        print(f"  Training Samples:  {metrics['train_samples']:,}")
        print(f"  Test Samples:      {metrics['test_samples']:,}")
        print()
        
        print("PREDICTION:")
        print(f"  Direction:  {prediction.get('direction', 'N/A')}")
        print(f"  Confidence: {prediction.get('confidence', 0):.2%}")
        if prediction.get('low_confidence', False):
            print("  [WARNING] Low confidence prediction - market conditions uncertain")
        print()
        
        print("=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        
        return metrics
        
    except Exception as e:
        print(f"[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    metrics = test_model_with_sample_data()
    
    if metrics:
        print()
        print("SUMMARY:")
        if metrics.get('walk_forward_n_splits', 0) > 0:
            print(f"Average Training Accuracy: {metrics['walk_forward_train_mean']:.2%}")
            print(f"Average Testing Accuracy:  {metrics['walk_forward_test_mean']:.2%}")
        else:
            print(f"Training Accuracy: {metrics['train_accuracy']:.2%}")
            print(f"Testing Accuracy:  {metrics['test_accuracy']:.2%}")

