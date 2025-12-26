"""
LLM helper for AI Q&A using OpenAI API.
"""
import os
from typing import Dict, Optional
import json


def build_context(indicator_summary: Dict, ml_prediction: Dict, 
                 ml_metrics: Dict, risk_metrics: Dict, 
                 backtest_results: Dict) -> str:
    """
    Build structured context from all analysis components.
    
    Args:
        indicator_summary: Summary of technical indicators
        ml_prediction: ML model prediction results
        ml_metrics: ML model performance metrics
        risk_metrics: Risk management metrics
        backtest_results: Backtesting results
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    # Technical Indicators
    context_parts.append("=== TECHNICAL INDICATORS ===")
    if indicator_summary:
        cp = indicator_summary.get('current_price')
        context_parts.append(f"Current Price: ${cp:.2f}" if cp is not None else "Current Price: N/A")
        sma20 = indicator_summary.get('sma_20')
        context_parts.append(f"SMA 20: ${sma20:.2f}" if sma20 is not None else "SMA 20: N/A")
        sma50 = indicator_summary.get('sma_50')
        context_parts.append(f"SMA 50: ${sma50:.2f}" if sma50 is not None else "SMA 50: N/A")
        ema20 = indicator_summary.get('ema_20')
        context_parts.append(f"EMA 20: ${ema20:.2f}" if ema20 is not None else "EMA 20: N/A")
        rsi = indicator_summary.get('rsi_14')
        rsi_sig = indicator_summary.get('rsi_signal', 'N/A')
        context_parts.append(f"RSI (14): {rsi:.2f} ({rsi_sig})" if rsi is not None else f"RSI (14): N/A ({rsi_sig})")
        macd = indicator_summary.get('macd')
        macd_sig = indicator_summary.get('macd_signal_direction', 'N/A')
        context_parts.append(f"MACD: {macd:.4f} (Signal: {macd_sig})" if macd is not None else f"MACD: N/A (Signal: {macd_sig})")
        atr = indicator_summary.get('atr_14')
        context_parts.append(f"ATR (14): ${atr:.2f}" if atr is not None else "ATR (14): N/A")
        dr = indicator_summary.get('daily_return')
        context_parts.append(f"Daily Return: {dr:.2%}" if dr is not None else "Daily Return: N/A")
    
    # ML Predictions
    context_parts.append("\n=== MACHINE LEARNING PREDICTION ===")
    if ml_prediction:
        context_parts.append(f"Predicted Direction: {ml_prediction.get('direction', 'N/A')}")
        context_parts.append(f"Probability (Up): {ml_prediction.get('probability_up', 0):.2%}")
        context_parts.append(f"Probability (Down): {ml_prediction.get('probability_down', 0):.2%}")
        context_parts.append(f"Confidence: {ml_prediction.get('confidence', 0):.2%}")
    
    if ml_metrics:
        context_parts.append(f"Model Accuracy (Test): {ml_metrics.get('test_accuracy', 0):.2%}")
        context_parts.append(f"Model Type: {ml_metrics.get('model_type', 'N/A')}")
    
    # Risk Management
    context_parts.append("\n=== RISK MANAGEMENT ===")
    if risk_metrics and 'error' not in risk_metrics:
        ep = risk_metrics.get('entry_price')
        context_parts.append(f"Entry Price: ${ep:.2f}" if ep is not None else "Entry Price: N/A")
        sl = risk_metrics.get('stop_loss')
        context_parts.append(f"Stop-Loss: ${sl:.2f}" if sl is not None else "Stop-Loss: N/A")
        tg = risk_metrics.get('target')
        context_parts.append(f"Target: ${tg:.2f}" if tg is not None else "Target: N/A")
        rp = risk_metrics.get('risk_percentage')
        context_parts.append(f"Risk Percentage: {rp:.2f}%" if rp is not None else "Risk Percentage: N/A")
        rwp = risk_metrics.get('reward_percentage')
        context_parts.append(f"Reward Percentage: {rwp:.2f}%" if rwp is not None else "Reward Percentage: N/A")
        rr = risk_metrics.get('risk_reward_ratio')
        context_parts.append(f"Risk-Reward Ratio: {rr:.2f}" if rr is not None else "Risk-Reward Ratio: N/A")
        context_parts.append(f"Risk Level: {risk_metrics.get('risk_level', 'N/A')}")
    
    # Backtesting Results
    context_parts.append("\n=== BACKTESTING RESULTS ===")
    if backtest_results:
        context_parts.append(f"Strategy Total Return: {backtest_results.get('total_return', 0):.2f}%")
        context_parts.append(f"Buy & Hold Return: {backtest_results.get('buy_hold_return', 0):.2f}%")
        context_parts.append(f"Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
        context_parts.append(f"Total Trades: {backtest_results.get('total_trades', 0)}")
        context_parts.append(f"Win Rate: {backtest_results.get('win_rate', 0):.2f}%")
        context_parts.append(f"Average Win: ${backtest_results.get('avg_win', 0):.2f}")
        context_parts.append(f"Average Loss: ${backtest_results.get('avg_loss', 0):.2f}")
    
    return "\n".join(context_parts)


def query_llm(question: str, context: str, api_key: Optional[str] = None) -> str:
    """
    Query OpenAI API with context and question.
    
    Args:
        question: User's question
        context: Formatted context string
        api_key: OpenAI API key (if None, reads from environment)
        
    Returns:
        LLM response string
    """
    try:
        from openai import OpenAI
    except ImportError:
        return "Error: OpenAI library not installed. Please install it with: pip install openai"
    
    # Get API key (check Streamlit secrets first, then environment variable)
    if api_key is None:
        try:
            import streamlit as st
            # Try to get from Streamlit secrets (for deployed app)
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets['OPENAI_API_KEY']
        except:
            pass
        
        # Fall back to environment variable (for local development)
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        return "Error: OpenAI API key not found. Please set it in Streamlit Secrets (for deployed app) or as OPENAI_API_KEY environment variable (for local development)."
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Build system prompt with guidelines
        system_prompt = """You are a professional quantitative trading assistant. Your role is to provide analytical insights based on technical indicators, machine learning predictions, risk metrics, and backtesting results.

IMPORTANT GUIDELINES:
1. Use probabilistic language - never guarantee outcomes
2. Never provide direct buy/sell commands
3. Frame all advice as analysis and probabilities
4. Consider multiple factors when answering
5. Acknowledge uncertainty and market risks
6. Suggest considering multiple indicators together
7. Remind users that past performance doesn't guarantee future results

Provide thoughtful, balanced analysis that helps users understand the data and make informed decisions."""
        
        user_prompt = f"""Context from analysis:
{context}

User Question: {question}

Please provide a helpful, analytical response based on the context above. Remember to use probabilistic language and avoid direct trading commands."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error querying LLM: {str(e)}. Please check your API key and internet connection."


def get_llm_response(question: str, indicator_summary: Dict, 
                    ml_prediction: Dict, ml_metrics: Dict,
                    risk_metrics: Dict, backtest_results: Dict) -> str:
    """
    Complete LLM query pipeline.
    
    Args:
        question: User's question
        indicator_summary: Indicator summary
        ml_prediction: ML prediction
        ml_metrics: ML metrics
        risk_metrics: Risk metrics
        backtest_results: Backtest results
        
    Returns:
        LLM response
    """
    context = build_context(
        indicator_summary, ml_prediction, ml_metrics, 
        risk_metrics, backtest_results
    )
    
    return query_llm(question, context)

