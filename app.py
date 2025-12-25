"""
AI Stock Trading Assistant (LLM-Powered)
Main Streamlit application with dashboard UI.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import prepare_data
from src.indicators import calculate_indicators, get_indicator_summary
from src.ml_model import run_ml_analysis
from src.risk import get_risk_recommendations
from src.backtest import BacktestEngine
from src.llm_helper import get_llm_response

# Page configuration - Force sidebar to be visible
st.set_page_config(
    page_title="AI Stock Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "AI Stock Trading Assistant - Professional Trading Analysis Platform"
    }
)

# Force sidebar to be expanded (can't be collapsed)
if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

# Dark Mode Premium CSS Styling
st.markdown("""
    <style>
    /* Dark Mode Base */
    .stApp {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    
    /* Main content area */
    .main > div {
        padding-top: 2rem;
        background-color: transparent;
    }
    
    /* Header styling - Modern gradient */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #5b86e5 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }
    
    /* Sidebar - Dark theme - Always visible and accessible */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        border-right: 1px solid #334155 !important;
        min-width: 300px !important;
    }
    
    /* Force sidebar to be visible */
    [data-testid="stSidebar"][aria-expanded="false"] {
        display: block !important;
        visibility: visible !important;
    }
    
    /* Sidebar content visibility */
    [data-testid="stSidebar"] > * {
        visibility: visible !important;
        display: block !important;
    }
    
    /* Ensure sidebar toggle button is always visible and prominent */
    button[kind="header"] {
        background-color: #5b86e5 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(91, 134, 229, 0.4) !important;
        z-index: 9999 !important;
    }
    
    button[kind="header"]:hover {
        background-color: #6b96f5 !important;
        transform: scale(1.05);
    }
    
    /* Make sure sidebar header is visible */
    [data-testid="stSidebar"] [data-testid="stHeader"] {
        display: block !important;
        visibility: visible !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #f1f5f9;
    }
    
    /* Text colors for dark mode */
    .stMarkdown, .stText, p, li, td, th {
        color: #e2e8f0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    
    /* Metric cards - Enhanced for dark mode */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #00d4ff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.95rem;
    }
    
    /* Button styling - Modern dark theme */
    .stButton > button {
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #5b86e5 0%, #a855f7 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(91, 134, 229, 0.4);
        background: linear-gradient(135deg, #6b96f5 0%, #b865ff 100%);
    }
    
    /* Tab styling - Dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        color: #94a3b8;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #5b86e5 0%, #a855f7 100%);
        color: white !important;
    }
    
    /* Expander styling - Dark theme */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #f1f5f9 !important;
        background-color: #1e293b;
        border-radius: 6px;
    }
    
    .streamlit-expanderContent {
        background-color: #0f172a;
        border-radius: 6px;
    }
    
    /* Info boxes - Dark theme with colored borders */
    .stInfo {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
        color: #e2e8f0;
        border-radius: 6px;
    }
    
    .stSuccess {
        background-color: #1e293b;
        border-left: 4px solid #10b981;
        color: #e2e8f0;
        border-radius: 6px;
    }
    
    .stWarning {
        background-color: #1e293b;
        border-left: 4px solid #f59e0b;
        color: #e2e8f0;
        border-radius: 6px;
    }
    
    .stError {
        background-color: #1e293b;
        border-left: 4px solid #ef4444;
        color: #e2e8f0;
        border-radius: 6px;
    }
    
    /* Dataframe styling - Dark theme */
    .dataframe {
        border-radius: 8px;
        background-color: #1e293b;
    }
    
    .dataframe thead {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    .dataframe tbody tr {
        background-color: #1e293b;
        color: #e2e8f0;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #0f172a;
    }
    
    /* Input fields - Dark theme */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: #1e293b;
        color: #e2e8f0;
        border: 1px solid #334155;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #5b86e5;
        box-shadow: 0 0 0 2px rgba(91, 134, 229, 0.2);
    }
    
    /* File uploader - Dark theme */
    .stFileUploader > div {
        background-color: #1e293b;
        border: 2px dashed #334155;
        border-radius: 8px;
    }
    
    .stFileUploader > div:hover {
        border-color: #5b86e5;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #5b86e5 0%, #a855f7 100%);
    }
    
    /* Selectbox and other inputs */
    .stSelectbox > div > div {
        background-color: #1e293b;
        color: #e2e8f0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar styling for dark mode */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'indicators_calculated' not in st.session_state:
    st.session_state.indicators_calculated = False
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None
if 'risk_metrics' not in st.session_state:
    st.session_state.risk_metrics = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

# Premium header
st.markdown('<h1 class="main-header">AI Stock Trading Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional Trading Analysis Platform with ML & AI-Powered Insights</p>', unsafe_allow_html=True)

# Sidebar - Control Panel (Always Visible)
with st.sidebar:
    st.markdown("## 📊 Control Panel")
    st.markdown("*All controls and features are in this panel*")
    st.markdown("---")
    
    # Data Upload Section
    st.markdown("### 📁 Data Input")
    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=['csv'],
        help="CSV must contain: date, open, high, low, close, volume columns",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            df_clean = prepare_data(df_raw)
            st.session_state.df_processed = df_clean
            st.session_state.data_processed = True
            st.success(f"✅ **{len(df_clean)}** rows loaded")
            with st.expander("📅 Data Range", expanded=False):
                st.write(f"**Start:** {df_clean['date'].min().date()}")
                st.write(f"**End:** {df_clean['date'].max().date()}")
                st.write(f"**Period:** {(df_clean['date'].max() - df_clean['date'].min()).days} days")
        except Exception as e:
            st.error(f"❌ **Error:** {str(e)}")
            st.session_state.data_processed = False
    else:
        st.info("👆 Upload a CSV file to begin")
    
    st.markdown("---")
    
    # Entry Price
    st.markdown("### 💰 Position Settings")
    entry_price = st.number_input(
        "Entry Price ($)",
        min_value=0.01,
        value=100.0,
        step=0.01,
        format="%.2f",
        help="Assumed entry price for risk calculations"
    )
    
    st.markdown("---")
    
    # Analysis Control
    st.markdown("### ⚙️ Analysis")
    run_analysis = st.button(
        "🚀 Run Complete Analysis",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.data_processed
    )
    
    if run_analysis and st.session_state.data_processed:
        with st.spinner("🔄 Processing..."):
            try:
                # Calculate indicators
                df_with_indicators = calculate_indicators(st.session_state.df_processed)
                st.session_state.df_processed = df_with_indicators
                st.session_state.indicators_calculated = True
                
                # Run ML analysis
                model, ml_metrics, ml_prediction = run_ml_analysis(df_with_indicators)
                st.session_state.ml_results = {
                    'model': model,
                    'metrics': ml_metrics,
                    'prediction': ml_prediction
                }
                
                # Calculate risk metrics
                risk_metrics = get_risk_recommendations(df_with_indicators, entry_price)
                st.session_state.risk_metrics = risk_metrics
                
                # Run backtest
                backtest_engine = BacktestEngine(initial_capital=10000.0)
                backtest_results = backtest_engine.run_backtest(df_with_indicators)
                st.session_state.backtest_results = backtest_results
                
                st.success("✅ Analysis complete!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ **Error:** {str(e)}")
    
    st.markdown("---")
    
    # Status indicator
    if st.session_state.indicators_calculated:
        st.markdown("### ✅ Status")
        st.success("**Analysis Ready**")
    else:
        st.markdown("### ⏳ Status")
        st.info("**Waiting for Analysis**")
    
    st.markdown("---")
    
    # Info section
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **Features:**
        - Technical Indicators
        - ML Predictions
        - Risk Management
        - Strategy Backtesting
        - AI Q&A Assistant
        """)

# Main content area - Show sidebar instruction if needed
st.markdown("""
<div style='background: linear-gradient(135deg, #5b86e5 0%, #a855f7 100%); padding: 1rem; border-radius: 8px; margin-bottom: 2rem; text-align: center;'>
    <p style='color: white; font-weight: 600; margin: 0; font-size: 1.1rem;'>
        📊 <strong>Control Panel is in the LEFT SIDEBAR</strong> → Use ☰ button (top-left) if you don't see it
    </p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.data_processed:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <h2 style='color: #f1f5f9; margin-bottom: 2rem;'>Welcome</h2>
            <p style='font-size: 1.1rem; color: #94a3b8; line-height: 1.8; margin-bottom: 2rem;'>
                Upload a CSV file with OHLCV data to begin your analysis.
            </p>
            <div style='background: #1e293b; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #5b86e5; margin-top: 2rem;'>
                <p style='color: #e2e8f0; font-size: 0.95rem; margin: 0;'>
                    <strong>💡 Tip:</strong> Use the <strong>☰ menu button</strong> (top-left) to open/close the Control Panel sidebar.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 Expected CSV Format", expanded=True):
            st.markdown("""
            Your CSV file should contain the following columns:
            
            | Column | Description | Example |
            |--------|-------------|---------|
            | `date` | Date in any parseable format | 2024-01-01 |
            | `open` | Opening price | 100.50 |
            | `high` | High price | 105.00 |
            | `low` | Low price | 99.00 |
            | `close` | Closing price | 103.50 |
            | `volume` | Trading volume | 1000000 |
            
            **Note:** Column names are case-insensitive.
            """)
else:
    df = st.session_state.df_processed
    
    # Create tabs for main sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📈 Charts",
        "🤖 ML Prediction",
        "🛡️ Risk Management",
        "📉 Backtesting",
        "💬 AI Assistant"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.markdown("## 📊 Data Overview")
        st.markdown("---")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            days = (df['date'].max() - df['date'].min()).days
            st.metric("Date Range", f"{days} days")
        with col3:
            st.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
        with col4:
            st.metric("Latest Volume", f"{df['volume'].iloc[-1]:,.0f}")
        
        st.markdown("### Recent Data")
        st.dataframe(
            df[['date', 'open', 'high', 'low', 'close', 'volume']].tail(10),
            width='stretch',
            hide_index=True
        )
        
        # Indicator summary if available
        if st.session_state.indicators_calculated:
            st.markdown("### 📈 Current Indicators")
            indicator_summary = get_indicator_summary(df)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if indicator_summary.get('sma_20'):
                    st.metric("SMA 20", f"${indicator_summary['sma_20']:.2f}")
            with col2:
                if indicator_summary.get('sma_50'):
                    st.metric("SMA 50", f"${indicator_summary['sma_50']:.2f}")
            with col3:
                if indicator_summary.get('rsi_14'):
                    rsi = indicator_summary['rsi_14']
                    delta_color = "normal" if 30 <= rsi <= 70 else "inverse"
                    st.metric("RSI (14)", f"{rsi:.1f}", delta=indicator_summary.get('rsi_signal', ''), delta_color=delta_color)
            with col4:
                if indicator_summary.get('atr_14'):
                    st.metric("ATR (14)", f"${indicator_summary['atr_14']:.2f}")
            
            with st.expander("📋 Full Indicator Details", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Trend Indicators:**")
                    st.write(f"- EMA 20: ${indicator_summary.get('ema_20', 'N/A'):.2f}" if indicator_summary.get('ema_20') else "- EMA 20: N/A")
                    st.write(f"- MACD: {indicator_summary.get('macd', 'N/A'):.4f}" if indicator_summary.get('macd') else "- MACD: N/A")
                    st.write(f"- MACD Signal: {indicator_summary.get('macd_signal_direction', 'N/A')}")
                with col2:
                    st.write("**Momentum:**")
                    st.write(f"- Daily Return: {indicator_summary.get('daily_return', 0):.2%}" if indicator_summary.get('daily_return') else "- Daily Return: N/A")
        else:
            st.info("👆 Run analysis to see indicator summary")
    
    # Tab 2: Charts
    with tab2:
        if st.session_state.indicators_calculated:
            st.markdown("## 📈 Technical Analysis Charts")
            st.markdown("---")
            
            # Price chart with moving averages
            fig_price = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=('Price with Moving Averages', 'RSI (14)', 'MACD'),
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Price and moving averages
            fig_price.add_trace(
                go.Scatter(x=df['date'], y=df['close'], name='Close Price', 
                          line=dict(color='#1f77b4', width=2.5)),
                row=1, col=1
            )
            if 'sma_20' in df.columns:
                fig_price.add_trace(
                    go.Scatter(x=df['date'], y=df['sma_20'], name='SMA 20', 
                              line=dict(color='#ff7f0e', width=2)),
                    row=1, col=1
                )
            if 'sma_50' in df.columns:
                fig_price.add_trace(
                    go.Scatter(x=df['date'], y=df['sma_50'], name='SMA 50', 
                              line=dict(color='#d62728', width=2)),
                    row=1, col=1
                )
            if 'ema_20' in df.columns:
                fig_price.add_trace(
                    go.Scatter(x=df['date'], y=df['ema_20'], name='EMA 20', 
                              line=dict(color='#2ca02c', width=2, dash='dash')),
                    row=1, col=1
                )
            
            # RSI
            if 'rsi_14' in df.columns:
                fig_price.add_trace(
                    go.Scatter(x=df['date'], y=df['rsi_14'], name='RSI', 
                              line=dict(color='#9467bd', width=2.5)),
                    row=2, col=1
                )
                fig_price.add_hline(y=70, line_dash="dash", line_color="red", 
                                   annotation_text="Overbought (70)", row=2, col=1)
                fig_price.add_hline(y=30, line_dash="dash", line_color="green", 
                                   annotation_text="Oversold (30)", row=2, col=1)
                fig_price.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2, 
                                   layer="below", row=2, col=1)
            
            # MACD
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                fig_price.add_trace(
                    go.Scatter(x=df['date'], y=df['macd'], name='MACD', 
                              line=dict(color='#1f77b4', width=2.5)),
                    row=3, col=1
                )
                fig_price.add_trace(
                    go.Scatter(x=df['date'], y=df['macd_signal'], name='Signal', 
                              line=dict(color='#d62728', width=2.5)),
                    row=3, col=1
                )
                if 'macd_diff' in df.columns:
                    colors = ['green' if x >= 0 else 'red' for x in df['macd_diff']]
                    fig_price.add_trace(
                        go.Bar(x=df['date'], y=df['macd_diff'], name='Histogram', 
                              marker_color=colors, opacity=0.6),
                        row=3, col=1
                    )
            
            fig_price.update_layout(
                height=850,
                showlegend=True,
                hovermode='x unified',
                template='plotly_dark',
                title_text="",
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig_price.update_xaxes(title_text="Date", row=3, col=1)
            fig_price.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig_price.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            fig_price.update_yaxes(title_text="MACD", row=3, col=1)
            
            st.plotly_chart(fig_price, width='stretch')
        else:
            st.info("👆 Run analysis to view charts")
    
    # Tab 3: ML Prediction
    with tab3:
        if st.session_state.ml_results:
            st.markdown("## 🤖 Machine Learning Prediction")
            st.markdown("---")
            
            ml_metrics = st.session_state.ml_results['metrics']
            ml_prediction = st.session_state.ml_results['prediction']
            
            # Main prediction metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                direction = ml_prediction.get('direction', 'N/A')
                direction_emoji = "📈" if direction == 'Up' else "📉"
                st.metric("Predicted Direction", f"{direction_emoji} {direction}")
            with col2:
                confidence = ml_prediction.get('confidence', 0)
                st.metric("Confidence", f"{confidence:.1%}")
            with col3:
                test_acc = ml_metrics.get('test_accuracy', 0)
                st.metric("Test Accuracy", f"{test_acc:.1%}")
            with col4:
                train_acc = ml_metrics.get('train_accuracy', 0)
                st.metric("Train Accuracy", f"{train_acc:.1%}")
            
            st.markdown("### Prediction Probabilities")
            col1, col2 = st.columns(2)
            with col1:
                prob_up = ml_prediction.get('probability_up', 0)
                st.progress(prob_up, text=f"**Up:** {prob_up:.1%}")
            with col2:
                prob_down = ml_prediction.get('probability_down', 0)
                st.progress(prob_down, text=f"**Down:** {prob_down:.1%}")
            
            with st.expander("🔍 Model Details", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Model Information:**")
                    st.write(f"- Type: {ml_metrics.get('model_type', 'N/A')}")
                    st.write(f"- Training Samples: {ml_metrics.get('train_samples', 0):,}")
                    st.write(f"- Test Samples: {ml_metrics.get('test_samples', 0):,}")
                with col2:
                    st.markdown("**Top Features (Importance):**")
                    top_features = ml_metrics.get('top_features', [])
                    for feature, importance in top_features[:5]:
                        st.write(f"- {feature}: `{importance:.4f}`")
        else:
            st.info("👆 Run analysis to see ML predictions")
    
    # Tab 4: Risk Management
    with tab4:
        if st.session_state.risk_metrics and 'error' not in st.session_state.risk_metrics:
            st.markdown("## 🛡️ Risk Management")
            st.markdown("---")
            
            risk = st.session_state.risk_metrics
            
            # Key risk metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Entry Price", f"${risk.get('entry_price', 0):.2f}")
            with col2:
                stop_loss = risk.get('stop_loss', 0)
                risk_pct = risk.get('risk_percentage', 0)
                st.metric("Stop-Loss", f"${stop_loss:.2f}", 
                         delta=f"-{risk_pct:.2f}%", delta_color="inverse")
            with col3:
                target = risk.get('target', 0)
                reward_pct = risk.get('reward_percentage', 0)
                st.metric("Target", f"${target:.2f}",
                         delta=f"+{reward_pct:.2f}%")
            with col4:
                rr_ratio = risk.get('risk_reward_ratio', 0)
                st.metric("Risk-Reward Ratio", f"{rr_ratio:.2f}")
            
            # Additional metrics
            st.markdown("### Risk Assessment")
            col1, col2, col3 = st.columns(3)
            with col1:
                risk_level = risk.get('risk_level', 'Unknown')
                color_map = {'Low': '🟢', 'Moderate': '🟡', 'High': '🔴'}
                st.metric("Risk Level", f"{color_map.get(risk_level, '⚪')} {risk_level}")
            with col2:
                st.metric("Risk per Share", f"${risk.get('risk_per_share', 0):.2f}")
            with col3:
                st.metric("Reward per Share", f"${risk.get('reward_per_share', 0):.2f}")
            
            # Visual representation
            if risk.get('stop_loss') and risk.get('target'):
                st.markdown("### Price Levels Visualization")
                fig_risk = go.Figure()
                entry = risk.get('entry_price')
                stop = risk.get('stop_loss')
                target = risk.get('target')
                current = risk.get('current_price', entry)
                
                fig_risk.add_trace(go.Scatter(
                    x=['Stop-Loss', 'Entry', 'Current', 'Target'],
                    y=[stop, entry, current, target],
                    mode='markers+lines',
                    marker=dict(size=[20, 20, 20, 20], 
                              color=['#dc3545', '#007bff', '#28a745', '#28a745']),
                    line=dict(width=3, color='#6c757d', dash='dash'),
                    name='Price Levels',
                    hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
                ))
                fig_risk.update_layout(
                    title="",
                    yaxis_title="Price ($)",
                    height=450,
                    template='plotly_dark',
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_risk, width='stretch')
            
            with st.expander("📊 Detailed Risk Metrics", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ATR Value:** ${risk.get('atr_value', 0):.2f}")
                    st.write(f"**ATR Multiplier:** {risk.get('atr_multiplier', 0):.1f}x")
                    st.write(f"**Desired Risk-Reward:** {risk.get('desired_risk_reward', 0):.1f}:1")
                with col2:
                    st.write(f"**Total Risk:** ${risk.get('total_risk', 0):.2f}")
                    st.write(f"**Total Reward:** ${risk.get('total_reward', 0):.2f}")
                    st.write(f"**Current Price:** ${risk.get('current_price', 0):.2f}")
        else:
            st.info("👆 Run analysis to see risk management recommendations")
    
    # Tab 5: Backtesting
    with tab5:
        if st.session_state.backtest_results:
            st.markdown("## 📉 Strategy Backtesting")
            st.markdown("---")
            
            bt = st.session_state.backtest_results
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                strategy_return = bt.get('total_return', 0)
                buy_hold_return = bt.get('buy_hold_return', 0)
                delta = strategy_return - buy_hold_return
                st.metric("Strategy Return", f"{strategy_return:.2f}%",
                         delta=f"{delta:+.2f}% vs Buy & Hold")
            with col2:
                st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{bt.get('sharpe_ratio', 0):.2f}")
            with col4:
                st.metric("Win Rate", f"{bt.get('win_rate', 0):.1f}%")
            
            # Trade statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Trades", bt.get('total_trades', 0))
            with col2:
                st.metric("Avg Win", f"${bt.get('avg_win', 0):.2f}")
            with col3:
                st.metric("Avg Loss", f"${bt.get('avg_loss', 0):.2f}")
            
            # Equity curve
            if 'equity_curve' in bt and not bt['equity_curve'].empty:
                st.markdown("### Equity Curve")
                fig_equity = go.Figure()
                equity_df = bt['equity_curve']
                fig_equity.add_trace(go.Scatter(
                    x=equity_df['date'],
                    y=equity_df['equity'],
                    name='Strategy Equity',
                    line=dict(color='#1f77b4', width=3),
                    hovertemplate='Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
                ))
                fig_equity.add_trace(go.Scatter(
                    x=equity_df['date'],
                    y=equity_df['buy_hold_equity'],
                    name='Buy & Hold Equity',
                    line=dict(color='#6c757d', width=2.5, dash='dash'),
                    hovertemplate='Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
                ))
                fig_equity.update_layout(
                    title="",
                    xaxis_title="Date",
                    yaxis_title="Equity ($)",
                    height=500,
                    template='plotly_dark',
                    hovermode='x unified',
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_equity, width='stretch')
            
            # Trade details
            if bt.get('trades'):
                with st.expander("📋 Trade History", expanded=False):
                    trades_df = pd.DataFrame(bt['trades'])
                    if not trades_df.empty:
                        display_cols = ['entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl', 'pnl_percentage']
                        available_cols = [col for col in display_cols if col in trades_df.columns]
                        st.dataframe(
                            trades_df[available_cols].tail(20),
                            width='stretch',
                            hide_index=True
                        )
        else:
            st.info("👆 Run analysis to see backtesting results")
    
    # Tab 6: AI Assistant
    with tab6:
        st.markdown("## 💬 AI Q&A Assistant")
        st.markdown("---")
        
        # Check for API key
        api_key_set = os.getenv('OPENAI_API_KEY') is not None
        
        if not api_key_set:
            st.warning("⚠️ **OpenAI API Key Not Set**")
            st.info("""
            To enable AI Q&A features, set your OpenAI API key as an environment variable:
            
            **Windows PowerShell:**
            ```powershell
            $env:OPENAI_API_KEY="your-api-key-here"
            ```
            
            **Windows CMD:**
            ```cmd
            set OPENAI_API_KEY=your-api-key-here
            ```
            
            Then restart the application.
            """)
        else:
            st.success("✅ **AI Assistant Ready**")
        
        st.markdown("### Ask a Question")
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What does the RSI indicate about the current trend?",
            disabled=not api_key_set,
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("Ask AI", disabled=not api_key_set or not question, use_container_width=True)
        
        if ask_button:
            if st.session_state.indicators_calculated:
                with st.spinner("🤔 AI is analyzing..."):
                    try:
                        indicator_summary = get_indicator_summary(df)
                        ml_pred = st.session_state.ml_results['prediction'] if st.session_state.ml_results else {}
                        ml_met = st.session_state.ml_results['metrics'] if st.session_state.ml_results else {}
                        risk_met = st.session_state.risk_metrics if st.session_state.risk_metrics else {}
                        bt_res = st.session_state.backtest_results if st.session_state.backtest_results else {}
                        
                        response = get_llm_response(
                            question,
                            indicator_summary,
                            ml_pred,
                            ml_met,
                            risk_met,
                            bt_res
                        )
                        
                        st.markdown("### 💡 AI Response")
                        st.markdown("---")
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"❌ **Error:** {str(e)}")
            else:
                st.warning("⚠️ Please run the analysis first.")
        
        if api_key_set:
            st.markdown("---")
            st.info("💡 **Note:** The AI assistant uses probabilistic language and provides analytical insights. It does not provide direct trading advice.")
