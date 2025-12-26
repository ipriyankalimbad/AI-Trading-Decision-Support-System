# AI Trading Decision Support System

A comprehensive, production-ready trading analysis platform that combines advanced machine learning ensemble models, technical analysis, dynamic risk management, and strategy backtesting. This system provides data-driven insights to support trading decisions through a complete analytical pipeline from raw market data to actionable recommendations. Built with a focus on modularity, scalability, and professional-grade code architecture.

**Live Demo**: [https://ai-trading-decision-support-system-pri.streamlit.app/](https://ai-trading-decision-support-system-pri.streamlit.app/)

---

## Overview

A production-ready Streamlit web application that provides end-to-end trading analysis:

- **Technical Indicators**: SMA, EMA, RSI, MACD, ATR with interactive visualizations
- **Advanced ML Predictions**: Ensemble model (RandomForest + GradientBoosting) achieving 58-68% test accuracy
- **Risk Management**: ATR-based dynamic stop-loss and risk-reward target calculations
- **Strategy Backtesting**: SMA crossover strategy with comprehensive performance metrics
- **Intelligent Q&A**: Context-aware analytical assistant

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Data Processing** | Automatic CSV validation, cleaning, and OHLCV standardization with robust error handling |
| **Technical Analysis** | 6+ professional indicators with interactive Plotly visualizations and real-time updates |
| **Advanced Machine Learning** | Ensemble model combining RandomForest and GradientBoosting with 60+ engineered features, feature scaling, and selection for optimal performance |
| **Risk Management** | Dynamic ATR-based stop-loss calculation, risk-reward target pricing, and comprehensive risk assessment metrics |
| **Backtesting Engine** | Complete strategy performance analysis with equity curves, Sharpe ratio, win rate, and trade-by-trade statistics |
| **Intelligent Assistant** | Context-aware Q&A system that analyzes all components of the analysis pipeline |

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/ipriyankalimbad/AI-Trading-Decision-Support-System.git
cd AI-Trading-Decision-Support-System

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Usage

1. **Upload CSV**: Upload OHLCV data with columns: `date`, `open`, `high`, `low`, `close`, `volume`
2. **Set Entry Price**: Enter your assumed entry price in the sidebar for risk calculations
3. **Run Analysis**: Click "Run Complete Analysis" to process all features
4. **Explore Results**: Navigate through tabs to view indicators, ML predictions, risk metrics, and backtesting results

---

## Architecture

```
PROJECT1/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
└── src/                   # Source modules
    ├── utils.py          # Data validation and cleaning
    ├── indicators.py     # Technical indicator calculations
    ├── ml_model.py       # ML model training and prediction
    ├── risk.py           # Risk management calculations
    ├── backtest.py       # Backtesting engine
    └── llm_helper.py     # Intelligent assistant integration
```

### Data Flow

```
CSV Upload → Data Validation → Indicator Calculation → ML Analysis
                                                              ↓
Risk Management ← Backtesting ← Technical Indicators ← Feature Engineering
```

---

## Technical Details

### Machine Learning

- **Model Architecture**: Ensemble Voting Classifier combining RandomForest (300 estimators) and GradientBoosting (200 estimators)
- **Feature Engineering**: 60+ features including volume trends, price position, RSI signals, MACD momentum, lagged features, and volatility measures
- **Preprocessing**: RobustScaler for feature normalization, SelectKBest for feature selection (top 50 features)
- **Target**: Binary classification (next day price direction)
- **Split**: Time-series aware (80/20, preserves temporal order)
- **Performance**: 58-68% test accuracy with comprehensive metrics (precision, recall, F1-score)
- **Metrics**: Accuracy, feature importance, prediction probabilities, model performance analysis

### Backtesting Strategy

- **Strategy**: SMA 20/50 Crossover (Long-only trend-following)
- **Signals**: Buy on bullish crossover, sell on bearish crossover
- **Metrics**: Total return, Sharpe ratio, win rate, average win/loss, equity curve analysis

### Risk Management

- **Stop-Loss**: ATR-based dynamic calculation (Entry Price - ATR × 2.0)
- **Target**: Risk-reward based (Entry Price + Risk × 2.0)
- **Classification**: Low (<2%), Moderate (2-5%), High (≥5%)

---

## Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (RandomForest, GradientBoosting, VotingClassifier)
- **Technical Analysis**: TA-Lib (ta library)
- **Visualization**: Plotly
- **Development**: Python 3.8+

---

## Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
ta>=0.11.0
matplotlib>=3.7.0
plotly>=5.17.0
openai>=1.3.0
```

---

## Project Highlights

This project stands out through several key differentiators:

- **Advanced Ensemble ML Architecture**: Unlike single-model approaches, this system employs a sophisticated ensemble combining RandomForest and GradientBoosting classifiers. With 60+ engineered features, feature scaling, and intelligent feature selection, the model achieves 58-68% test accuracy - professional-grade performance for financial prediction.

- **Comprehensive Feature Engineering**: The system goes beyond basic indicators, incorporating volume trends, price position analysis, momentum signals, lagged features, and volatility measures. This multi-dimensional feature space captures market dynamics more effectively than traditional approaches.

- **Production-Ready Codebase**: Built with modular architecture, comprehensive error handling, and professional code organization. Each component is independently testable and maintainable, following industry best practices.

- **Dynamic Risk Management**: Unlike static percentage-based stop-losses, the system uses ATR-based dynamic risk calculation that adapts to market volatility, providing more intelligent risk assessment.

- **Complete Analytical Pipeline**: From raw data ingestion through feature engineering, model training, risk assessment, strategy backtesting, and intelligent Q&A - this is a complete end-to-end system, not just isolated components.

- **Interactive Professional UI**: Dark-themed, responsive interface with real-time Plotly visualizations, comprehensive metrics display, and intuitive navigation - designed for professional use.

- **Deployed and Accessible**: Fully deployed on Streamlit Cloud with automatic updates, demonstrating real-world deployment capabilities.

---

## Disclaimer

This application is a **decision support system** designed for educational and analytical purposes. It provides analytical insights, technical indicators, and data-driven recommendations to assist in trading decisions. This system does not predict stock prices or guarantee trading outcomes. It is a tool for analysis and should be used in conjunction with your own research and professional financial advice. Trading stocks involves substantial risk, and past performance does not guarantee future results. Always conduct thorough research and consult with qualified financial professionals before making investment decisions.

---

## License

This project is provided as-is for educational and analytical purposes. Not financial advice.
