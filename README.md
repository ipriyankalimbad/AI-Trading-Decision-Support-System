# 🤖 AI Trading Decision Support System

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-trading-decision-support-system-pri.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

**🌐 [Live Demo](https://ai-trading-decision-support-system-pri.streamlit.app/) | 💻 [GitHub Repository](https://github.com/ipriyankalimbad/AI-Trading-Decision-Support-System)**

*A comprehensive trading analysis platform combining Machine Learning, Technical Analysis, Risk Management, and AI-powered insights.*

</div>

---

## 📸 Screenshots

### Main Dashboard
![Dashboard](screenshots/dashboard.png)

### Technical Analysis Charts
![Charts](screenshots/charts.png)

### ML Predictions
![ML Predictions](screenshots/ml-predictions.png)

### Backtesting Results
![Backtesting](screenshots/backtesting.png)

---

## 🎯 Overview

A production-ready **Streamlit web application** that provides end-to-end trading analysis:

- **📊 Technical Indicators**: SMA, EMA, RSI, MACD, ATR
- **🤖 ML Predictions**: RandomForest-based price direction forecasting
- **🛡️ Risk Management**: ATR-based stop-loss and target calculations
- **📉 Strategy Backtesting**: SMA crossover strategy with performance metrics
- **💬 AI Q&A Assistant**: Context-aware analysis using OpenAI GPT

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Data Processing** | Automatic CSV validation, cleaning, and OHLCV standardization |
| **Technical Analysis** | 6+ indicators with interactive Plotly visualizations |
| **Machine Learning** | RandomForest classifier with feature importance analysis |
| **Risk Management** | Dynamic stop-loss, target prices, and risk-reward ratios |
| **Backtesting Engine** | Strategy performance vs buy-and-hold with equity curves |
| **AI Assistant** | GPT-3.5 powered Q&A with context from all analyses |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (optional, for AI Q&A feature)

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

1. **Upload CSV**: Upload OHLCV data (columns: `date`, `open`, `high`, `low`, `close`, `volume`)
2. **Set Entry Price**: Enter your entry price in the sidebar
3. **Run Analysis**: Click "Run Complete Analysis" to process all features
4. **Explore Results**: Navigate through tabs to view indicators, ML predictions, risk metrics, and backtesting results

---

## 🏗️ Architecture

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
    └── llm_helper.py     # LLM integration
```

### Data Flow

```
CSV Upload → Data Validation → Indicator Calculation → ML Analysis
                                                              ↓
Risk Management ← Backtesting ← Technical Indicators ← Feature Engineering
```

---

## 🔬 Technical Details

### Machine Learning
- **Model**: RandomForestClassifier (100 estimators)
- **Features**: Technical indicators + lagged features (close, RSI, MACD)
- **Target**: Binary classification (next day price direction)
- **Split**: Time-series aware (80/20, preserves temporal order)
- **Metrics**: Accuracy, feature importance, prediction probabilities

### Backtesting Strategy
- **Strategy**: SMA 20/50 Crossover (Long-only)
- **Signals**: Buy on bullish crossover, sell on bearish crossover
- **Metrics**: Total return, Sharpe ratio, win rate, average win/loss

### Risk Management
- **Stop-Loss**: ATR-based (Entry Price - ATR × 2.0)
- **Target**: Risk-reward based (Entry Price + Risk × 2.0)
- **Classification**: Low (<2%), Moderate (2-5%), High (≥5%)

---

## 📊 Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Technical Analysis**: TA-Lib (ta library)
- **Visualization**: Plotly
- **AI Integration**: OpenAI API (GPT-3.5-turbo)

---

## 📝 Requirements

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

## 🎓 Project Highlights

- ✅ **End-to-end ML pipeline**: From data ingestion to prediction
- ✅ **Production-ready code**: Modular architecture, error handling
- ✅ **Interactive visualizations**: Real-time charts with Plotly
- ✅ **Risk-aware design**: ATR-based stop-loss and risk metrics
- ✅ **AI integration**: Context-aware Q&A with safety guidelines
- ✅ **Deployed application**: Live on Streamlit Cloud

---

## 📸 Screenshots Setup

To add screenshots to your README:

1. Create a `screenshots/` folder in your project
2. Take screenshots of your app:
   - Dashboard view
   - Charts tab
   - ML Predictions tab
   - Backtesting tab
3. Save them as: `dashboard.png`, `charts.png`, `ml-predictions.png`, `backtesting.png`
4. The README will automatically display them

**Quick Screenshot Guide:**
- Open your app: https://ai-trading-decision-support-system-pri.streamlit.app/
- Take screenshots using Windows Snipping Tool (Win + Shift + S)
- Save in `screenshots/` folder
- Push to GitHub

---

## 🔗 Links

- **🌐 Live Application**: [https://ai-trading-decision-support-system-pri.streamlit.app/](https://ai-trading-decision-support-system-pri.streamlit.app/)
- **💻 GitHub Repository**: [https://github.com/ipriyankalimbad/AI-Trading-Decision-Support-System](https://github.com/ipriyankalimbad/AI-Trading-Decision-Support-System)

---

## ⚠️ Disclaimer

This application is for **educational and analytical purposes only**. It does not constitute financial advice. Trading stocks involves risk, and past performance does not guarantee future results. Always conduct your own research and consult with financial professionals before making investment decisions.

---

## 📄 License

This project is provided as-is for educational and analytical purposes. Not financial advice.

---

<div align="center">

**Built with ❤️ using Streamlit, Python, and Machine Learning**

⭐ Star this repo if you find it helpful!

</div>
