# AI Stock Trading Assistant (LLM-Powered)

A comprehensive, production-ready Streamlit application that combines technical analysis, machine learning predictions, risk management, strategy backtesting, and AI-powered Q&A for stock trading analysis.

## Why this project?

This project was built to showcase:
- End-to-end **data → ML → decision support** pipelines
- Practical use of **machine learning for time-series classification**
- **Risk-aware trading analysis** instead of naive predictions
- Safe and explainable **LLM integration** for analytical Q&A
- Clean, modular, production-style Python architecture

It is designed for **educational, analytical, and portfolio demonstration purposes**.


## Features

### 1. **Data Processing & Validation**
- CSV upload with automatic validation
- OHLCV data cleaning and standardization
- Date parsing and sorting
- Data quality checks (OHLC relationships, missing values)

### 2. **Technical Indicators**
- **SMA 20/50**: Simple Moving Averages for trend identification
- **EMA 20**: Exponential Moving Average for responsive trend signals
- **RSI 14**: Relative Strength Index with 30/70 overbought/oversold bands
- **MACD**: Moving Average Convergence Divergence with signal line
- **ATR 14**: Average True Range for volatility measurement
- **Daily Returns**: Price change calculations

### 3. **Machine Learning Prediction**
- **RandomForestClassifier** for next-step price direction prediction
- Time-series aware train/test split (preserves temporal order)
- Feature engineering with lagged indicators
- Probability outputs for up/down predictions
- Model accuracy metrics and feature importance analysis

### 4. **Risk Management**
- **ATR-based Stop-Loss**: Dynamic stop-loss calculation using Average True Range
- **Target Price**: Risk-reward ratio based target calculation
- **Risk Metrics**: Comprehensive risk assessment including:
  - Risk per share and percentage
  - Reward per share and percentage
  - Risk-reward ratio
  - Risk level classification (Low/Moderate/High)

### 5. **Strategy Backtesting**
- **SMA 20/50 Crossover Strategy**: Long-only trend-following strategy
- **Equity Curve**: Visual comparison with buy-and-hold
- **Performance Metrics**:
  - Total return vs buy-and-hold
  - Sharpe ratio
  - Win rate
  - Average win/loss
  - Individual trade statistics

### 6. **AI Q&A Assistant**
- **OpenAI Integration**: GPT-3.5-turbo powered assistant
- **Context-Aware**: Reasons over all analysis outputs (indicators, ML, risk, backtest)
- **Safety-First Design**: Uses probabilistic language, no direct trading advice
- **Structured Context**: Builds comprehensive context from quantitative analysis

## Architecture

### Project Structure
```
PROJECT1/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── src/                  # Source modules
    ├── utils.py          # Data validation and cleaning
    ├── indicators.py     # Technical indicator calculations
    ├── ml_model.py       # ML model training and prediction
    ├── risk.py           # Risk management calculations
    ├── backtest.py       # Backtesting engine
    └── llm_helper.py     # LLM integration and context building
```

### Data Flow

1. **Data Input**: User uploads CSV → `utils.py` validates and cleans
2. **Indicator Calculation**: Clean data → `indicators.py` computes all technical indicators
3. **ML Analysis**: Indicators → `ml_model.py` trains model and predicts direction
4. **Risk Management**: Indicators + Entry Price → `risk.py` calculates stop-loss and targets
5. **Backtesting**: Indicators → `backtest.py` runs SMA crossover strategy
6. **LLM Q&A**: All outputs → `llm_helper.py` builds context and queries OpenAI

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (for LLM features)

### Setup

1. **Clone or download the project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set OpenAI API key** (optional, for AI Q&A):
   ```bash
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your-api-key-here"
   
   # Linux/Mac
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### 1. Upload Data
- Click "Upload CSV file" in the sidebar
- CSV must contain: `date`, `open`, `high`, `low`, `close`, `volume` columns
- Column names are case-insensitive

### 2. Set Entry Price
- Enter your assumed entry price in the sidebar
- This is used for risk management calculations

### 3. Run Analysis
- Click "Run Complete Analysis" button
- The system will:
  - Calculate all technical indicators
  - Train ML model and make predictions
  - Calculate risk management metrics
  - Run backtesting on SMA crossover strategy

### 4. Review Results
- **Data Preview**: View uploaded data summary
- **Price & Indicator Charts**: Interactive charts with all indicators
- **ML Prediction Summary**: Model predictions and probabilities
- **Risk Management**: Stop-loss, target, and risk-reward metrics
- **Strategy Backtesting**: Equity curve and trade statistics
- **AI Q&A**: Ask questions about the analysis

## Technical Details

### Machine Learning Approach

**Model**: RandomForestClassifier
- **Algorithm**: Ensemble of decision trees (100 estimators)
- **Features**: Technical indicators + lagged features (close, RSI, MACD)
- **Target**: Binary classification (next day price up/down)
- **Train/Test Split**: Time-series split (80/20, preserving temporal order)
- **Hyperparameters**:
  - `n_estimators=100`
  - `max_depth=10`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
  - `random_state=42`

**Why RandomForest?**
- Handles non-linear relationships well
- Provides feature importance
- Robust to overfitting with proper parameters
- Fast training and prediction

### Backtesting Logic

**Strategy**: SMA 20/50 Crossover (Long-Only)

1. **Signal Generation**:
   - Buy signal: When SMA 20 crosses above SMA 50 (bullish crossover)
   - Sell signal: When SMA 20 crosses below SMA 50 (bearish crossover)

2. **Position Management**:
   - Long-only: Only enters long positions
   - Maintains position until sell signal
   - No shorting or leverage

3. **Returns Calculation**:
   - Strategy returns = position × daily returns
   - Cumulative returns = (1 + strategy_returns).cumprod()
   - Equity = initial_capital × cumulative_returns

4. **Metrics**:
   - Total return: Final equity / initial capital - 1
   - Sharpe ratio: (mean return / std return) × √252
   - Win rate: Winning trades / total trades
   - Average win/loss: Mean P&L of winning/losing trades

### Risk Management Logic

**ATR-Based Stop-Loss**:
- Stop-loss = Entry Price - (ATR × Multiplier)
- Default multiplier: 2.0
- ATR provides volatility-adjusted stop distance

**Target Calculation**:
- Target = Entry Price + (Risk × Risk-Reward Ratio)
- Default risk-reward ratio: 2.0
- Risk = Entry Price - Stop-Loss

**Risk Classification**:
- Low: Risk < 2%
- Moderate: 2% ≤ Risk < 5%
- High: Risk ≥ 5%

### LLM Safety Design

**System Prompt Guidelines**:
1. Use probabilistic language - never guarantee outcomes
2. Never provide direct buy/sell commands
3. Frame all advice as analysis and probabilities
4. Consider multiple factors when answering
5. Acknowledge uncertainty and market risks
6. Suggest considering multiple indicators together
7. Remind users that past performance doesn't guarantee future results

**Context Building**:
- Structured context from all analysis components
- Includes indicators, ML predictions, risk metrics, backtest results
- Formatted for clear LLM understanding

**Error Handling**:
- Graceful degradation if API key missing
- Clear error messages for API failures
- Fallback behavior when LLM unavailable

## File Descriptions

### `app.py`
Main Streamlit application with dashboard UI. Handles:
- User interface (sidebar + main content)
- File upload and data processing orchestration
- Visualization with Plotly charts
- Integration of all analysis modules
- Session state management

### `src/utils.py`
Data validation and cleaning utilities:
- `validate_csv()`: Checks for required OHLCV columns
- `clean_data()`: Standardizes column names, converts types, validates relationships
- `prepare_data()`: Main pipeline for data preparation

### `src/indicators.py`
Technical indicator calculations:
- `calculate_indicators()`: Computes all indicators using `ta` library
- `get_indicator_summary()`: Extracts current indicator values and interpretations

### `src/ml_model.py`
Machine learning pipeline:
- `prepare_features()`: Feature engineering with lagged indicators
- `train_model()`: Time-series split and RandomForest training
- `predict_next_direction()`: Makes predictions with probabilities
- `run_ml_analysis()`: Complete ML pipeline

### `src/risk.py`
Risk management calculations:
- `calculate_atr_based_stop_loss()`: ATR-based stop-loss calculation
- `calculate_target_price()`: Risk-reward based target
- `calculate_risk_metrics()`: Comprehensive risk assessment
- `get_risk_recommendations()`: Complete risk analysis pipeline

### `src/backtest.py`
Backtesting engine:
- `BacktestEngine` class: Main backtesting engine
- `sma_crossover_strategy()`: Generates buy/sell signals
- `calculate_returns()`: Computes strategy and buy-hold returns
- `get_trade_stats()`: Extracts individual trade statistics
- `run_backtest()`: Complete backtesting pipeline

### `src/llm_helper.py`
LLM integration:
- `build_context()`: Structures analysis outputs into LLM context
- `query_llm()`: OpenAI API integration with safety prompts
- `get_llm_response()`: Complete Q&A pipeline

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning (RandomForestClassifier)
- **ta**: Technical analysis library
- **matplotlib**: Plotting (used by ta)
- **plotly**: Interactive charts
- **openai**: OpenAI API client

## Notes

- The application requires at least 50+ rows of data for reliable indicator calculations
- ML model needs sufficient historical data for training (recommended: 100+ rows)
- ATR-based stop-loss requires at least 14 periods of data
- OpenAI API key is optional - all other features work without it
- The application uses session state to cache analysis results

## License

This project is provided as-is for educational and analytical purposes. Not financial advice.

## Disclaimer

This application is for educational and analytical purposes only. It does not constitute financial advice. Trading stocks involves risk, and past performance does not guarantee future results. Always conduct your own research and consult with financial professionals before making investment decisions.

