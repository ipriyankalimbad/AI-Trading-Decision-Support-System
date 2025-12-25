# How to Use Backtesting Feature

## Quick Guide to Backtesting

The app includes a **SMA 20/50 Crossover Strategy** backtesting engine. Here's how to use it:

---

## Step-by-Step: Run Backtesting

### Step 1: Upload Your Data
1. In the **left sidebar**, click **"📁 Data Input"**
2. Upload a CSV file with OHLCV data
3. Wait for: **"✅ X rows loaded"**

### Step 2: Run Complete Analysis
1. Set your **Entry Price** (optional, default is fine)
2. Click **"🚀 Run Complete Analysis"** button
3. Wait for processing (10-30 seconds)
4. You'll see: **"✅ Analysis complete!"**

### Step 3: View Backtesting Results
1. Click on the **"📉 Backtesting"** tab
2. You'll see all backtesting results!

---

## What You'll See in Backtesting Tab

### Performance Metrics:
- **Strategy Return:** Total return of the SMA crossover strategy
- **Buy & Hold Return:** Return if you just bought and held
- **Sharpe Ratio:** Risk-adjusted performance measure
- **Win Rate:** Percentage of winning trades

### Trade Statistics:
- **Total Trades:** Number of trades executed
- **Avg Win:** Average profit per winning trade
- **Avg Loss:** Average loss per losing trade

### Equity Curve:
- Visual chart showing strategy performance over time
- Comparison with buy-and-hold strategy
- Interactive Plotly chart

### Trade History:
- Click **"📋 Trade History"** expander
- See individual trades with:
  - Entry date
  - Exit date
  - Entry price
  - Exit price
  - Profit/Loss (P&L)
  - P&L percentage
  - Holding period

---

## How the Strategy Works

### SMA 20/50 Crossover Strategy:

1. **Buy Signal:**
   - When SMA 20 crosses **above** SMA 50
   - This is a "bullish crossover"
   - Strategy enters a **long position**

2. **Sell Signal:**
   - When SMA 20 crosses **below** SMA 50
   - This is a "bearish crossover"
   - Strategy exits the position

3. **Long-Only:**
   - Only buys (no shorting)
   - Maintains position until sell signal
   - No leverage

### Example:
```
Day 1: SMA 20 = 100, SMA 50 = 105 (SMA 20 < SMA 50) → No position
Day 2: SMA 20 = 102, SMA 50 = 104 (SMA 20 < SMA 50) → No position
Day 3: SMA 20 = 106, SMA 50 = 105 (SMA 20 > SMA 50) → BUY! 🟢
Day 4: SMA 20 = 108, SMA 50 = 106 (SMA 20 > SMA 50) → Hold position
Day 5: SMA 20 = 104, SMA 50 = 107 (SMA 20 < SMA 50) → SELL! 🔴
```

---

## Understanding the Results

### Strategy Return vs Buy & Hold:
- **Positive difference:** Strategy outperformed buy-and-hold
- **Negative difference:** Strategy underperformed buy-and-hold
- **0% difference:** Same performance

### Sharpe Ratio:
- **> 1:** Good risk-adjusted returns
- **> 2:** Excellent risk-adjusted returns
- **< 1:** Poor risk-adjusted returns
- **< 0:** Strategy loses money

### Win Rate:
- **> 50%:** More winning trades than losing
- **< 50%:** More losing trades than winning
- **Note:** Win rate alone doesn't tell the full story (check Avg Win vs Avg Loss)

### Equity Curve:
- **Blue line:** Your strategy's equity over time
- **Gray dashed line:** Buy-and-hold equity
- **Above gray line:** Strategy is outperforming
- **Below gray line:** Strategy is underperforming

---

## Tips for Better Backtesting

### 1. Use Enough Data:
- **Minimum:** 50-100 rows
- **Recommended:** 200+ rows for reliable results
- More data = more trades = more reliable statistics

### 2. Check Data Quality:
- Make sure dates are in order
- No missing prices
- Valid OHLC relationships (High ≥ Low, etc.)

### 3. Understand Limitations:
- **Past performance ≠ Future results**
- Backtesting assumes perfect execution (no slippage, fees)
- Results are theoretical, not guaranteed

### 4. Compare Strategies:
- Try different entry prices
- Compare with buy-and-hold
- Look at risk metrics (Sharpe ratio)

---

## Common Questions

### Q: Why is my strategy return negative?
**A:** The SMA crossover strategy may not work well for all stocks/timeframes. Some stocks are better suited for trend-following strategies than others.

### Q: Why are there no trades?
**A:** Your data might not have enough periods for SMA 50 to calculate, or there were no crossovers in your data range.

### Q: Can I change the strategy?
**A:** The current version uses SMA 20/50 crossover. To change it, you'd need to modify `src/backtest.py` (advanced).

### Q: What's a good Sharpe ratio?
**A:** Generally:
- **< 1:** Poor
- **1-2:** Good
- **> 2:** Excellent
- **> 3:** Outstanding

### Q: How do I interpret the equity curve?
**A:**
- **Steady upward:** Good strategy
- **Volatile:** High risk
- **Below buy-hold:** Strategy underperforming
- **Above buy-hold:** Strategy outperforming

---

## Example Workflow

1. **Upload stock data** (e.g., AAPL, TSLA, etc.)
2. **Run analysis** → Wait for completion
3. **Go to Backtesting tab**
4. **Review metrics:**
   - Is strategy return positive?
   - Is Sharpe ratio > 1?
   - Is win rate > 50%?
5. **Check equity curve:**
   - Is it above buy-hold line?
   - Is it smooth or volatile?
6. **Review trade history:**
   - How many trades?
   - Average holding period?
   - Biggest win/loss?

---

## Summary

**To use backtesting:**
1. Upload CSV → 2. Run Analysis → 3. Click "📉 Backtesting" tab → 4. Review results!

**The backtesting runs automatically** when you click "Run Complete Analysis" - no extra steps needed!

Just make sure you have enough data (50+ rows) and the analysis completes successfully.

Happy backtesting! 📊

