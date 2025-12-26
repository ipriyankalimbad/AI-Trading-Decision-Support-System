# 📉 Backtest Script - Complete Thought Process & Implementation Guide

## 🎯 What is Backtesting?

**Backtesting** = Testing a trading strategy on historical data to see how it would have performed.

**Why?**
- See if a strategy works before risking real money
- Understand strategy performance (returns, risk, win rate)
- Compare different strategies
- Learn from past market behavior

---

## 💭 The Thought Process: How We Built It

### **Step 1: Choose a Strategy**

**Question**: What trading strategy should we test?

**Answer**: **SMA Crossover Strategy** (Simple Moving Average)
- **Why?**: 
  - Simple and well-known
  - Easy to understand
  - Works well for trend-following
  - Good for beginners to understand

**The Strategy Logic:**
- **Buy Signal**: When fast SMA (20-day) crosses ABOVE slow SMA (50-day)
  - This means: Short-term trend is rising faster than long-term trend = BULLISH
- **Sell Signal**: When fast SMA (20-day) crosses BELOW slow SMA (50-day)
  - This means: Short-term trend is falling faster than long-term trend = BEARISH

**Visual Example:**
```
Price Chart:
    ↑
    |     /\
    |    /  \    ← Buy here (fast crosses above slow)
    |   /    \
    |  /      \___
    | /           \___
    |_________________\___  ← Sell here (fast crosses below slow)
    |                    \___
    |                       \___
    +---------------------------→ Time
```

---

### **Step 2: Design the Architecture**

**Question**: How should we structure the code?

**Answer**: **Object-Oriented Design (Class-based)**

**Why?**
- Clean organization
- Reusable components
- Easy to extend (add more strategies later)
- Professional structure

**Structure:**
```
BacktestEngine (Class)
├── __init__()          # Initialize with starting capital
├── sma_crossover_strategy()  # Generate buy/sell signals
├── calculate_returns()       # Calculate strategy performance
├── get_trade_stats()         # Extract individual trades
└── run_backtest()            # Main function (orchestrates everything)
```

---

### **Step 3: Generate Trading Signals**

**Question**: How do we detect when SMAs cross?

**Answer**: Compare previous day vs current day

**The Logic:**
```python
For each day:
    Previous day: fast_SMA <= slow_SMA
    Current day:  fast_SMA > slow_SMA
    → This is a CROSSOVER! → BUY SIGNAL
    
    Previous day: fast_SMA >= slow_SMA
    Current day:  fast_SMA < slow_SMA
    → This is a CROSSOVER! → SELL SIGNAL
```

**Code Implementation:**
```python
for i in range(1, len(df_signal)):
    prev_fast = df_signal[fast_col].iloc[i-1]  # Yesterday's fast SMA
    prev_slow = df_signal[slow_col].iloc[i-1]  # Yesterday's slow SMA
    curr_fast = df_signal[fast_col].iloc[i]    # Today's fast SMA
    curr_slow = df_signal[slow_col].iloc[i]    # Today's slow SMA
    
    # Bullish crossover (buy signal)
    if prev_fast <= prev_slow and curr_fast > curr_slow:
        signal = 1  # BUY
    
    # Bearish crossover (sell signal)
    elif prev_fast >= prev_slow and curr_fast < curr_slow:
        signal = -1  # SELL
```

**Why compare previous vs current?**
- We need to detect the MOMENT of crossover
- Just checking if fast > slow doesn't tell us if it just crossed
- We need to see the CHANGE from previous day

---

### **Step 4: Track Position State**

**Question**: How do we know if we're in a trade or not?

**Answer**: Use a position tracker

**The Logic:**
```python
position = 0  # Start with no position

For each day:
    If signal == BUY:
        position = 1  # Enter long position (we own the stock)
    
    If signal == SELL:
        position = 0  # Exit position (we sold the stock)
    
    Store position for this day
```

**Why track position?**
- We need to know if we're "in the market" or "out of the market"
- Only make returns when we're in a position
- Can't sell if we don't own anything

**Code Implementation:**
```python
position = 0
for i in range(len(df_signal)):
    if df_signal['signal'].iloc[i] == 1:  # Buy signal
        position = 1  # Enter long
    elif df_signal['signal'].iloc[i] == -1:  # Sell signal
        position = 0  # Exit long
    
    df_signal.loc[df_signal.index[i], 'position'] = position
```

---

### **Step 5: Calculate Strategy Returns**

**Question**: How do we calculate how much money we made?

**Answer**: Multiply position by daily returns

**The Formula:**
```
Strategy Return (Day) = Position × Daily Return

If position = 1 (we own stock):
    Strategy Return = 1 × Daily Return = Daily Return
    (We make/lose money based on price movement)

If position = 0 (we don't own stock):
    Strategy Return = 0 × Daily Return = 0
    (We make nothing, we're out of the market)
```

**Why shift position by 1 day?**
- We enter position at the CLOSE of the day we get the signal
- We start earning returns the NEXT day
- This is realistic (can't trade on the same day we see the signal)

**Code Implementation:**
```python
# Shift position by 1 (enter position at close, earn returns next day)
df_returns['strategy_returns'] = df_returns['position'].shift(1) * df_returns['daily_returns']
```

**Cumulative Returns:**
```
Day 1: 1 + return1
Day 2: (1 + return1) × (1 + return2)
Day 3: (1 + return1) × (1 + return2) × (1 + return3)
...
```

**Code:**
```python
df_returns['cumulative_returns'] = (1 + df_returns['strategy_returns']).cumprod()
```

**Equity (Total Money):**
```
Equity = Initial Capital × Cumulative Returns

Example:
- Start with $10,000
- Cumulative return = 1.15 (15% gain)
- Equity = $10,000 × 1.15 = $11,500
```

**Code:**
```python
df_returns['equity'] = self.initial_capital * df_returns['cumulative_returns']
```

---

### **Step 6: Compare with Buy & Hold**

**Question**: Is our strategy better than just buying and holding?

**Answer**: Calculate buy-and-hold returns for comparison

**Buy & Hold Strategy:**
- Buy on day 1
- Hold until the end
- No trading, just hold

**Why Compare?**
- If strategy can't beat buy-and-hold, it's not worth the effort
- Buy-and-hold is the baseline (simplest strategy)
- Good strategies should outperform

**Code:**
```python
# Buy and hold: just compound daily returns
df_returns['buy_hold_returns'] = (1 + df_returns['daily_returns']).cumprod()
df_returns['buy_hold_equity'] = self.initial_capital * df_returns['buy_hold_returns']
```

---

### **Step 7: Extract Individual Trades**

**Question**: What were the individual trades?

**Answer**: Track entry and exit points

**The Logic:**
```python
in_position = False
entry_price = None
entry_date = None

For each day:
    If not in_position and position == 1:
        # Enter trade
        in_position = True
        entry_price = current_price
        entry_date = current_date
    
    If in_position and position == 0:
        # Exit trade
        exit_price = current_price
        exit_date = current_date
        pnl = exit_price - entry_price
        Save this trade
        in_position = False
```

**What We Track:**
- Entry date and price
- Exit date and price
- Profit/Loss (PnL)
- PnL percentage
- Holding period (how many days)

**Code:**
```python
trades = []
in_position = False
entry_idx = None
entry_price = None

for i in range(len(df)):
    position = df['position'].iloc[i]
    price = df['close'].iloc[i]
    date = df['date'].iloc[i]
    
    # Enter position
    if not in_position and position == 1:
        in_position = True
        entry_idx = i
        entry_price = price
    
    # Exit position
    elif in_position and position == 0:
        exit_price = price
        pnl = exit_price - entry_price
        pnl_pct = (pnl / entry_price) * 100
        
        trades.append({
            'entry_date': df['date'].iloc[entry_idx],
            'exit_date': date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_percentage': pnl_pct,
            'holding_period': i - entry_idx
        })
        
        in_position = False
```

---

### **Step 8: Calculate Performance Metrics**

**Question**: How do we measure strategy performance?

**Answer**: Calculate key metrics

**1. Total Return:**
```
Total Return = (Final Equity / Initial Capital - 1) × 100%

Example:
- Start: $10,000
- End: $12,000
- Return: (12000/10000 - 1) × 100% = 20%
```

**2. Sharpe Ratio:**
```
Sharpe Ratio = (Mean Return / Std Dev of Returns) × √252

Why?
- Measures risk-adjusted returns
- Higher = better (more return per unit of risk)
- √252 = annualize (252 trading days per year)
```

**3. Win Rate:**
```
Win Rate = (Winning Trades / Total Trades) × 100%

Example:
- 10 trades total
- 6 winning trades
- Win rate = 60%
```

**4. Average Win/Loss:**
```
Average Win = Mean PnL of all winning trades
Average Loss = Mean PnL of all losing trades

Why?
- Want average win > average loss
- Even with 50% win rate, can be profitable if wins are bigger
```

**Code:**
```python
# Total return
total_return = (df_results['equity'].iloc[-1] / self.initial_capital - 1) * 100

# Sharpe ratio
strategy_returns = df_results['strategy_returns'].dropna()
sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)

# Win rate
winning_trades = [t for t in trades if t['pnl'] > 0]
win_rate = (len(winning_trades) / len(trades)) * 100

# Average win/loss
avg_win = np.mean([t['pnl'] for t in winning_trades])
avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0])
```

---

## 🔄 Complete Flow Diagram

```
1. Input Data (OHLCV + Indicators)
   ↓
2. Generate Signals (SMA Crossover Detection)
   ↓
3. Track Positions (Enter/Exit based on signals)
   ↓
4. Calculate Returns (Position × Daily Returns)
   ↓
5. Calculate Cumulative Returns & Equity
   ↓
6. Extract Individual Trades
   ↓
7. Calculate Metrics (Return, Sharpe, Win Rate)
   ↓
8. Return Results Dictionary
```

---

## 🎓 Key Concepts Explained

### **1. Why Long-Only Strategy?**

**Long-Only** = We only buy (go long), never short

**Why?**
- Simpler to understand
- Less risky for beginners
- Most common strategy type
- Can extend to shorting later

**What it means:**
- When we get a buy signal → Buy stock
- When we get a sell signal → Sell stock (exit position)
- We never bet against the stock (shorting)

---

### **2. Why Time-Series Split Matters**

**Important**: We test on FUTURE data, not past data

**Why?**
- Realistic: We can't use future information
- Prevents "look-ahead bias"
- Tests if strategy works on unseen data

**How:**
- Train on first 80% of data
- Test on last 20% of data
- This simulates real trading

---

### **3. Why Shift Position by 1 Day?**

**Problem**: We see signal at end of day, but can't trade until next day

**Solution**: Shift position by 1 day

**Example:**
```
Day 1 (Close): Signal = BUY
Day 2 (Open):  We actually buy (enter position)
Day 2 (Close): We start earning returns
```

**Code:**
```python
strategy_returns = position.shift(1) * daily_returns
```

---

### **4. Why Cumulative Product?**

**Cumulative Product** = Multiply all returns together

**Why?**
- Returns compound over time
- Day 1: $100 → $110 (10% gain)
- Day 2: $110 → $115.50 (5% gain)
- Total: $100 → $115.50 (15.5% gain, not 15%)

**Formula:**
```
Final = Initial × (1+r1) × (1+r2) × (1+r3) × ...
```

**Code:**
```python
cumulative = (1 + returns).cumprod()
```

---

## 🐛 Common Issues & Solutions

### **Issue 1: No Trades Generated**

**Problem**: Strategy never generates buy/sell signals

**Causes:**
- SMAs never cross (not enough data)
- Data doesn't have enough variation
- SMAs are too close together

**Solution:**
- Need at least 50+ days of data
- Check if SMAs are calculated correctly
- Try different SMA periods

---

### **Issue 2: Negative Returns**

**Problem**: Strategy loses money

**Why?**
- Strategy might not work for this stock
- Market conditions (sideways, choppy)
- Strategy needs optimization

**Solution:**
- This is normal! Not all strategies work for all stocks
- Try different parameters
- Compare with buy-and-hold

---

### **Issue 3: Too Many Trades**

**Problem**: Strategy trades too frequently (whipsaws)

**Why?**
- SMAs cross back and forth frequently
- Market is choppy/sideways
- Need filters (e.g., minimum holding period)

**Solution:**
- Add minimum holding period
- Use longer SMA periods
- Add confirmation signals

---

## 📊 Example Walkthrough

**Let's trace through an example:**

**Data:**
```
Day 1: Close = $100, SMA20 = $98, SMA50 = $95
Day 2: Close = $102, SMA20 = $99, SMA50 = $96
Day 3: Close = $105, SMA20 = $101, SMA50 = $97  ← CROSSOVER! (SMA20 > SMA50)
Day 4: Close = $103, SMA20 = $102, SMA50 = $98
Day 5: Close = $101, SMA20 = $101, SMA50 = $99
Day 6: Close = $99,  SMA20 = $100, SMA50 = $100  ← CROSSOVER! (SMA20 < SMA50)
```

**Signals:**
```
Day 1: No signal
Day 2: No signal
Day 3: BUY signal (SMA20 crossed above SMA50)
Day 4: Hold (in position)
Day 5: Hold (in position)
Day 6: SELL signal (SMA20 crossed below SMA50)
```

**Position:**
```
Day 1: position = 0
Day 2: position = 0
Day 3: position = 1 (entered at close of Day 3 = $105)
Day 4: position = 1
Day 5: position = 1
Day 6: position = 0 (exited at close of Day 6 = $99)
```

**Returns:**
```
Day 1: return = 0 (not in position)
Day 2: return = 0 (not in position)
Day 3: return = 0 (entered at close, no return yet)
Day 4: return = (103-105)/105 = -1.9% (in position)
Day 5: return = (101-103)/103 = -1.9% (in position)
Day 6: return = (99-101)/101 = -2.0% (in position, then exited)
```

**Trade Summary:**
```
Entry: Day 3 at $105
Exit:  Day 6 at $99
PnL:   $99 - $105 = -$6 (loss)
PnL%:  -5.7%
Holding Period: 3 days
```

---

## 🎯 Summary

**The Backtest Script Was Built With This Thought Process:**

1. ✅ **Choose Strategy**: SMA Crossover (simple, effective)
2. ✅ **Design Architecture**: Class-based, modular
3. ✅ **Generate Signals**: Detect SMA crossovers
4. ✅ **Track Positions**: Know when we're in/out of market
5. ✅ **Calculate Returns**: Position × Daily Returns
6. ✅ **Compare Performance**: vs Buy & Hold
7. ✅ **Extract Trades**: Individual trade analysis
8. ✅ **Calculate Metrics**: Return, Sharpe, Win Rate

**Key Principles:**
- Realistic (no look-ahead bias)
- Modular (easy to extend)
- Comprehensive (tracks everything)
- Professional (industry-standard metrics)

**This is how professional trading systems are built!** 🚀

