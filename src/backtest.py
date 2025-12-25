"""
Backtesting engine for SMA crossover strategy.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class BacktestEngine:
    """
    Backtesting engine for trading strategies.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
    
    def sma_crossover_strategy(self, df: pd.DataFrame, 
                               fast_period: int = 20, 
                               slow_period: int = 50) -> pd.DataFrame:
        """
        Generate signals for SMA crossover strategy (long-only).
        
        Args:
            df: DataFrame with price data and indicators
            fast_period: Fast SMA period
            slow_period: Slow SMA period
            
        Returns:
            DataFrame with signals added
        """
        df_signal = df.copy()
        
        # Ensure we have the required SMAs
        fast_col = f'sma_{fast_period}'
        slow_col = f'sma_{slow_period}'
        
        if fast_col not in df_signal.columns or slow_col not in df_signal.columns:
            raise ValueError(f"Missing required SMA columns: {fast_col}, {slow_col}")
        
        # Initialize signals
        df_signal['signal'] = 0
        df_signal['position'] = 0
        
        # Generate signals: 1 for buy, 0 for hold, -1 for sell
        # Buy when fast SMA crosses above slow SMA
        # Sell when fast SMA crosses below slow SMA
        
        for i in range(1, len(df_signal)):
            prev_fast = df_signal[fast_col].iloc[i-1]
            prev_slow = df_signal[slow_col].iloc[i-1]
            curr_fast = df_signal[fast_col].iloc[i]
            curr_slow = df_signal[slow_col].iloc[i]
            
            # Bullish crossover
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                df_signal.loc[df_signal.index[i], 'signal'] = 1  # Buy
            
            # Bearish crossover
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                df_signal.loc[df_signal.index[i], 'signal'] = -1  # Sell
        
        # Long-only: maintain position until sell signal
        position = 0
        for i in range(len(df_signal)):
            if df_signal['signal'].iloc[i] == 1:
                position = 1  # Enter long
            elif df_signal['signal'].iloc[i] == -1:
                position = 0  # Exit long
            
            df_signal.loc[df_signal.index[i], 'position'] = position
        
        return df_signal
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy returns and equity curve.
        
        Args:
            df: DataFrame with signals and positions
            
        Returns:
            DataFrame with returns and equity curve
        """
        df_returns = df.copy()
        
        # Calculate strategy returns
        df_returns['strategy_returns'] = df_returns['position'].shift(1) * df_returns['daily_returns']
        df_returns['strategy_returns'] = df_returns['strategy_returns'].fillna(0)
        
        # Calculate cumulative returns
        df_returns['cumulative_returns'] = (1 + df_returns['strategy_returns']).cumprod()
        df_returns['equity'] = self.initial_capital * df_returns['cumulative_returns']
        
        # Buy and hold returns for comparison
        df_returns['buy_hold_returns'] = (1 + df_returns['daily_returns']).cumprod()
        df_returns['buy_hold_equity'] = self.initial_capital * df_returns['buy_hold_returns']
        
        return df_returns
    
    def get_trade_stats(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract individual trade statistics.
        
        Args:
            df: DataFrame with positions and returns
            
        Returns:
            List of trade dictionaries
        """
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
                exit_date = date
                pnl = exit_price - entry_price
                pnl_pct = (pnl / entry_price) * 100
                holding_period = i - entry_idx
                
                trades.append({
                    'entry_date': df['date'].iloc[entry_idx],
                    'exit_date': exit_date,
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'pnl': float(pnl),
                    'pnl_percentage': float(pnl_pct),
                    'holding_period': int(holding_period)
                })
                
                in_position = False
        
        # Handle open position at the end
        if in_position:
            exit_price = df['close'].iloc[-1]
            pnl = exit_price - entry_price
            pnl_pct = (pnl / entry_price) * 100
            holding_period = len(df) - entry_idx - 1
            
            trades.append({
                'entry_date': df['date'].iloc[entry_idx],
                'exit_date': df['date'].iloc[-1],
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'pnl': float(pnl),
                'pnl_percentage': float(pnl_pct),
                'holding_period': int(holding_period),
                'open': True
            })
        
        return trades
    
    def run_backtest(self, df: pd.DataFrame, 
                    fast_period: int = 20, 
                    slow_period: int = 50) -> Dict:
        """
        Run complete backtest and return results.
        
        Args:
            df: DataFrame with price data and indicators
            fast_period: Fast SMA period
            slow_period: Slow SMA period
            
        Returns:
            Dictionary with backtest results
        """
        # Generate signals
        df_signals = self.sma_crossover_strategy(df, fast_period, slow_period)
        
        # Calculate returns
        df_results = self.calculate_returns(df_signals)
        
        # Get trade statistics
        trades = self.get_trade_stats(df_results)
        
        # Calculate performance metrics
        total_return = float(df_results['cumulative_returns'].iloc[-1] - 1) * 100
        buy_hold_return = float(df_results['buy_hold_returns'].iloc[-1] - 1) * 100
        
        final_equity = float(df_results['equity'].iloc[-1])
        final_buy_hold_equity = float(df_results['buy_hold_equity'].iloc[-1])
        
        # Calculate Sharpe ratio (simplified, assuming 252 trading days)
        strategy_returns = df_results['strategy_returns'].dropna()
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Win rate
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = (len(winning_trades) / len(trades)) * 100
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if any(t['pnl'] <= 0 for t in trades) else 0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
        
        results = {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'final_equity': final_equity,
            'final_buy_hold_equity': final_buy_hold_equity,
            'sharpe_ratio': float(sharpe_ratio),
            'total_trades': len(trades),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'trades': trades,
            'equity_curve': df_results[['date', 'equity', 'buy_hold_equity']].copy(),
            'strategy_data': df_results.copy()
        }
        
        return results

