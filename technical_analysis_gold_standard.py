#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Refined Technical Analysis Strategy (Trend Following with Trailing Stop)

This strategy focuses on trend following using EMA crossovers, confirmed by 
a longer-term EMA slope for regime filtering, and RSI for pullback entries. 
It utilizes an ATR-based initial stop loss and an ATR trailing stop for exits.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Union
import traceback

# -----------------------------------------------------------------------------
# Parameter Class (Refined)
# -----------------------------------------------------------------------------

class TechnicalGoldStandardParameters:
    """ Parameters for the Refined Technical Analysis Strategy """
    def __init__(self):
        # Moving Averages (Trend & Entry)
        self.ema_fast_period = 9
        self.ema_slow_period = 21
        self.ema_trend_period = 50
        self.ema_slope_threshold = 0.0001 # Minimal slope to consider trend established (adjust based on timeframe/asset)
        
        # RSI (Pullback Entry)
        self.rsi_period = 14
        self.rsi_pullback_long = 60 # Enter long on pullback if RSI < this level
        self.rsi_pullback_short = 40 # Enter short on pullback if RSI > this level
        
        # ATR (Risk Management)
        self.atr_period = 14
        self.atr_stop_loss_multiplier = 2.0
        self.atr_trailing_stop_multiplier = 1.5 # Multiplier for the trailing stop distance
        
        # Risk Management
        self.initial_capital = 10000.0
        self.risk_per_trade = 0.02 # Risk 2% of capital per trade
        self.min_atr_value = 1e-6 # Minimum ATR value to prevent division by zero

# -----------------------------------------------------------------------------
# Indicator Calculation Functions (Simplified)
# -----------------------------------------------------------------------------

def calculate_rsi(prices: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
    """Calculate RSI using Wilder's smoothing method."""
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices, name='close')
    elif not isinstance(prices, pd.Series):
        raise TypeError("Input 'prices' must be a pandas Series or numpy ndarray")

    delta = prices.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps) 
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50)

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate ATR using Wilder's smoothing method."""
    if not all(isinstance(s, pd.Series) for s in [high, low, close]):
        raise TypeError("Inputs 'high', 'low', 'close' must be pandas Series")

    high_low = high - low
    high_close_prev = np.abs(high - close.shift())
    low_close_prev = np.abs(low - close.shift())
    
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    return atr

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average (EMA)."""
    if not isinstance(prices, pd.Series):
        raise TypeError("Input 'prices' must be a pandas Series")
    return prices.ewm(span=period, adjust=False).mean()

def calculate_ema_slope(ema_series: pd.Series, period: int = 5) -> pd.Series:
    """Calculate the slope of an EMA series using linear regression over a short period."""
    # Use simple difference for approximation (more robust than linear regression on noisy data)
    # Normalize by EMA value to make it comparable across different price levels
    slope = ema_series.diff(period) / (period * ema_series.replace(0, np.finfo(float).eps))
    return slope.fillna(0)

# -----------------------------------------------------------------------------
# Backtesting Function (Refined Logic)
# -----------------------------------------------------------------------------

def run_technical_gold_standard_strategy_backtest(
    symbol: str, 
    timeframe: str, 
    initial_capital: float, 
    start_date: str, 
    end_date: str, 
    params: Optional[TechnicalGoldStandardParameters] = None, 
    plot: bool = False
) -> Tuple[Dict, pd.DataFrame]:
    """
    Runs a backtest for the Refined Technical Analysis strategy.
    Focuses on EMA trend following with RSI pullbacks and ATR trailing stops.
    """
    
    if params is None:
        params = TechnicalGoldStandardParameters()
        
    print(f"\n--- Starting Refined TA backtest for {symbol} on {timeframe} ---")
    print(f"Params: EMAs({params.ema_fast_period},{params.ema_slow_period},{params.ema_trend_period}), Slope Thresh({params.ema_slope_threshold}), "
          f"RSI({params.rsi_period},<{params.rsi_pullback_long},>{params.rsi_pullback_short}), ATR({params.atr_period},SL={params.atr_stop_loss_multiplier},Trail={params.atr_trailing_stop_multiplier})")
    
    # --- 1. Fetch Data ---
    try:
        # Assume fetch_crypto_data is available via the testing script
        from app import fetch_crypto_data 
        df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
        if df is None or df.empty or len(df) < params.ema_trend_period:
             raise ValueError(f"Insufficient data fetched for {symbol} {timeframe}")
    except ImportError:
        print("Warning: Could not import 'fetch_crypto_data'. Backtest cannot run.")
        return {"error": "Data fetching failed"}, pd.DataFrame()
    except ValueError as ve:
        print(f"Data Error: {ve}")
        return {"error": str(ve)}, pd.DataFrame()
    except Exception as e:
        print(f"Error fetching data: {e}")
        traceback.print_exc()
        return {"error": f"Data fetching error: {e}"}, pd.DataFrame()

    # --- 2. Calculate Indicators ---
    try:
        df['ema_fast'] = calculate_ema(df['close'], params.ema_fast_period)
        df['ema_slow'] = calculate_ema(df['close'], params.ema_slow_period)
        df['ema_trend'] = calculate_ema(df['close'], params.ema_trend_period)
        df['ema_trend_slope'] = calculate_ema_slope(df['ema_trend'])
        df['rsi'] = calculate_rsi(df['close'], params.rsi_period)
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], params.atr_period)
        
        # Shift indicators for signal generation (use previous bar's data)
        df['ema_fast_prev'] = df['ema_fast'].shift(1)
        df['ema_slow_prev'] = df['ema_slow'].shift(1)
        df['rsi_prev'] = df['rsi'].shift(1)
        
        # Drop initial rows with NaNs from indicator calculations
        df.dropna(inplace=True)
        if df.empty:
            raise ValueError("DataFrame became empty after indicator calculation and NaN drop.")
            
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        traceback.print_exc()
        return {"error": f"Indicator calculation error: {e}"}, pd.DataFrame()

    # --- 3. Backtest Loop ---
    capital = params.initial_capital
    equity_curve = [capital]
    trades = []
    position = None  # Can be 'long', 'short', or None
    entry_price = 0.0
    stop_loss_price = 0.0
    trailing_stop_price = 0.0
    position_size = 0.0

    for i in range(1, len(df)): # Start from 1 to use shifted values
        current_index = df.index[i]
        current_price = df['close'].iloc[i]
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_atr = df['atr'].iloc[i]
        current_rsi = df['rsi'].iloc[i]
        current_rsi_prev = df['rsi_prev'].iloc[i]
        current_ema_fast = df['ema_fast'].iloc[i]
        current_ema_slow = df['ema_slow'].iloc[i]
        prev_ema_fast = df['ema_fast_prev'].iloc[i]
        prev_ema_slow = df['ema_slow_prev'].iloc[i]
        current_slope = df['ema_trend_slope'].iloc[i]

        # --- Exit Logic ---
        if position == 'long':
            # Update trailing stop loss
            trailing_stop_price = max(trailing_stop_price, current_high - params.atr_trailing_stop_multiplier * current_atr)
            # Check exits: initial SL or trailing SL
            if current_low <= stop_loss_price or current_low <= trailing_stop_price:
                exit_price = stop_loss_price if current_low <= stop_loss_price else trailing_stop_price
                exit_price = min(exit_price, current_high) # Ensure exit isn't above high
                pnl = (exit_price - entry_price) * position_size
                capital += pnl
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_index,
                    'direction': 'long',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position_size,
                    'pnl': pnl,
                    'capital': capital,
                    'exit_reason': 'Stop Loss' if current_low <= stop_loss_price else 'Trailing Stop'
                })
                position = None
                position_size = 0

        elif position == 'short':
            # Update trailing stop loss
            trailing_stop_price = min(trailing_stop_price, current_low + params.atr_trailing_stop_multiplier * current_atr)
            # Check exits: initial SL or trailing SL
            if current_high >= stop_loss_price or current_high >= trailing_stop_price:
                exit_price = stop_loss_price if current_high >= stop_loss_price else trailing_stop_price
                exit_price = max(exit_price, current_low) # Ensure exit isn't below low
                pnl = (entry_price - exit_price) * position_size
                capital += pnl
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_index,
                    'direction': 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': position_size,
                    'pnl': pnl,
                    'capital': capital,
                    'exit_reason': 'Stop Loss' if current_high >= stop_loss_price else 'Trailing Stop'
                })
                position = None
                position_size = 0

        # --- Entry Logic ---
        if position is None:
            # Long Entry Conditions
            is_bullish_trend = current_slope > params.ema_slope_threshold
            is_fast_cross_up = current_ema_fast > current_ema_slow and prev_ema_fast <= prev_ema_slow
            is_rsi_pullback_long = current_rsi < params.rsi_pullback_long and current_rsi > current_rsi_prev # RSI rising from below threshold

            if is_bullish_trend and is_fast_cross_up and is_rsi_pullback_long:
                if current_atr > params.min_atr_value:
                    entry_price = current_price
                    stop_loss_price = entry_price - params.atr_stop_loss_multiplier * current_atr
                    risk_amount = capital * params.risk_per_trade
                    dollar_risk_per_unit = entry_price - stop_loss_price
                    if dollar_risk_per_unit > 0:
                        position_size = risk_amount / dollar_risk_per_unit
                        capital -= position_size * entry_price * 0.001 # Approx fee
                        position = 'long'
                        entry_time = current_index
                        trailing_stop_price = stop_loss_price # Initialize trailing stop near initial stop

            # Short Entry Conditions
            is_bearish_trend = current_slope < -params.ema_slope_threshold
            is_fast_cross_down = current_ema_fast < current_ema_slow and prev_ema_fast >= prev_ema_slow
            is_rsi_pullback_short = current_rsi > params.rsi_pullback_short and current_rsi < current_rsi_prev # RSI falling from above threshold

            if is_bearish_trend and is_fast_cross_down and is_rsi_pullback_short:
                 if current_atr > params.min_atr_value:
                    entry_price = current_price
                    stop_loss_price = entry_price + params.atr_stop_loss_multiplier * current_atr
                    risk_amount = capital * params.risk_per_trade
                    dollar_risk_per_unit = stop_loss_price - entry_price
                    if dollar_risk_per_unit > 0:
                        position_size = risk_amount / dollar_risk_per_unit
                        capital -= position_size * entry_price * 0.001 # Approx fee
                        position = 'short'
                        entry_time = current_index
                        trailing_stop_price = stop_loss_price # Initialize trailing stop near initial stop

        equity_curve.append(capital)

    # --- 4. Performance Calculation ---
    if not trades:
        print("No trades executed.")
        stats = {
            'Initial Capital': params.initial_capital,
            'Final Capital': capital,
            'Total Return [%]': 0.0,
            'Total Trades': 0,
            'Win Rate [%]': 0.0,
            'Profit Factor': 0.0,
            'Max Drawdown [%]': 0.0,
            'Avg Win': 0.0,
            'Avg Loss': 0.0
        }
        return stats, pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    total_return_pct = ((capital / params.initial_capital) - 1) * 100
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    win_rate = (len(wins) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    
    total_profit = wins['pnl'].sum()
    total_loss = abs(losses['pnl'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0

    # Calculate Max Drawdown
    equity_s = pd.Series(equity_curve, index=df.index)
    rolling_max = equity_s.cummax()
    drawdown = (equity_s - rolling_max) / rolling_max
    max_drawdown_pct = abs(drawdown.min()) * 100

    stats = {
        'Initial Capital': params.initial_capital,
        'Final Capital': capital,
        'Total Return [%]': round(total_return_pct, 2),
        'Total Trades': len(trades_df),
        'Win Rate [%]': round(win_rate, 2),
        'Profit Factor': round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
        'Max Drawdown [%]': round(max_drawdown_pct, 2),
        'Avg Win': round(avg_win, 2),
        'Avg Loss': round(avg_loss, 2)
    }

    print("--- Backtest Results ---")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("------------------------")

    # --- 5. Plotting (Optional & Simplified) ---
    if plot:
        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1, 2]})
            fig.suptitle(f'{symbol} {timeframe} - Refined TA Strategy Backtest', fontsize=16)

            # Plot Price and EMAs
            axes[0].plot(df.index, df['close'], label='Close Price', color='black', alpha=0.7)
            axes[0].plot(df.index, df['ema_fast'], label=f'EMA {params.ema_fast_period}', color='blue', alpha=0.8)
            axes[0].plot(df.index, df['ema_slow'], label=f'EMA {params.ema_slow_period}', color='orange', alpha=0.8)
            axes[0].plot(df.index, df['ema_trend'], label=f'EMA {params.ema_trend_period}', color='purple', linestyle='--', alpha=0.6)
            axes[0].set_ylabel('Price')
            axes[0].set_title('Price and EMAs')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            # Plot Trades
            long_entries = trades_df[trades_df['direction'] == 'long']
            short_entries = trades_df[trades_df['direction'] == 'short']
            axes[0].scatter(long_entries['entry_time'], long_entries['entry_price'], marker='^', color='lime', s=100, label='Long Entry', zorder=5)
            axes[0].scatter(short_entries['entry_time'], short_entries['entry_price'], marker='v', color='red', s=100, label='Short Entry', zorder=5)
            
            long_exits = trades_df[trades_df['direction'] == 'long']
            short_exits = trades_df[trades_df['direction'] == 'short']
            axes[0].scatter(long_exits['exit_time'], long_exits['exit_price'], marker='x', color='green', s=80, label='Long Exit', zorder=5)
            axes[0].scatter(short_exits['exit_time'], short_exits['exit_price'], marker='x', color='maroon', s=80, label='Short Exit', zorder=5)
            
            # Plot RSI
            axes[1].plot(df.index, df['rsi'], label=f'RSI ({params.rsi_period})', color='cyan')
            axes[1].axhline(params.rsi_pullback_long, color='red', linestyle='--', alpha=0.5, label=f'RSI Pullback Long ({params.rsi_pullback_long})')
            axes[1].axhline(params.rsi_pullback_short, color='green', linestyle='--', alpha=0.5, label=f'RSI Pullback Short ({params.rsi_pullback_short})')
            axes[1].set_ylabel('RSI')
            axes[1].set_ylim(0, 100)
            axes[1].set_title('RSI')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            # Plot Equity Curve and Drawdown
            axes[2].plot(equity_s.index, equity_s, label='Equity Curve', color='blue')
            axes[2].set_ylabel('Equity', color='blue')
            axes[2].tick_params(axis='y', labelcolor='blue')
            axes[2].grid(True, alpha=0.3)

            ax2b = axes[2].twinx()
            ax2b.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, color='red', label='Drawdown')
            ax2b.set_ylabel('Drawdown (%)', color='red')
            ax2b.tick_params(axis='y', labelcolor='red')
            ax2b.set_ylim(-100, 0)

            axes[2].set_title('Equity Curve and Drawdown')
            lines, labels = axes[2].get_legend_handles_labels()
            lines2, labels2 = ax2b.get_legend_handles_labels()
            axes[2].legend(lines + lines2, labels + labels2, loc='upper left')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
            # Save or show plot
            plot_filename = f"./{symbol.replace('/', '_')}_{timeframe}_backtest_plot.png"
            try:
                plt.savefig(plot_filename)
                print(f"Plot saved to {plot_filename}")
            except Exception as plot_error:
                print(f"Warning: Could not save plot: {plot_error}")
            # plt.show() # Uncomment to display plot interactively
            plt.close(fig) # Close plot figure to free memory
        except Exception as e:
            print(f"Error during plotting: {e}")
            traceback.print_exc()

    return stats, trades_df

# --- Example Usage (Optional) ---
if __name__ == "__main__":
    # Example: Run backtest for BTC/USDT 1h
    try:
        # Note: This requires fetch_crypto_data to be accessible
        # Define parameters or use defaults
        params = TechnicalGoldStandardParameters()
        params.ema_trend_period = 100 # Example override
        params.atr_trailing_stop_multiplier = 2.0 # Example override
        
        # Define date range
        end_dt = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        start_dt = end_dt - pd.Timedelta(days=90)
        start_date_str = start_dt.strftime("%Y-%m-%d")
        end_date_str = end_dt.strftime("%Y-%m-%d")

        stats, trades = run_technical_gold_standard_strategy_backtest(
            symbol='BTC/USDT',
            timeframe='1h',
            initial_capital=10000.0,
            start_date=start_date_str,
            end_date=end_date_str,
            params=params,
            plot=True
        )
        
        print("\n--- Final Stats ---")
        print(stats)
        # print("\n--- Trades --- ")
        # print(trades.head())
        
    except NameError as ne:
         print(f"Example Usage Error: {ne}. Ensure 'fetch_crypto_data' is importable or mock data is used.")
    except Exception as e:
        print(f"An error occurred during example execution: {e}")
        traceback.print_exc() 