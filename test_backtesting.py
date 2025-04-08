"""
Test script to run the GitHub reference implementation directly
"""
import sys
import os
import pandas as pd
import numpy as np

# Add NumPy bool8 compatibility for Bokeh
import numpy_patch

# Use the reference implementation instead of ours
sys.path.insert(0, os.path.join(os.getcwd(), 'reference_code'))

# Import from reference code
from reference_code.quantum_crypto_backtest import fetch_crypto_data, QuantumRSIStrategy
from backtesting import Backtest

def run_reference_backtest():
    # Use ETH/USDT with default parameters
    symbol = 'ETH/USDT'
    timeframe = '1h'
    start_date = '2024-01-01'
    end_date = '2024-07-01'
    capital = 10000
    
    # Fetch data
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")
    data = fetch_crypto_data(symbol, timeframe, start_date=start_date, end_date=end_date)
    print(f"Data fetched: {len(data)} candles from {data.index[0]} to {data.index[-1]}")
    
    # Create relaxed parameters for more trades
    params = {
        'rsi_period': 14,
        'rsi_high': 65,         # Lower threshold
        'rsi_low': 35,          # Higher threshold
        'v0': 1.5,              # Lower barrier
        'mass': 1.0,            # Lower mass
        'hbar': 1.5,            # Higher hbar
        'volume_threshold': 1.2, # Lower volume requirement
        'vwap_deviation': 0.008, # Lower deviation
        'stop_loss_atr': 1.0,
        'take_profit_atr': 2.0,
        'cooldown_period': 6,    # Reduced cooldown
        'use_fixed_position_size': True,
        'fixed_position_size': 0.2,
        'debug_mode': True,      # Enable debug
    }
    
    # Create strategy class with modified parameters
    class OptimizedStrategy(QuantumRSIStrategy):
        pass
        
    # Apply parameter overrides
    for param, value in params.items():
        setattr(OptimizedStrategy, param, value)
    
    # Run backtest
    bt = Backtest(data, OptimizedStrategy, cash=capital, commission=0.001, trade_on_close=True)
    stats = bt.run(trace=True)
    
    print(f"\n----- Backtest Results for {symbol} -----")
    print(f"Net Profit: {stats['Return [%]']:.2f}%")
    print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"# Trades: {stats['# Trades']}")
    print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
    print(f"Profit Factor: {stats['Profit Factor']:.2f}")
    print(f"Avg. Trade [%]: {stats['Avg. Trade [%]']:.2f}%")
    
    # Plot with custom info
    bt.plot(resample=False, show_legend=True, open_browser=True)
    
    return stats, bt

if __name__ == "__main__":
    run_reference_backtest() 