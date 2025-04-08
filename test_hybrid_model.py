#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Quantum Harmonic-Wave Hybrid model.
This script runs a backtest of the hybrid model on ETH/USDT with a 4h timeframe.
"""

import sys
from datetime import datetime, timedelta
from quantum_harmonic_wave_hybrid import run_quantum_hybrid_strategy_backtest, QuantumHybridParameters

def main():
    # Test parameters
    symbol = "ETH/USDT"
    timeframe = "4h"
    initial_capital = 10000.0
    
    # Use a longer period (3 months) to give more trading opportunities
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    print(f"Running hybrid model backtest for {symbol} on {timeframe} timeframe...")
    
    # Create custom parameters with more sensitive settings
    params = QuantumHybridParameters()
    params.entry_threshold = 0.5  # Even lower threshold for entry
    params.exit_threshold = 0.3   # Lower threshold for exit
    params.trend_threshold = 0.1  # Make trend detection more sensitive
    params.cooldown_period = 2    # Allow more frequent trading
    
    # Run the backtest with these parameters
    performance, trades = run_quantum_hybrid_strategy_backtest(
        symbol, timeframe, initial_capital, start_date, end_date, 
        params=params, plot=True
    )
    
    # Print additional trade information if available
    if trades is not None and not trades.empty:
        print("\nTrade Summary:")
        print(f"Total Trades: {len(trades)}")
        winning_trades = trades[trades['profit'] > 0]
        losing_trades = trades[trades['profit'] <= 0]
        print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
        print(f"Average Profit: {trades['profit'].mean():.2f}")
        print(f"Largest Win: {trades['profit'].max():.2f}")
        print(f"Largest Loss: {trades['profit'].min():.2f}")
    else:
        print("\nNo trades were generated during the backtest period.")
        print("Consider adjusting parameters or using a different time period.")

if __name__ == "__main__":
    main() 