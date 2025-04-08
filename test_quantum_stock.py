#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified Test Script for Quantum Stock Models

This script tests the quantum stock models on a single token and timeframe
to verify that the improved models work correctly.
"""

import os
import pandas as pd
from datetime import datetime
import json

# Import app modules for strategy running
from app import (
    run_quantum_stock_strategy_backtest,
    run_quantum_stock_improved_conservative_strategy_backtest,
    run_quantum_stock_improved_balanced_strategy_backtest, 
    run_quantum_stock_improved_aggressive_strategy_backtest,
    fetch_crypto_data
)

# Import parameter classes
from quantum_stock_model import QuantumStockParameters

# Ensure results directory exists
os.makedirs("improved_Quantum_stock_strategy", exist_ok=True)

def safe_run_strategy(strategy_name, strategy_func, *args, **kwargs):
    """Run a strategy with enhanced error handling to prevent crashes."""
    print(f"Running {strategy_name}...")
    try:
        stats, _ = strategy_func(*args, **kwargs)
        
        # Add check for None stats
        if stats is None:
            print(f"Warning: {strategy_name} returned None stats")
            return {
                'error': 'Strategy returned None stats',
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
        else:
            print(f"Completed {strategy_name} successfully.")
            return {
                'total_trades': stats.get('Total Trades', 0),
                'win_rate': stats.get('Win Rate [%]', 0),
                'total_return': stats.get('Return [%]', 0),
                'max_drawdown': stats.get('Max Drawdown [%]', 0),
                'profit_factor': stats.get('Profit Factor', 0)
            }
            
    except Exception as e:
        print(f"Error in {strategy_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'profit_factor': 0
        }

def test_quantum_stock_models():
    """Test the quantum stock models with the Paper model."""
    # Parameters
    token = "ETH/USDT"
    timeframe = "1h"
    start_date = "2023-01-01"
    end_date = "2023-01-07"  # Shorter period (1 week)
    initial_capital = 10000
    
    results = {}
    
    # Base parameters for Paper model
    paper_params = QuantumStockParameters()
    # Set parameters based on Zhang & Huang paper
    paper_params.m = 1e-30
    paper_params.hbar = 1.0
    paper_params.d = 0.2
    paper_params.e = 1e-19
    paper_params.F = 1e-19
    paper_params.omega = 1e-4
    paper_params.use_cosine_distribution = True
    paper_params.signal_threshold = 0.5
    paper_params.stop_loss_atr = 1.5
    paper_params.take_profit_atr = 1.5
    paper_params.cooldown_period = 8
    
    # Test Paper model first (physics-based model)
    results["Quantum Stock Paper Model"] = safe_run_strategy(
        "Quantum Stock Paper Model",
        run_quantum_stock_improved_balanced_strategy_backtest,
        token, timeframe, initial_capital, start_date, end_date, paper_params
    )
    
    # Test Conservative model
    conservative_params = QuantumStockParameters()
    conservative_params.m = 2.2e-30
    conservative_params.hbar = 0.85
    conservative_params.F = 1.1e-19
    conservative_params.omega = 0.8e-4
    conservative_params.signal_threshold = 0.7
    conservative_params.min_amplitude = 0.025
    conservative_params.stop_loss_atr = 1.4
    conservative_params.take_profit_atr = 1.8
    conservative_params.cooldown_period = 8
    
    results["Quantum Stock Conservative"] = safe_run_strategy(
        "Quantum Stock Conservative",
        run_quantum_stock_improved_conservative_strategy_backtest,
        token, timeframe, initial_capital, start_date, end_date, conservative_params
    )
    
    # Test Aggressive model
    aggressive_params = QuantumStockParameters()
    aggressive_params.m = 0.4e-30
    aggressive_params.hbar = 1.5
    aggressive_params.F = 1.5e-19
    aggressive_params.omega = 2.0e-4
    aggressive_params.signal_threshold = 0.4
    aggressive_params.min_amplitude = 0.015
    aggressive_params.stop_loss_atr = 1.0
    aggressive_params.take_profit_atr = 2.5
    aggressive_params.cooldown_period = 3
    
    results["Quantum Stock Aggressive"] = safe_run_strategy(
        "Quantum Stock Aggressive",
        run_quantum_stock_improved_aggressive_strategy_backtest,
        token, timeframe, initial_capital, start_date, end_date, aggressive_params
    )
    
    # Print summary
    print("\n===== Testing Results =====")
    for strategy, stats in results.items():
        print(f"\n{strategy}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    with open(f'improved_Quantum_stock_strategy/test_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, default=str)
    
    print(f"\nResults saved to improved_Quantum_stock_strategy/test_results_{timestamp}.json")

if __name__ == "__main__":
    test_quantum_stock_models() 