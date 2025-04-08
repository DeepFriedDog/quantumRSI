"""
Run Quantum Probability Model Backtest

This script runs a backtest of the Quantum Probability model on cryptocurrency data
with different parameter settings to evaluate performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Add numpy bool8 patch to avoid issues
np.bool8 = np.bool_ if not hasattr(np, 'bool8') else np.bool8
print("NumPy bool8 patch applied")

from quantum_probability_model import (
    QuantumProbabilityParameters,
    fetch_crypto_data,
    run_backtest,
    generate_probability_distribution
)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

def run_parameter_test(symbol="ETH/USDT", timeframe="1h", start_date="2023-01-01", end_date="2023-07-01"):
    """
    Run a series of tests with different parameter sets
    to determine the optimal configuration
    """
    print(f"Quantum Probability Theoretic Model Backtest")
    print(f"Symbol: {symbol}, Timeframe: {timeframe}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 60)
    
    # Fetch data
    print(f"Fetching data...")
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    print(f"Data fetched: {len(df)} candles")
    
    # Define parameter sets
    parameter_sets = {
        "Conservative": {
            "alpha": 2.5,         # Higher α parameters = stronger mean reversion
            "delta": 0.4,         # Lower δ = less influence of extreme returns
            "h": 0.8,             # Lower h = lower stochastic influence
            "omega_level": 1,     # Default to normal distribution
            "stop_loss_atr": 1.8, # Wider stop loss
            "take_profit_atr": 1.2, # Moderate take profit
            "cooldown_period": 12  # Longer cooldown
        },
        "Balanced": {
            "alpha": 2.0,
            "delta": 0.5,
            "h": 1.0,
            "omega_level": 3,     # Use bimodal distribution
            "stop_loss_atr": 1.5,
            "take_profit_atr": 1.3,
            "cooldown_period": 8
        },
        "Aggressive": {
            "alpha": 1.5,         # Lower α = less mean reversion
            "delta": 0.8,         # Higher δ = more influence of extreme returns
            "h": 1.5,             # Higher h = higher stochastic influence
            "omega_level": 5,     # Use trimodal distribution
            "stop_loss_atr": 1.2, # Tighter stop loss
            "take_profit_atr": 1.5, # Higher take profit
            "cooldown_period": 5   # Shorter cooldown
        }
    }
    
    # Run backtest for each parameter set
    results = {}
    
    for name, params_dict in parameter_sets.items():
        print(f"\nTesting {name} parameter set:")
        for key, value in params_dict.items():
            print(f"  {key}: {value}")
        
        # Create parameter object and update values
        params = QuantumProbabilityParameters()
        for key, value in params_dict.items():
            setattr(params, key, value)
        
        # Run backtest
        backtest_results = run_backtest(df, params, initial_capital=10000, plot=True)
        
        # Save results
        results[name] = backtest_results
        
        # Save to CSV
        trade_df = pd.DataFrame(backtest_results['trade_log'])
        if not trade_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trade_df.to_csv(f"results/quantum_probability_{name.lower()}_{timestamp}.csv", index=False)
    
    # Compare results
    compare_results(results, symbol, timeframe)
    
    return results

def compare_results(results, symbol, timeframe):
    """
    Compare performance metrics across parameter sets
    and create visualization
    """
    # Compile comparison data
    comparison = pd.DataFrame({
        strategy: {
            'Total Return (%)': data['total_return'],
            'Win Rate (%)': data['win_rate'],
            'Profit Factor': data['profit_factor'],
            'Avg Win (%)': data['avg_win'],
            'Avg Loss (%)': data['avg_loss'],
            'Max Drawdown (%)': data['max_drawdown'] * 100,
            'Trades': data['trades']
        }
        for strategy, data in results.items()
    })
    
    print("\n=== Strategy Comparison ===")
    print(comparison)
    
    # Save comparison to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison.to_csv(f"results/quantum_probability_comparison_{timestamp}.csv")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot equity curves
    for strategy, data in results.items():
        equity = data['equity_curve']
        # Normalize to percentage return
        equity_pct = [(e - equity[0]) / equity[0] * 100 for e in equity]
        plt.plot(equity_pct, label=f"{strategy} ({data['win_rate']:.1f}% win)")
    
    plt.title(f'Quantum Probability Model: Strategy Comparison\n{symbol} {timeframe}')
    plt.xlabel('Bars')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f"results/quantum_probability_comparison_{timestamp}.png")
    
    # Create a distribution visualization for different energy levels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get a sample QuantumProbabilityParameters
    params = QuantumProbabilityParameters()
    
    # Plot distributions for different energy levels
    for i, level in enumerate([1, 3, 5]):
        dist = generate_probability_distribution(None, params, level)
        axes[i].plot(dist['r_values'], dist['prob_density'])
        axes[i].set_title(f'Energy Level Ω = {level}')
        
        # Mark modes
        for mode_pos, mode_height in dist['modes']:
            axes[i].scatter(mode_pos, mode_height, color='red', s=50, marker='o')
            
        axes[i].set_xlabel('Normalized Return')
        if i == 0:
            axes[i].set_ylabel('Probability Density')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/quantum_probability_distributions_{timestamp}.png")
    
    print(f"\nVisualizations saved to results/")
    
    # Identify the best parameter set
    best_strategy = comparison.loc['Total Return (%)'].idxmax()
    print(f"\nBest parameter set based on return: {best_strategy}")
    
    # Consider win rate as a factor too
    best_winrate = comparison.loc['Win Rate (%)'].idxmax()
    print(f"Best parameter set based on win rate: {best_winrate}")
    
    return comparison

if __name__ == "__main__":
    # Run the parameter test
    run_parameter_test(
        symbol="ETH/USDT",
        timeframe="1h",
        start_date="2023-01-01", 
        end_date="2023-07-01"
    ) 