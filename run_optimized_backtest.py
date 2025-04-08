"""
Run optimized Quantum RSI backtest with NumPy patch

This script applies the NumPy patch and runs the optimized Quantum RSI strategy
across multiple cryptocurrency pairs on a 1-hour timeframe.
"""

# Apply NumPy patch to fix bool8 issue first
import numpy_patch

# Import the optimized Quantum RSI implementation
from quantum_rsi_optimized import (
    QuantumRSIParameters, 
    fetch_crypto_data, 
    run_backtest,
    run_multi_pair_test
)

def main():
    print("=== Quantum RSI Strategy with Maxwell's Demon Filter ===")
    print("Executing backtests with optimized parameters...")
    
    # Strategy parameter options
    strategy_options = {
        "standard": QuantumRSIParameters(),
        "aggressive": QuantumRSIParameters(),
        "conservative": QuantumRSIParameters()
    }
    
    # Adjust parameters for different profiles
    strategy_options["aggressive"].rsi_high = 65
    strategy_options["aggressive"].rsi_low = 35
    strategy_options["aggressive"].volume_threshold = 1.5
    strategy_options["aggressive"].take_profit_atr = 1.0
    strategy_options["aggressive"].stop_loss_atr = 1.8
    
    strategy_options["conservative"].rsi_high = 78
    strategy_options["conservative"].rsi_low = 22
    strategy_options["conservative"].volume_threshold = 2.0
    strategy_options["conservative"].take_profit_atr = 1.0
    strategy_options["conservative"].stop_loss_atr = 1.2
    
    # Test configuration
    symbol = "ETH/USDT"
    timeframe = "1h"
    start_date = "2023-01-01"
    end_date = "2023-07-01"
    
    # Single pair backtest with all strategy profiles
    print(f"\nRunning backtests on {symbol} from {start_date} to {end_date}...")
    
    results = {}
    
    for strategy_name, params in strategy_options.items():
        print(f"\nTesting {strategy_name.upper()} profile...")
        df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
        strategy_result = run_backtest(df, params, plot=True)
        results[strategy_name] = strategy_result
    
    # Compare strategy profiles
    print("\n=== Strategy Profile Comparison ===")
    for strategy_name, result in results.items():
        print(f"{strategy_name.upper()}:")
        print(f"  Win Rate: {result['win_rate']:.2f}%")
        print(f"  Total Return: {result['total_return']:.2f}%")
        print(f"  Profit Factor: {result['profit_factor']:.2f}")
        print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
        print(f"  Total Trades: {result['trades']}")
    
    # Optional: Multi-pair test using the best parameter set
    # Determine the best parameter set based on win rate or other metrics
    best_win_rate = 0
    best_strategy = "standard"
    
    for strategy_name, result in results.items():
        if result['win_rate'] > best_win_rate:
            best_win_rate = result['win_rate']
            best_strategy = strategy_name
    
    print(f"\nBest strategy profile: {best_strategy.upper()} with {best_win_rate:.2f}% win rate")
    
    # Ask user if they want to run multi-pair test
    run_multi = input("\nWould you like to run a multi-pair test with the best parameters? (y/n): ")
    
    if run_multi.lower() == 'y':
        pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
        print(f"\nRunning multi-pair test with {best_strategy.upper()} parameters...")
        multi_results = run_multi_pair_test(pairs, timeframe, start_date, end_date, 
                                          strategy_options[best_strategy])

if __name__ == "__main__":
    main() 