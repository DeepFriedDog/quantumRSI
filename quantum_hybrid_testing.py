import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json
import os
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from app import (
    run_quantum_stock_strategy_backtest,
    run_quantum_hybrid_strategy_backtest,
    fetch_crypto_data
)

# Token categories by market cap
HIGH_MARKET_CAP = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']
MEDIUM_MARKET_CAP = ['LINK/USDT', 'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT', 'ATOM/USDT', 'UNI/USDT', 'AAVE/USDT']
LOW_MARKET_CAP = ['ICP/USDT', 'FIL/USDT', 'NEAR/USDT', 'GRT/USDT', 'ZIL/USDT', 'ONE/USDT', 'KSM/USDT', 'ZRX/USDT']

# Optimal timeframes from previous testing
OPTIMAL_TIMEFRAMES = ['1h', '4h', '1d']

def get_market_cap_based_mass(symbol):
    """
    Adjust quantum mass parameter based on token market cap category
    """
    if symbol in HIGH_MARKET_CAP:
        return 0.85  # More stable, higher mass (conservative)
    elif symbol in MEDIUM_MARKET_CAP:
        return 1.1   # Balanced mass
    else:
        return 1.5   # Lower mass for smaller caps (more aggressive)

def select_diverse_tokens(num_tokens=8, high_cap=3, medium_cap=3, low_cap=2):
    """
    Select a diverse set of tokens with specified distribution across market caps
    """
    selected_high = random.sample(HIGH_MARKET_CAP, min(high_cap, len(HIGH_MARKET_CAP)))
    selected_medium = random.sample(MEDIUM_MARKET_CAP, min(medium_cap, len(MEDIUM_MARKET_CAP)))
    selected_low = random.sample(LOW_MARKET_CAP, min(low_cap, len(LOW_MARKET_CAP)))
    
    return selected_high + selected_medium + selected_low

def generate_random_month_period(min_days=28, max_days=45, start_year=2022, end_year=2024):
    """
    Generate a random testing period of approximately one month
    """
    start_date = datetime(random.randint(start_year, end_year), 
                          random.randint(1, 12), 
                          random.randint(1, 28))
    
    duration = random.randint(min_days, max_days)
    end_date = start_date + timedelta(days=duration)
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def create_summary_table(token_results):
    """
    Create a summary table of all strategy results
    """
    summary_data = []
    
    for token, timeframe_results in token_results.items():
        for timeframe, strategies in timeframe_results.items():
            for strategy_name, results in strategies.items():
                if results is not None:
                    summary_data.append({
                        'Token': token,
                        'Timeframe': timeframe,
                        'Strategy': strategy_name,
                        'Total Trades': results['Total Trades'],
                        'Win Rate [%]': results['Win Rate [%]'],
                        'Return [%]': results['Return [%]'],
                        'Max Drawdown [%]': results['Max Drawdown [%]'],
                        'Profit Factor': results['Profit Factor']
                    })
    
    return pd.DataFrame(summary_data)

def create_strategy_comparison_heatmap(summary_df, metric='Return [%]'):
    """
    Create a heatmap comparing strategies across tokens and timeframes
    """
    pivot_table = summary_df.pivot_table(
        index=['Token', 'Timeframe'],
        columns='Strategy',
        values=metric,
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, len(pivot_table) * 0.4))
    sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
    plt.title(f'Strategy Comparison - {metric}')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('hybrid_strategy_results', exist_ok=True)
    plt.savefig(f'hybrid_strategy_results/strategy_comparison_{metric.replace(" ", "_").replace("[", "").replace("]", "")}.png')
    plt.close()

def create_timeframe_comparison(summary_df, metric='Return [%]'):
    """
    Create a bar chart comparing performance across timeframes
    """
    timeframe_performance = summary_df.groupby(['Timeframe', 'Strategy'])[metric].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Timeframe', y=metric, hue='Strategy', data=timeframe_performance)
    plt.title(f'Performance by Timeframe - {metric}')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('hybrid_strategy_results', exist_ok=True)
    plt.savefig(f'hybrid_strategy_results/timeframe_comparison_{metric.replace(" ", "_").replace("[", "").replace("]", "")}.png')
    plt.close()

def create_strategy_boxplot(summary_df, metric='Return [%]'):
    """
    Create box plots showing the distribution of metrics for each strategy
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Strategy', y=metric, data=summary_df)
    plt.title(f'Strategy Performance Distribution - {metric}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('hybrid_strategy_results', exist_ok=True)
    plt.savefig(f'hybrid_strategy_results/strategy_boxplot_{metric.replace(" ", "_").replace("[", "").replace("]", "")}.png')
    plt.close()

def create_equity_curve_plot(token, timeframe, strategy_results):
    """
    Create equity curve plots comparing different strategies
    """
    plt.figure(figsize=(12, 6))
    
    for strategy_name, results in strategy_results.items():
        if results is not None and 'Equity Curve' in results:
            equity_curve = results['Equity Curve']
            plt.plot(equity_curve, label=strategy_name)
    
    plt.title(f'Equity Curves - {token} ({timeframe})')
    plt.xlabel('Trades')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs('hybrid_strategy_results', exist_ok=True)
    plt.savefig(f'hybrid_strategy_results/equity_curve_{token.replace("/", "_")}_{timeframe}.png')
    plt.close()

def create_visualizations(token_results, summary_df):
    """
    Create comprehensive visualizations of the testing results
    """
    # Create heatmaps for key metrics
    create_strategy_comparison_heatmap(summary_df, 'Return [%]')
    create_strategy_comparison_heatmap(summary_df, 'Win Rate [%]')
    create_strategy_comparison_heatmap(summary_df, 'Profit Factor')
    
    # Create timeframe comparisons
    create_timeframe_comparison(summary_df, 'Return [%]')
    create_timeframe_comparison(summary_df, 'Win Rate [%]')
    
    # Create strategy box plots
    create_strategy_boxplot(summary_df, 'Return [%]')
    create_strategy_boxplot(summary_df, 'Win Rate [%]')
    create_strategy_boxplot(summary_df, 'Profit Factor')
    
    # Create equity curves for each token-timeframe combination
    for token, timeframe_results in token_results.items():
        for timeframe, strategies in timeframe_results.items():
            create_equity_curve_plot(token, timeframe, strategies)

def run_strategies_for_token(token, timeframe, start_date, end_date, initial_capital=10000):
    """
    Run all strategies for a single token and timeframe
    """
    print(f"\n===== Testing {token} on {timeframe} timeframe =====")
    results = {}
    
    # Test if we have data for this token and timeframe
    test_data = fetch_crypto_data(token, timeframe, start_date, end_date)
    if test_data is None or len(test_data) < 50:
        print(f"Insufficient data for {token} on {timeframe} timeframe, skipping...")
        return None
    
    # Import parameters class for all tests
    from quantum_stock_model import QuantumStockParameters
    
    # Run Quantum Stock - Original model
    print(f"\nRunning Quantum Stock - Original for {token}...")
    original_params = QuantumStockParameters()  # Create default parameters
    original_stats, _ = run_quantum_stock_strategy_backtest(
        token, timeframe, initial_capital, start_date, end_date, original_params
    )
    results['Quantum Stock - Original'] = original_stats
    
    # Run Quantum Stock - Conservative variant
    print(f"\nRunning Quantum Stock - Conservative for {token}...")
    conservative_params = QuantumStockParameters()
    conservative_params.m = get_market_cap_based_mass(token) 
    conservative_params.hbar = 0.85
    conservative_params.signal_threshold = 0.7
    conservative_params.stop_loss_atr = 1.2
    conservative_params.take_profit_atr = 1.8
    
    conservative_stats, _ = run_quantum_stock_strategy_backtest(
        token, timeframe, initial_capital, start_date, end_date, conservative_params
    )
    results['Quantum Stock - Conservative'] = conservative_stats
    
    # Run Quantum Stock - Aggressive variant
    print(f"\nRunning Quantum Stock - Aggressive for {token}...")
    aggressive_params = QuantumStockParameters()
    aggressive_params.m = get_market_cap_based_mass(token) * 1.3
    aggressive_params.hbar = 1.5
    aggressive_params.signal_threshold = 0.4
    aggressive_params.stop_loss_atr = 1.0
    aggressive_params.take_profit_atr = 2.5
    
    aggressive_stats, _ = run_quantum_stock_strategy_backtest(
        token, timeframe, initial_capital, start_date, end_date, aggressive_params
    )
    results['Quantum Stock - Aggressive'] = aggressive_stats
    
    # Run Quantum Hybrid model
    print(f"\nRunning Quantum Hybrid model for {token}...")
    from quantum_stock_hybrid_model import QuantumHybridParameters
    hybrid_params = QuantumHybridParameters()
    hybrid_params.m = get_market_cap_based_mass(token)
    hybrid_params.mass = get_market_cap_based_mass(token)
    
    hybrid_stats, _ = run_quantum_hybrid_strategy_backtest(
        token, timeframe, initial_capital, start_date, end_date, hybrid_params
    )
    results['Quantum Hybrid'] = hybrid_stats
    
    return results

def run_all_strategies_testing():
    """
    Run a comprehensive test of all quantum strategies
    """
    # Generate a random time period for testing
    start_date, end_date = generate_random_month_period()
    print(f"Testing period: {start_date} to {end_date}")
    
    # Select diverse tokens
    tokens = select_diverse_tokens(num_tokens=8, high_cap=3, medium_cap=3, low_cap=2)
    print(f"Selected tokens: {tokens}")
    
    # Container for all results
    all_results = {}
    
    # For each token, test on optimal timeframes
    for token in tokens:
        all_results[token] = {}
        for timeframe in OPTIMAL_TIMEFRAMES:
            results = run_strategies_for_token(token, timeframe, start_date, end_date)
            all_results[token][timeframe] = results
            
    # Create summary and visualizations
    summary_df = create_summary_table(all_results)
    create_visualizations(all_results, summary_df)
    
    # Save results to JSON for future analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('hybrid_strategy_results', exist_ok=True)
    
    # Save summary to CSV
    summary_df.to_csv(f'hybrid_strategy_results/summary_{timestamp}.csv', index=False)
    
    # Filter results to make them JSON serializable
    serializable_results = {}
    for token, timeframe_results in all_results.items():
        serializable_results[token] = {}
        for timeframe, strategy_results in timeframe_results.items():
            if strategy_results is not None:
                serializable_results[token][timeframe] = {}
                for strategy_name, results in strategy_results.items():
                    if results is not None:
                        # Extract serializable elements
                        serializable_results[token][timeframe][strategy_name] = {
                            'Total Trades': results['Total Trades'],
                            'Win Rate [%]': float(results['Win Rate [%]']),
                            'Return [%]': float(results['Return [%]']),
                            'Max Drawdown [%]': float(results['Max Drawdown [%]']),
                            'Profit Factor': float(results['Profit Factor']),
                            # Skip Trades and Equity Curve as they may not be serializable
                        }
    
    # Save to JSON
    with open(f'hybrid_strategy_results/analysis_results_{timestamp}.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Create summary report
    top_strategies = summary_df.groupby('Strategy')['Return [%]'].mean().sort_values(ascending=False)
    top_win_rates = summary_df.groupby('Strategy')['Win Rate [%]'].mean().sort_values(ascending=False)
    top_timeframes = summary_df.groupby('Timeframe')['Return [%]'].mean().sort_values(ascending=False)
    
    print("\n===== TESTING SUMMARY =====")
    print("\nTop Strategies by Average Return:")
    print(top_strategies)
    
    print("\nTop Strategies by Win Rate:")
    print(top_win_rates)
    
    print("\nBest Timeframes:")
    print(top_timeframes)
    
    # Find best individual strategy combinations
    top_individual = summary_df.sort_values('Return [%]', ascending=False).head(5)
    print("\nTop Individual Strategy-Token-Timeframe Combinations:")
    print(top_individual[['Token', 'Timeframe', 'Strategy', 'Return [%]', 'Win Rate [%]', 'Profit Factor']])
    
    return all_results, summary_df

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('hybrid_strategy_results', exist_ok=True)
    
    # Run comprehensive testing
    results, summary = run_all_strategies_testing()
    
    print("\nResults saved to 'hybrid_strategy_results' directory.")
    print("Done!") 