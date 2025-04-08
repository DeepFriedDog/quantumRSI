#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Quantum Harmonic-Wave Hybrid Strategy Testing

This script runs extensive backtests on the Quantum Harmonic-Wave Hybrid strategy
across multiple tokens and timeframes, providing detailed visualizations and analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import json
import traceback
import pickle
from pathlib import Path
import gc
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import time
import sys

# Import strategy modules
from quantum_harmonic_wave_hybrid import run_quantum_hybrid_strategy_backtest, QuantumHybridParameters
from app import fetch_crypto_data

# Fix pandas warnings
pd.options.mode.chained_assignment = None

# Define token categories by market cap
HIGH_MARKET_CAP = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
MEDIUM_MARKET_CAP = ['LINK/USDT', 'DOT/USDT', 'AVAX/USDT', 'ATOM/USDT', 'NEAR/USDT', 'MATIC/USDT']
SMALL_MARKET_CAP = ['ALGO/USDT', 'RNDR/USDT', 'FTM/USDT', 'ONE/USDT', 'ZIL/USDT', 'ROSE/USDT', 'CHZ/USDT', 'ENJ/USDT', 'OCEAN/USDT', 'API3/USDT']

# Add ADA, DOGE, and SHIB based on user's prior interest
HIGH_MARKET_CAP.extend(['ADA/USDT', 'DOGE/USDT'])
MEDIUM_MARKET_CAP.append('SHIB/USDT')

# Add data caching directory
DATA_CACHE_DIR = 'data_cache'

# Ensure results directory exists
RESULTS_DIR = 'quantum_harmonic_hybrid_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define a global variable for storing intermediate results for recovery
recovery_results = []

def select_diverse_tokens(num_from_each_category=3):
    """Select tokens from different market cap categories."""
    high_cap = random.sample(HIGH_MARKET_CAP, min(num_from_each_category, len(HIGH_MARKET_CAP)))
    medium_cap = random.sample(MEDIUM_MARKET_CAP, min(num_from_each_category, len(MEDIUM_MARKET_CAP)))
    small_cap = random.sample(SMALL_MARKET_CAP, min(num_from_each_category, len(SMALL_MARKET_CAP)))
    
    selected_tokens = high_cap + medium_cap + small_cap
    
    print(f"Selected tokens:")
    print(f"  High Market Cap: {high_cap}")
    print(f"  Medium Market Cap: {medium_cap}")
    print(f"  Low Market Cap: {small_cap}")
    
    return selected_tokens


def generate_random_month_period(min_year=2023, max_year=2024):
    """Generate a random 1-month period."""
    year = random.randint(min_year, max_year)
    month = random.randint(1, 12)
    
    # Ensure we don't pick a future period
    current_date = datetime.now()
    if year == current_date.year and month > current_date.month:
        month = random.randint(1, current_date.month)
    
    # Create start date
    start_date = datetime(year, month, 1)
    
    # Create end date (30 days after start)
    end_date = start_date + timedelta(days=30)
    
    # Format dates for API
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Analysis period: {start_date_str} to {end_date_str}")
    return start_date_str, end_date_str


def prefetch_and_cache_data(tokens, timeframes, start_date, end_date):
    """
    Prefetch data for all token-timeframe combinations and cache it locally.
    Returns a dictionary of availability status.
    """
    print("Prefetching and caching all required data...")
    
    # Create cache directory if it doesn't exist
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    availability = {}
    
    for token in tokens:
        token_safe = token.replace('/', '_')
        availability[token] = {}
        
        for timeframe in timeframes:
            cache_file = os.path.join(DATA_CACHE_DIR, f"{token_safe}_{timeframe}_{start_date}_{end_date}.pkl")
            
            # Check if we already have this data cached
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    candle_count = len(data) if data is not None else 0
                    print(f"{token} {timeframe}: ✅ (Using cached data: {candle_count} candles)")
                    
                    availability[token][timeframe] = {
                        "available": candle_count > 100,
                        "candle_count": candle_count,
                        "cache_file": cache_file
                    }
                    continue
                except Exception as e:
                    print(f"Error loading cached data for {token} {timeframe}: {str(e)}")
            
            # Fetch new data
            try:
                print(f"Fetching {token} {timeframe} data from {start_date} to {end_date}...")
                data = fetch_crypto_data(token, timeframe, start_date, end_date)
                candle_count = len(data) if data is not None else 0
                
                if candle_count > 100:
                    # Cache the data
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data, f)
                    
                    print(f"{token} {timeframe}: ✅ ({candle_count} candles, cached to {cache_file})")
                    
                    availability[token][timeframe] = {
                        "available": True,
                        "candle_count": candle_count,
                        "cache_file": cache_file
                    }
                else:
                    print(f"{token} {timeframe}: ❌ (Insufficient data: {candle_count} candles)")
                    
                    availability[token][timeframe] = {
                        "available": False,
                        "candle_count": candle_count
                    }
            except Exception as e:
                print(f"{token} {timeframe}: ❌ Error: {str(e)}")
                
                availability[token][timeframe] = {
                    "available": False,
                    "error": str(e)
                }
    
    return availability


def get_cached_data(token, timeframe, start_date, end_date):
    """Get data from cache if available, otherwise fetch and cache it."""
    token_safe = token.replace('/', '_')
    cache_file = os.path.join(DATA_CACHE_DIR, f"{token_safe}_{timeframe}_{start_date}_{end_date}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Using cached data for {token} {timeframe} ({len(data)} candles)")
            return data
        except Exception as e:
            print(f"Error loading cached data: {str(e)}")
    
    # If we get here, we need to fetch the data
    print(f"Cache miss - fetching {token} {timeframe} data...")
    data = fetch_crypto_data(token, timeframe, start_date, end_date)
    
    # Cache the data for future use
    if data is not None and len(data) > 0:
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Cached {len(data)} candles to {cache_file}")
    
    return data


def patch_fetch_functions():
    """
    Monkey patch the fetch_crypto_data function to use cached data.
    """
    import app
    
    # Save original function
    original_fetch = app.fetch_crypto_data
    
    # Create patched function
    def patched_fetch_crypto_data(symbol, timeframe, start_date, end_date=None, exchange_id='binance'):
        """Patched version that uses cached data if available."""
        return get_cached_data(symbol, timeframe, start_date, end_date)
    
    # Apply the patch
    app.fetch_crypto_data = patched_fetch_crypto_data
    
    print("Patched fetch_crypto_data to use cached data")
    
    return original_fetch


def extract_key_metrics(stats):
    """Extract key metrics from strategy results with improved error handling."""
    if stats is None:
        print("Warning: Stats object is None, returning default values")
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'risk_adjusted_return': 0
        }
    
    try:
        # Convert string percentages to floats if needed
        win_rate = stats.get('Win Rate [%]', 0)
        if isinstance(win_rate, str):
            win_rate = float(win_rate.replace('%', ''))
        
        total_return = stats.get('Total Return [%]', 0)
        if isinstance(total_return, str):
            total_return = float(total_return.replace('%', ''))
        
        max_drawdown = stats.get('Max Drawdown [%]', 0)
        if isinstance(max_drawdown, str):
            max_drawdown = float(max_drawdown.replace('%', ''))
        
        profit_factor = stats.get('Profit Factor', 0)
        if isinstance(profit_factor, str):
            profit_factor = float(profit_factor)
        
        # Handle infinity profit factor
        if profit_factor == float('inf'):
            profit_factor = 999.99
        
        # Calculate Sharpe-like ratio (return / drawdown) if possible
        risk_adjusted_return = 0
        if max_drawdown > 0:
            risk_adjusted_return = total_return / max_drawdown
        
        avg_profit = stats.get('Average Profit', 0)
        avg_loss = stats.get('Average Loss', 0)
        
        return {
            'total_trades': stats.get('Total Trades', 0),
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'risk_adjusted_return': risk_adjusted_return,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss
        }
    except Exception as e:
        print(f"Error extracting metrics: {str(e)}")
        traceback.print_exc()
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'risk_adjusted_return': 0,
            'error': str(e)
        }


def safe_run_strategy(strategy_name, token, timeframe, initial_capital, start_date, end_date, params):
    """Run a strategy with enhanced error handling to prevent crashes."""
    print(f"Running {strategy_name} on {token} ({timeframe})...")
    
    try:
        # Set a maximum timeout for each strategy run
        max_runtime = 600  # 10 minutes max per strategy
        start_time = time.time()
        
        # Use a try-except with proper timeouts to catch hanging processes
        try:
            stats, trades = run_quantum_hybrid_strategy_backtest(
                token, timeframe, initial_capital, start_date, end_date, params, plot=False
            )
            
            # Check if we're taking too long
            if time.time() - start_time > max_runtime:
                raise TimeoutError(f"Strategy execution exceeded {max_runtime} seconds")
            
            # Add check for None stats
            if stats is None:
                print(f"Warning: {strategy_name} returned None stats")
                return {
                    'error': 'Strategy returned None stats',
                    'token': token,
                    'timeframe': timeframe,
                    'strategy': strategy_name,
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'max_drawdown': 0,
                    'profit_factor': 0,
                    'risk_adjusted_return': 0
                }
            else:
                metrics = extract_key_metrics(stats)
                print(f"Completed {strategy_name} on {token} ({timeframe}) successfully.")
                
                # Add token and timeframe to metrics
                metrics['token'] = token
                metrics['timeframe'] = timeframe
                metrics['strategy'] = strategy_name
                
                return metrics
        except TimeoutError as te:
            print(f"Timeout running {strategy_name} on {token} ({timeframe}): {str(te)}")
            return {
                'error': f'Timeout: {str(te)}',
                'token': token,
                'timeframe': timeframe,
                'strategy': strategy_name,
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'risk_adjusted_return': 0
            }
            
    except Exception as e:
        print(f"Error in {strategy_name} on {token} ({timeframe}): {str(e)}")
        traceback.print_exc()
        return {
            'error': str(e),
            'token': token,
            'timeframe': timeframe,
            'strategy': strategy_name,
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'risk_adjusted_return': 0
        }


def get_param_variants(token):
    """
    Generate parameter variants (Conservative, Balanced, Aggressive) 
    for the Quantum Harmonic-Wave Hybrid strategy.
    """
    # Base parameters
    base_params = QuantumHybridParameters()
    
    # Conservative parameters
    conservative_params = QuantumHybridParameters()
    conservative_params.conservative_weight = 0.8
    conservative_params.aggressive_weight = 0.2
    conservative_params.trend_threshold = 0.2
    conservative_params.entry_threshold = 0.8
    conservative_params.exit_threshold = 0.5
    conservative_params.stop_loss_atr = 1.8
    conservative_params.take_profit_atr = 1.5
    conservative_params.cooldown_period = 5
    
    # Balanced parameters
    balanced_params = QuantumHybridParameters()
    # Default values are already balanced
    
    # Aggressive parameters
    aggressive_params = QuantumHybridParameters()
    aggressive_params.conservative_weight = 0.3
    aggressive_params.aggressive_weight = 0.7
    aggressive_params.trend_threshold = 0.1
    aggressive_params.entry_threshold = 0.4
    aggressive_params.exit_threshold = 0.3
    aggressive_params.stop_loss_atr = 1.2
    aggressive_params.take_profit_atr = 2.5
    aggressive_params.cooldown_period = 2
    
    return {
        "Quantum Harmonic - Conservative": conservative_params,
        "Quantum Harmonic - Balanced": balanced_params,
        "Quantum Harmonic - Aggressive": aggressive_params
    }


def run_comprehensive_backtest(tokens, timeframes, start_date, end_date, initial_capital=10000):
    """Run comprehensive backtest of the Quantum Harmonic-Wave Hybrid strategy."""
    # Apply patches to use cached data
    original_fetch = patch_fetch_functions()
    
    try:
        results = []
        
        # Total number of tests
        total_tests = len(tokens) * len(timeframes) * 3  # 3 strategy variants
        test_count = 0
        
        # Process in smaller batches to prevent memory issues
        batch_size = 5  # Process 5 tests at a time
        
        for token_idx, token in enumerate(tokens):
            print(f"\n===== Processing token {token_idx+1}/{len(tokens)}: {token} =====")
            
            # Explicitly clear memory after each token
            if token_idx > 0:
                gc.collect()
            
            # Get parameter variants for this token
            param_variants = get_param_variants(token)
            
            for timeframe in timeframes:
                # Clear memory after each timeframe
                gc.collect()
                
                print(f"\n----- Testing {token} on {timeframe} timeframe -----")
                
                for strategy_name, params in param_variants.items():
                    test_count += 1
                    print(f"\n[{test_count}/{total_tests}] Testing {strategy_name} on {token} ({timeframe})...")
                    
                    try:
                        # Run the strategy with a timeout
                        metrics = safe_run_strategy(
                            strategy_name, token, timeframe, initial_capital, 
                            start_date, end_date, params
                        )
                        
                        # Add to results
                        results.append(metrics)
                        
                        # Save after each individual test
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        try:
                            with open(f'{RESULTS_DIR}/interim_results_{test_count}_{timestamp}.json', 'w') as f:
                                json.dump(results, f, default=str)
                        except Exception as save_error:
                            print(f"Warning: Could not save interim results: {save_error}")
                        
                    except Exception as e:
                        print(f"Error testing {strategy_name} on {token} ({timeframe}): {str(e)}")
                        traceback.print_exc()
                        results.append({
                            'error': str(e),
                            'token': token,
                            'timeframe': timeframe,
                            'strategy': strategy_name,
                            'total_trades': 0,
                            'win_rate': 0,
                            'total_return': 0,
                            'max_drawdown': 0,
                            'profit_factor': 0,
                            'risk_adjusted_return': 0
                        })
                    
                    # Force garbage collection after each test
                    gc.collect()
                
                # Save after completing each timeframe
                timeframe_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f'{RESULTS_DIR}/interim_results_{token}_{timeframe}_{timeframe_timestamp}.json', 'w') as f:
                    json.dump(results, f, default=str)
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{RESULTS_DIR}/comprehensive_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, default=str)
        
        print(f"\nAnalysis complete! Results saved to {filename}")
        return results, filename
    
    except Exception as e:
        print(f"ERROR in comprehensive backtest: {str(e)}")
        traceback.print_exc()
        
        # Try to save partial results
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recovery_file = f'{RESULTS_DIR}/recovery_results_{timestamp}.json'
            with open(recovery_file, 'w') as f:
                json.dump(results, f, default=str)
            print(f"Recovered partial results saved to {recovery_file}")
            return results, recovery_file
        except Exception as save_error:
            print(f"Failed to save recovery results: {save_error}")
            return results, None
    
    finally:
        # Restore original function when done
        import app
        app.fetch_crypto_data = original_fetch
        print("Restored original fetch_crypto_data function")


def create_strategy_comparison_visualizations(results, output_file):
    """Create comprehensive visualizations comparing strategy variants."""
    if not results:
        print("No results to visualize")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Remove rows with errors
    df = df[~df.astype(str).apply(lambda x: 'error' in x.values, axis=1)]
    
    # Ensure numeric columns are numeric
    numeric_cols = ['total_trades', 'win_rate', 'total_return', 'max_drawdown', 
                    'profit_factor', 'risk_adjusted_return']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create figure with multiple panels
    plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.5])
    
    # 1. Heatmap of win rates by strategy and timeframe
    ax1 = plt.subplot(gs[0, 0])
    win_rate_pivot = df.pivot_table(
        values='win_rate', 
        index='strategy', 
        columns='timeframe', 
        aggfunc='mean'
    )
    sns.heatmap(win_rate_pivot, annot=True, cmap='RdYlGn', fmt='.1f', ax=ax1, vmin=20, vmax=80)
    ax1.set_title('Average Win Rate (%) by Strategy and Timeframe', fontsize=14)
    
    # 2. Heatmap of returns by strategy and timeframe
    ax2 = plt.subplot(gs[0, 1])
    return_pivot = df.pivot_table(
        values='total_return', 
        index='strategy', 
        columns='timeframe', 
        aggfunc='mean'
    )
    sns.heatmap(return_pivot, annot=True, cmap='RdYlGn', fmt='.1f', ax=ax2, vmin=-10, vmax=20)
    ax2.set_title('Average Return (%) by Strategy and Timeframe', fontsize=14)
    
    # 3. Box plot of returns by strategy
    ax3 = plt.subplot(gs[1, 0])
    sns.boxplot(x='strategy', y='total_return', data=df, ax=ax3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.set_title('Distribution of Returns (%) by Strategy', fontsize=14)
    ax3.set_ylabel('Return (%)')
    
    # 4. Box plot of win rates by strategy
    ax4 = plt.subplot(gs[1, 1])
    sns.boxplot(x='strategy', y='win_rate', data=df, ax=ax4)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.set_title('Distribution of Win Rates (%) by Strategy', fontsize=14)
    ax4.set_ylabel('Win Rate (%)')
    
    # 5. Scatter plot of win rate vs return
    ax5 = plt.subplot(gs[2, 0])
    scatter = sns.scatterplot(
        x='win_rate', 
        y='total_return', 
        hue='strategy', 
        size='total_trades',
        sizes=(50, 200),
        alpha=0.7,
        data=df, 
        ax=ax5
    )
    ax5.set_title('Win Rate vs Return (size = # of trades)', fontsize=14)
    ax5.set_xlabel('Win Rate (%)')
    ax5.set_ylabel('Return (%)')
    scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Bar chart of risk-adjusted returns
    ax6 = plt.subplot(gs[2, 1])
    risk_adj_pivot = df.pivot_table(
        values='risk_adjusted_return', 
        index='strategy', 
        aggfunc='mean'
    ).sort_values('risk_adjusted_return', ascending=False)
    sns.barplot(
        x=risk_adj_pivot.index, 
        y='risk_adjusted_return', 
        data=risk_adj_pivot.reset_index(),
        ax=ax6
    )
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    ax6.set_title('Average Risk-Adjusted Return (Return/Drawdown)', fontsize=14)
    ax6.set_ylabel('Return/Drawdown Ratio')
    
    # 7. Combined performance table
    ax7 = plt.subplot(gs[3, :])
    
    # Calculate overall metrics by strategy
    strategy_summary = df.groupby('strategy').agg({
        'total_trades': 'sum',
        'win_rate': 'mean',
        'total_return': 'mean',
        'max_drawdown': 'mean',
        'profit_factor': 'mean',
        'risk_adjusted_return': 'mean'
    }).sort_values('total_return', ascending=False)
    
    # Add count of profitable tests
    profitable_tests = df[df['total_return'] > 0].groupby('strategy').size()
    strategy_summary['profitable_tests'] = profitable_tests
    strategy_summary['total_tests'] = df.groupby('strategy').size()
    strategy_summary['profitable_pct'] = (strategy_summary['profitable_tests'] / strategy_summary['total_tests'] * 100).round(1)
    
    # Format the table data
    table_data = []
    for idx, row in strategy_summary.iterrows():
        table_data.append([
            idx,
            f"{row['win_rate']:.1f}%",
            f"{row['total_return']:.1f}%",
            f"{row['max_drawdown']:.1f}%",
            f"{row['profit_factor']:.2f}",
            f"{row['risk_adjusted_return']:.2f}",
            f"{row['profitable_tests']}/{row['total_tests']} ({row['profitable_pct']}%)",
            f"{row['total_trades']}"
        ])
    
    # Create the table
    table = ax7.table(
        cellText=table_data,
        colLabels=['Strategy', 'Avg Win Rate', 'Avg Return', 'Avg Max DD', 'Profit Factor', 'Risk/Reward', 'Profitable Tests', 'Total Trades'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color code the table based on return
    for i in range(len(table_data)):
        # Color code based on return
        return_val = strategy_summary.iloc[i]['total_return']
        if return_val > 10:
            cell_color = (0.1, 0.6, 0.1)  # Dark green
        elif return_val > 5:
            cell_color = (0.2, 0.8, 0.2)  # Green
        elif return_val > 0:
            cell_color = (0.8, 0.8, 0.2)  # Yellow
        else:
            cell_color = (0.8, 0.2, 0.2)  # Red
        
        # Apply the color to the strategy name cell
        table[(i+1, 0)].set_facecolor(cell_color)
        
        # Highlight the win rate and return cells
        win_rate = strategy_summary.iloc[i]['win_rate']
        if win_rate > 50:
            table[(i+1, 1)].set_facecolor((0.2, 0.8, 0.2))  # Green
        else:
            table[(i+1, 1)].set_facecolor((0.8, 0.2, 0.2))  # Red
    
    ax7.set_title('Strategy Performance Summary', fontsize=16)
    ax7.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()


def create_token_performance_analysis(results, output_file):
    """Create analysis of token performance across strategies."""
    if not results:
        print("No results to visualize")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Remove rows with errors
    df = df[~df.astype(str).apply(lambda x: 'error' in x.values, axis=1)]
    
    # Ensure numeric columns are numeric
    numeric_cols = ['total_trades', 'win_rate', 'total_return', 'max_drawdown', 
                    'profit_factor', 'risk_adjusted_return']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Calculate average return by token and strategy
    token_pivot = df.pivot_table(
        values='total_return', 
        index='token', 
        columns='strategy', 
        aggfunc='mean'
    ).sort_values(by='Quantum Harmonic - Balanced', ascending=False)
    
    # Plot as a grouped bar chart
    token_pivot.plot(kind='bar', ax=plt.gca())
    plt.title('Average Return (%) by Token and Strategy', fontsize=16)
    plt.ylabel('Return (%)')
    plt.xlabel('Token')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Strategy')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Token performance analysis saved to {output_file}")
    plt.close()


def main():
    """Main function to run the analysis."""
    print("Starting Quantum Harmonic-Wave Hybrid Strategy Comprehensive Backtest")
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Add signal handlers for graceful shutdown
    import signal
    
    # Global variable to store results for recovery
    global recovery_results
    recovery_results = []
    
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, performing graceful shutdown...")
        # Try to save recovery data
        try:
            if recovery_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                recovery_file = f'{RESULTS_DIR}/interrupted_results_{timestamp}.json'
                with open(recovery_file, 'w') as f:
                    json.dump(recovery_results, f, default=str)
                print(f"Saved interrupted results to {recovery_file}")
        except Exception as e:
            print(f"Failed to save recovery data: {e}")
        sys.exit(0)
    
    # Register the signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Select tokens (4 from high cap, 3 from medium, 3 from low)
        tokens = []
        tokens.extend(random.sample(HIGH_MARKET_CAP, min(4, len(HIGH_MARKET_CAP))))
        tokens.extend(random.sample(MEDIUM_MARKET_CAP, min(3, len(MEDIUM_MARKET_CAP))))
        tokens.extend(random.sample(SMALL_MARKET_CAP, min(3, len(SMALL_MARKET_CAP))))
        
        # Choose timeframes as specified
        timeframes = ['5m', '30m', '1h', '4h', '1d']
        
        # Generate random 1-month period
        start_date, end_date = generate_random_month_period()
        
        # Prefetch and cache data
        availability = prefetch_and_cache_data(tokens, timeframes, start_date, end_date)
        
        # Filter tokens and timeframes to only include those with available data
        valid_tokens = []
        for token, timeframe_data in availability.items():
            valid_timeframes = [tf for tf, data in timeframe_data.items() if data.get('available', False)]
            if len(valid_timeframes) == len(timeframes):
                valid_tokens.append(token)
        
        # If not enough tokens have data, try a different date range
        if len(valid_tokens) < 5:
            print("Not enough tokens with valid data. Trying a different date range...")
            start_date, end_date = generate_random_month_period()
            availability = prefetch_and_cache_data(tokens, timeframes, start_date, end_date)
            
            valid_tokens = []
            for token, timeframe_data in availability.items():
                valid_timeframes = [tf for tf, data in timeframe_data.items() if data.get('available', False)]
                if len(valid_timeframes) == len(timeframes):
                    valid_tokens.append(token)
        
        # If still not enough tokens, use what we have
        if len(valid_tokens) < 3:
            print(f"Warning: Only {len(valid_tokens)} tokens have data for all timeframes.")
            if not valid_tokens:
                print("No tokens with valid data for all timeframes. Exiting.")
                return
        
        print(f"Running analysis on {len(valid_tokens)} tokens...")
        
        # Run the comprehensive backtest
        results, results_file = run_comprehensive_backtest(valid_tokens, timeframes, start_date, end_date)
        
        # Store in global recovery variable
        recovery_results = results
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        create_strategy_comparison_visualizations(
            results, 
            f'{RESULTS_DIR}/comprehensive_analysis_{timestamp}.png'
        )
        
        # Create token performance analysis
        create_token_performance_analysis(
            results,
            f'{RESULTS_DIR}/token_performance_{timestamp}.png'
        )
        
        print(f"Analysis complete! Results saved to {results_file}")
        print(f"Visualizations saved to {RESULTS_DIR}/")
        
    except Exception as e:
        print(f"ERROR: Analysis failed with error: {str(e)}")
        traceback.print_exc()
        print("Attempting to save partial results...")
        
        # Try to save whatever results we have
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            with open(f'{RESULTS_DIR}/partial_results_{timestamp}.json', 'w') as f:
                json.dump({"error": str(e), "partial_results": recovery_results}, f, default=str)
            print(f"Partial results saved to {RESULTS_DIR}/partial_results_{timestamp}.json")
        except Exception as save_error:
            print(f"Failed to save partial results: {str(save_error)}")


if __name__ == "__main__":
    main() 