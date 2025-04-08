#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved Quantum Anharmonic Testing Script

This script runs comprehensive tests on all versions of the Quantum Anharmonic model
(original and improved variants) across multiple tokens and timeframes.
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

# Import app modules for strategy running
from app import (
    run_quantum_anharmonic_strategy_backtest,
    run_quantum_anharmonic_improved_conservative_strategy_backtest,
    run_quantum_anharmonic_improved_balanced_strategy_backtest, 
    run_quantum_anharmonic_improved_aggressive_strategy_backtest,
    fetch_crypto_data
)

# Import parameter class
from quantum_anharmonic_oscillator_model import QuantumAnharmonicParameters

# Fix pandas to avoid SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Define token categories by market cap
HIGH_MARKET_CAP = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
MEDIUM_MARKET_CAP = ['LINK/USDT', 'DOT/USDT', 'AVAX/USDT', 'ATOM/USDT', 'NEAR/USDT', 'MATIC/USDT']
SMALL_MARKET_CAP = ['ALGO/USDT', 'RNDR/USDT', 'FTM/USDT', 'ONE/USDT', 'ZIL/USDT', 'ROSE/USDT', 'CHZ/USDT', 'ENJ/USDT', 'OCEAN/USDT', 'API3/USDT']

# Add a data caching directory constant
DATA_CACHE_DIR = 'data_cache'

# Ensure results directory exists
os.makedirs("improved_Quantum_anharmonic_strategy", exist_ok=True)

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
    if year == datetime.now().year and month > datetime.now().month:
        month = random.randint(1, datetime.now().month)
    
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

def safe_run_strategy(strategy_name, strategy_func, strategy_results, *args, **kwargs):
    """Run a strategy with enhanced error handling to prevent crashes."""
    print(f"Running {strategy_name}...")
    try:
        stats, _ = strategy_func(*args, **kwargs)
        
        # Add check for None stats
        if stats is None:
            print(f"Warning: {strategy_name} returned None stats")
            strategy_results[strategy_name] = {
                'error': 'Strategy returned None stats',
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'risk_adjusted_return': 0
            }
        else:
            strategy_results[strategy_name] = extract_key_metrics(stats)
            print(f"Completed {strategy_name} successfully.")
            
        # Save intermediate results after each strategy
        try:
            with open(f'improved_Quantum_anharmonic_strategy/intermediate_{args[0].split("/")[0]}_{args[1]}.json', 'w') as f:
                json.dump(strategy_results, f, default=str)
        except Exception as save_error:
            print(f"Warning: Could not save intermediate results: {save_error}")
            
    except Exception as e:
        print(f"Error in {strategy_name}: {str(e)}")
        traceback.print_exc()
        strategy_results[strategy_name] = {
            'error': str(e),
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'risk_adjusted_return': 0
        }
    
    return strategy_results

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
        
        total_return = stats.get('Return [%]', 0)
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
        
        return {
            'total_trades': stats.get('Total Trades', 0),
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'risk_adjusted_return': risk_adjusted_return
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

def adjust_parameters_for_timeframe(params, timeframe):
    """Adjust quantum parameters based on timeframe to maintain scaling relationships."""
    # Parameters need to be adjusted based on the timeframe to keep quantum effects consistent
    multipliers = {
        '1m': 0.2,    # Higher frequency needs more dampened parameters
        '5m': 0.4,    # Moderately high frequency
        '15m': 0.7,   # Medium frequency
        '30m': 0.85,  # Medium-low frequency
        '1h': 1.0,    # Base parameters designed for 1h timeframe
        '4h': 1.3,    # Lower frequency needs amplified parameters
        '1d': 1.5     # Daily timeframe needs stronger parameters
    }
    
    multiplier = multipliers.get(timeframe, 1.0)
    
    # Adjust anharmonic parameters
    params.beta *= multiplier
    params.gamma *= multiplier
    
    # Adjust frequency parameter based on timeframe
    # Normalize omega to keep the effect consistent across timeframes
    timeframe_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
    minutes = timeframe_minutes.get(timeframe, 60)
    base_minutes = 60  # 1h is our reference timeframe
    
    # Scale omega inversely with timeframe length
    if hasattr(params, 'omega') and params.omega is not None:
        if isinstance(params.omega, float):
            params.omega = params.omega * (base_minutes / minutes)
        else:
            # Handle case where omega is in format 1/X
            omega_value = 1/252  # default value
            params.omega = omega_value * (base_minutes / minutes)
    
    return params

def run_all_quantum_anharmonic_strategies(tokens, timeframes, start_date, end_date):
    """Run all versions of the Quantum Anharmonic model for each token-timeframe combination."""
    # Apply patches to use cached data
    original_fetch = patch_fetch_functions()
    
    try:
        results = {}
        
        # Total number of tests
        total_tests = len(tokens) * len(timeframes)
        test_count = 0
        
        for token in tokens:
            results[token] = {}
            
            for timeframe in timeframes:
                test_count += 1
                print(f"\n[{test_count}/{total_tests}] Testing {token} on {timeframe} timeframe...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                
                try:
                    # Run all strategies
                    strategy_results = run_strategies_for_token(
                        token, timeframe, start_date, end_date
                    )
                    
                    # Store results
                    results[token][timeframe] = strategy_results
                    
                except Exception as e:
                    print(f"Error testing {token} on {timeframe}: {str(e)}")
                    traceback.print_exc()
                    results[token][timeframe] = {"error": str(e)}
                
                # Save interim results after EACH timeframe to prevent data loss
                with open(f'improved_Quantum_anharmonic_strategy/analysis_results_{timestamp}_interim.json', 'w') as f:
                    json.dump(results, f, default=str)
                
                # Force garbage collection after each timeframe to free memory
                gc.collect()
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f'improved_Quantum_anharmonic_strategy/analysis_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, default=str)
        
        print(f"\nAnalysis complete! Results saved to {filename}")
        return results, filename
    
    finally:
        # Restore original function when done
        import app
        app.fetch_crypto_data = original_fetch
        print("Restored original fetch_crypto_data function")
        
def run_strategies_for_token(token, timeframe, start_date, end_date):
    """Run all versions of the Quantum Anharmonic model for a single token-timeframe combination."""
    strategy_results = {}
    
    # Legacy Conservative Quantum Anharmonic
    conservative_params = QuantumAnharmonicParameters()
    conservative_params.beta = 0.10
    conservative_params.gamma = 0.05
    conservative_params.lambda_param = 0.3
    conservative_params.set_spread_from_volatility(0.010)
    conservative_params.stop_loss_atr = 1.7
    conservative_params.take_profit_atr = 1.2
    conservative_params.cooldown_period = 10
    
    # Adjust parameters for the timeframe
    conservative_params = adjust_parameters_for_timeframe(conservative_params, timeframe)
    
    strategy_results = safe_run_strategy(
        'Quantum Anharmonic - Conservative',
        run_quantum_anharmonic_strategy_backtest,
        strategy_results,
        token, timeframe, start_date, end_date, conservative_params
    )
    
    # Legacy Balanced Quantum Anharmonic
    balanced_params = QuantumAnharmonicParameters()
    # Using default parameters
    balanced_params.beta = 0.1
    balanced_params.gamma = 0.05
    balanced_params.lambda_param = 0.25
    balanced_params.omega = 1/252
    
    # Adjust parameters for the timeframe
    balanced_params = adjust_parameters_for_timeframe(balanced_params, timeframe)
    
    strategy_results = safe_run_strategy(
        'Quantum Anharmonic - Balanced',
        run_quantum_anharmonic_strategy_backtest,
        strategy_results,
        token, timeframe, start_date, end_date, balanced_params
    )
    
    # Legacy Aggressive Quantum Anharmonic
    aggressive_params = QuantumAnharmonicParameters()
    aggressive_params.beta = 0.15
    aggressive_params.gamma = 0.08
    aggressive_params.lambda_param = 0.4
    aggressive_params.set_spread_from_volatility(0.012)
    aggressive_params.stop_loss_atr = 1.2
    aggressive_params.take_profit_atr = 1.5
    aggressive_params.cooldown_period = 5
    
    # Adjust parameters for the timeframe
    aggressive_params = adjust_parameters_for_timeframe(aggressive_params, timeframe)
    
    strategy_results = safe_run_strategy(
        'Quantum Anharmonic - Aggressive',
        run_quantum_anharmonic_strategy_backtest,
        strategy_results,
        token, timeframe, start_date, end_date, aggressive_params
    )
    
    # Improved Conservative Quantum Anharmonic
    improved_conservative_params = None  # Let the function set default parameters
    
    strategy_results = safe_run_strategy(
        'Quantum Anharmonic - Improved Conservative',
        run_quantum_anharmonic_improved_conservative_strategy_backtest,
        strategy_results,
        token, timeframe, start_date, end_date, improved_conservative_params
    )
    
    # Improved Balanced Quantum Anharmonic
    improved_balanced_params = None  # Let the function set default parameters
    
    strategy_results = safe_run_strategy(
        'Quantum Anharmonic - Improved Balanced',
        run_quantum_anharmonic_improved_balanced_strategy_backtest,
        strategy_results,
        token, timeframe, start_date, end_date, improved_balanced_params
    )
    
    # Improved Aggressive Quantum Anharmonic
    improved_aggressive_params = None  # Let the function set default parameters
    
    strategy_results = safe_run_strategy(
        'Quantum Anharmonic - Improved Aggressive',
        run_quantum_anharmonic_improved_aggressive_strategy_backtest,
        strategy_results,
        token, timeframe, start_date, end_date, improved_aggressive_params
    )
    
    return strategy_results

def create_visualizations(results, output_file='improved_Quantum_anharmonic_strategy/strategy_analysis.png'):
    """Create comprehensive visualizations of the results."""
    print("Creating visualizations...")
    
    # Convert results to a flat DataFrame for easier plotting
    rows = []
    for token, timeframe_data in results.items():
        for timeframe, strategy_data in timeframe_data.items():
            if isinstance(strategy_data, dict) and 'error' not in strategy_data:
                for strategy, metrics in strategy_data.items():
                    # Skip entries with errors
                    if 'error' in metrics:
                        continue
                        
                    # Add market cap category
                    market_cap = "High" if token in HIGH_MARKET_CAP else "Medium" if token in MEDIUM_MARKET_CAP else "Low"
                    
                    rows.append({
                        'token': token,
                        'market_cap': market_cap,
                        'timeframe': timeframe,
                        'strategy': strategy,
                        'total_trades': metrics.get('total_trades', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'total_return': metrics.get('total_return', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'profit_factor': metrics.get('profit_factor', 0),
                        'risk_adjusted_return': metrics.get('risk_adjusted_return', 0)
                    })
    
    if not rows:
        print("No valid results to visualize")
        return
    
    df = pd.DataFrame(rows)
    
    # Create a multi-panel figure
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
    sns.heatmap(win_rate_pivot, annot=True, cmap='RdYlGn', fmt='.1f', ax=ax1)
    ax1.set_title('Average Win Rate (%) by Strategy and Timeframe', fontsize=14)
    
    # 2. Heatmap of returns by strategy and timeframe
    ax2 = plt.subplot(gs[0, 1])
    return_pivot = df.pivot_table(
        values='total_return', 
        index='strategy', 
        columns='timeframe', 
        aggfunc='mean'
    )
    sns.heatmap(return_pivot, annot=True, cmap='RdYlGn', fmt='.1f', ax=ax2)
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
    sns.scatterplot(
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
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
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
    
    # Create market cap analysis
    create_market_cap_analysis(df, 'improved_Quantum_anharmonic_strategy/market_cap_analysis.png')
    
    # Create timeframe analysis
    create_timeframe_analysis(df, 'improved_Quantum_anharmonic_strategy/timeframe_analysis.png')

def create_market_cap_analysis(df, output_file):
    """Create analysis of strategy performance by market cap."""
    plt.figure(figsize=(14, 10))
    
    # Calculate average return by strategy and market cap
    market_cap_pivot = df.pivot_table(
        values='total_return', 
        index='strategy', 
        columns='market_cap', 
        aggfunc='mean'
    )
    
    # Plot as a grouped bar chart
    market_cap_pivot.plot(kind='bar', ax=plt.gca())
    plt.title('Average Return (%) by Strategy and Market Cap', fontsize=16)
    plt.ylabel('Return (%)')
    plt.xlabel('Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Market Cap')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Market cap analysis saved to {output_file}")
    plt.close()

def create_timeframe_analysis(df, output_file):
    """Create analysis of strategy performance by timeframe."""
    plt.figure(figsize=(14, 10))
    
    # Calculate average return by strategy and timeframe
    timeframe_pivot = df.pivot_table(
        values='total_return', 
        index='strategy', 
        columns='timeframe', 
        aggfunc='mean'
    )
    
    # Plot as a grouped bar chart
    timeframe_pivot.plot(kind='bar', ax=plt.gca())
    plt.title('Average Return (%) by Strategy and Timeframe', fontsize=16)
    plt.ylabel('Return (%)')
    plt.xlabel('Strategy')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Timeframe')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Timeframe analysis saved to {output_file}")
    plt.close()

def main():
    """Main function to run the analysis."""
    # Create results directory if it doesn't exist
    os.makedirs('improved_Quantum_anharmonic_strategy', exist_ok=True)
    
    # Select 9 tokens (3 from each market cap category)
    tokens = select_diverse_tokens(3)
    
    # Choose timeframes
    timeframes = ['1m', '5m', '15m', '30m', '1h']
    
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
    
    try:
        # Run the analysis on valid tokens
        results, results_file = run_all_quantum_anharmonic_strategies(valid_tokens, timeframes, start_date, end_date)
        
        # Create visualizations
        create_visualizations(results, f'improved_Quantum_anharmonic_strategy/comprehensive_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        
        print("Analysis complete!")
    except Exception as e:
        print(f"ERROR: Analysis failed with error: {str(e)}")
        traceback.print_exc()
        print("Attempting to save partial results...")
        
        # Try to save whatever results we have
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        try:
            with open(f'improved_Quantum_anharmonic_strategy/analysis_results_PARTIAL_{timestamp}.json', 'w') as f:
                json.dump({"error": str(e), "partial_results": {}}, f, default=str)
            print(f"Partial results saved to improved_Quantum_anharmonic_strategy/analysis_results_PARTIAL_{timestamp}.json")
        except Exception as save_error:
            print(f"Failed to save partial results: {str(save_error)}")

if __name__ == "__main__":
    main() 