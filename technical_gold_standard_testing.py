#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone Testing Script for Technical Analysis Gold Standard Strategy

This script runs tests on the Technical Analysis Gold Standard strategy 
across multiple tokens and timeframes.
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
import warnings
import time
import sys
import signal
import psutil  # For memory monitoring

warnings.filterwarnings("ignore")

# --- Strategy and Data Imports ---
# Import the new strategy function and parameters
from technical_analysis_gold_standard import (
    run_technical_gold_standard_strategy_backtest, 
    TechnicalGoldStandardParameters
)
# Import data fetching function
from app import fetch_crypto_data

# Fix pandas to avoid SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# --- Constants and Configuration ---
# Define token categories by market cap (adjust as needed)
HIGH_MARKET_CAP = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT']
MEDIUM_MARKET_CAP = ['LINK/USDT', 'DOT/USDT', 'AVAX/USDT', 'ATOM/USDT', 'NEAR/USDT', 'MATIC/USDT', 'UNI/USDT', 'LTC/USDT']
SMALL_MARKET_CAP = ['ALGO/USDT', 'FTM/USDT', 'ONE/USDT', 'ZIL/USDT', 'AAVE/USDT', 'ETC/USDT', 'KSM/USDT', 'CHZ/USDT']

DATA_CACHE_DIR = 'data_cache'  # Directory for caching historical data
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d'] # Timeframes to test
RESULTS_DIR = "technical_gold_standard_results" # Directory for saving results
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global variable for storing intermediate results for recovery
recovery_results = {}

# --- Helper Functions (Adapted from comprehensive script) ---

def select_diverse_tokens(num_high=4, num_medium=3, num_small=3):
    """Select a diverse set of tokens from different market cap categories."""
    selected_tokens = []
    try:
        selected_tokens.extend(random.sample(HIGH_MARKET_CAP, min(num_high, len(HIGH_MARKET_CAP))))
        selected_tokens.extend(random.sample(MEDIUM_MARKET_CAP, min(num_medium, len(MEDIUM_MARKET_CAP))))
        selected_tokens.extend(random.sample(SMALL_MARKET_CAP, min(num_small, len(SMALL_MARKET_CAP))))
    except ValueError as e:
        print(f"Warning: Could not select requested number of tokens. {e}")
        # Fallback: Select all available tokens if sampling fails
        selected_tokens = list(set(HIGH_MARKET_CAP + MEDIUM_MARKET_CAP + SMALL_MARKET_CAP))

    # Print selection summary
    print("Selected tokens for testing:")
    print(f"  Total: {len(selected_tokens)}")
    print(f"  High Market Cap: {[t for t in selected_tokens if t in HIGH_MARKET_CAP]}")
    print(f"  Medium Market Cap: {[t for t in selected_tokens if t in MEDIUM_MARKET_CAP]}")
    print(f"  Low Market Cap: {[t for t in selected_tokens if t in SMALL_MARKET_CAP]}")
    
    return list(set(selected_tokens)) # Ensure uniqueness

def generate_random_month_period(min_year=2023, max_year=2024):
    """Generate a random 1-month period within the specified year range."""
    current_time = datetime.now()
    
    # Adjust max_year and potentially max_month if current year is selected
    effective_max_year = min(max_year, current_time.year)
    if min_year > effective_max_year:
        min_year = effective_max_year # Ensure min_year is not in the future
        
    year = random.randint(min_year, effective_max_year)
    
    max_month = 12
    if year == current_time.year:
        max_month = current_time.month 
        if max_month == 1: # Avoid getting stuck if it's January 1st
             year -= 1
             max_month = 12
             if year < min_year:
                 print("Warning: Cannot select a valid past month. Using default.")
                 year = 2023
                 month = 1
             else:
                month = random.randint(1, max_month)
        else:
            # Select month from 1 up to the previous month to ensure a full month is available
             month = random.randint(1, max_month -1 if max_month > 1 else 1) 
    else:
        month = random.randint(1, 12)

    start_date = datetime(year, month, 1)
    # Calculate end date (last day of the month)
    next_month = start_date.replace(day=28) + timedelta(days=4) # Go to approx next month
    end_date = next_month - timedelta(days=next_month.day) # Go to last day of current month

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    print(f"Analysis period: {start_date_str} to {end_date_str}")
    return start_date_str, end_date_str

def get_cached_data(token, timeframe, start_date, end_date):
    """Get data from cache if available, otherwise fetch and cache it."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    token_safe = token.replace('/', '_')
    cache_file = os.path.join(DATA_CACHE_DIR, f"{token_safe}_{timeframe}_{start_date}_{end_date}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Using cached data for {token} {timeframe} ({len(data)} candles)")
            return data
        except Exception as e:
            print(f"Warning: Error loading cached data: {str(e)}. Refetching...")
    
    print(f"Fetching {token} {timeframe} data ({start_date} to {end_date})...")
    data = fetch_crypto_data(token, timeframe, start_date, end_date)
    
    if data is not None and not data.empty:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Cached {len(data)} candles to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to cache data for {token} {timeframe}: {e}")
            
    return data

def patch_fetch_functions():
    """Monkey patch the fetch_crypto_data function to use cached data."""
    import app # Ensure app is importable
    original_fetch = app.fetch_crypto_data
    def patched_fetch_crypto_data(symbol, timeframe, start_date, end_date=None, exchange_id='binance'):
        return get_cached_data(symbol, timeframe, start_date, end_date)
    app.fetch_crypto_data = patched_fetch_crypto_data
    print("Patched fetch_crypto_data to use cached data.")
    return original_fetch

def prefetch_and_cache_data(tokens, timeframes, start_date, end_date):
    """Prefetch data for all token-timeframe combinations and cache it locally."""
    print("Prefetching and caching all required data...")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    availability = {}
    total_combinations = len(tokens) * len(timeframes)
    completed = 0

    for token in tokens:
        token_safe = token.replace('/', '_')
        availability[token] = {}
        for timeframe in timeframes:
            completed += 1
            progress = (completed / total_combinations) * 100
            print(f"Progress: {progress:.1f}% - Processing {token} {timeframe} ({completed}/{total_combinations})")

            # Use the generated date range directly for all timeframes in this test setup
            fetch_start_date = start_date
            fetch_end_date = end_date
            
            cache_file = os.path.join(DATA_CACHE_DIR, f"{token_safe}_{timeframe}_{fetch_start_date}_{fetch_end_date}.pkl")

            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f: data = pickle.load(f)
                    candle_count = len(data) if data is not None else 0
                    print(f"  {token} {timeframe}: ✅ (Using cached data: {candle_count} candles)")
                    availability[token][timeframe] = {"available": candle_count > 100, "candle_count": candle_count, "cache_file": cache_file, "start_date": fetch_start_date, "end_date": fetch_end_date}
                    continue
                except Exception as e:
                    print(f"  Warning: Error loading cache for {token} {timeframe}: {e}. Refetching...")
            
            try:
                print(f"  Fetching {token} {timeframe} data from {fetch_start_date} to {fetch_end_date}...")
                # Use the original fetch function for prefetching
                data = fetch_crypto_data(token, timeframe, fetch_start_date, fetch_end_date)
                candle_count = len(data) if data is not None else 0
                
                min_candles = 100 # Minimum required candles for a valid test
                is_available = candle_count >= min_candles

                if is_available:
                    with open(cache_file, 'wb') as f: pickle.dump(data, f)
                    print(f"  {token} {timeframe}: ✅ ({candle_count} candles, cached to {cache_file})")
                else:
                    print(f"  {token} {timeframe}: ❌ (Insufficient data: {candle_count} candles, needed {min_candles})")
                
                availability[token][timeframe] = {"available": is_available, "candle_count": candle_count, "cache_file": cache_file if is_available else None, "start_date": fetch_start_date, "end_date": fetch_end_date}
            
            except Exception as e:
                print(f"  {token} {timeframe}: ❌ Error during fetch: {str(e)}")
                availability[token][timeframe] = {"available": False, "error": str(e), "start_date": fetch_start_date, "end_date": fetch_end_date}
            
            time.sleep(0.5) # Small delay to avoid potential API rate limits during prefetch

    print("Prefetching complete!")
    return availability

def safe_run_strategy(strategy_name, strategy_func, strategy_results, *args, **kwargs):
    """Run a strategy with enhanced error handling, memory monitoring, and timeout."""
    print(f"\nRunning {strategy_name}...")
    result_key = f"{args[0]}_{args[1]}" # Token_Timeframe
    try:
        max_runtime = 600  # 10 minutes max per strategy run
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        try:
            gc.collect()
            stats, _ = strategy_func(*args, **kwargs)
            if stats is None:
                raise ValueError(f"{strategy_name} returned None stats")

            if time.time() - start_time > max_runtime:
                raise TimeoutError(f"Strategy execution exceeded {max_runtime} seconds")

            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            print(f"Final memory usage: {current_memory:.1f} MB (increase: {memory_increase:.1f} MB)")
            if current_memory > 1500: print(f"WARNING: High memory usage detected: {current_memory:.1f} MB")

            strategy_results[result_key] = extract_key_metrics(stats)
            print(f"Completed {strategy_name} for {result_key} successfully.")
            
            # Save intermediate checkpoint immediately after successful run
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_token = args[0].replace("/", "_")
                filename = f'{RESULTS_DIR}/checkpoint_{safe_token}_{args[1]}_{timestamp}.json'
                with open(filename, 'w') as f:
                    json.dump({result_key: strategy_results[result_key]}, f, default=str)
                # print(f"Saved intermediate results checkpoint to {filename}") # Can be verbose
            except Exception as save_error:
                print(f"Warning: Could not save intermediate results checkpoint: {save_error}")

        except TimeoutError as te:
            print(f"Timeout running {strategy_name} for {result_key}: {str(te)}")
            strategy_results[result_key] = {'error': f'Timeout: {str(te)}', **extract_key_metrics(None)}
        
    except Exception as e:
        print(f"Error in {strategy_name} for {result_key}: {str(e)}")
        traceback.print_exc()
        strategy_results[result_key] = {'error': str(e), **extract_key_metrics(None)}
    
    gc.collect()
    return strategy_results

def extract_key_metrics(stats):
    """Extract key metrics from strategy results dict, handling missing keys/errors."""
    if stats is None or not isinstance(stats, dict):
        return {
            'total_trades': 0, 'win_rate': 0.0, 'total_return': 0.0,
            'max_drawdown': 0.0, 'profit_factor': 0.0, 'risk_adjusted_return': 0.0
        }
    try:
        metrics = {}
        metrics['total_trades'] = stats.get('Total Trades', 0)
        
        win_rate = stats.get('Win Rate [%]', 0.0)
        metrics['win_rate'] = float(str(win_rate).replace('%', '')) if isinstance(win_rate, (str, float, int)) else 0.0
        
        total_return = stats.get('Total Return [%]', 0.0)
        metrics['total_return'] = float(str(total_return).replace('%', '')) if isinstance(total_return, (str, float, int)) else 0.0

        max_drawdown = stats.get('Max Drawdown [%]', 0.0)
        md = float(str(max_drawdown).replace('%', '')) if isinstance(max_drawdown, (str, float, int)) else 0.0
        metrics['max_drawdown'] = md * 100 if 0 < md <= 1.0 else md # Ensure percentage
        
        profit_factor = stats.get('Profit Factor', 0.0)
        pf = float(profit_factor) if isinstance(profit_factor, (str, float, int)) else 0.0
        metrics['profit_factor'] = 999.99 if pf == float('inf') else pf

        metrics['risk_adjusted_return'] = (metrics['total_return'] / metrics['max_drawdown']) if metrics['max_drawdown'] > 0 else 0.0
        
        return metrics
    except Exception as e:
        print(f"Warning: Error extracting metrics: {e}")
        return {
            'total_trades': 0, 'win_rate': 0.0, 'total_return': 0.0,
            'max_drawdown': 0.0, 'profit_factor': 0.0, 'risk_adjusted_return': 0.0, 'extraction_error': str(e)
        }

# --- Main Execution Logic ---

def run_all_tests():
    """Run the Technical Analysis Gold Standard strategy on selected tokens/timeframes."""
    start_time_total = time.time()
    print("Starting Technical Analysis Gold Standard Strategy backtest run...")
    print("-" * 70)

    # Setup
    tokens = select_diverse_tokens()
    start_date, end_date = generate_random_month_period()
    initial_capital = 10000.0
    all_results = {} # Store results keyed by 'Token_Timeframe'
    strategy_name = "Technical Gold Standard" # Single strategy name

    original_fetch = patch_fetch_functions()
    
    global recovery_results
    def signal_handler(sig, frame):
        print(f"\nReceived signal {sig}, performing graceful shutdown...")
        try:
            if recovery_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                recovery_file = f'{RESULTS_DIR}/interrupted_results_{timestamp}.json'
                with open(recovery_file, 'w') as f: json.dump(recovery_results, f, default=str)
                print(f"Saved interrupted results to {recovery_file}")
        except Exception as e:
            print(f"Failed to save recovery data: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Prefetch data
        # Use original fetch for prefetching before patching
        import app
        app.fetch_crypto_data = original_fetch # Temporarily restore for prefetch
        print("\nPrefetching data...")
        availability = prefetch_and_cache_data(tokens, TIMEFRAMES, start_date, end_date)
        app.fetch_crypto_data = patch_fetch_functions() # Re-apply patch
        
        valid_tests = []
        for token, timeframe_data in availability.items():
            for timeframe, data_info in timeframe_data.items():
                if data_info.get('available', False):
                    valid_tests.append((token, timeframe, data_info['start_date'], data_info['end_date']))
        
        if not valid_tests:
            print("Error: No valid data available for any selected token/timeframe combination.")
            return {}, None

        print(f"\nStarting tests on {len(valid_tests)} valid token/timeframe combinations...")
        total_tests = len(valid_tests)
        
        # Main testing loop
        for i, (token, timeframe, tf_start_date, tf_end_date) in enumerate(valid_tests):
            print(f"\n----- Test {i+1}/{total_tests}: {token} on {timeframe} ----- ")
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024
            print(f"Memory before test: {mem_before:.1f} MB")
            if mem_before > 2000:
                 print("High memory usage detected before test. Forcing GC...")
                 gc.collect()
                 time.sleep(3)

            # Run the single strategy
            all_results = safe_run_strategy(
                strategy_name,
                run_technical_gold_standard_strategy_backtest,
                all_results,
                token, 
                timeframe, 
                initial_capital, 
                tf_start_date, 
                tf_end_date,
                params=TechnicalGoldStandardParameters(), # Use default params for now
                plot=False # Keep plotting off for batch runs
            )
            recovery_results = all_results # Update recovery data
            time.sleep(1) # Small pause
            gc.collect() # Clean up memory

        # Final results saving and processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        final_filename = f'{RESULTS_DIR}/analysis_results_{timestamp}.json'
        print(f"\nSaving final results to {final_filename}...")
        with open(final_filename, 'w') as f:
            json.dump(all_results, f, default=str)

        # Create Summary and Visualizations
        summary_df = create_summary_table(all_results)
        if summary_df is not None and not summary_df.empty:
            create_visualizations(summary_df, timestamp)
        else:
            print("Skipping visualization generation due to empty summary.")

        total_time = time.time() - start_time_total
        print(f"\nAnalysis complete! Total time: {total_time:.2f} seconds.")
        return all_results, final_filename

    except Exception as e:
        print(f"\nERROR: Analysis failed with unhandled error: {e}")
        traceback.print_exc()
        print("Attempting to save partial results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            partial_file = f'{RESULTS_DIR}/partial_results_{timestamp}.json'
            with open(partial_file, 'w') as f:
                json.dump({"error": str(e), "partial_results": recovery_results}, f, default=str)
            print(f"Partial results saved to {partial_file}")
        except Exception as save_error:
            print(f"Failed to save partial results: {save_error}")
        return recovery_results, None
    finally:
        # Ensure original function is restored
        try:
            import app
            app.fetch_crypto_data = original_fetch
            print("Restored original fetch_crypto_data function.")
        except Exception as restore_error:
            print(f"Warning: Failed to restore fetch_crypto_data: {restore_error}")

def create_summary_table(results_dict):
    """Create a summary table DataFrame from the results dictionary."""
    rows = []
    if not isinstance(results_dict, dict):
        print("Error: create_summary_table expects a dictionary.")
        return None
        
    for key, metrics in results_dict.items():
        if 'error' in metrics or not isinstance(metrics, dict):
            continue # Skip failed runs
        
        try:
            token, timeframe = key.split('_') # Assumes key format 'TOKEN/USDT_TIMEFRAME'
        except ValueError:
            print(f"Warning: Skipping result with unexpected key format: {key}")
            continue
        
        # Determine Market Cap
        market_cap = "High" if token in HIGH_MARKET_CAP else "Medium" if token in MEDIUM_MARKET_CAP else "Low"
        
        rows.append({
            'Token': token,
            'Market Cap': market_cap,
            'Timeframe': timeframe,
            'Strategy': 'Technical Gold Standard', # Only one strategy
            'Total Trades': metrics.get('total_trades', 0),
            'Win Rate [%]': metrics.get('win_rate', 0.0),
            'Return [%]': metrics.get('total_return', 0.0),
            'Max Drawdown [%]': metrics.get('max_drawdown', 0.0),
            'Profit Factor': metrics.get('profit_factor', 0.0),
            'Risk/Reward': metrics.get('risk_adjusted_return', 0.0)
        })

    if not rows:
        print("No valid results found to create summary table.")
        return None
        
    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    try:
        df.to_csv(f'{RESULTS_DIR}/summary_{timestamp}.csv', index=False)
        print(f"Summary table saved to {RESULTS_DIR}/summary_{timestamp}.csv")
    except Exception as e:
        print(f"Error saving summary CSV: {e}")
        
    return df

def create_visualizations(summary_df, timestamp):
    """Create visualizations adapted for a single strategy analysis."""
    if summary_df is None or summary_df.empty:
        print("Cannot create visualizations with empty summary data.")
        return
        
    print("\nCreating visualizations...")
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    
    try:
        plt.figure(figsize=(20, 24))
        # Increased grid size for more plots
        gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.5]) 
        gs.update(hspace=0.4, wspace=0.3) # Adjust spacing

        # --- Row 1: Heatmaps (Token vs Timeframe) ---
        ax1 = plt.subplot(gs[0, 0])
        try:
            win_rate_pivot = summary_df.pivot_table(values='Win Rate [%]', index='Token', columns='Timeframe', aggfunc='mean')
            sns.heatmap(win_rate_pivot.fillna(0), annot=True, cmap='viridis', fmt='.1f', ax=ax1, cbar=True, linewidths=.5)
            ax1.set_title('Average Win Rate (%) by Token and Timeframe', fontsize=14)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error creating Win Rate heatmap:\n{e}', ha='center', va='center')
            print(f"Error creating Win Rate heatmap: {e}")

        ax2 = plt.subplot(gs[0, 1])
        try:
            return_pivot = summary_df.pivot_table(values='Return [%]', index='Token', columns='Timeframe', aggfunc='mean')
            sns.heatmap(return_pivot.fillna(0), annot=True, cmap='RdYlGn', fmt='.1f', ax=ax2, cbar=True, linewidths=.5)
            ax2.set_title('Average Return (%) by Token and Timeframe', fontsize=14)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.setp(ax2.yaxis.get_majorticklabels(), rotation=0)
        except Exception as e:
            ax2.text(0.5, 0.5, f'Error creating Return heatmap:\n{e}', ha='center', va='center')
            print(f"Error creating Return heatmap: {e}")

        # --- Row 2: Box Plots (Distribution across Tests) ---
        ax3 = plt.subplot(gs[1, 0])
        try:
            sns.boxplot(y='Return [%]', data=summary_df, ax=ax3, palette="coolwarm")
            ax3.set_title('Distribution of Returns (%) Across All Tests', fontsize=14)
            ax3.set_ylabel('Return (%)')
            # Define strategy_name for the label
            strategy_name = summary_df['Strategy'].iloc[0] if not summary_df.empty else 'Strategy'
            ax3.set_xlabel(strategy_name) # Label x-axis with strategy name
        except Exception as e:
            ax3.text(0.5, 0.5, f'Error creating Return boxplot:\n{e}', ha='center', va='center')
            print(f"Error creating Return boxplot: {e}")

        ax4 = plt.subplot(gs[1, 1])
        try:
            sns.boxplot(y='Win Rate [%]', data=summary_df, ax=ax4, palette="viridis")
            ax4.set_title('Distribution of Win Rates (%) Across All Tests', fontsize=14)
            ax4.set_ylabel('Win Rate (%)')
            # Use the defined strategy_name
            ax4.set_xlabel(strategy_name)
        except Exception as e:
            ax4.text(0.5, 0.5, f'Error creating Win Rate boxplot:\n{e}', ha='center', va='center')
            print(f"Error creating Win Rate boxplot: {e}")

        # --- Row 3: Scatter & Bar ---
        ax5 = plt.subplot(gs[2, 0])
        try:
            # Ensure numeric types for scatter plot
            summary_df['Win Rate [%]'] = pd.to_numeric(summary_df['Win Rate [%]'], errors='coerce')
            summary_df['Return [%]'] = pd.to_numeric(summary_df['Return [%]'], errors='coerce')
            summary_df['Total Trades'] = pd.to_numeric(summary_df['Total Trades'], errors='coerce').fillna(0)
            
            scatter = sns.scatterplot(
                x='Win Rate [%]', y='Return [%]', hue='Timeframe', 
                size='Total Trades', sizes=(30, 300), alpha=0.7,
                data=summary_df, ax=ax5, palette='tab10'
            )
            ax5.set_title('Win Rate vs Return (Color=Timeframe, Size=Trades)', fontsize=14)
            ax5.set_xlabel('Win Rate (%)')
            ax5.set_ylabel('Return (%)')
            ax5.axhline(0, color='grey', linestyle='--', lw=1)
            ax5.axvline(50, color='grey', linestyle='--', lw=1) # Mark 50% win rate
            ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Timeframe / Trades')
        except Exception as e:
            ax5.text(0.5, 0.5, f'Error creating Scatter plot:\n{e}', ha='center', va='center')
            print(f"Error creating Scatter plot: {e}")

        ax6 = plt.subplot(gs[2, 1])
        try:
            # Average Risk/Reward by Timeframe
            rr_pivot = summary_df.pivot_table(values='Risk/Reward', index='Timeframe', aggfunc='mean').sort_values('Risk/Reward', ascending=False)
            sns.barplot(x=rr_pivot.index, y='Risk/Reward', data=rr_pivot.reset_index(), ax=ax6, palette='coolwarm')
            ax6.set_title('Average Risk-Adjusted Return by Timeframe', fontsize=14)
            ax6.set_ylabel('Return/Drawdown Ratio')
            ax6.set_xlabel('Timeframe')
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        except Exception as e:
             ax6.text(0.5, 0.5, f'Error creating R/R Bar plot:\n{e}', ha='center', va='center')
             print(f"Error creating R/R Bar plot: {e}")

        # --- Row 4: Overall Performance Summary Table ---
        ax7 = plt.subplot(gs[3, :])
        try:
            # Calculate overall average metrics
            overall_summary = summary_df.agg({
                'Total Trades': 'sum',
                'Win Rate [%]': 'mean',
                'Return [%]': 'mean',
                'Max Drawdown [%]': 'mean',
                'Profit Factor': 'mean', # Need careful aggregation for PF (e.g., from total profit/loss)
                'Risk/Reward': 'mean' 
            })
            # Recalculate overall profit factor from sums if possible
            # This requires access to individual trade data, which we don't have here.
            # Mean of profit factors might be misleading.
            
            profitable_tests_count = (summary_df['Return [%]'] > 0).sum()
            total_tests_count = len(summary_df)
            profitable_pct = (profitable_tests_count / total_tests_count * 100) if total_tests_count > 0 else 0

            table_data = [[
                f"{overall_summary['Win Rate [%]']:.1f}%",
                f"{overall_summary['Return [%]']:.1f}%",
                f"{overall_summary['Max Drawdown [%]']:.1f}%",
                f"{overall_summary['Profit Factor']:.2f}",
                f"{overall_summary['Risk/Reward']:.2f}",
                f"{profitable_tests_count}/{total_tests_count} ({profitable_pct:.1f}%)",
                f"{int(overall_summary['Total Trades'])}"
            ]]
            
            header = ['Avg Win Rate', 'Avg Return', 'Avg Max DD', 'Avg Profit Factor', 'Avg Risk/Reward', 'Profitable Tests', 'Total Trades']
            table = ax7.table(cellText=table_data, colLabels=header, cellLoc='center', loc='center', colWidths=[0.12]*6 + [0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.8)
            
             # Color coding can be added here based on overall_summary values
            # Example: Color the Avg Return cell
            return_val = overall_summary['Return [%]']
            if return_val > 5: cell_color = 'lightgreen'
            elif return_val > 0: cell_color = 'lightyellow'
            else: cell_color = 'salmon'
            table[(1, 1)].set_facecolor(cell_color) # Row 1 (after header), Col 1 (Avg Return)

            # Use the defined strategy_name
            ax7.set_title(f'Overall Performance Summary - {strategy_name}', fontsize=16, pad=20)
            ax7.axis('off')
        except Exception as e:
            ax7.text(0.5, 0.5, f'Error creating Summary Table:\n{e}', ha='center', va='center')
            print(f"Error creating Summary Table: {e}")

        # Final adjustments and save
        plt.suptitle(f'Technical Analysis Gold Standard - Strategy Performance Analysis ({timestamp})', fontsize=18, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
        
        output_file = f'{RESULTS_DIR}/ta_gold_standard_analysis_{timestamp}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to {output_file}")
        plt.close()
    except Exception as e:
        print(f"\nError during visualization creation: {e}")
        traceback.print_exc()
        plt.close() # Ensure plot is closed even on error

# --- Script Entry Point ---
if __name__ == "__main__":
    run_all_tests() 