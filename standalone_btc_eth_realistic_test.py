#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone Quantum Hybrid Enhanced Strategy Backtesting Script - Realistic BTC/ETH 2024 Test

This script runs tests specifically on the Quantum Hybrid Enhanced strategy
for BTC/USDT and ETH/USDT for the year 2024, incorporating commission and slippage.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random # Keep for potential future use, though not needed for token selection now
import json
import traceback
import pickle
from pathlib import Path
import gc
import warnings
import time
import sys
import signal
import psutil  # For memory monitoring

warnings.filterwarnings("ignore")

# --- Strategy and Data Imports ---
from app import fetch_crypto_data

# Import ONLY the Enhanced Hybrid strategy function and parameters
try:
    # Ensure the strategy file has been updated with commission/slippage logic
    from quantum_hybrid_enhanced import run_quantum_hybrid_enhanced_strategy_backtest, QuantumHybridEnhancedParameters
except ImportError:
    print("ERROR: Could not import 'run_quantum_hybrid_enhanced_strategy_backtest' or 'QuantumHybridEnhancedParameters'.")
    print("Ensure 'quantum_hybrid_enhanced.py' exists, contains these, and includes realism logic.")
    sys.exit(1)

# Fix pandas to avoid SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# --- Constants and Configuration ---
# Define specific tokens for this run
TARGET_TOKENS = ['BTC/USDT', 'ETH/USDT']

# Define token categories by market cap (needed for parameter adjustment)
# Both target tokens are High Cap
HIGH_MARKET_CAP = ['BTC/USDT', 'ETH/USDT']
MEDIUM_MARKET_CAP = []
SMALL_MARKET_CAP = []

DATA_CACHE_DIR = 'data_cache'
# Define specific timeframes for this run
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
# New results directory for this specific test
RESULTS_DIR = "realistic_hybrid_btc_eth_results_2024"
os.makedirs(RESULTS_DIR, exist_ok=True)
STRATEGY_NAME_DISPLAY = "Quantum Hybrid Enhanced (Realistic)" # Updated display name

# Global variable for storing intermediate results for recovery
recovery_results = {}

# --- Helper Functions --- #

def select_specific_tokens():
    """Return the hardcoded list of target tokens for this test."""
    print(f"Using specific tokens for testing: {TARGET_TOKENS}")
    return TARGET_TOKENS

def generate_one_year_period(year=2024):
    """Generate the testing period for the specified year (default 2024)."""
    start_date = datetime(year, 1, 1)
    # Set end_date to current date if testing partial current year, else Dec 31st
    current_date = datetime.now()
    if year == current_date.year:
        end_date = current_date - timedelta(days=1) # Use up to yesterday
    else:
        end_date = datetime(year, 12, 31)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    print(f"Analysis period: {start_date_str} to {end_date_str}")
    return start_date_str, end_date_str

def get_cached_data(token, timeframe, start_date, end_date, original_fetch_func):
    """Get data from cache if available, otherwise fetch using the original function."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    token_safe = token.replace('/', '_')
    cache_file = os.path.join(DATA_CACHE_DIR, f"{token_safe}_{timeframe}_{start_date}_{end_date}.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            # Basic check if loaded data seems valid (e.g., is a DataFrame)
            if isinstance(data, pd.DataFrame) and not data.empty:
                print(f"Using cached data for {token} {timeframe} ({len(data)} candles)")
                return data
            else:
                print(f"Cached data for {token} {timeframe} is invalid/empty. Refetching...")
                # Optionally remove the invalid cache file
                # os.remove(cache_file)
        except Exception as e:
            print(f"Error loading cached data: {str(e)}. Refetching...")
            # Optionally remove the corrupted cache file
            # os.remove(cache_file)

    # Cache miss or invalid cache - Fetch using the provided original function
    print(f"Fetching {token} {timeframe} data using original fetcher...")
    try:
        data = original_fetch_func(token, timeframe, start_date, end_date)
    except Exception as fetch_error:
        print(f"  ERROR during fetch: {fetch_error}")
        return None # Return None explicitly on fetch error

    # Validate fetched data before caching
    if isinstance(data, pd.DataFrame) and not data.empty:
        print(f"  Fetched {len(data)} candles.")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"  Cached {len(data)} candles to {cache_file}")
        except Exception as cache_error:
            print(f"  Warning: Failed to cache data for {token} {timeframe}: {cache_error}")
        return data
    elif data is None:
         print(f"  Fetched no data for {token} {timeframe}.")
         return None
    else:
        # Handle unexpected return types (like bool) from fetcher
        print(f"  Warning: Original fetcher returned unexpected type ({type(data)}), not a DataFrame. Returning None.")
        return None

def patch_fetch_functions():
    """Monkey patch the fetch_crypto_data function to use cached data."""
    import app
    # Ensure app.fetch_crypto_data exists before trying to patch
    if not hasattr(app, 'fetch_crypto_data') or not callable(app.fetch_crypto_data):
        print("ERROR: Cannot patch - 'app.fetch_crypto_data' not found or not callable.")
        # Decide how to handle this: raise error, exit, or return None
        raise ImportError("Failed to find the function to patch.")

    original_fetch = app.fetch_crypto_data
    print(f"Original fetch function type: {type(original_fetch)}")

    # Define the patched function that calls get_cached_data
    def patched_fetch_crypto_data(symbol, timeframe, start_date, end_date=None, exchange_id='binance'):
        # Pass the *original* fetch function to get_cached_data
        return get_cached_data(symbol, timeframe, start_date, end_date, original_fetch)

    # Apply the patch
    app.fetch_crypto_data = patched_fetch_crypto_data
    print("Patched fetch_crypto_data to use cached data wrapper.")
    # Return the original function so it can be restored later
    return original_fetch

# safe_run_strategy remains the same
def safe_run_strategy(strategy_name, strategy_func, strategy_results, *args, **kwargs):
    """Run a strategy with enhanced error handling, memory monitoring, and timeout."""
    print(f"\nRunning {strategy_name}...")
    result_key = f"{args[0]}_{args[1]}" # Token_Timeframe unique key for results dict
    try:
        max_runtime = 900  # Increased timeout slightly for potentially longer 1m runs
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"Initial memory usage: {initial_memory:.1f} MB")

        try:
            gc.collect()
            stats, _, _ = strategy_func(*args, **kwargs)
            if stats is None:
                raise ValueError(f"{strategy_name} returned None stats")

            if time.time() - start_time > max_runtime:
                raise TimeoutError(f"Strategy execution exceeded {max_runtime} seconds")

            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            print(f"Final memory usage: {current_memory:.1f} MB (increase: {memory_increase:.1f} MB)")
            if current_memory > 2000: print(f"WARNING: High memory usage detected: {current_memory:.1f} MB")

            strategy_results[result_key] = extract_key_metrics(stats)
            print(f"Completed {strategy_name} for {result_key} successfully.")

            # Save intermediate checkpoint immediately after successful run
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_token = args[0].replace("/", "_")
                filename = f'{RESULTS_DIR}/checkpoint_{safe_token}_{args[1]}_{timestamp}.json'
                with open(filename, 'w') as f:
                    json.dump({result_key: strategy_results[result_key]}, f, default=str)
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

# extract_key_metrics remains the same
def extract_key_metrics(stats):
    """Extract key metrics from strategy results dict, handling missing keys/errors."""
    if stats is None or not isinstance(stats, dict):
        return {
            'total_trades': 0, 'win_rate': 0.0, 'total_return': 0.0,
            'max_drawdown': 0.0, 'profit_factor': 0.0, 'risk_adjusted_return': 0.0
        }
    try:
        metrics = {}
        metrics['total_trades'] = stats.get('Total Trades', stats.get('total_trades', 0))

        win_rate = stats.get('Win Rate [%]', stats.get('win_rate', 0))
        metrics['win_rate'] = float(str(win_rate).replace('%', '')) if isinstance(win_rate, (str, float, int)) else 0.0

        total_return = stats.get('Return [%]', stats.get('total_return', 0))
        metrics['total_return'] = float(str(total_return).replace('%', '')) if isinstance(total_return, (str, float, int)) else 0.0

        max_drawdown = stats.get('Max Drawdown [%]', stats.get('max_drawdown', 0))
        if isinstance(max_drawdown, str):
             md = float(max_drawdown.replace('%', ''))
        elif isinstance(max_drawdown, (float, int)):
             md = float(max_drawdown)
        else:
            md = 0.0
        # Handle case where drawdown might be exactly 0 after fees, avoid multiplying 0
        metrics['max_drawdown'] = md * 100 if 0 < md <= 1.0 else md

        profit_factor = stats.get('Profit Factor', stats.get('profit_factor', 0))
        pf = float(profit_factor) if isinstance(profit_factor, (str, float, int)) else 0.0
        metrics['profit_factor'] = 999.99 if pf == float('inf') else pf

        # Calculate Risk/Reward using the potentially modified max_drawdown
        metrics['risk_adjusted_return'] = (metrics['total_return'] / metrics['max_drawdown']) if metrics['max_drawdown'] > 0 else 0.0

        return metrics
    except Exception as e:
        print(f"Warning: Error extracting metrics: {e}")
        return {
            'total_trades': 0, 'win_rate': 0.0, 'total_return': 0.0,
            'max_drawdown': 0.0, 'profit_factor': 0.0, 'risk_adjusted_return': 0.0, 'extraction_error': str(e)
        }

# Modified adjust_parameters_for_timeframe
def adjust_parameters_for_timeframe(token, timeframe):
    """
    Adjust parameters for the *Enhanced* Hybrid strategy for BTC/ETH.
    Uses default realistic commission/slippage.
    Avoids potentially overfit 1m omega from previous tests.
    """
    params = QuantumHybridEnhancedParameters() # Use Enhanced parameters (with realism defaults)

    # Set mass multiplier (both are High Cap)
    mass_multiplier = 2.0
    base_mass = 1.0e-30 * mass_multiplier
    params.m = base_mass
    if hasattr(params, 'mass'): params.mass = base_mass

    # Adjust omega frequency based on timeframe
    # *** Use 5m omega value for 1m to test overfitting hypothesis ***
    if timeframe == '1d': params.omega = 5.0e-4
    elif timeframe == '4h': params.omega = 2.5e-4
    elif timeframe == '1h': params.omega = 1.1e-4
    elif timeframe == '30m': params.omega = 0.8e-4
    elif timeframe == '15m': params.omega = 0.6e-4
    elif timeframe == '5m': params.omega = 0.3e-4
    elif timeframe == '1m': params.omega = 0.3e-4 # Using 5m value for 1m test

    print(f"Adjusted Enhanced Params for {token} {timeframe}: mass_mult={mass_multiplier}, omega={params.omega:.1e}, commission={params.commission_pct:.4f}, slippage={params.slippage_pct:.4f}")
    return params

# Renamed run function for clarity
def run_realistic_enhanced_strategy(token, timeframe, start_date, end_date, initial_capital=10000):
    """Run the realistic Quantum Hybrid Enhanced strategy for a single token-timeframe."""
    strategy_results = {} # Dictionary to hold results for this combo
    strategy_name = STRATEGY_NAME_DISPLAY # Use global display name

    # Get parameters adjusted for token/timeframe
    params = adjust_parameters_for_timeframe(token, timeframe)

    # Run the enhanced strategy with realism parameters included in 'params'
    strategy_results = safe_run_strategy(
        strategy_name,
        run_quantum_hybrid_enhanced_strategy_backtest,
        strategy_results,
        token,
        timeframe,
        initial_capital,
        start_date,
        end_date,
        params=params # Pass the params object containing commission/slippage
    )

    result_key = f"{token}_{timeframe}"
    return strategy_results.get(result_key, {'error': 'Strategy run failed or key mismatch'})

# prefetch_and_cache_data remains the same
def prefetch_and_cache_data(tokens, timeframes, start_date, end_date, original_fetch_func):
    """
    Prefetch data using the original fetch function.
    Pass the original_fetch_func explicitly.
    """
    print("Prefetching and caching all required data using original fetcher...")
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
            fetch_start_date = start_date
            fetch_end_date = end_date
            full_year = True
            days_back = None
            if timeframe == '1m':
                 days_back = 30 # e.g., 30 days for 1m
            elif timeframe in ['5m', '15m', '30m']:
                 days_back = 60 # 60 days for 5m, 15m, 30m

            if days_back:
                full_year = False
                try:
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    adjusted_start_dt = end_dt - timedelta(days=days_back)
                    fetch_start_date = adjusted_start_dt.strftime('%Y-%m-%d')
                    print(f"  Using adjusted period ({days_back} days) for {timeframe}: {fetch_start_date} to {fetch_end_date}")
                except ValueError:
                     print(f" Error parsing end_date {end_date}, using full period.")
                     full_year=True # Revert if error
                     fetch_start_date=start_date

            cache_file = os.path.join(DATA_CACHE_DIR, f"{token_safe}_{timeframe}_{fetch_start_date}_{fetch_end_date}.pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f: data = pickle.load(f)
                    candle_count = len(data) if data is not None else 0
                    print(f"  {token} {timeframe}: ✅ (Using cached data: {candle_count} candles)")
                    availability[token][timeframe] = {"available": candle_count > 100, "candle_count": candle_count, "cache_file": cache_file, "start_date": fetch_start_date, "end_date": fetch_end_date, "full_year": full_year}
                    continue
                except Exception as e: print(f"  Error loading cache: {e}")
            try:
                print(f"  Fetching {token} {timeframe} data from {fetch_start_date} to {fetch_end_date}...")
                # *** Use the passed original_fetch_func directly ***
                data = original_fetch_func(token, timeframe, fetch_start_date, fetch_end_date)
                candle_count = len(data) if isinstance(data, pd.DataFrame) else 0
                is_available = candle_count > 100
                if is_available:
                    # Cache the data fetched by the original function
                    with open(cache_file, 'wb') as f: pickle.dump(data, f)
                    print(f"  {token} {timeframe}: ✅ ({candle_count} candles, cached)")
                else: print(f"  {token} {timeframe}: ❌ (Insufficient data: {candle_count})")
                availability[token][timeframe] = {"available": is_available, "candle_count": candle_count, "cache_file": cache_file if is_available else None, "start_date": fetch_start_date, "end_date": fetch_end_date, "full_year": full_year}
            except Exception as e:
                print(f"  {token} {timeframe}: ❌ Error fetching: {e}")
                availability[token][timeframe] = {"available": False, "error": str(e), "start_date": fetch_start_date, "end_date": fetch_end_date, "full_year": full_year}
    print("Prefetching complete!")
    return availability

# run_all_tests function modified for new context
def run_all_tests():
    """Run the realistic Quantum Hybrid Enhanced strategy on BTC/ETH for 2024."""
    start_time_total = time.time()
    print(f"Starting {STRATEGY_NAME_DISPLAY} backtest run for 2024...")
    print("-" * 70)

    # Setup
    tokens = select_specific_tokens()
    start_date, end_date = generate_one_year_period(year=2024) # Set year to 2024
    initial_capital = 10000.0
    all_results = {} # Store results keyed by token, then timeframe

    original_fetch = None # Initialize
    try:
        original_fetch = patch_fetch_functions() # Get the original function while patching

        # Signal handling remains the same
        global recovery_results
        def signal_handler(sig, frame):
            print(f"\nReceived signal {sig}, performing graceful shutdown...")
            try:
                if recovery_results:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    recovery_file = f'{RESULTS_DIR}/interrupted_results_{timestamp}.json'
                    with open(recovery_file, 'w') as f: json.dump(recovery_results, f, default=str)
                    print(f"Saved interrupted results to {recovery_file}")
            except Exception as e: print(f"Failed to save recovery data: {e}")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Prefetch data using the original function
            print("\nPrefetching data for 2024...")
            # *** Pass original_fetch to prefetch ***
            availability = prefetch_and_cache_data(tokens, TIMEFRAMES, start_date, end_date, original_fetch)

            # The patch applied by patch_fetch_functions is now active for the main test loop

            valid_tests = []
            for token, timeframe_data in availability.items():
                for timeframe, data_info in timeframe_data.items():
                    if data_info.get('available', False):
                        valid_tests.append((token, timeframe, data_info['start_date'], data_info['end_date']))

            if not valid_tests:
                print("Error: No valid data available for any selected token/timeframe.")
                return {}, None

            print(f"\nStarting tests on {len(valid_tests)} valid token/timeframe combinations for 2024...")
            total_tests = len(valid_tests)

            # Main testing loop
            for i, (token, timeframe, tf_start_date, tf_end_date) in enumerate(valid_tests):
                print(f"\n----- Test {i+1}/{total_tests}: {token} on {timeframe} (Period: {tf_start_date} to {tf_end_date}) ----- ")
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024
                print(f"Memory before test: {mem_before:.1f} MB")
                if mem_before > 2500: # Increased threshold slightly
                     print("High memory usage. Forcing GC...")
                     gc.collect()
                     time.sleep(3)

                # Run the single strategy using the renamed function
                result = run_realistic_enhanced_strategy(
                    token, timeframe, tf_start_date, tf_end_date, initial_capital
                )

                # Store result correctly
                if token not in all_results: all_results[token] = {}
                all_results[token][timeframe] = result # Store the direct result for this combo

                recovery_results = all_results # Update recovery data
                time.sleep(1)
                gc.collect()

            # Final results saving and processing
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            final_filename = f'{RESULTS_DIR}/analysis_results_{timestamp}.json'
            print(f"\nSaving final 2024 results to {final_filename}...")
            with open(final_filename, 'w') as f:
                json.dump(all_results, f, default=str)

            # Create Summary
            summary_df = create_summary_table(all_results, STRATEGY_NAME_DISPLAY)
            summary_filename = None
            if summary_df is not None and not summary_df.empty:
                summary_filename = f'{RESULTS_DIR}/summary_{timestamp}.csv'
                try:
                    summary_df.to_csv(summary_filename, index=False)
                    print(f"\nSummary table saved to {summary_filename}")
                    # Instruct user to run generate_visualization.py
                    print(f"\nTo visualize the 2024 results, run:\npython generate_visualization.py {summary_filename}")
                except Exception as e:
                     print(f"Error saving summary CSV: {e}")
                     summary_filename = None # Reset on error
            else:
                print("Skipping summary CSV generation due to empty results.")

            total_time = time.time() - start_time_total
            print(f"\nAnalysis complete! Total time: {total_time:.2f} seconds.")
            return all_results, final_filename # Return JSON filename

        except Exception as e:
            print(f"\nERROR: Analysis failed with unhandled error: {e}")
            traceback.print_exc()
            print("Attempting to save partial results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                partial_file = f'{RESULTS_DIR}/partial_results_{timestamp}.json'
                with open(partial_file, 'w') as f: json.dump({"error": str(e), "partial_results": recovery_results}, f, default=str)
                print(f"Partial results saved to {partial_file}")
            except Exception as save_error: print(f"Failed to save partial results: {save_error}")
            return recovery_results, None
    finally:
        # Restore original function if it was successfully retrieved
        if original_fetch is not None:
            try:
                import app
                app.fetch_crypto_data = original_fetch
                print("Restored original fetch_crypto_data function.")
            except Exception as restore_error:
                print(f"Warning: Failed to restore original fetch function: {restore_error}")
        else:
            print("Warning: Original fetch function was not captured, cannot restore.")

# create_summary_table remains the same
def create_summary_table(results_dict, strategy_name):
    """Create a summary DataFrame for the single strategy results."""
    rows = []
    if not isinstance(results_dict, dict): return None

    for token, timeframe_data in results_dict.items():
         if not isinstance(timeframe_data, dict): continue
         for timeframe, metrics in timeframe_data.items():
            if not isinstance(metrics, dict) or 'error' in metrics:
                 continue # Skip failed or errored runs

            # Determine market cap based on the reduced list
            market_cap = "High" if token in HIGH_MARKET_CAP else "Unknown"

            rows.append({
                'Token': token,
                'Market Cap': market_cap,
                'Timeframe': timeframe,
                'Strategy': strategy_name,
                'Total Trades': metrics.get('total_trades', 0),
                'Win Rate [%]': metrics.get('win_rate', 0.0),
                'Return [%]': metrics.get('total_return', 0.0),
                'Max Drawdown [%]': metrics.get('max_drawdown', 0.0),
                'Profit Factor': metrics.get('profit_factor', 0.0),
                'Risk/Reward': metrics.get('risk_adjusted_return', 0.0)
            })

    if not rows: return None
    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # Save to the correct directory
    summary_filepath = Path(RESULTS_DIR) / f'summary_{timestamp}.csv'
    try:
        df.to_csv(summary_filepath, index=False)
        print(f"Summary table saved to {summary_filepath}")
    except Exception as e: print(f"Error saving summary CSV: {e}")
    return df

# --- Script Entry Point ---
if __name__ == "__main__":
    run_all_tests() 