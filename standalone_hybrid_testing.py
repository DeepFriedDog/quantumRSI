#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone Quantum Hybrid Enhanced Strategy Backtesting Script

This script runs tests specifically on the Quantum Hybrid Enhanced strategy
across multiple timeframes for BTC, ETH, MATIC, and SOL.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
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
# Assuming the function exists in quantum_hybrid_enhanced.py
try:
    from quantum_hybrid_enhanced import run_quantum_hybrid_enhanced_strategy_backtest, QuantumHybridEnhancedParameters
except ImportError:
    print("ERROR: Could not import 'run_quantum_hybrid_enhanced_strategy_backtest' or 'QuantumHybridEnhancedParameters'.")
    print("Ensure 'quantum_hybrid_enhanced.py' exists and contains these.")
    sys.exit(1)

# Fix pandas to avoid SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# --- Constants and Configuration ---
# Define specific tokens for this run
TARGET_TOKENS = ['BTC/USDT', 'ETH/USDT', 'MATIC/USDT', 'SOL/USDT']

# Define token categories by market cap (needed for parameter adjustment)
# Ensure TARGET_TOKENS are categorized correctly
HIGH_MARKET_CAP = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'] # Added SOL
MEDIUM_MARKET_CAP = ['MATIC/USDT'] # Only MATIC from target list
SMALL_MARKET_CAP = [] # None for this run

DATA_CACHE_DIR = 'data_cache'
# Define specific timeframes for this run
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
RESULTS_DIR = "enhanced_hybrid_select_tokens_results" # Unique results directory
os.makedirs(RESULTS_DIR, exist_ok=True)
STRATEGY_NAME_DISPLAY = "Quantum Hybrid Enhanced" # For outputs

# Global variable for storing intermediate results for recovery
recovery_results = {}

# --- Helper Functions (Mostly copied, some adapted) ---

def select_specific_tokens():
    """Return the hardcoded list of target tokens."""
    print("Using specific tokens for testing:")
    print(f"  {TARGET_TOKENS}")
    return TARGET_TOKENS

def generate_one_year_period(year=2023):
    """Generate a one-year testing period (same as comprehensive)."""
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    current_date = datetime.now()
    if end_date > current_date: end_date = current_date
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
            with open(cache_file, 'rb') as f: data = pickle.load(f)
            print(f"Using cached data for {token} {timeframe} ({len(data)} candles)")
            return data
        except Exception as e: print(f"Error loading cached data: {str(e)}")
    print(f"Fetching {token} {timeframe} data...")
    data = fetch_crypto_data(token, timeframe, start_date, end_date)
    if data is not None and len(data) > 0:
        with open(cache_file, 'wb') as f: pickle.dump(data, f)
        print(f"Cached {len(data)} candles to {cache_file}")
    return data

def patch_fetch_functions():
    """Monkey patch the fetch_crypto_data function to use cached data."""
    import app
    original_fetch = app.fetch_crypto_data
    def patched_fetch_crypto_data(symbol, timeframe, start_date, end_date=None, exchange_id='binance'):
        return get_cached_data(symbol, timeframe, start_date, end_date)
    app.fetch_crypto_data = patched_fetch_crypto_data
    print("Patched fetch_crypto_data to use cached data")
    return original_fetch

def safe_run_strategy(strategy_name, strategy_func, strategy_results, *args, **kwargs):
    """Run a strategy with enhanced error handling, memory monitoring, and timeout."""
    print(f"\nRunning {strategy_name}...")
    result_key = f"{args[0]}_{args[1]}" # Token_Timeframe unique key for results dict
    try:
        max_runtime = 600  # 10 minutes max per strategy run
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        try:
            gc.collect()
            # Assuming the hybrid strategy function signature matches this call structure
            stats, _, _ = strategy_func(*args, **kwargs)
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
        metrics['max_drawdown'] = md * 100 if 0 < md <= 1.0 else md # Ensure percentage
        
        profit_factor = stats.get('Profit Factor', stats.get('profit_factor', 0))
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

def adjust_parameters_for_timeframe(token, timeframe):
    """
    Adjust parameters for the *Enhanced* Hybrid strategy based on token market cap and timeframe.
    Returns QuantumHybridEnhancedParameters object.
    (Parameter logic might need refinement based on the enhanced strategy's specifics)
    """
    params = QuantumHybridEnhancedParameters() # Use Enhanced parameters

    # Determine base mass based on market cap
    mass_multiplier = 1.0
    if token in HIGH_MARKET_CAP: mass_multiplier = 2.0
    elif token in MEDIUM_MARKET_CAP: mass_multiplier = 1.5
    # No small caps in this run, but keep logic just in case
    elif token in SMALL_MARKET_CAP: mass_multiplier = 0.8
    else:
        print(f"Warning: Token {token} not found in market cap lists. Using default mass multiplier.")

    base_mass = 1.0e-30 * mass_multiplier
    params.m = base_mass
    # Ensure other potential mass parameters in Enhanced class are set if needed
    if hasattr(params, 'mass'):
        params.mass = base_mass

    # Adjust omega frequency based on timeframe for Enhanced Hybrid
    # Add logic for '1m' timeframe
    if timeframe == '1d': params.omega = 5.0e-4
    elif timeframe == '4h': params.omega = 2.5e-4
    elif timeframe == '1h': params.omega = 1.1e-4
    elif timeframe == '30m': params.omega = 0.8e-4
    elif timeframe == '15m': params.omega = 0.6e-4
    elif timeframe == '5m': params.omega = 0.3e-4
    elif timeframe == '1m': params.omega = 0.1e-4 # Example value for 1m, adjust as needed

    # Adjust other Enhanced-specific parameters if necessary based on timeframe/token
    # Example:
    # if hasattr(params, 'anharmonic_coeff'):
    #     if timeframe in ['1m', '5m']: params.anharmonic_coeff = 0.01
    #     else: params.anharmonic_coeff = 0.005

    print(f"Adjusted Enhanced Params for {token} {timeframe}: mass_mult={mass_multiplier}, omega={params.omega:.1e}")
    return params

def run_enhanced_hybrid_strategy_for_token_timeframe(token, timeframe, start_date, end_date, initial_capital=10000):
    """Run ONLY the Quantum Hybrid Enhanced strategy for a single token-timeframe."""
    strategy_results = {} # Dictionary to hold results for this combo
    strategy_name = STRATEGY_NAME_DISPLAY # Use global display name

    # Get parameters adjusted for token/timeframe using the enhanced logic
    params = adjust_parameters_for_timeframe(token, timeframe)

    # Run the enhanced strategy
    strategy_results = safe_run_strategy(
        strategy_name,
        run_quantum_hybrid_enhanced_strategy_backtest, # Corrected function name here
        strategy_results, # Pass the dict to store the result under the specific key
        token,
        timeframe,
        initial_capital,
        start_date,
        end_date,
        params=params
    )

    # safe_run_strategy returns the dict with results under token_timeframe key
    # We want the results directly for this token/timeframe
    result_key = f"{token}_{timeframe}"
    return strategy_results.get(result_key, {'error': 'Strategy run failed or key mismatch'})

def prefetch_and_cache_data(tokens, timeframes, start_date, end_date):
    """
    Prefetch data, using adjusted dates for smaller timeframes (including 1m).
    """
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
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                adjusted_start_dt = end_dt - timedelta(days=days_back)
                fetch_start_date = adjusted_start_dt.strftime('%Y-%m-%d')
                print(f"  Using adjusted period ({days_back} days) for {timeframe}: {fetch_start_date} to {fetch_end_date}")

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
                data = fetch_crypto_data(token, timeframe, fetch_start_date, fetch_end_date)
                candle_count = len(data) if data is not None else 0
                is_available = candle_count > 100
                if is_available:
                    with open(cache_file, 'wb') as f: pickle.dump(data, f)
                    print(f"  {token} {timeframe}: ✅ ({candle_count} candles, cached)")
                else: print(f"  {token} {timeframe}: ❌ (Insufficient data: {candle_count})")
                availability[token][timeframe] = {"available": is_available, "candle_count": candle_count, "cache_file": cache_file if is_available else None, "start_date": fetch_start_date, "end_date": fetch_end_date, "full_year": full_year}
            except Exception as e:
                print(f"  {token} {timeframe}: ❌ Error fetching: {e}")
                availability[token][timeframe] = {"available": False, "error": str(e), "start_date": fetch_start_date, "end_date": fetch_end_date, "full_year": full_year}
    print("Prefetching complete!")
    return availability

def run_all_tests():
    """Run the Quantum Hybrid Enhanced strategy on selected tokens/timeframes."""
    start_time_total = time.time()
    print("Starting Quantum Hybrid Enhanced Strategy backtest run...")
    print("-" * 70)

    # Setup
    tokens = select_specific_tokens()
    start_date, end_date = generate_one_year_period()
    initial_capital = 10000.0
    all_results = {} # Store results keyed by token, then timeframe
    strategy_name_display = STRATEGY_NAME_DISPLAY # For outputs

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
        except Exception as e: print(f"Failed to save recovery data: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Prefetch data
        import app # Ensure fetch_crypto_data is accessible before patching
        app.fetch_crypto_data = original_fetch # Use original fetch for prefetch
        print("\nPrefetching data...")
        availability = prefetch_and_cache_data(tokens, TIMEFRAMES, start_date, end_date)
        app.fetch_crypto_data = patch_fetch_functions() # Re-apply patch
        
        valid_tests = []
        for token, timeframe_data in availability.items():
            for timeframe, data_info in timeframe_data.items():
                if data_info.get('available', False):
                    valid_tests.append((token, timeframe, data_info['start_date'], data_info['end_date']))
        
        if not valid_tests:
            print("Error: No valid data available for any selected token/timeframe.")
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
                 print("High memory usage. Forcing GC...")
                 gc.collect()
                 time.sleep(3)

            # Run the single strategy
            result = run_enhanced_hybrid_strategy_for_token_timeframe(
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
        print(f"\nSaving final results to {final_filename}...")
        with open(final_filename, 'w') as f:
            json.dump(all_results, f, default=str)

        # Create Summary
        summary_df = create_summary_table(all_results, STRATEGY_NAME_DISPLAY) # Use global name
        summary_filename = None
        if summary_df is not None and not summary_df.empty:
            summary_filename = f'{RESULTS_DIR}/summary_{timestamp}.csv'
            try:
                summary_df.to_csv(summary_filename, index=False)
                print(f"\nSummary table saved to {summary_filename}")
                # Instruct user to run generate_visualization.py
                print(f"\nTo visualize results, run:\npython generate_visualization.py {summary_filename}")
            except Exception as e:
                 print(f"Error saving summary CSV: {e}")
                 summary_filename = None # Reset on error
        else:
            print("Skipping summary CSV generation due to empty results.")

        total_time = time.time() - start_time_total
        print(f"\nAnalysis complete! Total time: {total_time:.2f} seconds.")
        return all_results, final_filename # Return JSON filename, not summary DF

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
        # Ensure original function is restored
        try:
            import app
            app.fetch_crypto_data = original_fetch
            print("Restored original fetch_crypto_data function.")
        except Exception as restore_error: print(f"Warning: Failed to restore: {restore_error}")

def create_summary_table(results_dict, strategy_name):
    """Create a summary DataFrame for the single strategy results."""
    rows = []
    if not isinstance(results_dict, dict): return None
        
    for token, timeframe_data in results_dict.items():
         if not isinstance(timeframe_data, dict): continue
         for timeframe, metrics in timeframe_data.items():
            if not isinstance(metrics, dict) or 'error' in metrics:
                 continue # Skip failed or errored runs
            
            market_cap = "High" if token in HIGH_MARKET_CAP else "Medium" if token in MEDIUM_MARKET_CAP else "Low"
            
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
    try:
        df.to_csv(f'{RESULTS_DIR}/summary_{timestamp}.csv', index=False)
        print(f"Summary table saved to {RESULTS_DIR}/summary_{timestamp}.csv")
    except Exception as e: print(f"Error saving summary CSV: {e}")
    return df

# --- Script Entry Point ---
if __name__ == "__main__":
    run_all_tests()