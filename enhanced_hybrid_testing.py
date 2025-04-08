#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Hybrid Strategy Comparative Backtesting Script

This script runs comparative tests on:
- Quantum Stock Hybrid (Original)
- Quantum Hybrid Enhanced (with Regime Filter & ATR Trailing Stop)

The script backtests across multiple timeframes (5m, 15m, 1h)
for a 1-year period (default 2023), focusing on a diverse selection of tokens
including ADA, DOGE, ETH and other high/medium/low market cap tokens.
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
import glob # <<< ADDED GLOB FOR FILE FINDING

warnings.filterwarnings("ignore")

# Import app modules for strategy running
from app import fetch_crypto_data

# Import the strategy functions to compare
from app import run_quantum_hybrid_strategy_backtest # Original Hybrid
from quantum_hybrid_enhanced import run_quantum_hybrid_enhanced_strategy_backtest # Enhanced Hybrid

# Import parameter classes
from quantum_stock_hybrid_model import QuantumHybridParameters # Original Params
from quantum_hybrid_enhanced import QuantumHybridEnhancedParameters # Enhanced Params

# Fix pandas to avoid SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Define token categories by market cap
HIGH_MARKET_CAP = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT']
MEDIUM_MARKET_CAP = ['LINK/USDT', 'DOT/USDT', 'AVAX/USDT', 'ATOM/USDT', 'NEAR/USDT', 'MATIC/USDT', 'UNI/USDT', 'LTC/USDT']
SMALL_MARKET_CAP = ['ALGO/USDT', 'FTM/USDT', 'ONE/USDT', 'ZIL/USDT', 'AAVE/USDT', 'ETC/USDT', 'KSM/USDT', 'CHZ/USDT']

# Add a data caching directory constant
DATA_CACHE_DIR = 'data_cache' # Reuse existing cache if possible

# Timeframes to test - Reduced set
TIMEFRAMES = ['5m', '15m', '1h']

# Ensure results directory exists
RESULTS_DIR = "enhanced_hybrid_results" # New directory for these results
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define a global variable for storing intermediate results for recovery
recovery_results = {}

def select_specific_tokens():
    """
    Select specific tokens for testing based on requirements:
    1. Always include ADA/USDT and DOGE/USDT
    2. Add ETH/USDT and two more high market cap tokens
    3. Add 3 medium market cap tokens
    4. Add 3 low market cap tokens
    """
    # Always include ADA and DOGE
    selected_tokens = ['ADA/USDT', 'DOGE/USDT', 'ETH/USDT']
    
    # Add 2 more high market cap tokens (excluding ones already selected)
    remaining_high_cap = [token for token in HIGH_MARKET_CAP if token not in selected_tokens]
    selected_tokens.extend(random.sample(remaining_high_cap, 2))
    
    # Add 3 medium market cap tokens
    selected_tokens.extend(random.sample(MEDIUM_MARKET_CAP, 3))
    
    # Add 3 low market cap tokens
    selected_tokens.extend(random.sample(SMALL_MARKET_CAP, 3))
    
    # Print selection summary
    print("Selected tokens for testing:")
    print(f"  High Market Cap: {[t for t in selected_tokens if t in HIGH_MARKET_CAP]}")
    print(f"  Medium Market Cap: {[t for t in selected_tokens if t in MEDIUM_MARKET_CAP]}")
    print(f"  Low Market Cap: {[t for t in selected_tokens if t in SMALL_MARKET_CAP]}")
    
    return selected_tokens

def generate_one_year_period(year=2023):
    """Generate a one-year testing period."""
    # Use the specified year (defaulting to 2023)
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    # Make sure we don't go into the future
    current_date = datetime.now()
    if end_date > current_date:
        end_date = current_date
    
    # Format dates for API
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Analysis period: {start_date_str} to {end_date_str}")
    return start_date_str, end_date_str

def get_cached_data(token, timeframe, start_date, end_date):
    """Get data from cache if available, otherwise fetch and cache it."""
    # Create cache directory if it doesn't exist
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    token_safe = token.replace('/', '_')
    # Use full date range in cache filename, as we fetch full year here
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
    print(f"Fetching {token} {timeframe} data from {start_date} to {end_date}...")
    # Add warning for potentially large fetches on small timeframes
    if timeframe in ['5m', '15m']:
        print(f"  WARNING: Fetching full year for {timeframe}. This might take time and consume significant disk space.")

    data = fetch_crypto_data(token, timeframe, start_date, end_date)
    
    # Cache the data for future use
    if data is not None and len(data) > 0:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Cached {len(data)} candles to {cache_file}")
    
    return data

def patch_fetch_functions():
    """Monkey patch the fetch_crypto_data function to use cached data."""
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
        # Set a maximum timeout for each strategy run
        max_runtime = 600  # 10 minutes max per strategy
        start_time = time.time()
        
        # Track memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        try:
            # Different strategies return different formats, normalize them
            if "quantum_oscillator" in strategy_func.__name__:
                # Force garbage collection before running
                gc.collect()
                
                stats, _ = strategy_func(*args, **kwargs)
                if stats is None:
                    raise ValueError(f"{strategy_name} returned None stats")
            elif "quantum_stock" in strategy_func.__name__ or "quantum_hybrid" in strategy_func.__name__:
                # Force garbage collection before running
                gc.collect()
                
                # Add initial_capital for stock strategies if not provided
                if "initial_capital" not in kwargs and len(args) < 3:
                    kwargs["initial_capital"] = 10000

                # Call the strategy function
                result = strategy_func(*args, **kwargs)

                # Check the number of returned values and unpack accordingly
                if isinstance(result, tuple):
                    if len(result) == 3:
                        stats, _, _ = result # Unpack 3, discard last two
                    elif len(result) == 2:
                        stats, _ = result    # Unpack 2, discard last one
                    else:
                        raise ValueError(f"{strategy_name} returned an unexpected number of values: {len(result)}")
                else:
                     # Handle case where only stats might be returned (or non-tuple)
                     if result is not None: # Assuming non-tuple means just stats
                          stats = result
                     else:
                          stats = None # If result itself is None
                     # Alternatively, raise error if non-tuple is unexpected:
                     # raise TypeError(f"{strategy_name} did not return a tuple as expected.")

                if stats is None:
                    raise ValueError(f"{strategy_name} returned None stats")
            else:
                raise ValueError(f"Unknown strategy function: {strategy_func.__name__}")
                
            # Check if we're taking too long
            if time.time() - start_time > max_runtime:
                raise TimeoutError(f"Strategy execution exceeded {max_runtime} seconds")
                
            # Check memory usage after running
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            print(f"Final memory usage: {current_memory:.1f} MB (increase: {memory_increase:.1f} MB)")
            
            # Alert if memory usage is high
            if current_memory > 1500:  # Over 1.5GB
                print(f"WARNING: High memory usage detected: {current_memory:.1f} MB")
                gc.collect()  # Force garbage collection
                
            strategy_results[strategy_name] = extract_key_metrics(stats)
            print(f"Completed {strategy_name} successfully.")
            
            # Save intermediate results after each strategy
            try:
                token = args[0] if len(args) > 0 else "unknown"
                timeframe = args[1] if len(args) > 1 else "unknown"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_token = token.replace("/", "_")
                filename = f'{RESULTS_DIR}/intermediate_{safe_token}_{timeframe}_{strategy_name.replace(" ", "_")}_{timestamp}.json'
                with open(filename, 'w') as f:
                    json.dump({strategy_name: strategy_results[strategy_name]}, f, default=str) # Save only current strategy result
                # print(f"Saved intermediate results to {filename}") # Less verbose
            except Exception as save_error:
                print(f"Warning: Could not save intermediate results: {save_error}")
                
        except TimeoutError as te:
            print(f"Timeout running {strategy_name}: {str(te)}")
            strategy_results[strategy_name] = {
                'error': f'Timeout: {str(te)}',
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'risk_adjusted_return': 0
            }
            
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
    
    # Force garbage collection after each strategy
    gc.collect()
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
        # Handle different stats formats
        total_trades = stats.get('Total Trades', stats.get('total_trades', 0))
        
        # Convert string percentages to floats if needed
        win_rate = stats.get('Win Rate [%]', stats.get('win_rate', 0))
        if isinstance(win_rate, str):
            win_rate = float(win_rate.replace('%', ''))
        
        total_return = stats.get('Return [%]', stats.get('total_return', 0))
        if isinstance(total_return, str):
            total_return = float(total_return.replace('%', ''))
        
        max_drawdown = stats.get('Max Drawdown [%]', stats.get('max_drawdown', 0))
        if isinstance(max_drawdown, str):
            max_drawdown = float(max_drawdown.replace('%', ''))
        # If max_drawdown is a decimal (0-1 range), convert to percentage
        elif max_drawdown <= 1.0 and max_drawdown > 0:
            max_drawdown = max_drawdown * 100
        
        profit_factor = stats.get('Profit Factor', stats.get('profit_factor', 0))
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
            'total_trades': total_trades,
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

def adjust_parameters_for_timeframe(token, timeframe, strategy_type):
    """
    Adjust parameters based on token market cap and timeframe.
    Handles 'hybrid' and 'hybrid_enhanced'.
    Only includes logic for 5m, 15m, 1h.
    """
    mass_multiplier = 1.0
    if token in HIGH_MARKET_CAP: mass_multiplier = 2.0
    elif token in MEDIUM_MARKET_CAP: mass_multiplier = 1.5
    else: mass_multiplier = 0.8

    params = None # Initialize params

    if strategy_type == 'hybrid':
        params = QuantumHybridParameters()
        base_mass = 1.0e-30 * mass_multiplier
        params.m = base_mass
        params.mass = base_mass

        # Adjust frequency based on timeframe
        if timeframe == '1h': params.omega = 1.1e-4
        elif timeframe == '15m': params.omega = 0.6e-4
        elif timeframe == '5m': params.omega = 0.3e-4
        # Default for safety, though should not be reached with current TIMEFRAMES
        else: params.omega = 1.1e-4

    elif strategy_type == 'hybrid_enhanced':
        params = QuantumHybridEnhancedParameters() # Use enhanced param class
        base_mass = 1.0e-30 * mass_multiplier
        params.m = base_mass
        params.mass = base_mass

        # Adjust frequency based on timeframe (using same logic as original hybrid for now)
        if timeframe == '1h': params.omega = 1.1e-4
        elif timeframe == '15m': params.omega = 0.6e-4
        elif timeframe == '5m': params.omega = 0.3e-4
        # Default for safety
        else: params.omega = 1.1e-4

        # Keep other enhanced params at their defaults unless specific tuning is needed
        # e.g., params.regime_slope_threshold could be adjusted per timeframe later

    return params

def run_strategies_for_token_timeframe(token, timeframe, start_date, end_date, initial_capital=10000):
    """Run the two hybrid strategies for a single token-timeframe combination."""
    strategy_results = {}
    overall_start_time = time.time()
    max_token_timeframe_runtime = 1800 # 30 minutes max per token-timeframe combo (2 strategies)
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024 # MB
    print(f"Token-timeframe starting memory: {initial_memory:.1f} MB")

    # Save the empty results first for checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_token = token.replace("/", "_")
    checkpoint_path = f'{RESULTS_DIR}/checkpoint_{safe_token}_{timeframe}_{timestamp}.json'
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump({"token": token, "timeframe": timeframe, "results": {}}, f, default=str)
    except Exception as e:
        print(f"Warning: Failed to save initial checkpoint: {e}")

    # List of strategies to run - ONLY the two hybrids
    strategies = [
        ('Quantum Stock Hybrid', run_quantum_hybrid_strategy_backtest,
         adjust_parameters_for_timeframe(token, timeframe, 'hybrid'), 'hybrid'),
        ('Quantum Hybrid Enhanced', run_quantum_hybrid_enhanced_strategy_backtest,
         adjust_parameters_for_timeframe(token, timeframe, 'hybrid_enhanced'), 'hybrid_enhanced')
    ]

    # Run each strategy with checkpoints
    for i, (strategy_name, strategy_func, params, strategy_type) in enumerate(strategies):
        current_memory = process.memory_info().rss / 1024 / 1024 # MB
        if current_memory > 2000: # Over 2GB
            print(f"WARNING: Memory usage too high ({current_memory:.1f} MB). Forcing GC...")
            gc.collect(); time.sleep(5)
            current_memory = process.memory_info().rss / 1024 / 1024
            if current_memory > 2500: # Still over 2.5GB
                print(f"CRITICAL: Memory usage still too high ({current_memory:.1f} MB). Skipping remaining strategies for {token}-{timeframe}.")
                strategy_results[strategy_name] = {'error': 'Memory limit exceeded before run'} # Mark as error
                break # Stop processing this token/timeframe

        if time.time() - overall_start_time > max_token_timeframe_runtime:
            print(f"WARNING: Token-timeframe {token}-{timeframe} exceeded time limit ({max_token_timeframe_runtime}s). Skipping remaining strategies.")
            strategy_results[strategy_name] = {'error': 'Time limit exceeded before run'} # Mark as error
            break # Stop processing this token/timeframe

        # Run the strategy
        if params is None:
             print(f"Warning: No parameters generated for {strategy_name} on {token}-{timeframe}. Skipping.")
             strategy_results[strategy_name] = {'error': 'Parameter generation failed'}
             continue

        strategy_results = safe_run_strategy(
            strategy_name, strategy_func, strategy_results,
            token, timeframe, initial_capital, start_date, end_date, params=params # Pass params explicitly
        )

        # Save checkpoint after each strategy completes (or fails)
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Checkpoint includes all results accumulated so far for this token/timeframe
            checkpoint_path = f'{RESULTS_DIR}/checkpoint_{safe_token}_{timeframe}_{i+1}of{len(strategies)}_{timestamp}.json'
            with open(checkpoint_path, 'w') as f:
                json.dump({"token": token, "timeframe": timeframe, "results": strategy_results,
                          "completed_strategies": i+1, "total_strategies": len(strategies)}, f, default=str)
            # print(f"Saved checkpoint after strategy {i+1}/{len(strategies)}") # Less verbose
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    final_memory = process.memory_info().rss / 1024 / 1024 # MB
    memory_change = final_memory - initial_memory
    print(f"Token-timeframe final memory: {final_memory:.1f} MB (change: {memory_change:.1f} MB)")

    return strategy_results

def prefetch_and_cache_data(tokens, timeframes, start_date, end_date):
    """
    Prefetch data for all token-timeframe combinations and cache it locally.
    Fetches full year for 1h, but only last 3 months for 5m/15m.
    """
    print("Prefetching and caching required data (adjusting for 5m/15m)...")
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    availability = {}
    total_combinations = len(tokens) * len(timeframes)
    completed = 0

    end_dt_obj = datetime.strptime(end_date, '%Y-%m-%d') # End date object for calculations

    for token in tokens:
        token_safe = token.replace('/', '_')
        availability[token] = {}

        for timeframe in timeframes:
            completed += 1
            progress = (completed / total_combinations) * 100
            print(f"Progress: {progress:.1f}% - Processing {token} {timeframe} ({completed}/{total_combinations})")

            # Adjust date range for 5m/15m
            fetch_start_date = start_date
            fetch_end_date = end_date
            full_period = True # Assume full period initially

            if timeframe in ['5m', '15m']:
                full_period = False
                # Calculate 3 months before end date
                adjusted_start_dt = end_dt_obj - timedelta(days=90) # Approx 3 months
                fetch_start_date = adjusted_start_dt.strftime('%Y-%m-%d')
                print(f"  Adjusted period for {timeframe}: {fetch_start_date} to {fetch_end_date}")

            # Cache filename reflects the *actual* dates fetched
            cache_file = os.path.join(DATA_CACHE_DIR, f"{token_safe}_{timeframe}_{fetch_start_date}_{fetch_end_date}.pkl")

            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f: data = pickle.load(f)
                    candle_count = len(data) if data is not None else 0
                    print(f"  {token} {timeframe}: ✅ (Using cached data: {candle_count} candles, Period: {'Full Year' if full_period else 'Last 3 Months'})")
                    availability[token][timeframe] = {
                        "available": candle_count > 100, # Basic check
                        "candle_count": candle_count,
                        "start_date": fetch_start_date,
                        "end_date": fetch_end_date,
                        "full_period": full_period
                    }
                    continue
                except Exception as e:
                    print(f"  Error loading cached data for {token} {timeframe}: {str(e)}")

            # Fetch new data using the potentially adjusted date range
            try:
                print(f"  Fetching {token} {timeframe} data from {fetch_start_date} to {fetch_end_date}...")

                data = fetch_crypto_data(token, timeframe, fetch_start_date, fetch_end_date)
                candle_count = len(data) if data is not None else 0

                if candle_count > 100:
                    with open(cache_file, 'wb') as f: pickle.dump(data, f)
                    print(f"  {token} {timeframe}: ✅ ({candle_count} candles, cached, Period: {'Full Year' if full_period else 'Last 3 Months'})")
                    availability[token][timeframe] = {
                        "available": True,
                        "candle_count": candle_count,
                        "start_date": fetch_start_date,
                        "end_date": fetch_end_date,
                        "full_period": full_period
                    }
                else:
                    print(f"  {token} {timeframe}: ❌ (Insufficient data: {candle_count} candles)")
                    availability[token][timeframe] = {
                        "available": False,
                        "candle_count": candle_count,
                        "start_date": fetch_start_date,
                        "end_date": fetch_end_date,
                        "full_period": full_period
                    }
            except Exception as e:
                print(f"  {token} {timeframe}: ❌ Error fetching: {str(e)}")
                availability[token][timeframe] = {
                    "available": False,
                    "error": str(e),
                    "start_date": fetch_start_date,
                    "end_date": fetch_end_date,
                    "full_period": full_period
                }

    print("Prefetching complete!")
    return availability

def run_all_strategies():
    """Run the comparative tests for the two hybrid strategies."""
    global_start_date, global_end_date = generate_one_year_period() # Keep the overall period definition
    tokens = select_specific_tokens()
    all_results = {}
    original_fetch = patch_fetch_functions()

    try:
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
            # Restore original function before exiting
            import app
            if 'original_fetch' in locals() and hasattr(app, 'fetch_crypto_data') and app.fetch_crypto_data != original_fetch:
                 app.fetch_crypto_data = original_fetch
                 print("Restored original fetch_crypto_data function")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"Prefetching data for {len(TIMEFRAMES)} timeframes: {TIMEFRAMES} (Using last 3 months for 5m/15m)...")
        # Pass global start/end dates to prefetcher, it will adjust internally
        availability = prefetch_and_cache_data(tokens, TIMEFRAMES, global_start_date, global_end_date)

        valid_tokens = [token for token, tf_data in availability.items()
                        if any(data.get('available', False) for data in tf_data.values())]
        if len(valid_tokens) < len(tokens):
            print(f"Warning: Only {len(valid_tokens)}/{len(tokens)} tokens have valid data for tested timeframes.")
        tokens = valid_tokens

        total_tests = sum(1 for t in tokens for tf in TIMEFRAMES if availability.get(t, {}).get(tf, {}).get('available', False))
        test_count = 0

        # Process tokens sequentially
        for token_idx, token in enumerate(tokens):
            print(f"\n===== Processing token {token_idx+1}/{len(tokens)}: {token} =====")
            gc.collect() # Clear memory before starting a new token
            all_results[token] = {}
            if token not in recovery_results: # Ensure token key exists
                recovery_results[token] = {}

            for timeframe_idx, timeframe in enumerate(TIMEFRAMES):
                gc.collect() # Clear memory before starting a new timeframe

                # Check availability from the prefetch results
                tf_info = availability.get(token, {}).get(timeframe, {})
                if not tf_info.get('available', False):
                    print(f"Skipping {token} on {timeframe} - insufficient data as per prefetch.")
                    # Log error in results and recovery data
                    error_msg = tf_info.get("error", "Insufficient data")
                    all_results[token][timeframe] = {"error": error_msg}
                    if timeframe not in recovery_results[token]: recovery_results[token][timeframe] = {}
                    recovery_results[token][timeframe] = {"error": error_msg}
                    continue

                # Get the specific start/end dates for this test from availability info
                test_start_date = tf_info['start_date']
                test_end_date = tf_info['end_date']
                is_full_period = tf_info['full_period']

                test_count += 1
                print(f"\n----- Testing {token} on {timeframe} timeframe ({test_count}/{total_tests}) -----")
                print(f"Date range for test: {test_start_date} to {test_end_date} ({'Full Year' if is_full_period else 'Last 3 Months'})")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M") # Timestamp for potential interim save

                try:
                    process = psutil.Process(os.getpid())
                    mem_before = process.memory_info().rss / 1024 / 1024 # MB
                    print(f"Memory usage before test: {mem_before:.1f} MB")

                    if mem_before > 2500: # Memory check
                        print("WARNING: High memory usage. Forcing GC...")
                        gc.collect(); time.sleep(5)
                        mem_after_gc = process.memory_info().rss / 1024 / 1024
                        print(f"Memory after GC: {mem_after_gc:.1f} MB")
                        if mem_after_gc > 2500:
                            print("CRITICAL: Memory still too high. Saving checkpoint and exiting.")
                            checkpoint_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                            # Save recovery_results as checkpoint
                            with open(f'{RESULTS_DIR}/memory_checkpoint_{checkpoint_time}.json', 'w') as f:
                                json.dump(recovery_results, f, default=str)
                            print(f"Saved results to memory_checkpoint_{checkpoint_time}.json. Restart manually.")
                            # Restore original function before exiting
                            import app
                            if 'original_fetch' in locals() and hasattr(app, 'fetch_crypto_data') and app.fetch_crypto_data != original_fetch:
                                 app.fetch_crypto_data = original_fetch
                                 print("Restored original fetch_crypto_data function")
                            return recovery_results, None # Return partial results

                    # Run the two strategies using the specific test dates
                    results = run_strategies_for_token_timeframe(
                        token, timeframe, test_start_date, test_end_date
                    )
                    all_results[token][timeframe] = results
                    # Ensure timeframe key exists before assignment
                    if timeframe not in recovery_results[token]: recovery_results[token][timeframe] = {}
                    recovery_results[token][timeframe] = results # Update recovery data

                except Exception as e:
                    print(f"Error testing {token} on {timeframe}: {str(e)}")
                    traceback.print_exc()
                    error_result = {"error": str(e)}
                    all_results[token][timeframe] = error_result
                    # Log error in recovery data
                    if timeframe not in recovery_results[token]: recovery_results[token][timeframe] = {}
                    recovery_results[token][timeframe] = error_result

                # Save interim results (remains the same, uses recovery_results)
                safe_token = token.replace("/", "_")
                interim_filename = f'{RESULTS_DIR}/interim_results_{safe_token}_{timeframe}_{timestamp}.json'
                try:
                    # Save only the result for the current token/timeframe in the interim file
                    current_result_to_save = recovery_results.get(token, {}).get(timeframe, {})
                    with open(interim_filename, 'w') as f:
                        json.dump({token: {timeframe: current_result_to_save}}, f, default=str)
                except Exception as save_error:
                     print(f"Warning: Failed to save interim results: {save_error}")

                time.sleep(1) # Reduced pause

        # Save final results using recovery_results which should be the most complete
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f'{RESULTS_DIR}/analysis_results_{timestamp}.json'
        final_results_to_save = recovery_results
        try:
            with open(filename, 'w') as f:
                json.dump(final_results_to_save, f, default=str)
        except Exception as save_error:
            print(f"ERROR Saving final JSON results: {save_error}")
            filename = None # Indicate final save failed

        summary_df = create_summary_table(final_results_to_save)
        # Check if summary is valid before creating plots
        if not summary_df.empty and filename is not None: # Also check if final save was successful
             try:
                 create_visualizations(final_results_to_save, summary_df, timestamp)
             except Exception as plot_error:
                  print(f"ERROR Creating visualizations: {plot_error}")
        elif summary_df.empty:
            print("Summary table is empty, skipping visualizations.")
        else:
            print("Final JSON save failed, skipping visualizations.")

        print(f"\nAnalysis complete!")
        if filename:
             print(f"Results saved to {filename}")
        else:
             print("Final JSON results could not be saved.")

        # <<< ADDED CLEANUP LOGIC >>>
        print("\nCleaning up intermediate checkpoint and JSON files...")
        try:
            # Find checkpoint files
            checkpoint_files_to_delete = glob.glob(os.path.join(RESULTS_DIR, '*_checkpoint_*.pkl'))
            # Find intermediate JSON files
            intermediate_json_files = glob.glob(os.path.join(RESULTS_DIR, 'intermediate_*.json'))

            files_to_delete = checkpoint_files_to_delete + intermediate_json_files

            if not files_to_delete:
                print("No intermediate files found to delete.")
            else:
                deleted_count = 0
                for file_path in files_to_delete:
                    try:
                        os.remove(file_path)
                        # print(f"Deleted: {os.path.basename(file_path)}") # Uncomment for verbose logging
                        deleted_count += 1
                    except OSError as e:
                        print(f"Error deleting file {os.path.basename(file_path)}: {e}")
                print(f"Successfully deleted {deleted_count} intermediate files.")
        except Exception as e:
            print(f"Error during file cleanup: {e}")

        return final_results_to_save, filename

    except Exception as e:
        print(f"ERROR: Analysis failed: {str(e)}")
        traceback.print_exc()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        try:
            # Save recovery_results on major failure
            partial_filename = f'{RESULTS_DIR}/partial_results_{timestamp}.json'
            with open(partial_filename, 'w') as f:
                json.dump({"error": str(e), "partial_results": recovery_results}, f, default=str)
            print(f"Partial results saved to {partial_filename}")
        except Exception as save_error:
            print(f"Failed to save partial results: {str(save_error)}")
        return recovery_results, None # Return partial results

    finally:
        # Ensure original function is restored
        import app
        if 'original_fetch' in locals() and hasattr(app, 'fetch_crypto_data') and app.fetch_crypto_data != original_fetch:
             app.fetch_crypto_data = original_fetch
             print("Restored original fetch_crypto_data function")

def create_summary_table(token_results):
    """Create a summary table for all token-timeframe-strategy combinations."""
    rows = []
    
    for token, timeframe_data in token_results.items():
        for timeframe, strategy_results in timeframe_data.items():
            if isinstance(strategy_results, dict) and 'error' not in strategy_results:
                for strategy, metrics in strategy_results.items():
                    # Skip entries with errors
                    if 'error' in metrics:
                        continue
                    
                    # Add market cap category
                    market_cap = "High" if token in HIGH_MARKET_CAP else "Medium" if token in MEDIUM_MARKET_CAP else "Low"
                    
                    rows.append({
                        'Token': token,
                        'Market Cap': market_cap,
                        'Timeframe': timeframe,
                        'Strategy': strategy,
                        'Total Trades': metrics.get('total_trades', 0),
                        'Win Rate [%]': metrics.get('win_rate', 0),
                        'Return [%]': metrics.get('total_return', 0),
                        'Max Drawdown [%]': metrics.get('max_drawdown', 0),
                        'Profit Factor': metrics.get('profit_factor', 0),
                        'Risk/Reward': metrics.get('risk_adjusted_return', 0)
                    })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    df.to_csv(f'{RESULTS_DIR}/summary_{timestamp}.csv', index=False)
    
    return df

def create_visualizations(token_results, summary_df, timestamp):
    """Create visualizations comparing the two hybrid strategies, including heatmaps."""
    print("Creating visualizations...")

    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style

    # Ensure summary_df is not empty and contains the expected strategies
    if summary_df.empty:
        print("Summary DataFrame is empty. Skipping visualization.")
        return

    strategies_to_compare = ['Quantum Stock Hybrid', 'Quantum Hybrid Enhanced']
    summary_df_filtered = summary_df[summary_df['Strategy'].isin(strategies_to_compare)]

    if summary_df_filtered.empty:
        print(f"No data found for strategies {strategies_to_compare} in summary_df. Skipping visualization.")
        return
    if len(summary_df_filtered['Strategy'].unique()) < 1:
         print(f"Warning: No strategies found in results. Plots will be empty.")
         # Allow continuing to generate empty plot structure if needed

    # Define a palette for consistency
    palette = sns.color_palette("viridis", n_colors=len(strategies_to_compare))
    strategy_palette = {strategy: color for strategy, color in zip(strategies_to_compare, palette)}

    # Create a multi-panel figure - Changed to 4 rows for heatmaps
    plt.figure(figsize=(18, 24)) # Wider and taller for heatmaps and table
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1.5]) # 4 rows layout

    # --- Row 1: Heatmaps --- #

    # Heatmap 1: Average Win Rate (%)
    ax_hm1 = plt.subplot(gs[0, 0])
    try:
        win_rate_pivot = summary_df_filtered.pivot_table(index='Strategy', columns='Timeframe', values='Win Rate [%]')
        # Ensure correct order
        timeframe_order = [tf for tf in TIMEFRAMES if tf in win_rate_pivot.columns]
        strategy_order = [s for s in strategies_to_compare if s in win_rate_pivot.index]
        win_rate_pivot = win_rate_pivot.reindex(index=strategy_order, columns=timeframe_order)

        sns.heatmap(win_rate_pivot, annot=True, fmt=".1f", cmap="RdYlGn", linewidths=.5, ax=ax_hm1, cbar_kws={'label': 'Win Rate (%)'})
        ax_hm1.set_title('Average Win Rate (%) by Strategy and Timeframe', fontsize=14)
        ax_hm1.set_ylabel('Strategy')
        ax_hm1.set_xlabel('Timeframe')
    except Exception as e:
        print(f"Error creating Win Rate heatmap: {e}")
        ax_hm1.text(0.5, 0.5, 'Error generating Win Rate Heatmap', ha='center', va='center')
        ax_hm1.set_title('Average Win Rate (%) - Error', fontsize=14)

    # Heatmap 2: Average Return (%)
    ax_hm2 = plt.subplot(gs[0, 1])
    try:
        return_pivot = summary_df_filtered.pivot_table(index='Strategy', columns='Timeframe', values='Return [%]')
        # Ensure correct order (use same timeframe/strategy order as above)
        return_pivot = return_pivot.reindex(index=strategy_order, columns=timeframe_order)

        sns.heatmap(return_pivot, annot=True, fmt=".1f", cmap="RdYlGn", linewidths=.5, ax=ax_hm2, center=0, cbar_kws={'label': 'Return (%)'})
        ax_hm2.set_title('Average Return (%) by Strategy and Timeframe', fontsize=14)
        ax_hm2.set_ylabel('') # No Y label needed as it matches HM1
        ax_hm2.set_yticks([]) # Remove Y ticks
        ax_hm2.set_xlabel('Timeframe')
    except Exception as e:
        print(f"Error creating Return heatmap: {e}")
        ax_hm2.text(0.5, 0.5, 'Error generating Return Heatmap', ha='center', va='center')
        ax_hm2.set_title('Average Return (%) - Error', fontsize=14)


    # --- Row 2: Box Plots --- #

    # Box plot 1: Distribution of Returns (%)
    ax1 = plt.subplot(gs[1, 0])
    if not summary_df_filtered.empty:
        sns.boxplot(x='Strategy', y='Return [%]', data=summary_df_filtered, ax=ax1, palette=strategy_palette, order=strategies_to_compare)
        ax1.set_title('Distribution of Returns (%) by Strategy', fontsize=14)
        ax1.set_ylabel('Return (%)')
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', rotation=10)
    else:
        ax1.text(0.5, 0.5, 'No data for Returns Box Plot', ha='center', va='center')

    # Box plot 2: Distribution of Win Rates (%)
    ax2 = plt.subplot(gs[1, 1])
    if not summary_df_filtered.empty:
        sns.boxplot(x='Strategy', y='Win Rate [%]', data=summary_df_filtered, ax=ax2, palette=strategy_palette, order=strategies_to_compare)
        ax2.set_title('Distribution of Win Rates (%) by Strategy', fontsize=14)
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_xlabel('')
        ax2.tick_params(axis='x', rotation=10)
    else:
        ax2.text(0.5, 0.5, 'No data for Win Rate Box Plot', ha='center', va='center')

    # --- Row 3: Scatter & Bar --- #

    # Scatter plot 3: Win Rate vs Return
    ax3 = plt.subplot(gs[2, 0])
    if not summary_df_filtered.empty:
        sns.scatterplot(
            x='Win Rate [%]', y='Return [%]', hue='Strategy', size='Total Trades',
            sizes=(50, 250), alpha=0.7, data=summary_df_filtered, ax=ax3, palette=strategy_palette,
            hue_order=strategies_to_compare
        )
        ax3.set_title('Win Rate vs Return (size = # trades)', fontsize=14)
        ax3.set_xlabel('Win Rate (%)')
        ax3.set_ylabel('Return (%)')
        ax3.legend(title='Strategy & Trades', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    else:
         ax3.text(0.5, 0.5, 'No data for Scatter Plot', ha='center', va='center')
         ax3.set_title('Win Rate vs Return (No Data)', fontsize=14)


    # Bar chart 4: Average Risk/Reward
    ax4 = plt.subplot(gs[2, 1])
    if not summary_df_filtered.empty:
        risk_reward_avg = summary_df_filtered.groupby('Strategy')['Risk/Reward'].mean().reset_index()
        # Ensure order matches
        risk_reward_avg = risk_reward_avg.set_index('Strategy').reindex(strategies_to_compare).reset_index()
        sns.barplot(x='Strategy', y='Risk/Reward', data=risk_reward_avg, ax=ax4, palette=strategy_palette, order=strategies_to_compare)
        ax4.set_title('Average Risk-Adjusted Return (Return/Drawdown)', fontsize=14)
        ax4.set_ylabel('Avg Return/Drawdown Ratio')
        ax4.set_xlabel('')
        ax4.tick_params(axis='x', rotation=10)
    else:
         ax4.text(0.5, 0.5, 'No data for Risk/Reward Plot', ha='center', va='center')
         ax4.set_title('Average Risk-Adjusted Return (No Data)', fontsize=14)


    # --- Row 4: Summary Table --- #
    ax5 = plt.subplot(gs[3, :]) # Span across bottom row

    if not summary_df_filtered.empty:
        strategy_summary = summary_df_filtered.groupby('Strategy').agg({
            'Total Trades': 'sum', 'Win Rate [%]': 'mean', 'Return [%]': 'mean',
            'Max Drawdown [%]': 'mean', 'Profit Factor': 'mean', 'Risk/Reward': 'mean'
        }).reindex(strategies_to_compare) # Ensure order

        strategy_summary = strategy_summary.fillna(0) # Handle missing strategies

        profitable_tests = summary_df_filtered[summary_df_filtered['Return [%]'] > 0].groupby('Strategy').size()
        strategy_summary['profitable_tests'] = profitable_tests.reindex(strategy_summary.index, fill_value=0)
        strategy_summary['total_tests'] = summary_df_filtered.groupby('Strategy').size().reindex(strategy_summary.index, fill_value=0)
        strategy_summary['profitable_pct'] = (
            strategy_summary['profitable_tests'].astype(float) /
            strategy_summary['total_tests'].replace(0, np.nan)
        ).fillna(0) * 100

        table_data = []
        for idx in strategies_to_compare:
            if idx in strategy_summary.index:
                row = strategy_summary.loc[idx]
                table_data.append([
                    idx, f"{row['Win Rate [%]']:.1f}%", f"{row['Return [%]']:.1f}%",
                    f"{row['Max Drawdown [%]']:.1f}%", f"{row['Profit Factor']:.2f}",
                    f"{row['Risk/Reward']:.2f}", f"{int(row['profitable_tests'])}/{int(row['total_tests'])} ({row['profitable_pct']:.1f}%)",
                    f"{int(row['Total Trades'])}"
                ])
            else:
                 table_data.append([idx, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])

        header = ['Strategy', 'Avg Win Rate', 'Avg Return', 'Avg Max DD', 'Profit Factor', 'Risk/Reward', 'Profitable Tests', 'Total Trades']

        table = ax5.table(
            cellText=table_data, colLabels=header, cellLoc='center', loc='center',
            colWidths=[0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.1]
        )
        table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.8)

        for j in range(len(header)): table[(0, j)].set_facecolor("#DDDDDD"); table[(0, j)].set_text_props(weight='bold')
        for i in range(len(table_data)): table[(i + 1, 0)].set_facecolor(strategy_palette.get(table_data[i][0], "#F0F0F0")); table[(i + 1, j)].set_height(0.06)

        ax5.set_title('Strategy Performance Summary Comparison', fontsize=16, pad=20)
    else:
         ax5.text(0.5, 0.5, 'No data available for Summary Table', ha='center', va='center')
         ax5.set_title('Strategy Performance Summary (No Data)', fontsize=16, pad=20)

    ax5.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.suptitle('Quantum Hybrid vs Enhanced Hybrid Performance Comparison', fontsize=18, y=0.99)

    output_file = f'{RESULTS_DIR}/enhanced_hybrid_analysis_{timestamp}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    print("Starting Enhanced Hybrid comparative backtest...")
    print(f"Comparing 'Quantum Stock Hybrid' vs 'Quantum Hybrid Enhanced'")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Results will be saved in: {RESULTS_DIR}")
    print("-" * 80)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results, filename = run_all_strategies()

    print("\nTesting completed!")
    if filename:
        print(f"Results saved to {filename}")
    else:
        print("Testing finished, but final results file might not have been generated due to errors or interruptions.")
    print("-" * 80) 