#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comprehensive Quantum Strategy Backtesting Script

This script runs comparative tests on the best quantum trading strategies:
- Quantum Stock - Improved Aggressive
- Quantum Stock - Improved Balanced
- Quantum Stock Hybrid
- Quantum Oscillator - Classic
- Quantum Oscillator - Improved Aggressive
- Quantum Oscillator - Improved Balanced

The script backtests across multiple timeframes (5m, 15m, 30m, 1h, 4h, 1d)
for a 1-year period, focusing on a diverse selection of tokens
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
import glob # Add glob for file searching
import argparse
import math
import optuna # Import Optuna
import datetime as dt # Use alias for datetime

# Import utility functions
# Attempt import from improved_quantum_stock_testing.py based on latest search results
try:
    # Assuming these are the correct names in the target file
    # get_start_date_for_period is NOT here, will be implemented locally
    from improved_quantum_stock_testing import get_market_cap_based_mass, prefetch_and_cache_data
    # Create an alias if comprehensive_script uses the old name
    get_market_cap_category = get_market_cap_based_mass
    print("Imported utility functions (excluding date calculation) from improved_quantum_stock_testing.py")
except ImportError as e:
    print(f"ERROR: Could not import utility functions from improved_quantum_stock_testing.py: {e}")
    print("Check the exact function names (prefetch_and_cache_data, get_market_cap_based_mass) in improved_quantum_stock_testing.py")
    sys.exit(1)

warnings.filterwarnings("ignore")

# Import app modules for strategy running
# Assuming app.py might contain fetch_crypto_data or it's globally available
# If fetch_crypto_data is defined elsewhere, adjust this import
try:
    # Attempt import assuming it's structured within an 'app' module/directory
    from app import fetch_crypto_data
except ImportError:
    # Fallback: Try importing from a top-level app.py if that's the structure
    try:
        import app as app_module
        fetch_crypto_data = app_module.fetch_crypto_data
        print("Imported fetch_crypto_data from top-level app.py")
    except (ImportError, AttributeError):
        print("ERROR: Could not find fetch_crypto_data. Ensure it's importable.")
        sys.exit(1)

# Import all the strategy functions from their respective files
# Quantum Stock Models
try:
    from app import run_quantum_stock_strategy_backtest, run_quantum_hybrid_strategy_backtest
    from quantum_stock_model import QuantumStockParameters
    from quantum_stock_hybrid_model import QuantumHybridParameters
    print("Imported Stock/Hybrid strategies from 'app' module/directory.")
except ImportError:
    # Fallback: Try importing directly if they are top-level files/importable
    try:
        from quantum_stock_model import run_quantum_stock_strategy_backtest, QuantumStockParameters
        from quantum_stock_hybrid_model import run_quantum_hybrid_strategy_backtest, QuantumHybridParameters
        print("Imported Stock/Hybrid strategies from top-level files.")
    except ImportError as e:
        print(f"ERROR: Could not import Stock/Hybrid strategies: {e}")
        sys.exit(1)

# Import Quantum Oscillator functions directly from testing files (or their primary source)
try:
    # Ideal: from quantum_oscillator_models import ...
    # Current: Using testing file as source
    from improved_quantum_oscillator_testing import (
        run_quantum_oscillator_strategy_backtest, # Assuming this is the 'Classic' one
        run_quantum_oscillator_improved_balanced_strategy_backtest,
        run_quantum_oscillator_improved_aggressive_strategy_backtest
    )
    # Need the base Parameter class for Oscillator
    from quantum_oscillator_model import QuantumOscillatorParameters
    print("Imported Oscillator strategies from improved_quantum_oscillator_testing.py")
    QUANTUM_OSCILLATOR_AVAILABLE = True # Defined only on successful import
except ImportError as e:
    print(f"WARNING: Could not import Oscillator strategies: {e}. Oscillator tests will be skipped.")
    # Define the variable as False instead of exiting
    QUANTUM_OSCILLATOR_AVAILABLE = False
    # sys.exit(1) # Don't exit here

# Import Quantum Hybrid Enhanced
try:
    from quantum_hybrid_enhanced import run_quantum_hybrid_enhanced_strategy_backtest, QuantumHybridEnhancedParameters
    print("Imported Enhanced Hybrid strategy.")
except ImportError as e:
    print(f"ERROR: Could not import Enhanced Hybrid strategy: {e}")
    sys.exit(1)

# Fix pandas to avoid SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Define token categories by market cap
HIGH_MARKET_CAP = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT']
MEDIUM_MARKET_CAP = ['LINK/USDT', 'DOT/USDT', 'AVAX/USDT', 'ATOM/USDT', 'NEAR/USDT', 'MATIC/USDT', 'UNI/USDT', 'LTC/USDT']
SMALL_MARKET_CAP = ['ALGO/USDT', 'FTM/USDT', 'ONE/USDT', 'ZIL/USDT', 'AAVE/USDT', 'ETC/USDT', 'KSM/USDT', 'CHZ/USDT']

# Add a data caching directory constant
DATA_CACHE_DIR = 'data_cache'

# Timeframes to test
TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

# Ensure results directory exists
RESULTS_DIR = "comprehensive_dex_realistic_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define a global variable for storing intermediate results for recovery
recovery_results = {}

# Global variable to hold strategy definitions (alternative to passing)
# Define it once, outside functions if it's truly constant for the script execution.
# Or define inside run_all_strategies and pass it down.
defined_strategies_list = [
    ('Quantum Stock - Balanced', run_quantum_stock_strategy_backtest, 'stock', True),
    ('Quantum Stock - Aggressive', run_quantum_stock_strategy_backtest, 'stock', True),
    ('Quantum Stock Hybrid', run_quantum_hybrid_strategy_backtest, 'hybrid', True),
    ('Quantum Oscillator - Classic', run_quantum_oscillator_strategy_backtest, 'oscillator', False),
    ('Quantum Hybrid Enhanced', run_quantum_hybrid_enhanced_strategy_backtest, 'enhanced_hybrid', True)
]

# --- Optuna Configuration ---
# Define the specific strategy name used internally for optimization targeting
OPTIMIZE_STRATEGY_NAME_INTERNAL = "Quantum Stock Hybrid"

# Define targets (token, timeframe) for optimization evaluation
# ---- MODIFIED TO FOCUS ONLY ON ETH AND SOL ----
# ---- REMOVING 1m TIMEFRAME DUE TO MEMORY ISSUES ----
OPTIMIZATION_TARGETS = [
    # ('BTC/USDT', '30m'),
    ('ETH/USDT', '30m'),
    # ('BTC/USDT', '1h'),
    ('ETH/USDT', '1h'),
    ('SOL/USDT', '1h'),
    # ('BTC/USDT', '4h'),
    ('ETH/USDT', '4h'),
    ('SOL/USDT', '4h'),
    ('SOL/USDT', '1d'),
    # ('DOGE/USDT', '1d'),
    # ('MATIC/USDT', '1d'),
    # Add other ETH/SOL timeframes if desired for evaluation
    ('ETH/USDT', '1d'),
    ('SOL/USDT', '30m'),
    ('SOL/USDT', '5m'), # Add lower timeframes if stable enough
    ('ETH/USDT', '5m')
    # ---- ADDING 1m TIMEFRAME ----
    # ('ETH/USDT', '1m'), # Excluded due to memory
    # ('SOL/USDT', '1m')  # Excluded due to memory
]
# Metric to optimize (from the stats dictionary returned by the backtest function)
OPTIMIZATION_METRIC = 'risk_adjusted_return' # Options: 'total_return', 'profit_factor', 'risk_adjusted_return', 'win_rate'
OPTIMIZATION_DIRECTION = 'maximize' # 'maximize' or 'minimize'
DEFAULT_N_TRIALS = 50 # Default number of optimization trials

# Define the specific tokens and timeframes for *individual* optimization
INDIVIDUAL_OPTIMIZATION_TOKENS = ['SOL/USDT', 'ETH/USDT']
INDIVIDUAL_OPTIMIZATION_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

# --- Utility Function for Parameter Filename ---
def get_optimized_params_filename(results_dir, strategy_name, token, timeframe):
    """Generates the specific filename for optimized parameters."""
    strategy_prefix = strategy_name.replace(' ', '_').replace('-', '_')
    token_prefix = token.replace('/', '_')
    return os.path.join(results_dir, f"best_params_{strategy_prefix}_{token_prefix}_{timeframe}.json")

def select_specific_tokens(num_total=10):
    """
    Select specific tokens for testing, ensuring BTC, ETH, SOL are included,
    plus a mix of market caps up to num_total.
    """
    # Always include BTC, ETH, SOL
    selected_tokens = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    needed = num_total - len(selected_tokens)
    if needed <= 0:
        print(f"Selected required tokens: {selected_tokens}")
        return selected_tokens

    # Determine how many more from each cap (roughly proportional, ensuring variety)
    num_high_needed = min(max(0, len(HIGH_MARKET_CAP) - 3), needed // 3)
    num_med_needed = min(max(0, len(MEDIUM_MARKET_CAP)), (needed - num_high_needed) // 2)
    num_low_needed = min(max(0, len(SMALL_MARKET_CAP)), needed - num_high_needed - num_med_needed)

    # Adjust if rounding/availability caused mismatch
    while num_high_needed + num_med_needed + num_low_needed < needed:
        # Prioritize adding medium/low cap if available
        if num_med_needed < len(MEDIUM_MARKET_CAP):
            num_med_needed += 1
        elif num_low_needed < len(SMALL_MARKET_CAP):
            num_low_needed += 1
        elif num_high_needed < len(HIGH_MARKET_CAP) - 3:
             num_high_needed +=1 # Add high cap if others exhausted
        else:
            break # Cannot add more

    # Add random high market cap tokens (excluding already selected)
    remaining_high_cap = [token for token in HIGH_MARKET_CAP if token not in selected_tokens]
    selected_tokens.extend(random.sample(remaining_high_cap, min(num_high_needed, len(remaining_high_cap))))

    # Add random medium market cap tokens
    remaining_medium_cap = [token for token in MEDIUM_MARKET_CAP if token not in selected_tokens]
    selected_tokens.extend(random.sample(remaining_medium_cap, min(num_med_needed, len(remaining_medium_cap))))

    # Add random low market cap tokens
    remaining_low_cap = [token for token in SMALL_MARKET_CAP if token not in selected_tokens]
    selected_tokens.extend(random.sample(remaining_low_cap, min(num_low_needed, len(remaining_low_cap))))

    # Ensure we don't exceed num_total due to sampling minimums
    selected_tokens = selected_tokens[:num_total]
    
    # Print selection summary
    print(f"Selected {len(selected_tokens)} tokens for testing:")
    print(f"  High Market Cap: {[t for t in selected_tokens if t in HIGH_MARKET_CAP]}")
    print(f"  Medium Market Cap: {[t for t in selected_tokens if t in MEDIUM_MARKET_CAP]}")
    print(f"  Low Market Cap: {[t for t in selected_tokens if t in SMALL_MARKET_CAP]}")
    
    return selected_tokens

def generate_random_one_month_period(year=2023):
    """Generate a random one-month testing period within the specified year."""
    # Select a random start month (1-12)
    start_month = random.randint(1, 12)
    start_date = datetime(year, start_month, 1)

    # Calculate end date (last day of the start month)
    if start_month == 12:
        end_date = datetime(year, 12, 31)
    else:
        # Go to the first day of the next month and subtract one day
        end_date = datetime(year, start_month + 1, 1) - timedelta(days=1)

    # Format dates
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Analysis period (Random Month): {start_date_str} to {end_date_str}")
    return start_date_str, end_date_str

def get_cached_data(token, timeframe, start_date, end_date, original_fetch_func):
    """Get data from cache if available, otherwise fetch using the original function."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    token_safe = token.replace('/', '_')
    # Cache filename includes the exact dates used for fetching
    cache_file = os.path.join(DATA_CACHE_DIR, f"{token_safe}_{timeframe}_{start_date}_{end_date}.pkl")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, pd.DataFrame) and not data.empty:
                print(f"Using cached data for {token} {timeframe} ({len(data)} candles)")
                return data
            else:
                print(f"Cached data for {token} {timeframe} is invalid/empty. Refetching...")
        except Exception as e:
            print(f"Error loading cached data: {str(e)}. Refetching...")

    # Cache miss or invalid cache - Fetch using the provided original function
    print(f"Fetching {token} {timeframe} data ({start_date} to {end_date}) using original fetcher...")
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
        print(f"  Warning: Original fetcher returned unexpected type ({type(data)}), not a DataFrame. Returning None.")
        return None

def patch_fetch_functions():
    """Monkey patch the fetch_crypto_data function to use cached data."""
    # Find the actual fetch_crypto_data function (handle potential import variations)
    fetch_func = None
    try:
        # Try importing from app first
        import app
        if hasattr(app, 'fetch_crypto_data') and callable(app.fetch_crypto_data):
            fetch_func = app.fetch_crypto_data
        else:
            raise ImportError # Trigger fallback
    except ImportError:
        # Fallback to global scope or wherever it might be defined
        if 'fetch_crypto_data' in globals() and callable(globals()['fetch_crypto_data']):
             fetch_func = globals()['fetch_crypto_data']
        else:
             # Last resort: try app module import again
             try:
                 import app as app_module
                 if hasattr(app_module, 'fetch_crypto_data') and callable(app_module.fetch_crypto_data):
                      fetch_func = app_module.fetch_crypto_data
             except (ImportError, AttributeError):
                 pass # fetch_func remains None

    if fetch_func is None:
        print("CRITICAL ERROR: Cannot find the 'fetch_crypto_data' function to patch.")
        sys.exit(1)

    original_fetch = fetch_func
    print(f"Found original fetch function: {original_fetch.__name__}")

    # Define the patched function that calls get_cached_data
    def patched_fetch_crypto_data(symbol, timeframe, start_date, end_date=None, exchange_id='binance'):
        # Pass the *original* fetch function to get_cached_data
        return get_cached_data(symbol, timeframe, start_date, end_date, original_fetch)

    # Apply the patch
    try:
        import app
        app.fetch_crypto_data = patched_fetch_crypto_data
        print("Patched fetch_crypto_data in 'app' module.")
    except ImportError:
        globals()['fetch_crypto_data'] = patched_fetch_crypto_data
        print("Patched fetch_crypto_data in global scope.")
    
    print("Applied patch to use cached data wrapper.")
    # Return the original function so it can be restored later
    return original_fetch

def safe_run_strategy(strategy_name, strategy_func, strategy_results, *args, **kwargs):
    """Run a strategy with enhanced error handling to prevent crashes."""
    print(f"\nRunning {strategy_name}...")
    result_key = f"{strategy_name}_{args[0]}_{args[1]}" # Use Strategy_Token_Timeframe key

    try:
        max_runtime = 900 # Increased timeout slightly
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        try:
            gc.collect()
            
            # --- Unpacking Logic --- Need to handle different return signatures
            stats = None
            if "quantum_hybrid_enhanced" in strategy_func.__name__:
                # Enhanced Hybrid returns 3 values (stats, df, trades_df)
                stats, _, _ = strategy_func(*args, **kwargs)
            elif "quantum_oscillator" in strategy_func.__name__:
                # Oscillators typically return 2 values (stats, df)
                stats, _ = strategy_func(*args, **kwargs)
            elif "quantum_stock" in strategy_func.__name__ or "quantum_hybrid" in strategy_func.__name__:
                 # Original Stock/Hybrid also return 2 (stats, df)
                 # Need to ensure initial_capital is passed correctly
                 temp_args = list(args)
                 temp_kwargs = kwargs.copy()
                 # Check if initial_capital is expected (position 2 in args for these funcs)
                 # Adjust if needed based on actual function signature if this fails
                 if len(temp_args) < 3 and "initial_capital" not in temp_kwargs:
                     # Find where initial_capital should be inserted (usually after token, timeframe)
                     if len(temp_args) >= 2:
                          temp_args.insert(2, 10000) # Default capital
                     else:
                          temp_kwargs["initial_capital"] = 10000 # Add as kwarg if args too short
                 stats, _ = strategy_func(*temp_args, **temp_kwargs)
            else:
                print(f"Warning: Unrecognized strategy function for unpacking: {strategy_func.__name__}. Assuming 2 return values.")
                stats, _ = strategy_func(*args, **kwargs) # Default assumption

            # --- Post-Execution Checks ---
            if stats is None:
                raise ValueError(f"{strategy_name} returned None stats")
            
            if time.time() - start_time > max_runtime:
                raise TimeoutError(f"Strategy execution exceeded {max_runtime} seconds")
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            print(f"Final memory usage: {current_memory:.1f} MB (increase: {memory_increase:.1f} MB)")
            if current_memory > 2000: print(f"WARNING: High memory usage detected: {current_memory:.1f} MB")

            # Store results using the defined result_key
            strategy_results[result_key] = extract_key_metrics(stats)
            print(f"Completed {strategy_name} for {args[0]} {args[1]} successfully.")

            # Save intermediate checkpoint (using result_key)
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Use parts of result_key for filename
                fname_parts = result_key.split('_')
                safe_fname = "_".join(fname_parts).replace("/", "_")
                filename = f'{RESULTS_DIR}/checkpoint_{safe_fname}_{timestamp}.json'
                with open(filename, 'w') as f:
                    # Save the specific result just obtained
                    json.dump({result_key: strategy_results[result_key]}, f, default=str)
                # Removed print statement for saving intermediate to reduce log noise
            except Exception as save_error:
                print(f"Warning: Could not save intermediate results checkpoint: {save_error}")
            
        except TimeoutError as te:
            print(f"Timeout running {strategy_name} for {args[0]} {args[1]}: {str(te)}")
            strategy_results[result_key] = {'error': f'Timeout: {str(te)}', **extract_key_metrics(None)}
            
    except Exception as e:
        print(f"Error in {strategy_name} for {args[0]} {args[1]}: {str(e)}")
        traceback.print_exc()
        strategy_results[result_key] = {'error': str(e), **extract_key_metrics(None)}

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

def adjust_parameters_for_timeframe_and_fees(token, timeframe, strategy_type):
    """
    Adjust base parameters based on timeframe and strategy type.
    Then, apply DEX-approximated fees based on the token type.
    
    Args:
        token (str): The trading pair (e.g., 'BTC/USDT').
        timeframe (str): The timeframe (e.g., '1h', '4h').
        strategy_type (str): 'stock', 'oscillator', 'hybrid', or 'enhanced_hybrid'.
        
    Returns:
        object: Appropriate parameters instance for the strategy with adjusted fees.
    """
    params = None
    base_mass = 1.0e-30 # Default base mass

    # Determine base mass multiplier based on market cap
    mass_multiplier = 1.0
    if token in HIGH_MARKET_CAP: mass_multiplier = 2.0
    elif token in MEDIUM_MARKET_CAP: mass_multiplier = 1.5
    else: mass_multiplier = 0.8 # Low cap
    base_mass *= mass_multiplier

    # --- Base Parameter Initialization & Timeframe Adjustments ---
    if strategy_type == 'stock':
        params = QuantumStockParameters()
        # Check if mass attributes exist before setting
        if hasattr(params, 'm'): params.m = base_mass
        if hasattr(params, 'mass'): params.mass = base_mass
        # Adjust omega based on timeframe
        if hasattr(params, 'omega'):
            if timeframe == '1d': params.omega = 2.0e-4
            elif timeframe == '4h': params.omega = 1.5e-4
            elif timeframe == '1h': params.omega = 1.0e-4
            elif timeframe == '30m': params.omega = 0.8e-4
            elif timeframe == '15m': params.omega = 0.6e-4
            elif timeframe == '5m': params.omega = 0.4e-4
            elif timeframe == '1m': params.omega = 0.3e-4 # Added 1m omega
            
    elif strategy_type == 'oscillator':
        params = QuantumOscillatorParameters()
        if hasattr(params, 'lambda_param'): params.lambda_param = 0.25
        # Adjust omega based on timeframe (using inverse periods)
        if hasattr(params, 'omega'):
            if timeframe == '1d': params.omega = 1/150
            elif timeframe == '4h': params.omega = 1/180
            elif timeframe == '1h': params.omega = 1/200
            elif timeframe == '30m': params.omega = 1/220
            elif timeframe == '15m': params.omega = 1/240
            elif timeframe == '5m': params.omega = 1/260
            elif timeframe == '1m': params.omega = 1/280 # Added 1m omega
    
    elif strategy_type == 'hybrid':
        params = QuantumHybridParameters()
        if hasattr(params, 'm'): params.m = base_mass
        if hasattr(params, 'mass'): params.mass = base_mass
        # Adjust omega based on timeframe
        if hasattr(params, 'omega'):
            if timeframe == '1d': params.omega = 5.0e-4
            elif timeframe == '4h': params.omega = 2.5e-4
            elif timeframe == '1h': params.omega = 1.1e-4
            elif timeframe == '30m': params.omega = 0.8e-4
            elif timeframe == '15m': params.omega = 0.6e-4
            elif timeframe == '5m': params.omega = 0.3e-4
            elif timeframe == '1m': params.omega = 0.2e-4 # Added 1m omega

    elif strategy_type == 'enhanced_hybrid':
        params = QuantumHybridEnhancedParameters()
        if hasattr(params, 'm'): params.m = base_mass
        if hasattr(params, 'mass'): params.mass = base_mass
        # Adjust omega based on timeframe (consistent with standalone test)
        if hasattr(params, 'omega'):
            if timeframe == '1d': params.omega = 5.0e-4
            elif timeframe == '4h': params.omega = 2.5e-4
            elif timeframe == '1h': params.omega = 1.1e-4
            elif timeframe == '30m': params.omega = 0.8e-4
            elif timeframe == '15m': params.omega = 0.6e-4
            elif timeframe == '5m': params.omega = 0.3e-4
            elif timeframe == '1m': params.omega = 0.3e-4 # Use 5m value for 1m test
        # Set default risk_pct if not already set (it should be in the class)
        if not hasattr(params, 'risk_pct') or params.risk_pct is None:
             setattr(params, 'risk_pct', 0.01) # Use setattr for safety
        # Ensure use_adaptive_position_sizing exists and is True
        if not hasattr(params, 'use_adaptive_position_sizing'):
             setattr(params, 'use_adaptive_position_sizing', True)
        elif getattr(params, 'use_adaptive_position_sizing') is not True:
             params.use_adaptive_position_sizing = True # Ensure it's enabled if exists but False

    else:
        print(f"Warning: Unknown strategy type '{strategy_type}' for parameter adjustment. Using defaults.")
        # Fallback or raise error? For now, maybe return None or default params
        params = QuantumStockParameters() # Example fallback
        # Apply default mass multiplier to fallback
        if hasattr(params, 'm'): params.m = base_mass
        if hasattr(params, 'mass'): params.mass = base_mass

    # --- Apply DEX Fee Approximations ---
    if params is not None:
        # Ensure fee attributes exist and set DEX fees
        if token == 'BTC/USDT':
            setattr(params, 'commission_pct', 0.0040) # 0.4% (P2P DEX + Mining Fee Approx)
            setattr(params, 'slippage_pct', 0.0010)   # 0.1% (Higher P2P Slippage Approx)
        elif token == 'SOL/USDT' or token in [t.split('/')[0] for t in SMALL_MARKET_CAP if 'SOL' in t]: # Basic check for Solana ecosystem
            setattr(params, 'commission_pct', 0.0030) # 0.3% (Raydium LP Fee + Gas Approx)
            setattr(params, 'slippage_pct', 0.0005)   # 0.05% (Standard AMM Slippage)
        else: # Assume ETH or other EVM token
            setattr(params, 'commission_pct', 0.0040) # 0.4% (Uniswap LP Fee + Gas Approx)
            setattr(params, 'slippage_pct', 0.0005)   # 0.05% (Standard AMM Slippage)

        # Ensure risk_pct exists for all applicable strategies (Stock, Hybrid, Enhanced Hybrid)
        if strategy_type in ['stock', 'hybrid', 'enhanced_hybrid']:
             if not hasattr(params, 'risk_pct'): setattr(params, 'risk_pct', 0.01)
             if not hasattr(params, 'use_adaptive_position_sizing'): setattr(params, 'use_adaptive_position_sizing', True)
             elif getattr(params, 'use_adaptive_position_sizing') is not True: params.use_adaptive_position_sizing = True

        print(f"Adjusted Params for {token} {timeframe} ({strategy_type}): omega={getattr(params, 'omega', 'N/A'):.1e}, commission={params.commission_pct:.4f}, slippage={params.slippage_pct:.4f}")

    return params

def find_latest_checkpoint(token, timeframe, results_dir):
    """Find the latest checkpoint file for a given token and timeframe."""
    safe_token = token.replace("/", "_")
    pattern = os.path.join(results_dir, f"checkpoint_{safe_token}_{timeframe}_*_*.json")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None, 0, {}

    # Find the latest file based on timestamp in the filename or modification time
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)

    # Try to extract completed count and load results
    try:
        # Extract completed count from filename (e.g., checkpoint_BTC_USDT_1m_3of5_....json)
        parts = Path(latest_checkpoint).stem.split('_')
        completed_count = 0
        for part in reversed(parts):
             if 'of' in part:
                 completed_count = int(part.split('of')[0])
                 break

        with open(latest_checkpoint, 'r') as f:
            data = json.load(f)
        # Ensure the loaded data is the results dict (handle old/different formats)
        results = data.get('results', data if isinstance(data, dict) and any(s in data for s in ['Quantum', 'Balanced', 'Conservative', 'Optimal']) else {})

        print(f"Found latest checkpoint: {latest_checkpoint} (completed {completed_count} strategies)")
        return latest_checkpoint, completed_count, results
    except Exception as e:
        print(f"Warning: Error reading checkpoint {latest_checkpoint}: {e}. Starting fresh.")
        return None, 0, {}

def run_strategies_for_token_timeframe(token, timeframe, start_date, end_date, initial_capital, results_dir, strategy_config, params_map=None):
    """Run all selected strategies for a single token-timeframe combination, resuming from checkpoint."""
    overall_start_time = time.time()
    max_token_timeframe_runtime = 3600  # 1 hour max per token-timeframe combo
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"Token-timeframe starting memory: {initial_memory:.1f} MB")

    # --- Checkpoint Loading --- #
    latest_checkpoint_file, completed_count, strategy_results = find_latest_checkpoint(token, timeframe, results_dir)
    if latest_checkpoint_file:
        print(f"Resuming {token} {timeframe} from checkpoint, starting after strategy #{completed_count}")
    else:
        print(f"No valid checkpoint found for {token} {timeframe}. Starting fresh.")
        strategy_results = {} # Ensure it's initialized if no checkpoint found
        completed_count = 0

    # Use the global/passed strategies list (assuming it will be passed or accessible)
    # Need to adjust how the strategies list is accessed if not global.
    # For now, assume it's accessible or passed in a future step.
    # We will define it in run_all_strategies and use it there.
    global defined_strategies_list # Temporary measure if needed, better to pass it
    strategies_to_run = defined_strategies_list # Use the list defined in the calling scope
    
    # Run each strategy with checkpoints
    for i, (strategy_name, strategy_func, param_type, needs_capital) in enumerate(strategies_to_run):

        # --- Resume Logic --- #
        current_strategy_index = i + 1 # 1-based index for comparison
        if current_strategy_index <= completed_count:
             print(f"Skipping strategy #{current_strategy_index}: {strategy_name} (already completed in checkpoint)")
             continue

        # Memory and time checks (as before)
        current_memory = process.memory_info().rss / 1024 / 1024
        if current_memory > 2000: # Handle high memory
             print(f"WARNING: High memory ({current_memory:.1f} MB). GC & Pause...")
             gc.collect(); time.sleep(5)
             current_memory = process.memory_info().rss / 1024 / 1024
             if current_memory > 2500:
                 print("CRITICAL: Memory still too high. Skipping remaining.")
                 break
        if time.time() - overall_start_time > max_token_timeframe_runtime:
            print("WARNING: Token-timeframe taking too long. Skipping remaining.")
            break
        
        # --- Get Adjusted Parameters (including fees) ---
        print(f"\n--- Running: {strategy_name} --- ")
        params = adjust_parameters_for_timeframe_and_fees(token, timeframe, param_type)

        # Apply specific parameter tweaks if needed (safer with hasattr)
        if strategy_name == 'Quantum Stock - Aggressive' and params and hasattr(params, 'hbar'):
            params.hbar = 1.5
        elif strategy_name == 'Quantum Stock - Balanced' and params and hasattr(params, 'hbar'):
             params.hbar = 1.1

        # Check if params object is valid for strategies that require it
        if params is None and param_type not in ['oscillator']: # Oscillator Classic might work without params?
             print(f"Warning: Could not generate parameters for {strategy_name}. Skipping.")
             continue

        # --- Prepare arguments for safe_run_strategy ---
        args_list = [token, timeframe]
        kwargs_dict = {}
        if needs_capital:
            args_list.append(initial_capital)
        args_list.extend([start_date, end_date])
        if params is not None:
             kwargs_dict = {"params": params}
        elif strategy_name == 'Quantum Oscillator - Classic':
             kwargs_dict = {}

        # --- Run the strategy ---
        strategy_results = safe_run_strategy(
            strategy_name,
            strategy_func,
            strategy_results, # Pass current results dict
            *args_list,
            **kwargs_dict
        )

        # --- Save Checkpoint --- #
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_token = token.replace("/", "_")
            # Use loop index `i` (which is 0-based) + 1 for the count in the filename
            checkpoint_path = f'{RESULTS_DIR}/checkpoint_{safe_token}_{timeframe}_{i+1}of{len(strategies_to_run)}_{timestamp}.json'
            with open(checkpoint_path, 'w') as f:
                json.dump({"token": token, "timeframe": timeframe, "results": strategy_results, 
                           "completed_strategies": i+1, "total_strategies": len(strategies_to_run)}, f, default=str)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint after {strategy_name}: {e}")
    
    # Final memory usage (as before)
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_change = final_memory - initial_memory
    print(f"Token-timeframe final memory: {final_memory:.1f} MB (change: {memory_change:.1f} MB)")
    
    return strategy_results # Return the final accumulated results for this combo

def prefetch_and_cache_data(tokens, timeframes, start_date, end_date, original_fetch_func):
    """
    Prefetch data using the original fetch function.
    Adjust date range for smaller timeframes.
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
            
            # --- REMOVE adjusted date range logic for 1m ---
            fetch_start_date = start_date
            fetch_end_date = end_date
            # days_back = None
            # if timeframe == '1m': days_back = 7 # Fetch 7 days for 1m
            # # Only fetch limited data for 1m to avoid excessive downloads
            # # Add more rules here if needed (e.g., 14 days for 5m)
            # # elif timeframe == '5m': days_back = 14

            # is_full_period = True
            # if days_back:
            #     is_full_period = False
            #     try:
            #         end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            #         # Ensure start date doesn't go before the overall period start
            #         overall_start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            #         adjusted_start_dt = max(overall_start_dt, end_dt - timedelta(days=days_back))
            #         fetch_start_date = adjusted_start_dt.strftime('%Y-%m-%d')
            #         # print(f"  Using adjusted period ({days_back} days) for {timeframe}: {fetch_start_date} to {fetch_end_date}")
            #     except ValueError:
            #          print(f" Error parsing dates, using full period for {timeframe}.")
            #          is_full_period = True # Revert if error
            #          fetch_start_date = start_date
            # Always use the full period now
            is_full_period = True

            # Cache file name includes the actual fetch dates
            cache_file = os.path.join(DATA_CACHE_DIR, f"{token_safe}_{timeframe}_{fetch_start_date}_{fetch_end_date}.pkl")
            
            # Check cache
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f: data = pickle.load(f)
                    candle_count = len(data) if isinstance(data, pd.DataFrame) else 0
                    print(f"  {token} {timeframe}: ✅ (Using cached data: {candle_count} candles)")
                    availability[token][timeframe] = {"available": candle_count > 100, "candle_count": candle_count, "cache_file": cache_file, "start_date": fetch_start_date, "end_date": fetch_end_date, "is_full_period": is_full_period}
                    continue
                except Exception as e: print(f"  Error loading cache: {e}")
            
            # Fetching part within prefetch:
            try:
                print(f"  Fetching {token} {timeframe} data from {fetch_start_date} to {fetch_end_date}...")
                data = original_fetch_func(token, timeframe, fetch_start_date, fetch_end_date)
                candle_count = len(data) if isinstance(data, pd.DataFrame) else 0
                is_available = candle_count > 100
                if is_available:
                    with open(cache_file, 'wb') as f: pickle.dump(data, f)
                    print(f"  {token} {timeframe}: ✅ ({candle_count} candles, cached)")
                else: print(f"  {token} {timeframe}: ❌ (Insufficient data: {candle_count})")
                availability[token][timeframe] = {"available": is_available, "candle_count": candle_count, "cache_file": cache_file if is_available else None, "start_date": fetch_start_date, "end_date": fetch_end_date, "is_full_period": is_full_period}
            except Exception as e:
                print(f"  {token} {timeframe}: ❌ Error fetching: {e}")
                availability[token][timeframe] = {"available": False, "error": str(e), "start_date": fetch_start_date, "end_date": fetch_end_date, "is_full_period": is_full_period}
    print("Prefetching complete!")
    return availability

# --- Optuna Objective Function Definition ---
def objective(trial, start_date, end_date, initial_capital):
    """Optuna objective function for Quantum Stock Hybrid."""

    # Define search space - adjust parameter names and ranges based on QuantumHybridParameters
    params = QuantumHybridParameters() # Start with default parameters

    # Suggest values for tunable hyperparameters
    params.conservative_weight = trial.suggest_float('conservative_weight', 0.1, 0.9, step=0.1)
    params.aggressive_weight = 1.0 - params.conservative_weight # Derived parameter

    # Use 'entry_threshold' as found in the class definition
    if hasattr(params, 'entry_threshold'):
        params.entry_threshold = trial.suggest_float('entry_threshold', 0.3, 1.0, step=0.05)

    # Use '..._factor' names as found in the class definition
    if hasattr(params, 'stop_loss_atr_factor') and hasattr(params, 'take_profit_atr_factor'):
        params.stop_loss_atr_factor = trial.suggest_float('stop_loss_atr_factor', 1.0, 4.0, step=0.2)
        params.take_profit_atr_factor = trial.suggest_float('take_profit_atr_factor', 1.5, 6.0, step=0.2)
    # Example: Suggest ATR period if tunable
    if hasattr(params, 'atr_period'):
         params.atr_period = trial.suggest_int('atr_period', 10, 30)

    target_metric_values = []
    valid_runs = 0

    print(f"--- Starting Trial {trial.number} ---")
    print(f"Parameters: {trial.params}")

    for token, timeframe in OPTIMIZATION_TARGETS:
        print(f"  Trial {trial.number}: Running target {token} {timeframe}...") # ADDED: More verbose logging
        try:
            # Run backtest with suggested parameters for this target
            # The backtest function should use the passed params object
            stats, _ = run_quantum_hybrid_strategy_backtest(
                symbol=token,
                timeframe=timeframe,
                initial_capital=initial_capital,
                start_date=start_date,
                end_date=end_date,
                params=params # Pass the object with suggested values
            )

            if stats:
                metrics = extract_key_metrics(stats)
                metric_value = metrics.get(OPTIMIZATION_METRIC)
                if metric_value is not None and np.isfinite(metric_value):
                    target_metric_values.append(metric_value)
                    valid_runs += 1
                    # print(f"    Trial {trial.number} - {token} {timeframe}: {OPTIMIZATION_METRIC} = {metric_value:.4f}") # Verbose
                else:
                    pass # Silently ignore missing/invalid metrics for this target
                    # print(f"    Trial {trial.number} - {token} {timeframe}: Metric '{OPTIMIZATION_METRIC}' missing or invalid.")
            # else:
                # print(f"    Trial {trial.number} - {token} {timeframe}: Backtest returned no stats.")

        except Exception as e:
            print(f"    Trial {trial.number} - {token} {timeframe}: Error during backtest - {e}")
            traceback.print_exc() # Uncommented for full error details
            # Don't add to target_metric_values if error occurs
            pass
        finally:
            gc.collect() # Collect garbage after each target run

    # Calculate average metric value across successful runs
    if valid_runs > 0:
        average_metric = np.mean(target_metric_values)
        print(f"--- Trial {trial.number} Result: Avg {OPTIMIZATION_METRIC} = {average_metric:.4f} (over {valid_runs}/{len(OPTIMIZATION_TARGETS)} targets) ---")
        # Handle potential non-finite average
        if not np.isfinite(average_metric):
             print(f"--- Trial {trial.number} Warning: Non-finite average metric. Returning poor score. ---")
             return -np.inf # Return poor score if average is non-finite
        return average_metric
    else:
        print(f"--- Trial {trial.number} Failed: No valid target runs completed. Returning poor score. ---")
        # Return a very poor score if maximizing, or very high if minimizing
        return -np.inf # Assuming maximization

# --- END Optuna Objective Function ---

def run_all_strategies(tokens, timeframes, start_date, end_date, results_dir, oscillator_available, optimize_strategy=None, n_trials=DEFAULT_N_TRIALS):
    """
    Runs all configured strategies for the given tokens, timeframes, and date range.
    Can optionally run optimization for a specified strategy instead of full backtesting.
    Accepts oscillator_available flag to conditionally include oscillator strategies.
    """
    results_file_path = os.path.join(results_dir, f"partial_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    completed_combos_file = os.path.join(results_dir, "completed_combinations.json")
    overall_results = {}
    completed_combos = set()

    # Load previous results if they exist
    # Search for the latest analysis_results or partial_results file
    existing_results_files = list(Path(results_dir).glob("analysis_results_*.json")) + list(Path(results_dir).glob("partial_results_*.json"))
    if existing_results_files:
        latest_results_file = max(existing_results_files, key=os.path.getctime)
        print(f"Loading previous results from: {latest_results_file}")
        try:
            with open(latest_results_file, 'r') as f:
                overall_results = json.load(f)
            print(f"Loaded {len(overall_results)} token results.")
        except Exception as e:
            print(f"Error loading previous results: {e}. Starting fresh.")
            overall_results = {}

    # Load completed combinations if file exists
    if os.path.exists(completed_combos_file):
        try:
            with open(completed_combos_file, 'r') as f:
                completed_combos = set(json.load(f))
            print(f"Loaded {len(completed_combos)} completed combinations.")
        except Exception as e:
            print(f"Error loading completed combinations file: {e}. Assuming none completed.")
            completed_combos = set()


    initial_capital_setting = 1000.0 # Keep the realistic capital setting

    # Define the strategies to run (adjust based on imports and availability)
    strategy_config = {
        "Quantum Stock - Balanced": run_quantum_stock_strategy_backtest,
        "Quantum Stock - Aggressive": run_quantum_stock_strategy_backtest,
        "Quantum Stock Hybrid": run_quantum_hybrid_strategy_backtest,
        "Quantum Hybrid Enhanced": run_quantum_hybrid_enhanced_strategy_backtest,
    }
    if oscillator_available:
        strategy_config["Quantum Oscillator - Classic"] = run_quantum_oscillator_strategy_backtest
    else:
        print("Oscillator strategies are unavailable or failed to import, skipping...")


    # --- Optimization Mode ---
    if optimize_strategy:
        if optimize_strategy != OPTIMIZE_STRATEGY_NAME_INTERNAL:
             print(f"Error: Optimization only implemented for '{OPTIMIZE_STRATEGY_NAME_INTERNAL}'. Requested: '{optimize_strategy}'")
             return

        print(f"--- Starting Optimization for {optimize_strategy} ---")
        print(f"Targets: {OPTIMIZATION_TARGETS}")
        print(f"Metric: {OPTIMIZATION_METRIC} ({OPTIMIZATION_DIRECTION})")
        print(f"Trials: {n_trials}")

        # --- Define Study Name and Storage --- #
        study_name = f"{optimize_strategy.replace(' ', '_')}_optimization"
        # Use SQLite for persistent storage
        # First, construct the actual database file path
        db_file_path = os.path.join(results_dir, f'{study_name}.db')
        # Then, create the storage connection string
        storage_name = f"sqlite:///{db_file_path}"
        print(f"Using Optuna study '{study_name}' with storage: {storage_name}")
        # ------------------------------------ #

        # Optional: Prefetch data for optimization targets here if needed
        print("Prefetching data for optimization targets...")
        # Extract unique tokens and timeframes from optimization targets
        optimization_tokens = list(set([t[0] for t in OPTIMIZATION_TARGETS]))
        optimization_timeframes = list(set([t[1] for t in OPTIMIZATION_TARGETS]))
        # Correctly call prefetch_and_cache_data with all required arguments
        prefetch_and_cache_data(optimization_tokens, optimization_timeframes, start_date, end_date, fetch_crypto_data)
        print("Data prefetching complete.")

        # Create or load the study using persistent storage
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True, # Load existing study to resume
            direction=OPTIMIZATION_DIRECTION
        )

        # Prune completed trials if they exist (optional, good for clean restarts)
        completed_trial_numbers = [t.number for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trial_numbers:
             print(f"Resuming study. Found {len(completed_trial_numbers)} completed trials.")
             # Logic to potentially skip already completed trials can be added here if needed,
             # but Optuna's default optimize loop handles this fairly well.

        # Use a lambda to pass static arguments (dates, capital) to the objective function
        objective_with_args = lambda trial: objective(trial, start_date, end_date, initial_capital_setting)

        try:
            # Determine remaining trials needed
            n_completed = len(completed_trial_numbers)
            n_remaining = max(0, n_trials - n_completed)
            if n_remaining == 0:
                print("Study already has the target number of trials or more. No optimization needed.")
            else:
                print(f"Optimizing for {n_remaining} more trials (target: {n_trials}, completed: {n_completed})...")
                study.optimize(objective_with_args, n_trials=n_remaining, timeout=None) # Optimize only for remaining trials
        except KeyboardInterrupt:
             print("\nOptimization interrupted by user.")
        except Exception as opt_e:
             print(f"\nAn error occurred during optimization: {opt_e}")
             print(traceback.format_exc())


        print("\n--- Optimization Finished ---")
        if study.trials and study.best_trial: # Check if any trials completed
            print(f"Best Trial Number: {study.best_trial.number}")
            print(f"Best Value (Avg {OPTIMIZATION_METRIC}): {study.best_value:.4f}")
            print("Best Parameters Found:")
            best_params_dict = study.best_trial.params
            for key, value in best_params_dict.items():
                print(f"  {key}: {value}")

            # Save the best params to a JSON file
            best_params_file = os.path.join(results_dir, f"best_params_{optimize_strategy.replace(' ', '_')}.json")
            try:
                # Convert params to a serializable format if needed (Optuna usually returns basic types)
                serializable_params = {}
                # Reconstruct full params object with defaults + optimized values
                best_params_obj = QuantumHybridParameters() # Start with default
                for key, value in best_params_dict.items():
                    if hasattr(best_params_obj, key):
                        setattr(best_params_obj, key, value)
                    else:
                        print(f"Warning: Best param '{key}' not found in QuantumHybridParameters class during save.")
                # Add derived/fixed params back if needed (like aggressive_weight)
                if 'conservative_weight' in best_params_dict:
                     setattr(best_params_obj, 'aggressive_weight', 1.0 - best_params_dict['conservative_weight'])

                # Convert object to dict for saving
                serializable_params = best_params_obj.__dict__

                # Add meta-info
                serializable_params['_optimization_info'] = {
                     'strategy': optimize_strategy,
                     'metric': OPTIMIZATION_METRIC,
                     'direction': OPTIMIZATION_DIRECTION,
                     'best_value': study.best_value,
                     'completed_trials': len(study.trials),
                     'targets': OPTIMIZATION_TARGETS,
                     'timestamp': datetime.now().isoformat()
                }

                with open(best_params_file, 'w') as f:
                    json.dump(serializable_params, f, indent=4)
                print(f"Best parameters saved to {best_params_file}")

            except Exception as save_e:
                print(f"Error saving best parameters to JSON: {save_e}")
        else:
            print("No successful trials completed or best trial not found.")

    # --- Normal Backtesting Mode ---
    else:
        print("--- Running Normal Backtesting ---")
        # Prefetch data for all combinations first (existing logic)
        all_symbols_timeframes = [(token, tf) for token in tokens for tf in timeframes]
        prefetch_and_cache_data(all_symbols_timeframes, start_date, end_date)

        # --- Load Optimized Params if available ---
        optimized_params_map = {}
        hybrid_params_file = os.path.join(results_dir, f"best_params_{OPTIMIZE_STRATEGY_NAME_INTERNAL.replace(' ', '_')}.json")
        if os.path.exists(hybrid_params_file):
             try:
                 with open(hybrid_params_file, 'r') as f:
                     loaded_params_dict = json.load(f)
                     print(f"Loading optimized parameters for {OPTIMIZE_STRATEGY_NAME_INTERNAL} from {hybrid_params_file}")
                     # Reconstruct the params object
                     opt_params_obj = QuantumHybridParameters()
                     for key, value in loaded_params_dict.items():
                          if key == '_optimization_info': continue # Skip metadata
                          if hasattr(opt_params_obj, key):
                               setattr(opt_params_obj, key, value)
                          else:
                               print(f"Warning: Unknown parameter '{key}' in loaded file.")
                     optimized_params_map[OPTIMIZE_STRATEGY_NAME_INTERNAL] = opt_params_obj
             except Exception as load_e:
                  print(f"Error loading optimized parameters for {OPTIMIZE_STRATEGY_NAME_INTERNAL}: {load_e}")
        # Add similar loading blocks here if you optimize other strategies
        # --- End Load Optimized Params ---


        # Main loop through tokens and timeframes (existing logic)
        for token in tokens:
            if token not in overall_results:
                overall_results[token] = {}
            for timeframe in timeframes:
                combo_key = f"{token}_{timeframe}"
                if combo_key in completed_combos:
                    if combo_key in overall_results.get(token,{}).get(timeframe,{}):
                         print(f"Skipping completed combination: {token} {timeframe}")
                         continue
                    else:
                         print(f"Combination {token} {timeframe} marked complete but missing results, rerunning...")
                         # Allow rerun by removing from completed set temporarily
                         # completed_combos.discard(combo_key)
                         # Or just proceed to run it

                print(f"\nProcessing combination: {token} {timeframe} with Initial Capital: ${initial_capital_setting:.2f}")

                results_for_combo = run_strategies_for_token_timeframe(
                    token,
                    timeframe,
                    start_date,
                    end_date,
                    initial_capital_setting,
                    results_dir, # Pass results_dir for checkpointing
                    strategy_config,
                    params_map=optimized_params_map # Pass the map of loaded optimized params
                )

                # Process results (existing logic, ensure update is safe)
                if results_for_combo is not None:
                     if timeframe not in overall_results.get(token, {}):
                         overall_results[token][timeframe] = {}
                     overall_results[token][timeframe].update(results_for_combo)
                     completed_combos.add(combo_key)

                     # Save intermediate results
                     try:
                          with open(results_file_path, 'w') as f:
                               json.dump(overall_results, f, indent=4)
                          # Save completed combos
                          with open(completed_combos_file, 'w') as f:
                               json.dump(list(completed_combos), f)
                     except Exception as e:
                          print(f"Error saving intermediate results: {e}")
                else:
                     print(f"No results returned for {token} {timeframe}. Combination might have failed entirely.")

        # Final save after loop finishes
        final_results_file = os.path.join(results_dir, f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print(f"\nSaving final results to {final_results_file}...")
        try:
            with open(final_results_file, 'w') as f:
                json.dump(overall_results, f, indent=4)
            # Clean up partial results file if desired and successful
            if os.path.exists(results_file_path):
                 try:
                      os.remove(results_file_path)
                      print(f"Removed intermediate file: {results_file_path}")
                 except OSError as e:
                      print(f"Error removing intermediate file {results_file_path}: {e}")
            print("Final results saved.")
        except Exception as e:
            print(f"Error saving final results: {e}")


# --- NEW Local Helper Function for Date Calculation ---
def calculate_start_date_from_period(period_str, end_date_str):
    """Calculates the start date string based on a period string (e.g., '1y', '6m') and an end date string."""
    try:
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        period_str = period_str.lower()
        num = int(period_str[:-1]) # Extract number
        unit = period_str[-1]     # Extract unit (y, m, d, w)

        if unit == 'y':
            start_date = end_date - timedelta(days=num * 365)
        elif unit == 'm':
            # Approximate months as 30 days for simplicity, adjust if precise month boundaries needed
            start_date = end_date - timedelta(days=num * 30)
        elif unit == 'w':
            start_date = end_date - timedelta(weeks=num)
        elif unit == 'd':
            start_date = end_date - timedelta(days=num)
        elif period_str == 'ytd':
             start_date = datetime(end_date.year, 1, 1)
        else:
            print(f"Warning: Unsupported period format '{period_str}'. Defaulting to 1 year.")
            start_date = end_date - timedelta(days=365)

        return start_date.strftime('%Y-%m-%d')

    except Exception as e:
        print(f"Error calculating start date from period '{period_str}': {e}. Defaulting to 1 year.")
        # Fallback to 1 year before end date
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            start_date = end_date - timedelta(days=365)
            return start_date.strftime('%Y-%m-%d')
        except:
             # Absolute fallback if end_date is also bad
             return (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
# --- END NEW Function ---

# --- Argument Parsing and Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive backtests or parameter optimization for crypto strategies.")
    parser.add_argument("--tokens", nargs='+', default=['BTC/USDT', 'ETH/USDT'], help="List of tokens (trading pairs) to backtest in normal mode.")
    parser.add_argument("--timeframes", nargs='+', default=['1h', '4h'], help="List of timeframes to backtest in normal mode (e.g., 1m, 5m, 1h, 1d).")
    parser.add_argument("--period", type=str, default="1y", help="Lookback period (e.g., 1y, 6m, 3m, 1m, YTD). Used for both modes.")
    parser.add_argument("--results_dir", type=str, default="comprehensive_results", help="Directory to save results, checkpoints, and optimized parameters.")
    parser.add_argument('--start_date', type=str, default=None, help='Start date in YYYY-MM-DD format (overrides --period).')
    parser.add_argument('--end_date', type=str, default=None, help='End date in YYYY-MM-DD format (defaults to today).')

    # Optuna arguments
    parser.add_argument('--optimize-strategy', type=str, default=None,
                        help=f'Specify strategy name to optimize individually per token/timeframe (e.g., "{OPTIMIZE_STRATEGY_NAME_INTERNAL}"). Runs optimization instead of backtest. Uses internal token/timeframe lists.')
    parser.add_argument('--n-trials', type=int, default=DEFAULT_N_TRIALS,
                        help='Number of trials PER token/timeframe combination during optimization.')

    args = parser.parse_args()

    # Determine date range
    end_dt_str = args.end_date if args.end_date else datetime.now().strftime('%Y-%m-%d')
    # Use the NEW local function if start_date is not provided
    start_dt_str = args.start_date if args.start_date else calculate_start_date_from_period(args.period, end_dt_str)

    print("-" * 50)
    print(f"Starting Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results Directory: {args.results_dir}")
    print(f"Data Period: {start_dt_str} to {end_dt_str}")

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

    # --- Apply Caching Patch ---
    # Ensure the fetch function used during backtests checks the cache first
    print("Applying data caching patch to fetch function...")
    original_fetch_func = patch_fetch_functions() # Call the patcher
    # -------------------------

    if args.optimize_strategy:
        print(f"Mode: Individual Parameter Optimization")
        print(f"Strategy to Optimize: {args.optimize_strategy}")
        print(f"Tokens for Optimization: {INDIVIDUAL_OPTIMIZATION_TOKENS}")
        print(f"Timeframes for Optimization: {INDIVIDUAL_OPTIMIZATION_TIMEFRAMES}")
        print(f"Trials per Combination: {args.n_trials}")
        # Run optimization - tokens/timeframes args are ignored, internal lists are used
        run_all_strategies(
            tokens=[], # Not used in optimization mode
            timeframes=[], # Not used in optimization mode
            start_date=start_dt_str,
            end_date=end_dt_str,
            results_dir=args.results_dir,
            oscillator_available=QUANTUM_OSCILLATOR_AVAILABLE, # Pass flag
            optimize_strategy=args.optimize_strategy,
            n_trials=args.n_trials
        )
    else:
        print(f"Mode: Normal Backtesting")
        print(f"Tokens: {args.tokens}")
        print(f"Timeframes: {args.timeframes}")
        # Run normal backtest using specified tokens/timeframes
        run_all_strategies(
            tokens=args.tokens,
            timeframes=args.timeframes,
            start_date=start_dt_str,
            end_date=end_dt_str,
            results_dir=args.results_dir,
            oscillator_available=QUANTUM_OSCILLATOR_AVAILABLE, # Pass flag
            optimize_strategy=None, # Ensure optimization mode is off
            n_trials=0 # Not used
        )

    # Optional: Restore original function if needed (usually not necessary for script runs)
    # if original_fetch_func:
    #     try:
    #         import app
    #         app.fetch_crypto_data = original_fetch_func
    #     except ImportError:
    #         globals()['fetch_crypto_data'] = original_fetch_func
    #     print("Restored original fetch function.")

    print("-" * 50)
    print(f"Run Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Comprehensive analysis run finished.") 