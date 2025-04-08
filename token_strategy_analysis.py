import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import io
import base64
import traceback
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import json
import ccxt
import pickle
from pathlib import Path

# Fix pandas to avoid SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Import app modules for strategy running
from app import (
    run_optimized_strategy_backtest, 
    run_simple_strategy_backtest,
    run_quantum_probability_strategy_backtest,
    run_quantum_stock_strategy_backtest,
    run_quantum_oscillator_strategy_backtest,
    run_quantum_anharmonic_strategy_backtest,
    run_quantum_tunneling_strategy_backtest,
    fetch_crypto_data
)

# Import parameter classes
from quantum_rsi_optimized import QuantumRSIParameters
from quantum_probability_model import QuantumProbabilityParameters
from quantum_stock_model import QuantumStockParameters
from quantum_oscillator_model import QuantumOscillatorParameters
from quantum_anharmonic_oscillator_model import QuantumAnharmonicParameters
from quantum_tunneling_model import QuantumTunnelingParameters

# Define token categories by market cap
HIGH_MARKET_CAP = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
MEDIUM_MARKET_CAP = ['LINK/USDT', 'DOT/USDT', 'AVAX/USDT', 'ATOM/USDT', 'NEAR/USDT', 'MATIC/USDT']
SMALL_MARKET_CAP = ['ALGO/USDT', 'RNDR/USDT', 'FTM/USDT', 'ONE/USDT', 'ZIL/USDT', 'ROSE/USDT', 'CHZ/USDT', 'ENJ/USDT', 'OCEAN/USDT', 'API3/USDT']

# Add a data caching directory constant after the market cap lists
DATA_CACHE_DIR = 'data_cache'

# Function to select a diverse set of tokens
def select_diverse_tokens(num_tokens=10):
    """Select a diverse set of tokens based on market cap."""
    # Ensure we don't request more tokens than available
    max_tokens = len(HIGH_MARKET_CAP) + len(MEDIUM_MARKET_CAP) + len(SMALL_MARKET_CAP)
    num_tokens = min(num_tokens, max_tokens)
    
    # Calculate distribution: 30% high, 30% medium, 40% small cap
    high_count = max(1, int(num_tokens * 0.3))
    medium_count = max(1, int(num_tokens * 0.3))
    small_count = num_tokens - high_count - medium_count
    
    selected_tokens = []
    selected_tokens.extend(random.sample(HIGH_MARKET_CAP, high_count))
    selected_tokens.extend(random.sample(MEDIUM_MARKET_CAP, medium_count))
    selected_tokens.extend(random.sample(SMALL_MARKET_CAP, small_count))
    
    print(f"Selected tokens: {selected_tokens}")
    return selected_tokens

# Function to generate a random 1-month period
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

# Function to prefetch and cache data
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
                    # Will fetch again below
            
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

# Function to get cached data
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

# Function to run all strategies for token-timeframe combinations
def run_all_strategies(tokens, timeframes, start_date, end_date, initial_capital=10000):
    """Run all strategies for each token-timeframe combination using cached data."""
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
                    # Run all strategies as in the "compare_all" option
                    strategy_results = run_all_strategies_for_token(
                        token, timeframe, initial_capital, start_date, end_date
                    )
                    
                    # Store results
                    results[token][timeframe] = strategy_results
                    
                except Exception as e:
                    print(f"Error testing {token} on {timeframe}: {str(e)}")
                    traceback.print_exc()
                    results[token][timeframe] = {"error": str(e)}
                
                # Save interim results after EACH timeframe to prevent data loss
                with open(f'analysis_results_{timestamp}_interim.json', 'w') as f:
                    json.dump(results, f, default=str)
                
                # Force garbage collection after each timeframe to free memory
                import gc
                gc.collect()
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f'analysis_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, default=str)
        
        print(f"\nAnalysis complete! Results saved to {filename}")
        return results, filename
    
    finally:
        # Restore original function when done
        import app
        app.fetch_crypto_data = original_fetch
        print("Restored original fetch_crypto_data function")

# Function to run all strategies for a single token-timeframe combination
def run_all_strategies_for_token(token, timeframe, initial_capital, start_date, end_date):
    """Run all strategies for a single token-timeframe combination."""
    strategy_results = {}
    
    # Conservative Quantum RSI - Make less conservative
    conservative_params = QuantumRSIParameters()
    conservative_params.rsi_high = 70  # Lower from 72
    conservative_params.rsi_low = 30   # Higher from 28
    conservative_params.volume_threshold = 1.3  # Lower from 1.5
    conservative_params.take_profit_atr = 1.2
    conservative_params.stop_loss_atr = 1.5  # Lower from 1.6
    conservative_params.cooldown_period = 4  # Lower from 6
    conservative_params.maxwell_demon_filter = True
    conservative_params.barrier_height = 1.0  # Lower barrier for easier crossover
    
    strategy_results = safe_run_strategy(
        'Conservative',
        run_optimized_strategy_backtest,
        strategy_results,
        token, timeframe, initial_capital, start_date, end_date, conservative_params
    )
    
    # Balanced Quantum RSI - Make more active
    balanced_params = QuantumRSIParameters()
    balanced_params.rsi_high = 65  # Lower threshold
    balanced_params.rsi_low = 35   # Higher threshold
    balanced_params.volume_threshold = 1.0
    balanced_params.cooldown_period = 3  # Reduced cooldown
    balanced_params.maxwell_demon_filter = True
    balanced_params.barrier_height = 0.8  # Lower barrier
    
    strategy_results = safe_run_strategy(
        'Balanced',
        run_optimized_strategy_backtest,
        strategy_results,
        token, timeframe, initial_capital, start_date, end_date, balanced_params
    )
    
    # Optimal strategy - Fix to produce trades
    optimal_params = {
        'rsi_period': 14,
        'rsi_high': 62,           # Lower from 65
        'rsi_low': 38,            # Higher from 35
        'v0': 1.2,                # Lower from 1.5
        'a': 0.8,                 # Lower from 1.0
        'mass': 1.0,              # Lower from 1.2
        'hbar': 1.3,              # Higher from 1.2
        'stop_loss_atr': 1.5,     # Same
        'take_profit_atr': 1.4,   # Same
        'cooldown_period': 3,     # Shorter from 6
        'max_consecutive_losses': 3,
        'debug_mode': True
    }
    
    strategy_results = safe_run_strategy(
        'Optimal',
        run_simple_strategy_backtest,
        strategy_results,
        token, timeframe, initial_capital, start_date, end_date, optimal_params
    )
    
    # Quantum Prob - Conservative - Make less conservative
    qp_conservative = QuantumProbabilityParameters()
    qp_conservative.alpha = 2.0  # Lower from 2.5
    qp_conservative.delta = 0.5  # Higher from 0.4
    qp_conservative.h = 1.0      # Higher from 0.8
    qp_conservative.omega_level = 2  # Higher from 1
    qp_conservative.stop_loss_atr = 1.6  # Lower from 1.8
    qp_conservative.take_profit_atr = 1.3  # Higher from 1.2
    qp_conservative.cooldown_period = 8  # Lower from 12
    
    strategy_results = safe_run_strategy(
        'Quantum Prob - Conservative',
        run_quantum_probability_strategy_backtest,
        strategy_results,
        token, timeframe, initial_capital, start_date, end_date, qp_conservative
    )
    
    # Quantum Prob - Aggressive - Already good but tweak
    qp_aggressive = QuantumProbabilityParameters()
    qp_aggressive.alpha = 1.4   # Lower from 1.5
    qp_aggressive.delta = 0.85  # Higher from 0.8
    qp_aggressive.h = 1.6       # Higher from 1.5
    qp_aggressive.omega_level = 5  # Same
    qp_aggressive.stop_loss_atr = 1.1  # Lower from 1.2
    qp_aggressive.take_profit_atr = 1.6  # Higher from 1.5
    qp_aggressive.cooldown_period = 4  # Lower from 5
    
    strategy_results = safe_run_strategy(
        'Quantum Prob - Aggressive',
        run_quantum_probability_strategy_backtest,
        strategy_results,
        token, timeframe, initial_capital, start_date, end_date, qp_aggressive
    )
    
    # Quantum Stock Model - Conservative - Already good but tweak
    qs_conservative = QuantumStockParameters()
    qs_conservative.m = 1.8e-30  # Lower from 2e-30
    qs_conservative.hbar = 0.9  # Higher from 0.8
    qs_conservative.F = 0.9e-19  # Higher from 0.8e-19
    qs_conservative.omega = 0.9e-4  # Higher from 0.8e-4
    qs_conservative.stop_loss_atr = 1.7  # Lower from 1.8
    qs_conservative.take_profit_atr = 1.3  # Higher from 1.2
    qs_conservative.cooldown_period = 10  # Lower from 12
    qs_conservative.signal_threshold = 0.65  # Lower from 0.7
    
    strategy_results = safe_run_strategy(
        'Quantum Stock - Conservative',
        run_quantum_stock_strategy_backtest,
        strategy_results,
        token, timeframe, initial_capital, start_date, end_date, qs_conservative
    )
    
    # Quantum Stock Model - Aggressive - Already produces trades
    qs_aggressive = QuantumStockParameters()
    qs_aggressive.m = 0.5e-30
    qs_aggressive.hbar = 1.2
    qs_aggressive.F = 1.2e-19
    qs_aggressive.omega = 1.2e-4
    qs_aggressive.stop_loss_atr = 1.2
    qs_aggressive.take_profit_atr = 1.5
    qs_aggressive.cooldown_period = 5
    qs_aggressive.signal_threshold = 0.5
    
    strategy_results = safe_run_strategy(
        'Quantum Stock - Aggressive',
        run_quantum_stock_strategy_backtest,
        strategy_results,
        token, timeframe, initial_capital, start_date, end_date, qs_aggressive
    )
    
    # Add Quantum Oscillator - Conservative
    try:
        # First check if the function exists in app
        import inspect
        from app import run_quantum_oscillator_strategy_backtest
        if not callable(run_quantum_oscillator_strategy_backtest):
            raise ImportError("run_quantum_oscillator_strategy_backtest is not callable")
            
        from quantum_oscillator_model import QuantumOscillatorParameters
        
        osc_conservative = QuantumOscillatorParameters()
        osc_conservative.lambda_param = 0.5
        osc_conservative.hbar = 1.0
        osc_conservative.omega = 1/120
        osc_conservative.set_spread_from_volatility(0.012)
        osc_conservative.stop_loss_atr = 1.7
        osc_conservative.take_profit_atr = 1.3
        osc_conservative.cooldown_period = 10
        
        strategy_results = safe_run_strategy(
            'Quantum Oscillator - Conservative',
            run_quantum_oscillator_strategy_backtest,
            strategy_results,
            token, timeframe, start_date, end_date, osc_conservative
        )
        
        # Add Quantum Oscillator - Aggressive
        osc_aggressive = QuantumOscillatorParameters()
        osc_aggressive.lambda_param = 0.75
        osc_aggressive.hbar = 1.5
        osc_aggressive.omega = 1/80
        osc_aggressive.set_spread_from_volatility(0.018)
        osc_aggressive.stop_loss_atr = 1.0
        osc_aggressive.take_profit_atr = 2.0
        osc_aggressive.cooldown_period = 3
        
        strategy_results = safe_run_strategy(
            'Quantum Oscillator - Aggressive',
            run_quantum_oscillator_strategy_backtest,
            strategy_results,
            token, timeframe, start_date, end_date, osc_aggressive
        )
    except (ImportError, AttributeError) as e:
        print(f"Could not run Quantum Oscillator strategies: {str(e)}")
    except Exception as e:
        print(f"Error in Quantum Oscillator setup: {str(e)}")
    
    # Quantum Anharmonic - Conservative
    try:
        # First check if the function exists in app
        import inspect
        from app import run_quantum_anharmonic_strategy_backtest
        if not callable(run_quantum_anharmonic_strategy_backtest):
            raise ImportError("run_quantum_anharmonic_strategy_backtest is not callable")
            
        anharmonic_conservative = QuantumAnharmonicParameters()
        anharmonic_conservative.beta = 0.10
        anharmonic_conservative.gamma = 0.05
        anharmonic_conservative.lambda_param = 0.3
        anharmonic_conservative.set_spread_from_volatility(0.010)
        anharmonic_conservative.stop_loss_atr = 1.7
        anharmonic_conservative.take_profit_atr = 1.2
        anharmonic_conservative.cooldown_period = 10
        
        strategy_results = safe_run_strategy(
            'Quantum Anharmonic - Conservative',
            run_quantum_anharmonic_strategy_backtest,
            strategy_results,
            token, timeframe, start_date, end_date, anharmonic_conservative
        )
    except (ImportError, AttributeError) as e:
        print(f"Could not run Quantum Anharmonic - Conservative: {str(e)}")
    except Exception as e:
        print(f"Error in Quantum Anharmonic - Conservative: {str(e)}")
    
    # Quantum Anharmonic - Aggressive - Already present
    try:
        params = QuantumAnharmonicParameters()
        params.beta = 0.15
        params.gamma = 0.08
        params.lambda_param = 0.4
        params.set_spread_from_volatility(0.012)
        params.stop_loss_atr = 1.2
        params.take_profit_atr = 1.5
        params.cooldown_period = 5
        
        strategy_results = safe_run_strategy(
            'Quantum Anharmonic - Aggressive',
            run_quantum_anharmonic_strategy_backtest,
            strategy_results,
            token, timeframe, start_date, end_date, params
        )
    except Exception as e:
        print(f"Error in Quantum Anharmonic - Aggressive: {str(e)}")
    
    # Quantum Tunneling - Conservative
    try:
        # First check if the function exists in app
        import inspect
        from app import run_quantum_tunneling_strategy_backtest
        if not callable(run_quantum_tunneling_strategy_backtest):
            raise ImportError("run_quantum_tunneling_strategy_backtest is not callable")
            
        tunneling_conservative = QuantumTunnelingParameters()
        tunneling_conservative.volatility = 0.25
        tunneling_conservative.lambda_param = 0.7
        tunneling_conservative.vol_drop_threshold = 0.25
        tunneling_conservative.range_threshold = 0.25
        tunneling_conservative.min_range_days = 5
        tunneling_conservative.tunneling_threshold = 0.1
        tunneling_conservative.stop_loss_atr = 1.7
        tunneling_conservative.take_profit_atr = 1.3
        tunneling_conservative.lookback_period = 20
        tunneling_conservative.cooldown_period = 10
        
        strategy_results = safe_run_strategy(
            'Quantum Tunneling - Conservative',
            run_quantum_tunneling_strategy_backtest,
            strategy_results,
            token, timeframe, start_date, end_date, tunneling_conservative
        )
    except (ImportError, AttributeError) as e:
        print(f"Could not run Quantum Tunneling - Conservative: {str(e)}")
    except Exception as e:
        print(f"Error in Quantum Tunneling - Conservative: {str(e)}")
    
    # Quantum Tunneling - Aggressive - Make more aggressive to generate more trades
    try:
        tunneling_aggressive = QuantumTunnelingParameters()
        tunneling_aggressive.volatility = 0.3        # Lower from 0.35
        tunneling_aggressive.lambda_param = 0.45     # Lower from 0.5
        tunneling_aggressive.vol_drop_threshold = 0.08  # Lower from 0.1
        tunneling_aggressive.range_threshold = 0.12   # Lower from 0.15
        tunneling_aggressive.min_range_days = 2       # Lower from 3
        tunneling_aggressive.tunneling_threshold = 0.02 # Lower from 0.03
        tunneling_aggressive.stop_loss_atr = 1.1      # Lower from 1.2
        tunneling_aggressive.take_profit_atr = 2.2    # Higher from 2.0
        tunneling_aggressive.lookback_period = 8      # Lower from 10
        tunneling_aggressive.cooldown_period = 2      # Lower from 3
        
        strategy_results = safe_run_strategy(
            'Quantum Tunneling - Aggressive',
            run_quantum_tunneling_strategy_backtest,
            strategy_results,
            token, timeframe, start_date, end_date, tunneling_aggressive
        )
    except Exception as e:
        print(f"Error in Quantum Tunneling - Aggressive: {str(e)}")
    
    # Add momentum strategies if available
    try:
        # First check if the function exists in app
        from app import run_quantum_momentum_strategy_backtest
        if not callable(run_quantum_momentum_strategy_backtest):
            raise ImportError("run_quantum_momentum_strategy_backtest is not callable")
            
        from quantum_momentum_model import QuantumMomentumParameters
        
        # Conservative Momentum
        momentum_conservative = QuantumMomentumParameters()
        momentum_conservative.lookback_period = 20
        momentum_conservative.momentum_threshold = 0.01
        momentum_conservative.quantum_factor = 0.7
        momentum_conservative.stop_loss_atr = 1.7
        momentum_conservative.take_profit_atr = 1.3
        momentum_conservative.cooldown_period = 10
        
        strategy_results = safe_run_strategy(
            'Quantum Momentum - Conservative',
            run_quantum_momentum_strategy_backtest,
            strategy_results,
            token, timeframe, start_date, end_date, momentum_conservative
        )
        
        # Aggressive Momentum
        momentum_aggressive = QuantumMomentumParameters()
        momentum_aggressive.lookback_period = 10
        momentum_aggressive.momentum_threshold = 0.005
        momentum_aggressive.quantum_factor = 1.0
        momentum_aggressive.stop_loss_atr = 1.2
        momentum_aggressive.take_profit_atr = 1.8
        momentum_aggressive.cooldown_period = 4
        
        strategy_results = safe_run_strategy(
            'Quantum Momentum - Aggressive',
            run_quantum_momentum_strategy_backtest,
            strategy_results,
            token, timeframe, start_date, end_date, momentum_aggressive
        )
    except (ImportError, AttributeError) as e:
        print(f"Could not run Quantum Momentum strategies: {str(e)}")
    except Exception as e:
        print(f"Error in Quantum Momentum setup: {str(e)}")
        
    return strategy_results

# Helper to extract key metrics from strategy results
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

# Function to create comprehensive visualizations
def create_visualizations(results, output_file='strategy_analysis.png'):
    """Create comprehensive visualizations of the results."""
    print("Creating visualizations...")
    
    # Convert results to a flat DataFrame for easier plotting
    rows = []
    for token, timeframe_data in results.items():
        for timeframe, strategy_data in timeframe_data.items():
            if isinstance(strategy_data, dict) and 'error' not in strategy_data:
                for strategy, metrics in strategy_data.items():
                    rows.append({
                        'token': token,
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
    
    # Create token-specific visualization
    create_token_comparison(df, 'token_comparison.png')

# Function to create token-specific visualization
def create_token_comparison(df, output_file='token_comparison.png'):
    """Create token-specific visualization."""
    # Get top 3 strategies by average return
    top_strategies = df.groupby('strategy')['total_return'].mean().nlargest(3).index.tolist()
    
    # Filter data for these strategies
    top_df = df[df['strategy'].isin(top_strategies)]
    
    # Create visualization
    plt.figure(figsize=(16, 10))
    
    # 1. Token performance by strategy
    ax1 = plt.subplot(2, 1, 1)
    token_pivot = top_df.pivot_table(
        values='total_return', 
        index='token', 
        columns='strategy', 
        aggfunc='mean'
    ).sort_values(by=top_strategies[0], ascending=False)
    
    token_pivot.plot(kind='bar', ax=ax1)
    ax1.set_title('Token Performance by Top Strategies', fontsize=14)
    ax1.set_ylabel('Return (%)')
    ax1.set_xlabel('')
    ax1.legend(title='Strategy')
    
    # 2. Timeframe performance by strategy
    ax2 = plt.subplot(2, 1, 2)
    timeframe_pivot = top_df.pivot_table(
        values='total_return', 
        index='timeframe', 
        columns='strategy', 
        aggfunc='mean'
    )
    
    timeframe_pivot.plot(kind='bar', ax=ax2)
    ax2.set_title('Timeframe Performance by Top Strategies', fontsize=14)
    ax2.set_ylabel('Return (%)')
    ax2.set_xlabel('Timeframe')
    ax2.legend(title='Strategy')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Token comparison visualization saved to {output_file}")
    plt.close()

# Add this function to the top of the file, after imports
def setup_memory_management():
    """Set up memory management to prevent memory-related crashes."""
    import gc
    import sys
    
    # More aggressive garbage collection
    gc.set_threshold(100, 5, 5)  # Lower threshold for garbage collection
    
    # Print memory usage information
    try:
        import psutil
        def print_memory_usage():
            process = psutil.Process()
            print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
        
        # Set up a timer to periodically check memory usage
        import threading
        def memory_check():
            print_memory_usage()
            threading.Timer(60, memory_check).start()  # Check every 60 seconds
        
        memory_check()
    except ImportError:
        print("psutil not installed. Memory usage monitoring disabled.")

# Now patch all the backtest functions to use cached data
# Create a monkey-patching function to modify the app module's fetch_crypto_data temporarily
def patch_fetch_functions():
    """
    Monkey patch the fetch_crypto_data function to use cached data, 
    and apply similar patches to all strategy backtest functions.
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

# Update the main function to use memory management
def main():
    """Main function to run the analysis."""
    global token_data_cache  # Make the cache available to all functions
    token_data_cache = {}
    
    # Initialize memory management
    try:
        import psutil
        print("Setting up memory management...")
        setup_memory_management()
    except ImportError:
        print("psutil not installed. Continuing without memory monitoring.")
    
    # Create results directory if it doesn't exist
    os.makedirs('strategy_analysis', exist_ok=True)
    
    # Select 10 tokens from different market cap categories
    tokens = select_diverse_tokens(10)
    
    # Choose 3 timeframes
    timeframes = ['5m', '15m', '1h']
    
    # Generate random 1-month period
    start_date, end_date = generate_random_month_period()
    
    # Prefetch and check data availability - this will cache all data
    availability = prefetch_and_cache_data(tokens, timeframes, start_date, end_date)
    
    # Filter tokens and timeframes to only include those with available data
    valid_tokens = []
    for token, timeframe_data in availability.items():
        valid_timeframes = [tf for tf, data in timeframe_data.items() if data.get('available', False)]
        if len(valid_timeframes) == len(timeframes):
            valid_tokens.append(token)
    
    # Ensure we have enough tokens
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
    if len(valid_tokens) < 5:
        print(f"Warning: Only {len(valid_tokens)} tokens have data for all timeframes.")
        if not valid_tokens:
            print("No tokens with valid data for all timeframes. Exiting.")
            return
    
    # Limit token count for faster analysis but ensure minimum of 3 tokens
    max_tokens = min(5, max(3, len(valid_tokens)))
    valid_tokens = valid_tokens[:max_tokens]
    print(f"Running analysis on {len(valid_tokens)} tokens for faster completion...")
    
    # Store the availability info globally for the run_all_strategies function to use
    token_data_cache = {
        'start_date': start_date,
        'end_date': end_date,
        'availability': availability
    }
    
    try:
        # Run the analysis on valid tokens
        results, results_file = run_all_strategies(valid_tokens, timeframes, start_date, end_date)
        
        # Create visualizations
        create_visualizations(results, f'strategy_analysis/comprehensive_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.png')
        
        print("Analysis complete!")
    except Exception as e:
        print(f"ERROR: Analysis failed with error: {str(e)}")
        traceback.print_exc()
        print("Attempting to save partial results...")
        
        # Try to save whatever results we have
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        try:
            with open(f'analysis_results_PARTIAL_{timestamp}.json', 'w') as f:
                json.dump({"error": str(e), "partial_results": token_data_cache}, f, default=str)
            print(f"Partial results saved to analysis_results_PARTIAL_{timestamp}.json")
        except Exception as save_error:
            print(f"Failed to save partial results: {str(save_error)}")

# Replace the check_data_availability function with the below version
def check_data_availability(tokens, timeframes, start_date, end_date):
    """Check data availability for each token and timeframe, using cached data if available."""
    print("Checking data availability...")
    
    # Just call prefetch_and_cache_data which will handle both checking and caching
    return prefetch_and_cache_data(tokens, timeframes, start_date, end_date)

# Update the safe_run_strategy function to handle dimension mismatch for tunneling model too
def safe_run_strategy(strategy_name, strategy_func, strategy_results, *args, **kwargs):
    """Run a strategy with enhanced error handling to prevent crashes."""
    print(f"Running {strategy_name}...")
    try:
        # For quantum oscillator/tunneling specifically, add a try-except to catch dimension mismatch
        if "oscillator" in strategy_name.lower() or "tunneling" in strategy_name.lower():
            try:
                stats, _ = strategy_func(*args, **kwargs)
            except ValueError as ve:
                if "x and y must have same first dimension" in str(ve):
                    # This is the dimension mismatch error in plotting - extract results without plotting
                    print(f"Caught dimension mismatch error in {strategy_name}. Attempting to run without plotting...")
                    
                    # Add plot=False to kwargs if it exists and is True
                    if kwargs.get('plot', True):
                        # Create a copy of kwargs to avoid modifying the original
                        modified_kwargs = dict(kwargs)
                        modified_kwargs['plot'] = False
                        stats, _ = strategy_func(*args, **modified_kwargs)
                    else:
                        # Re-raise if we can't fix it
                        raise
                else:
                    # Re-raise other ValueError types
                    raise
        else:
            # Normal execution for non-oscillator strategies
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
            with open(f'token_analysis_intermediate_{args[0].split("/")[0]}_{args[1]}.json', 'w') as f:
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
        # Continue despite errors
    
    return strategy_results

# Add a function to monitor and manage system resources
def monitor_resources():
    """Monitor system resources and print warning if running low."""
    try:
        import psutil
        
        # Check current memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Log memory usage
        print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB ({memory_percent:.1f}%)")
        
        # If memory usage is high, force garbage collection
        if memory_percent > 70:
            print("WARNING: High memory usage detected. Forcing garbage collection...")
            import gc
            gc.collect()
            
            # Check if memory is still high after collection
            memory_percent_after = process.memory_percent()
            if memory_percent_after > 75:
                print("CRITICAL: Memory usage still high after garbage collection. Consider reducing analysis scope.")
                
        return memory_percent < 85  # Return True if memory usage is acceptable
        
    except ImportError:
        print("psutil not installed. Cannot monitor system resources.")
        return True  # Assume resources are OK if we can't check

if __name__ == "__main__":
    main() 