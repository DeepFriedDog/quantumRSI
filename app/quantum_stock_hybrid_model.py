import pandas as pd
import os
import psutil

def run_quantum_hybrid_strategy_backtest(symbol, timeframe, initial_capital, start_date, end_date, params=None):
    """
    Run the backtest for the Quantum Hybrid strategy.
    """
    if params is None:
        params = QuantumHybridParameters()

    # --- Data Fetching ---
    from .data_fetching import fetch_crypto_data # Ensure correct import path if needed
    print(f"Fetching data for {symbol} on {timeframe} timeframe from {start_date} to {end_date}...")
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    if df is None or df.empty:
        print(f"Error fetching data for {symbol} {timeframe} or data is empty.")
        return None, None
    print(f"Data fetched: {len(df)} candles from {df.index.min()} to {df.index.max()}")
    print(f"  Memory after fetch: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")

    # --- Indicator Calculations (with detailed logging) ---
    print("  Calculating volatility...")
    try:
        df = calculate_volatility(df, params)
        print("  Volatility calculated.")
        print(f"  Memory after volatility: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"  ERROR calculating volatility: {e}")
        return None, None

    print("  Calculating market imbalance...")
    try:
        df = calculate_market_imbalance(df, params)
        print("  Market imbalance calculated.")
        print(f"  Memory after imbalance: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"  ERROR calculating market imbalance: {e}")
        return None, None

    print("  Calculating energy levels...")
    try:
        df = calculate_energy_levels(df, params)
        print("  Energy levels calculated.")
        print(f"  Memory after energy: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"  ERROR calculating energy levels: {e}")
        return None, None

    print("  Calculating conservative signals...")
    try:
        df = calculate_conservative_signals(df, params)
        print("  Conservative signals calculated.")
        print(f"  Memory after conservative signals: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"  ERROR calculating conservative signals: {e}")
        return None, None

    print("  Calculating aggressive signals...")
    try:
        df = calculate_aggressive_signals(df, params)
        print("  Aggressive signals calculated.")
        print(f"  Memory after aggressive signals: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"  ERROR calculating aggressive signals: {e}")
        return None, None

    print("  Combining signals...")
    try:
        df['hybrid_signal'] = (params.conservative_weight * df['conservative_signal'] +
                               params.aggressive_weight * df['aggressive_signal'])
        print("  Signals combined.")
        print(f"  Memory after combining signals: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"  ERROR combining signals: {e}")
        return None, None

    # --- Backtest Initialization ---
    print("Starting backtest loop...")
    # ... (rest of the backtest logic remains the same) ... 