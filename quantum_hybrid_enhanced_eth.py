#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Quantum Harmonic-Wave Hybrid Model (ETH Specific Tuning)

This module implements an enhanced hybrid quantum model combining aspects of
the Quantum Stock Conservative and Aggressive models, with added regime 
filtering (EMA slope) and dynamic ATR trailing stops.

This version is specifically designated for tuning parameters for ETH/USDT.
"""

import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import norm
import math
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Helper function to calculate EMA slope
def calculate_ema_slope(series, period=5):
    """Calculates the slope of an EMA series over a lookback period."""
    if len(series) < period:
        return pd.Series([0.0] * len(series), index=series.index) # Not enough data
    x = np.arange(period)
    slopes = series.rolling(window=period).apply(
        lambda y: np.polyfit(x, y, 1)[0] if not y.isnull().any() else np.nan, 
        raw=True
    )
    avg_val = series.rolling(window=period).mean()
    normalized_slopes = slopes / (avg_val + 1e-9) 
    return normalized_slopes.fillna(0)

class QuantumHybridEnhancedETHParameters: # Renamed for clarity
    """
    Parameter class for the ETH-Tuned Enhanced Quantum Hybrid model.
    (Starts identical to the general enhanced version)
    """
    def __init__(self):
        # --- Core Quantum Parameters ---
        self.hbar = 1.0         
        self.m = 1.0            
        self.mass = 1.0         
        self.omega = 1.0e-4     
        self.lambda_param = 0.5 
        
        # --- Hybrid Weight Parameters ---
        self.conservative_weight = 0.55  
        self.aggressive_weight = 0.45    
        
        # --- Signal Generation Parameters ---
        self.entry_threshold = 0.6       
        self.exit_threshold = 0.5        
        self.volatility_adaptation = True  
        self.momentum_factor = 0.6       
        
        # --- Base Risk Management Parameters ---
        self.stop_loss_atr_factor = 1.4  
        self.atr_period = 14             
        self.rsi_period = 14             
        self.rsi_threshold_upper = 75    
        self.rsi_threshold_lower = 28    
        self.rsi_weight = 0.25           
        
        # --- Market Imbalance Parameters ---
        self.volume_ema_period = 20      
        self.imbalance_lookback = 6      
        self.imbalance_threshold = 0.4   
        
        # --- Enhancement Parameters ---
        self.use_regime_filter = True     
        self.regime_ema_period = 50       
        self.regime_slope_period = 5      
        self.regime_slope_threshold = 1e-5 
        
        self.use_atr_trailing_stop = True 
        self.trailing_stop_atr_multiplier = 1.8 
        
        # --- Other Parameters ---
        self.dynamic_timeframe = True    # Keep dynamic adjustments initially
        self.risk_pct = 1.0              
        self.use_adaptive_position_sizing = True  

# --- Calculation Functions (Identical to quantum_hybrid_enhanced.py) ---

def calculate_market_imbalance(df, params):
    df['volume_ema'] = df['volume'].ewm(span=params.volume_ema_period, adjust=False).mean()
    df['price_direction'] = np.sign(df['close'].diff(1)).fillna(0)
    df['buying_volume'] = df['volume'] * (df['price_direction'] > 0).astype(float)
    df['selling_volume'] = df['volume'] * (df['price_direction'] < 0).astype(float)
    rolling_vol_sum = df['volume'].rolling(window=params.imbalance_lookback).sum()
    df['buying_pressure'] = df['buying_volume'].rolling(window=params.imbalance_lookback).sum() / (rolling_vol_sum + 1e-8)
    df['selling_pressure'] = df['selling_volume'].rolling(window=params.imbalance_lookback).sum() / (rolling_vol_sum + 1e-8)
    df['market_imbalance'] = df['buying_pressure'] - df['selling_pressure']
    df['market_imbalance'].fillna(0, inplace=True)
    imbalance_rolling_min = df['market_imbalance'].rolling(window=100).min()
    imbalance_rolling_max = df['market_imbalance'].rolling(window=100).max()
    df['market_imbalance_normalized'] = 2 * (df['market_imbalance'] - imbalance_rolling_min) / \
                                       (imbalance_rolling_max - imbalance_rolling_min + 1e-8) - 1
    df['market_imbalance_normalized'].fillna(0, inplace=True)
    df['market_imbalance_normalized'] = df['market_imbalance_normalized'].clip(-1, 1)
    return df

def calculate_volatility(df, params):
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].ewm(span=params.atr_period, adjust=False).mean()
    df['returns'] = df['close'].pct_change()
    df['return_volatility'] = df['returns'].rolling(window=20).std()
    df['normalized_atr'] = df['atr'] / (df['close'] + 1e-8)
    df['volatility'] = (df['normalized_atr'] + df['return_volatility']) / 2
    df['volatility'] = df['volatility'].fillna(df['volatility'].mean())
    return df

def calculate_energy_levels(df, params):
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['price_displacement'] = (df['close'] - rolling_mean) / (rolling_std + 1e-8)
    df['price_velocity'] = df['price_displacement'].diff(1)
    df['kinetic_energy'] = 0.5 * params.m * df['price_velocity']**2
    df['potential_energy'] = 0.5 * params.m * params.omega**2 * df['price_displacement']**2
    df['total_energy'] = df['kinetic_energy'] + df['potential_energy']
    df['quantum_level'] = np.round(df['total_energy'] / (params.hbar * params.omega + 1e-8) - 0.5)
    df['quantum_level'] = df['quantum_level'].clip(0, 10) 
    max_levels = 10  
    df['normalized_energy'] = df['quantum_level'] / max_levels
    df['energy_change'] = df['quantum_level'].diff(1)
    df['significant_energy_change'] = (abs(df['energy_change']) >= 1).astype(int)
    return df

def calculate_conservative_signals(df, params):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(span=params.rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=params.rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8) 
    df['rsi'] = 100 - (100 / (1 + rs))
    rolling_min = df['close'].rolling(window=50).min()
    rolling_max = df['close'].rolling(window=50).max()
    df['price_position'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-8)
    df['conservative_signal_raw'] = (
        0.45 * (1 - df['normalized_energy']) +  
        0.35 * df['market_imbalance_normalized'] +  
        0.2 * (1 - df['price_position'])  
    )
    rsi_factor = np.where(
        df['rsi'] < params.rsi_threshold_lower, 1 + (params.rsi_threshold_lower - df['rsi']) / 50, 0.8 )
    df['conservative_signal_raw'] = df['conservative_signal_raw'] * rsi_factor
    rolling_min_sig = df['conservative_signal_raw'].rolling(window=20).min()
    rolling_max_sig = df['conservative_signal_raw'].rolling(window=20).max()
    df['conservative_signal'] = (df['conservative_signal_raw'] - rolling_min_sig) / \
                                (rolling_max_sig - rolling_min_sig + 1e-8)
    df['conservative_signal'] = df['conservative_signal'].fillna(0.5).clip(0, 1)
    return df

def calculate_aggressive_signals(df, params):
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    df['ema_diff'] = (df['ema_fast'] - df['ema_slow']) / (df['close'] + 1e-8)
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal_line']
    df['macd_direction'] = np.sign(df['macd_hist'].diff(3)).fillna(0)
    df['aggressive_signal_raw'] = (
        0.35 * df['market_imbalance_normalized'] +  
        0.25 * df['significant_energy_change'] +   
        0.25 * df['ema_diff'] +                    
        0.15 * df['macd_direction'] +              
        0.15 * (df['rsi'] - 50) / 50               
    )
    if params.volatility_adaptation:
        vol_mean = df['volatility'].rolling(window=100).mean()
        vol_ratio = df['volatility'] / (vol_mean + 1e-8)
        vol_scaling = vol_ratio.clip(0.5, 2.0)
        df['aggressive_signal_raw'] = df['aggressive_signal_raw'] * vol_scaling
    rolling_min_agg = df['aggressive_signal_raw'].rolling(window=10).min()
    rolling_max_agg = df['aggressive_signal_raw'].rolling(window=10).max()
    df['aggressive_signal'] = (df['aggressive_signal_raw'] - rolling_min_agg) / \
                              (rolling_max_agg - rolling_min_agg + 1e-8)
    df['aggressive_signal'] = df['aggressive_signal'].fillna(0.5).clip(0, 1)
    return df

def calculate_regime_filter(df, params):
    if params.use_regime_filter:
        df[f'ema_{params.regime_ema_period}'] = df['close'].ewm(span=params.regime_ema_period, adjust=False).mean()
        df['ema_slope'] = calculate_ema_slope(df[f'ema_{params.regime_ema_period}'], period=params.regime_slope_period)
        df['regime'] = 0 
        df.loc[df['ema_slope'] > params.regime_slope_threshold, 'regime'] = 1
        df.loc[df['ema_slope'] < -params.regime_slope_threshold, 'regime'] = -1
    else:
        df['regime'] = 1 
        df['ema_slope'] = 0
    return df

def run_quantum_hybrid_enhanced_eth_strategy_backtest(symbol, timeframe, initial_capital, start_date, end_date, params=None):
    """
    Run the backtest for the ETH-Tuned Enhanced Quantum Hybrid strategy.
    """
    if params is None:
        params = QuantumHybridEnhancedETHParameters() # Use ETH specific param class

    from app import fetch_crypto_data 
    df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
    if df is None or df.empty: return None, None

    print("Calculating indicators (ETH Enhanced)...")
    df = calculate_volatility(df, params)
    df = calculate_market_imbalance(df, params)
    df = calculate_energy_levels(df, params)
    df = calculate_conservative_signals(df, params)
    df = calculate_aggressive_signals(df, params)
    df = calculate_regime_filter(df, params) 
    df['hybrid_signal'] = (params.conservative_weight * df['conservative_signal'] +
                           params.aggressive_weight * df['aggressive_signal'])

    print("Starting backtest loop (ETH Enhanced)...")
    capital = initial_capital
    equity = [initial_capital]
    trades = []
    position = 0
    entry_price = 0
    initial_stop_loss_price = 0
    trailing_stop_loss_price = 0
    position_size_coins = 0 

    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_atr = df['atr'].iloc[i]
        current_regime = df['regime'].iloc[i]
        pnl = 0

        if position != 0:
            if params.use_atr_trailing_stop:
                if position == 1:
                    new_stop = current_high - params.trailing_stop_atr_multiplier * current_atr
                    trailing_stop_loss_price = max(trailing_stop_loss_price, new_stop, initial_stop_loss_price)
                    stop_price_to_check = trailing_stop_loss_price
                else: # position == -1
                    new_stop = current_low + params.trailing_stop_atr_multiplier * current_atr
                    trailing_stop_loss_price = min(trailing_stop_loss_price, new_stop, initial_stop_loss_price)
                    stop_price_to_check = trailing_stop_loss_price
            else:
                 stop_price_to_check = initial_stop_loss_price

            exit_signal = False
            if position == 1 and current_low <= stop_price_to_check:
                exit_price = stop_price_to_check; exit_signal = True; # print(f"{df.index[i]} - Long SL Hit")
            elif position == -1 and current_high >= stop_price_to_check:
                exit_price = stop_price_to_check; exit_signal = True; # print(f"{df.index[i]} - Short SL Hit")

            if not exit_signal:
                 hybrid_exit_long = position == 1 and df['aggressive_signal'].iloc[i] < (1 - params.exit_threshold)
                 hybrid_exit_short = position == -1 and df['aggressive_signal'].iloc[i] > params.exit_threshold
                 if hybrid_exit_long or hybrid_exit_short:
                     exit_price = current_price; exit_signal = True; # print(f"{df.index[i]} - Hybrid Exit Signal")
            
            if exit_signal:
                pnl = (exit_price - entry_price) * position_size_coins if position == 1 else (entry_price - exit_price) * position_size_coins
                capital += pnl
                trades.append({
                    'entry_date': df.index[entry_index],
                    'exit_date': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position_size_coins,
                    'pnl': pnl,
                    'type': 'long' if position == 1 else 'short',
                    'atr_at_entry': entry_atr,
                    'regime_at_entry': entry_regime,
                    'hybrid_signal_at_entry': entry_signal_value
                })
                position = 0; entry_price = 0; position_size_coins = 0

        if position == 0:
            long_entry_signal = df['hybrid_signal'].iloc[i] > params.entry_threshold
            short_entry_signal = df['hybrid_signal'].iloc[i] < (1 - params.entry_threshold)
            allow_long = (not params.use_regime_filter) or (current_regime == 1)
            allow_short = (not params.use_regime_filter) or (current_regime == -1)

            if long_entry_signal and allow_long: position = 1
            elif short_entry_signal and allow_short: position = -1
            
            if position != 0:
                entry_price = current_price
                entry_atr = current_atr
                entry_index = i 
                entry_regime = current_regime 
                entry_signal_value = df['hybrid_signal'].iloc[i] 
                stop_loss_distance = params.stop_loss_atr_factor * entry_atr
                risk_per_coin = stop_loss_distance
                risk_amount = capital * (params.risk_pct / 100.0)
                
                if risk_per_coin > 1e-9:
                    position_size_coins = risk_amount / risk_per_coin
                else:
                    position_size_coins = 0; position = 0; print(f"Warning: Invalid risk_per_coin at {df.index[i]}")
                
                if position == 1:
                    initial_stop_loss_price = entry_price - stop_loss_distance
                    trailing_stop_loss_price = initial_stop_loss_price 
                    # print(f"{df.index[i]} - Enter Long: {entry_price:.4f} | SL: {initial_stop_loss_price:.4f} | Regime: {entry_regime}")
                elif position == -1:
                    initial_stop_loss_price = entry_price + stop_loss_distance
                    trailing_stop_loss_price = initial_stop_loss_price 
                    # print(f"{df.index[i]} - Enter Short: {entry_price:.4f} | SL: {initial_stop_loss_price:.4f} | Regime: {entry_regime}")
        
        current_pnl_open = 0
        if position == 1: current_pnl_open = (current_price - entry_price) * position_size_coins
        elif position == -1: current_pnl_open = (entry_price - current_price) * position_size_coins
        equity.append(capital + current_pnl_open)

    print("Calculating performance metrics (ETH Enhanced)...")
    equity_curve = pd.Series(equity[1:], index=df.index[1:]) 
    
    if not trades: return {'Total Trades': 0, 'Win Rate [%]': 0, 'Return [%]': 0, 'Max Drawdown [%]': 0, 'Profit Factor': 0}, df

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    total_return = ((equity_curve.iloc[-1] / initial_capital) - 1) * 100
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / (peak + 1e-8)
    max_drawdown = abs(drawdown.min()) * 100
    gross_profit = winning_trades['pnl'].sum()
    gross_loss = abs(losing_trades['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    stats = {
        'Total Trades': total_trades, 'Win Rate [%]': win_rate, 'Return [%]': total_return,
        'Max Drawdown [%]': max_drawdown, 'Profit Factor': profit_factor,
        'Trades': trades_df, 'Equity Curve': equity_curve
    }
    df_result = df[[ 'close', 'hybrid_signal', 'regime', f'ema_{params.regime_ema_period}', 'atr']].copy()
    return stats, df_result

if __name__ == '__main__':
    symbol = 'ETH/USDT' # <<< Default to ETH for this version
    timeframe = '1h'
    initial_capital = 10000
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    params_eth = QuantumHybridEnhancedETHParameters()
    # --- Potential ETH Specific Tunings Here ---
    # params_eth.omega = 1.2e-4 # Example: Slightly different omega for ETH 1h?
    # params_eth.trailing_stop_atr_multiplier = 1.7 
    # params_eth.regime_slope_threshold = 0.8e-5
    # --- End ETH Specific Tunings ---

    stats, df_results = run_quantum_hybrid_enhanced_eth_strategy_backtest(
        symbol, timeframe, initial_capital, start_date, end_date, params_eth
    )

    if stats:
        print("\n--- ETH Enhanced Backtest Results ---")
        # ... (rest of printing and plotting identical to the general enhanced version) ...
        print(f"Total Trades: {stats['Total Trades']}")
        print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        print(f"Total Return: {stats['Return [%]']:.2f}%")
        print(f"Max Drawdown: {stats['Max Drawdown [%]']:.2f}%")
        print(f"Profit Factor: {stats['Profit Factor']:.2f}")
        
        if df_results is not None and not df_results.empty:
             import matplotlib.pyplot as plt
             fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
             axes[0].plot(df_results.index, df_results['close'], label='Close Price', color='blue', alpha=0.7)
             axes[0].plot(df_results.index, df_results[f'ema_{params_eth.regime_ema_period}'], label=f'EMA {params_eth.regime_ema_period}', color='orange', linestyle='--')
             axes[0].set_title(f'{symbol} {timeframe} - Price & Regime EMA (ETH Tuned)')
             axes[0].legend(); axes[0].grid(True)
             axes[1].plot(df_results.index, df_results['hybrid_signal'], label='Hybrid Signal', color='purple')
             regime_colors = {1: 'lightgreen', -1: 'lightcoral', 0: 'lightgray'}
             for regime_val, color in regime_colors.items():
                 axes[1].fill_between(df_results.index, 0, 1, where=df_results['regime'] == regime_val, color=color, alpha=0.3, label=f'Regime {regime_val}')
             axes[1].axhline(params_eth.entry_threshold, color='red', linestyle='--', label=f'Entry Thresh ({params_eth.entry_threshold:.1f})')
             axes[1].axhline(1 - params_eth.entry_threshold, color='red', linestyle='--')
             axes[1].set_title('Hybrid Signal & Market Regime (ETH Tuned)')
             axes[1].legend(); axes[1].set_ylim(0, 1); axes[1].grid(True)
             axes[2].plot(stats['Equity Curve'].index, stats['Equity Curve'], label='Equity Curve', color='green')
             axes[2].set_title('Equity Curve (ETH Tuned)')
             axes[2].grid(True)
             plt.tight_layout()
             plt.show()
```
```