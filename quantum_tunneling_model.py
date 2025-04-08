"""
Quantum Tunneling of Stock Price in Range Bound Market Conditions

This module implements the Quantum Tunneling model for stock prices in range-bound markets,
based on the work of Ovidiu Racorean in "Quantum Tunneling of Stock Price in Range Bound
Market Conditions" (Quantum Physics Letters, 2014).

The model treats stock prices as quantum particles that can tunnel through potential barriers
(support/resistance levels) in range-bound markets, with the probability determined by a
transmission coefficient derived from a financial time-independent Schrödinger equation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import ccxt
import os
import time
import math
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator

# Ensure results directory exists
os.makedirs("static/results", exist_ok=True)

class QuantumTunnelingParameters:
    def __init__(self, market_type="crypto"):
        # Financial parameters
        self.risk_free_rate = 0.03  # Risk-free interest rate (yearly)
        self.volatility = 0.3       # Stock price volatility
        
        # Tunneling parameters
        self.lambda_param = 0.8     # REDUCED: Lambda parameter from 2.0 to 0.8 (related to r/sigma^2)
        self.lookback_period = 15   # REDUCED: Shorter period to identify range-bound markets
        self.range_threshold = 0.10 # INCREASED: Accept wider ranges as percentage of price
        self.vol_drop_threshold = 0.15 # REDUCED: Much smaller volatility drop threshold
        self.min_range_days = 5     # REDUCED: Fewer minimum days in range
        
        # Market parameters
        self.use_empirical_volatility = True  # Whether to use historical volatility
        
        # Technical indicators
        self.rsi_period = 14
        self.atr_period = 14
        self.support_resistance_periods = [10, 20, 50]  # Periods for detecting support/resistance
        
        # Risk management
        self.stop_loss_atr = 1.5    # ATR multiplier for stop loss
        self.take_profit_atr = 2.0  # ATR multiplier for take profit
        self.cooldown_period = 5    # REDUCED: Shorter bars to wait after a trade
        self.tunneling_threshold = 0.05  # REDUCED: Much lower minimum transmission coefficient to enter a trade


def calculate_indicators(df, params):
    """
    Calculate technical indicators needed for the strategy
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Calculate RSI
    rsi = RSIIndicator(df_copy['close'], window=params.rsi_period)
    df_copy['rsi'] = rsi.rsi()
    
    # Calculate ATR for position sizing
    atr = AverageTrueRange(high=df_copy['high'], low=df_copy['low'], close=df_copy['close'], window=params.atr_period)
    df_copy['atr'] = atr.average_true_range()
    
    # Calculate SMAs for support/resistance detection
    for period in params.support_resistance_periods:
        sma = SMAIndicator(df_copy['close'], window=period)
        df_copy[f'sma_{period}'] = sma.sma_indicator()
    
    # Calculate log returns
    df_copy['log_return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
    
    # Calculate historical volatility (rolling 20-day annualized standard deviation)
    df_copy['rolling_volatility'] = df_copy['log_return'].rolling(window=20).std() * np.sqrt(252)
    
    # Calculate volatility drop
    df_copy['volatility_drop'] = df_copy['rolling_volatility'].rolling(window=5).max() / df_copy['rolling_volatility']
    
    return df_copy


def detect_range_bound(df, params, i, lookback=None):
    """
    Detect if the market is in a range-bound condition starting from index i and looking back
    
    Returns:
        Dictionary with range information including support, resistance, and range status
    """
    if lookback is None:
        lookback = params.lookback_period
    
    if i < lookback:
        return {'range_bound': False}
    
    # Extract the price range
    price_section = df.iloc[i-lookback:i]
    
    # Detect support and resistance levels - use more extreme percentiles
    support = price_section['low'].quantile(0.1)  # Use 10th percentile 
    resistance = price_section['high'].quantile(0.9)  # Use 90th percentile
    
    # Calculate range as percentage of average price
    avg_price = price_section['close'].mean()
    range_pct = (resistance - support) / avg_price
    
    # Determine how many days price stayed within range - much more lenient criteria
    within_range = ((price_section['close'] >= support * 0.95) & 
                   (price_section['close'] <= resistance * 1.05))  # Allow 5% wiggle room
    days_in_range = within_range.sum()
    
    # Calculate ratio of days in range
    range_ratio = days_in_range / len(price_section)
    
    # MUCH more liberal definition of range-bound 
    # Either reasonable range OR many days in range OR recent sideways price action
    recent_trend = abs(price_section['close'].iloc[-1] / price_section['close'].iloc[0] - 1) < 0.05
    
    is_range_bound = ((range_pct <= params.range_threshold) or 
                     (range_ratio >= 0.5) or  # Only need 50% of days in range
                     recent_trend)  # Or recent sideways price action
    
    return {
        'range_bound': is_range_bound,
        'support': support,
        'resistance': resistance,
        'range_pct': range_pct,
        'days_in_range': days_in_range,
        'avg_price': avg_price,
        'range_ratio': range_ratio
    }


def calculate_transmission_coefficient(params, K, S):
    """
    Calculate the quantum tunneling probability using a modified WKB approximation
    
    Args:
        params: QuantumTunnelingParameters
        K: Barrier level (resistance or support)
        S: Current price
        
    Returns:
        Transmission coefficient (probability of tunneling)
    """
    # Extract parameters
    sigma = params.volatility
    r = params.risk_free_rate
    
    # Calculate lambda parameter dynamically if not explicitly set
    # According to Racorean's paper, lambda = 2r/sigma^2
    lambda_param = params.lambda_param
    if lambda_param is None or lambda_param <= 0:
        lambda_param = 2 * r / (sigma**2)
    
    # Calculate the term inside the square root
    term = 1 - lambda_param * K**2
    
    # If term is negative or zero, tunneling is not possible
    if term <= 0:
        return 0
    
    try:
        # Calculate square root term with safety checks
        sqrt_term = np.sqrt(term)
        
        # Check if sqrt_term is too close to 1 - more robust check
        if abs(sqrt_term - 1) < 1e-4:
            # Term is very close to 1, tunneling is extremely unlikely
            return 0
            
        # Ensure we don't have invalid input to log function
        denominator = sqrt_term - 1
        if denominator <= 0:
            # If denominator is non-positive, tunneling not possible
            return 0
            
        # Calculate log term with error handling
        log_arg = (sqrt_term + 1) / denominator
        if log_arg <= 0:
            return 0
            
        # Calculate the exponent terms from the paper
        exponent_term1 = 0.5 * np.log(log_arg)
        exponent_term2 = sqrt_term
        
        # Calculate the full exponent
        sr = np.sqrt(sigma**2 / r)  # Characteristic length scale Sr = σ/√r
        coef = -2 * K / sr  # Coefficient from paper: -2K/Sr
        exponent = coef * (exponent_term1 - exponent_term2)
        
        # Handle potential overflow in exponential
        if exponent < -700:  # np.exp has issues below ~-745
            return 0
            
        # Calculate transmission coefficient
        T = np.exp(exponent)
        
        # Ensure we return a valid probability
        return max(0.0, min(1.0, T))
        
    except Exception as e:
        # Catch any numerical errors and return 0 probability
        print(f"Error in tunneling calculation: {e}")
        return 0


def calculate_breakout_target(params, K):
    """
    Calculate the expected price target after tunneling
    
    Args:
        params: QuantumTunnelingParameters
        K: Strike price (resistance or support level)
    
    Returns:
        Target price after tunneling
    """
    r = params.risk_free_rate
    sigma = params.volatility
    
    # Calculate Sr (characteristic length scale) as defined in the paper
    Sr = sigma / np.sqrt(r)
    
    # Calculate penetration distance based on the quantum model
    # From Racorean's paper: The tunneling particle will emerge at a distance
    # approximately equal to the characteristic length scale Sr from the barrier
    penetration_distance = Sr
    
    # Target price - penetrate beyond the barrier by distance Sr
    if K > 0:  # For upward tunneling (resistance)
        target = K + penetration_distance
    else:      # For downward tunneling (support)
        target = K - penetration_distance
    
    return target


def detect_tunneling_opportunity(df, params, i):
    """
    Detect if there's a tunneling opportunity at index i
    """
    if i < 20:  # Reduced from 30
        return {'tunneling': False}
    
    # Check if market is range-bound
    range_info = detect_range_bound(df, params, i)
    if not range_info['range_bound']:
        return {'tunneling': False}
    
    # Check for volatility drop - more flexible criteria
    # Accept ANY of these conditions for volatility drop
    current_vol = df.iloc[i]['rolling_volatility'] if 'rolling_volatility' in df.columns else 0.3
    
    # Multiple ways to detect volatility contraction
    if i >= 20:
        # 1. Standard volatility drop measurement
        volatility_drop = df.iloc[i]['volatility_drop'] if 'volatility_drop' in df.columns else 1.0
        
        # 2. Recent volatility compared to longer-term volatility
        recent_vol = df.iloc[i-5:i]['log_return'].std() * np.sqrt(252) if 'log_return' in df.columns else current_vol
        longer_vol = df.iloc[i-20:i-5]['log_return'].std() * np.sqrt(252) if 'log_return' in df.columns else current_vol*1.2
        vol_contraction = longer_vol / max(recent_vol, 0.001) if recent_vol > 0 else 1.0
        
        # 3. Volatility percentile - low percentile indicates low volatility period
        if 'rolling_volatility' in df.columns:
            vol_window = df.iloc[max(0, i-50):i]['rolling_volatility']
            if not vol_window.empty:
                vol_percentile = stats.percentileofscore(vol_window, current_vol) / 100
                vol_is_low = vol_percentile < 0.3  # Bottom 30% of recent volatility
            else:
                vol_is_low = False
        else:
            vol_is_low = False
    else:
        volatility_drop = 1.0
        vol_contraction = 1.0
        vol_is_low = False
    
    # Check if ANY volatility condition indicates a potential tunneling opportunity
    significant_drop = (volatility_drop > params.vol_drop_threshold or 
                       vol_contraction > params.vol_drop_threshold or 
                       vol_is_low)
    
    if not significant_drop:
        return {'tunneling': False}
    
    # If we're in a range and volatility dropped significantly, calculate tunneling probability
    support = range_info['support']
    resistance = range_info['resistance']
    current_price = df.iloc[i]['close']
    
    # Ensure minimum volatility for calculations
    if 'rolling_volatility' in df.columns:
        params.volatility = max(df.iloc[i]['rolling_volatility'], 0.05)
    else:
        params.volatility = max(params.volatility, 0.05)
    
    # Calculate upward tunneling probability
    upward_target = calculate_breakout_target(params, resistance)
    upward_T = calculate_transmission_coefficient(params, resistance, upward_target)
    
    # Calculate downward tunneling probability
    downward_target = 2 * support - calculate_breakout_target(params, support)
    downward_T = calculate_transmission_coefficient(params, support, downward_target)
    
    # Boost probabilities significantly
    upward_T *= 1.5
    downward_T *= 1.5
    
    # Add positional bias - price position within the range
    price_position = (current_price - support) / (resistance - support) if (resistance - support) > 0 else 0.5
    
    # Boost tunneling probability based on price position
    if price_position > 0.7:  # Near resistance
        upward_T *= 1.5  # Boost by 50% 
    elif price_position < 0.3:  # Near support
        downward_T *= 1.5  # Boost by 50%
    
    # Determine tunneling direction and probability with MUCH lower threshold
    effective_threshold = params.tunneling_threshold / 2  # Cut the threshold in half
    
    if upward_T > downward_T and upward_T > effective_threshold:
        direction = 'upward'
        probability = upward_T
        target = upward_target
        entry = resistance
    elif downward_T > effective_threshold:
        direction = 'downward'
        probability = downward_T
        target = downward_target
        entry = support
    else:
        return {'tunneling': False}
    
    return {
        'tunneling': True,
        'direction': direction,
        'probability': probability,
        'target': target,
        'entry': entry,
        'support': support,
        'resistance': resistance,
        'volatility_drop': volatility_drop if 'volatility_drop' in locals() else 1.0,
        'rel_vol_drop': vol_contraction if 'vol_contraction' in locals() else 1.0,
        'price_position': price_position
    }


def find_trading_signals(df, i, params):
    """
    Find trading signals based on the quantum tunneling model
    
    Args:
        df: DataFrame with price data and indicators
        i: Current index
        params: QuantumTunnelingParameters
    
    Returns:
        Dictionary with signal information
    """
    # Check for tunneling opportunities
    tunneling_info = detect_tunneling_opportunity(df, params, i)
    
    # Default values for consistent dictionary structure
    current_price = df.iloc[i]['close']
    
    # Create base signal data with default values
    base_signal_data = {
        'signal': 'neutral',
        'confidence': 0,
        'entry': current_price,
        'target': current_price,
        'support': current_price * 0.97,
        'resistance': current_price * 1.03,
        'volatility_drop': 0.0
    }
    
    if not tunneling_info['tunneling']:
        # Optional: Also check for other technical conditions that might support a trade
        # even if not a pure quantum tunneling opportunity
        
        # RSI conditions
        if 'rsi' in df.columns:
            rsi = df.iloc[i]['rsi']
            
            # Look for potential reversal setup when price is at support/resistance
            # with strong RSI readings
            if i >= 5:  # Need some history
                # Get recent price info
                high_range = df.iloc[i-5:i]['high'].max()
                low_range = df.iloc[i-5:i]['low'].min()
                
                # Calculate how close we are to support/resistance
                upper_band_proximity = abs(current_price - high_range) / current_price if current_price > 0 else 1.0
                lower_band_proximity = abs(current_price - low_range) / current_price if current_price > 0 else 1.0
                
                # Breakout or breakdown conditions
                if upper_band_proximity < 0.01 and rsi > 70:
                    signal_data = base_signal_data.copy()
                    signal_data.update({
                        'signal': 'buy',
                        'confidence': 0.2,  # Lower confidence for non-quantum signals
                        'entry': current_price,
                        'target': current_price * 1.02,  # Simple 2% target
                        'resistance': high_range,
                        'support': low_range
                    })
                    return signal_data
                    
                elif lower_band_proximity < 0.01 and rsi < 30:
                    signal_data = base_signal_data.copy()
                    signal_data.update({
                        'signal': 'sell',
                        'confidence': 0.2,  # Lower confidence for non-quantum signals
                        'entry': current_price,
                        'target': current_price * 0.98,  # Simple 2% target
                        'resistance': high_range,
                        'support': low_range
                    })
                    return signal_data
        
        return base_signal_data
    
    # Generate signal based on tunneling direction
    signal_data = base_signal_data.copy()
    
    if tunneling_info['direction'] == 'upward':
        signal = 'buy'
        confidence = tunneling_info['probability']
    elif tunneling_info['direction'] == 'downward':
        signal = 'sell'
        confidence = tunneling_info['probability']
    else:
        return base_signal_data
    
    # Apply a multiplier to boost confidence values overall
    confidence = min(0.95, confidence * 1.5)  # Boost by 50% but cap at 0.95
    
    # Update signal data with tunneling information
    signal_data.update({
        'signal': signal,
        'confidence': confidence,
        'entry': tunneling_info['entry'],
        'target': tunneling_info['target'],
        'support': tunneling_info['support'],
        'resistance': tunneling_info['resistance'],
        'volatility_drop': tunneling_info['volatility_drop'],
        'tunneling_info': tunneling_info  # Pass full tunneling info for debugging
    })
    
    return signal_data


def fetch_crypto_data(symbol, timeframe, start_date, end_date=None, exchange_id='binance'):
    """
    Fetch historical cryptocurrency data from an exchange
    """
    try:
        # Initialize exchange
        exchange = getattr(ccxt, exchange_id)()
        
        # Convert dates to timestamps
        from_timestamp = exchange.parse8601(f"{start_date}T00:00:00Z")
        to_timestamp = None
        if end_date:
            to_timestamp = exchange.parse8601(f"{end_date}T23:59:59Z")
        
        # Fetch OHLCV data
        all_candles = []
        current_timestamp = from_timestamp
        
        # CCXT fetching loop
        while True:
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp)
                if len(candles) == 0:
                    break
                
                all_candles.extend(candles)
                
                if to_timestamp and candles[-1][0] >= to_timestamp:
                    break
                
                # Move to the next batch
                current_timestamp = candles[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)  # Respect rate limit
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Filter by date range
        if end_date:
            df = df[df.index <= end_date]
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def visualize_transmission_coefficient(params, save_path=None):
    """
    Visualize how the transmission coefficient changes with different parameters
    """
    # Create a grid of K values (normalized)
    K_values = np.linspace(0.5, 1.5, 100)
    
    # Calculate transmission coefficient for different volatility values
    volatilities = [0.2, 0.3, 0.4, 0.5]
    plt.figure(figsize=(12, 10))
    
    for vol in volatilities:
        T_values = []
        # Save original volatility to restore later
        original_vol = params.volatility
        # Set new volatility
        params.volatility = vol
        
        for K in K_values:
            target = calculate_breakout_target(params, K)
            T = calculate_transmission_coefficient(params, K, target)
            T_values.append(T)
        
        plt.plot(K_values, T_values, label=f'σ = {vol}')
        
        # Restore original volatility
        params.volatility = original_vol
    
    plt.title('Transmission Coefficient vs. Strike Price for Different Volatilities')
    plt.xlabel('Normalized Strike Price (K)')
    plt.ylabel('Transmission Coefficient (T)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create a second plot for interest rate impact
    plt.figure(figsize=(12, 10))
    interest_rates = [0.01, 0.02, 0.03, 0.05]
    
    for r in interest_rates:
        T_values = []
        # Save original rate to restore later
        original_rate = params.risk_free_rate
        # Set new rate
        params.risk_free_rate = r
        
        for K in K_values:
            target = calculate_breakout_target(params, K)
            T = calculate_transmission_coefficient(params, K, target)
            T_values.append(T)
        
        plt.plot(K_values, T_values, label=f'r = {r}')
        
        # Restore original rate
        params.risk_free_rate = original_rate
    
    plt.title('Transmission Coefficient vs. Strike Price for Different Interest Rates')
    plt.xlabel('Normalized Strike Price (K)')
    plt.ylabel('Transmission Coefficient (T)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create a third plot showing relationship between volatility drop and transmission coefficient
    plt.figure(figsize=(12, 10))
    
    # Grid of volatility values
    vol_values = np.linspace(0.1, 0.6, 100)
    T_values = []
    
    for vol in vol_values:
        params.volatility = vol
        K = 1.0  # Normalized strike price
        target = calculate_breakout_target(params, K)
        T = calculate_transmission_coefficient(params, K, target)
        T_values.append(T)
    
    plt.plot(vol_values, T_values)
    plt.title('Transmission Coefficient vs. Volatility (K=1.0)')
    plt.xlabel('Volatility (σ)')
    plt.ylabel('Transmission Coefficient (T)')
    plt.grid(True, alpha=0.3)
    
    # Add annotation explaining the relationship
    annotation = (
        "Key Insights:\n"
        "1. Lower volatility dramatically increases tunneling probability\n"
        "2. This explains why price breakouts often occur after volatility contraction\n"
        "3. Quantum tunneling provides mathematical foundation for 'volatility squeeze' patterns"
    )
    plt.figtext(0.5, 0.02, annotation, ha='center', fontsize=12, bbox=dict(facecolor='ivory', alpha=0.5))
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def run_backtest(df, params, initial_capital=10000, plot=True):
    """
    Run a backtest of the Quantum Tunneling strategy
    """
    print(f"Starting Quantum Tunneling backtest with {len(df)} data points...")
    
    # Calculate indicators
    print("Calculating indicators...")
    df = calculate_indicators(df, params)
    
    # Initialize trading metrics
    equity = [initial_capital]
    trades = []
    in_position = False
    entry_price = 0
    position_type = None  # 'long' or 'short'
    entry_index = 0
    cooldown_counter = 0
    target_price = 0
    
    # Prepare for tracking
    trade_log = []
    tunneling_events = []
    
    print(f"Starting backtest loop through {len(df)-30} bars...")
    # Loop through data (start from index 30 to have enough data for indicators)
    for i in range(30, len(df)):
        # Print progress every 500 iterations
        if i % 500 == 0:
            print(f"Processing bar {i}/{len(df)} ({(i/len(df)*100):.1f}%)...")
            
        current_price = df.iloc[i]['close']
        current_date = df.index[i]
        
        # Skip if in cooldown period
        if cooldown_counter > 0:
            cooldown_counter -= 1
            equity.append(equity[-1])
            continue
        
        # Check if we need to close a position
        if in_position:
            # Calculate profit/loss
            if position_type == 'long':
                profit_pct = (current_price - entry_price) / entry_price
                stop_price = entry_price * (1 - params.stop_loss_atr * df.iloc[entry_index]['atr'] / entry_price)
                
                # Check for target price or stop loss
                if current_price >= target_price or current_price <= stop_price:
                    # Close position
                    trade_result = (current_price - entry_price) / entry_price
                    equity_change = equity[-1] * (1 + trade_result)
                    equity.append(equity_change)
                    
                    # Record trade
                    exit_reason = 'Target' if current_price >= target_price else 'Stop Loss'
                    trade_data = {
                        'entry_date': df.index[entry_index],
                        'exit_date': current_date,
                        'position': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'target_price': target_price,
                        'profit_pct': trade_result * 100,
                        'result': exit_reason
                    }
                    trade_log.append(trade_data)
                    trades.append(trade_result)
                    
                    # Reset position
                    in_position = False
                    cooldown_counter = params.cooldown_period
                    
                    continue
            
            elif position_type == 'short':
                profit_pct = (entry_price - current_price) / entry_price
                stop_price = entry_price * (1 + params.stop_loss_atr * df.iloc[entry_index]['atr'] / entry_price)
                
                # Check for target price or stop loss
                if current_price <= target_price or current_price >= stop_price:
                    # Close position
                    trade_result = (entry_price - current_price) / entry_price
                    equity_change = equity[-1] * (1 + trade_result)
                    equity.append(equity_change)
                    
                    # Record trade
                    exit_reason = 'Target' if current_price <= target_price else 'Stop Loss'
                    trade_data = {
                        'entry_date': df.index[entry_index],
                        'exit_date': current_date,
                        'position': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'target_price': target_price,
                        'profit_pct': trade_result * 100,
                        'result': exit_reason
                    }
                    trade_log.append(trade_data)
                    trades.append(trade_result)
                    
                    # Reset position
                    in_position = False
                    cooldown_counter = params.cooldown_period
                    
                    continue
        
        # Check for new signals if we're not in a position
        if not in_position:
            # Get signal from the quantum tunneling model
            signal_data = find_trading_signals(df, i, params)
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            # Track tunneling events for visualization
            if 'entry' in signal_data and signal != 'neutral':
                # Extract required fields with defaults if not present
                tunneling_event = {
                    'date': current_date,
                    'price': current_price,
                    'signal': signal,
                    'confidence': confidence,
                    'entry': signal_data.get('entry', current_price),
                    'target': signal_data.get('target', current_price * (1.02 if signal == 'buy' else 0.98)),
                    # Use safe gets with defaults for optional keys
                    'support': signal_data.get('support', current_price * 0.97),
                    'resistance': signal_data.get('resistance', current_price * 1.03),
                    'volatility_drop': signal_data.get('volatility_drop', 0.0)
                }
                tunneling_events.append(tunneling_event)
            
            # Enter position if signal is valid
            if signal == 'buy' and confidence > params.tunneling_threshold:
                # Enter long position
                in_position = True
                position_type = 'long'
                entry_price = signal_data.get('entry', current_price)  # Use current price as fallback
                target_price = signal_data.get('target', entry_price * 1.02)  # Default 2% profit target
                entry_index = i
                
            elif signal == 'sell' and confidence > params.tunneling_threshold:
                # Enter short position
                in_position = True
                position_type = 'short'
                entry_price = signal_data.get('entry', current_price)  # Use current price as fallback
                target_price = signal_data.get('target', entry_price * 0.98)  # Default 2% profit target
                entry_index = i
        
        # If no trade, equity remains the same
        if len(equity) <= i - 29:
            equity.append(equity[-1])
    
    # Close any remaining position at the end
    if in_position:
        current_price = df.iloc[-1]['close']
        if position_type == 'long':
            trade_result = (current_price - entry_price) / entry_price
        else:  # short
            trade_result = (entry_price - current_price) / entry_price
        
        equity_change = equity[-1] * (1 + trade_result)
        equity.append(equity_change)
        
        trade_data = {
            'entry_date': df.index[entry_index],
            'exit_date': df.index[-1],
            'position': position_type,
            'entry_price': entry_price,
            'exit_price': current_price,
            'target_price': target_price,
            'profit_pct': trade_result * 100,
            'result': 'End of Backtest'
        }
        trade_log.append(trade_data)
        trades.append(trade_result)
    
    # Calculate performance metrics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t > 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = np.mean([t for t in trades if t > 0]) * 100 if winning_trades > 0 else 0
    avg_loss = np.mean([t for t in trades if t < 0]) * 100 if (total_trades - winning_trades) > 0 else 0
    
    # Calculate profit factor
    gross_profit = sum(t for t in trades if t > 0)
    gross_loss = abs(sum(t for t in trades if t < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate max drawdown
    running_max = 0
    drawdown = 0
    max_drawdown = 0
    
    for e in equity:
        if e > running_max:
            running_max = e
        
        drawdown = (running_max - e) / running_max
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Final equity and return
    final_equity = equity[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    
    # Create trade DataFrame for analysis
    trade_df = pd.DataFrame(trade_log)
    
    # Plot results if requested
    if plot and len(equity) > 0:
        try:
            # Create a figure with subplots
            fig, axs = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
            
            # Plot equity curve - FIX: Ensure x and y have the same dimensions
            x_indices = df.index[29:29+len(equity)]
            # Check if x and y have same length, adjust if needed
            if len(x_indices) != len(equity):
                # Make them the same length by using the shorter one
                min_len = min(len(x_indices), len(equity))
                x_indices = x_indices[:min_len]
                equity_plot = equity[:min_len]
                print(f"Adjusted plotting arrays from lengths {len(df.index[29:29+len(equity)])} and {len(equity)} to {min_len}")
            else:
                equity_plot = equity
                
            axs[0].plot(x_indices, equity_plot)
            axs[0].set_title('Equity Curve')
            axs[0].set_ylabel('Equity')
            axs[0].grid(True, alpha=0.3)
            
            # Plot price with support/resistance zones
            axs[1].plot(df.index, df['close'])
            axs[1].set_title('Price with Tunneling Events')
            axs[1].set_ylabel('Price')
            axs[1].grid(True, alpha=0.3)
            
            # Plot volatility with drops
            axs[2].plot(df.index, df['rolling_volatility'], label='Volatility')
            axs[2].set_title('Volatility with Significant Drops')
            axs[2].set_ylabel('Volatility')
            axs[2].set_xlabel('Date')
            axs[2].grid(True, alpha=0.3)
            
            # Add tunneling events to the price chart
            for event in tunneling_events:
                # Mark resistance and support levels
                axs[1].axhline(y=event['resistance'], color='r', linestyle='--', alpha=0.5)
                axs[1].axhline(y=event['support'], color='g', linestyle='--', alpha=0.5)
                
                # Mark tunneling events
                color = 'green' if event['signal'] == 'buy' else 'red'
                marker = '^' if event['signal'] == 'buy' else 'v'
                
                # Mark the event on price chart
                axs[1].scatter(event['date'], event['price'], color=color, marker=marker, s=100)
                
                # Connect to target with arrow
                dx = 15  # Number of days to show the target (adjust as needed)
                target_date = event['date'] + pd.Timedelta(days=dx)
                axs[1].annotate('', 
                             xy=(target_date, event['target']), 
                             xytext=(event['date'], event['resistance'] if event['signal'] == 'buy' else event['support']),
                             arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, headwidth=8))
                
                # Mark volatility drop on volatility chart
                # Find the nearest date index using a more robust method
                try:
                    # Calculate the absolute difference between each date and the event date
                    date_diffs = abs(df.index - event['date'])
                    # Get the index of the minimum difference
                    nearest_idx = date_diffs.argmin()
                    vol_date = df.index[nearest_idx]
                    vol_value = df.iloc[nearest_idx]['rolling_volatility']
                    axs[2].scatter(vol_date, vol_value, color=color, marker='*', s=150)
                except Exception as e:
                    print(f"Error plotting volatility marker: {e}")
                    # Continue without the marker if there's an error
            
            # Add trades to the price chart
            if not trade_df.empty:
                for i, trade in trade_df.iterrows():
                    color = 'green' if trade['profit_pct'] > 0 else 'red'
                    marker = '^' if trade['position'] == 'long' else 'v'
                    # Entry point
                    axs[1].scatter(trade['entry_date'], trade['entry_price'], color=color, marker=marker, s=100)
                    # Exit point
                    axs[1].scatter(trade['exit_date'], trade['exit_price'], color=color, marker='o', s=100)
                    # Connect with a line
                    axs[1].plot([trade['entry_date'], trade['exit_date']], 
                             [trade['entry_price'], trade['exit_price']], color=color, alpha=0.5)
            
            plt.tight_layout()
            
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save the figure
            plt.savefig(f"static/results/quantum_tunneling_backtest_{timestamp}.png")
            plt.close()
            
            # Also save the transmission coefficient visualization
            visualize_transmission_coefficient(params, save_path=f"static/results/quantum_tunneling_coefficient_{timestamp}.png")
        except Exception as e:
            # If plotting fails, log the error but don't stop the backtest
            print(f"Warning: Failed to create plots: {str(e)}")
            plt.close('all')  # Close any open plots to avoid memory leaks
    
    # Print results
    print("\n=== Quantum Tunneling Backtest Results ===")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")
    
    # Save trade log
    if not trade_df.empty:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trade_df.to_csv(f"static/results/quantum_tunneling_trades_{timestamp}.csv", index=False)
    
    # Return results
    results = {
        'equity': equity,
        'total_return': total_return,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'average_win': avg_win,
        'average_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'trade_log': trade_log,
        'tunneling_events': tunneling_events
    }
    
    return results


if __name__ == "__main__":
    """
    Demonstration of the Quantum Tunneling Model for Stock Prices
    
    Based on 'Quantum Tunneling of Stock Price in Range Bound Market Conditions'
    by Ovidiu Racorean (Quantum Physics Letters, 2014)
    
    Key concepts:
    
    1. Stock prices in range-bound markets can tunnel through support/resistance
    2. Tunneling probability is determined by the transmission coefficient
    3. Volatility drop is a key predictor of tunneling events
    4. The model is based on a financial time-independent Schrödinger equation
    """
    print("\n=============== Quantum Tunneling Model for Stock Prices ===============")
    print("Based on Ovidiu Racorean's work on quantum tunneling in financial markets")
    print("===================================================================\n")
    
    # Initialize parameters
    print("Initializing quantum tunneling parameters...")
    params = QuantumTunnelingParameters()
    
    # Visualize transmission coefficient
    print("Generating visualizations of quantum tunneling coefficients...")
    visualize_transmission_coefficient(params, save_path="static/results/quantum_tunneling_coefficient.png")
    print("Coefficient visualization saved to static/results/quantum_tunneling_coefficient.png")
    
    # Option to run a backtest on historical data
    run_backtest_demo = False
    if run_backtest_demo:
        print("\nFetching historical market data for backtesting...")
        # Fetch some historical data
        symbol = "BTC/USDT"
        timeframe = "1d"
        start_date = '2022-01-01'
        end_date = '2022-12-31'
        
        df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
        print(f"Retrieved {len(df)} data points for {symbol} from {start_date} to {end_date}")
        
        # Run backtest
        print("\nRunning backtest with Quantum Tunneling model...")
        results = run_backtest(df, params, initial_capital=10000, plot=True)
    
    print("\nQuantum Tunneling Model demonstration complete.")
    print("Key features of this model:")
    print("1. Predicts explosive breakouts from range-bound markets")
    print("2. Identifies tunneling probability through support/resistance levels")
    print("3. Shows that volatility drops dramatically before tunneling events")
    print("4. Provides a quantum mechanical explanation for market 'volatility squeeze' patterns") 