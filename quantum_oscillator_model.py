"""
Quantum Oscillator Model of Stock Markets

This module implements the Quantum Oscillator Model described in the paper
'A Quantum Oscillator Model of Stock Markets' by David Orrell.

The model treats stock prices as quantum harmonic oscillators where energy 
levels correspond to integer numbers of transactions, and is derived by 
quantizing entropic forces representing the intentions of buyers and sellers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
from scipy import stats
import ccxt
import os
import time
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import math

# Ensure results directory exists
os.makedirs("static/results", exist_ok=True)

class QuantumOscillatorParameters:
    def __init__(self, market_type="crypto"):
        # Quantum parameters from Orrell's paper
        self.hbar = 1.0       # Financial version of Planck's constant (uncertainty measure)
        self.omega = 1/252    # Oscillator frequency (typical ~ once per year)
        self.lambda_param = 0.25  # Default energy level parameter (1/4 is the default per the paper)
        
        # Market spread parameters
        self.use_empirical_spread = True  # Whether to use empirical bid-ask spread
        self.mu = None        # Market spread (will be calculated from data)
        self.sigma = None     # Ground state volatility (= μ/2)
        
        # Transaction parameters
        self.dt = 1.0         # Time step (default: 1 day)
        
        # Strategy parameters
        self.rsi_period = 14
        self.atr_period = 14
        self.volume_window = 20
        
        # Risk management
        self.stop_loss_atr = 1.5   # ATR multiplier for stop loss
        self.take_profit_atr = 1.2 # ATR multiplier for take profit
        self.cooldown_period = 10  # Bars to wait after a trade
    
    def calculate_mass(self):
        """Calculate oscillator mass from parameters"""
        if self.sigma is None:
            raise ValueError("Sigma must be set before calculating mass")
        return self.hbar / (2 * self.omega * self.sigma**2)
    
    def set_spread_from_volatility(self, daily_volatility):
        """Set the spread and sigma parameters based on daily volatility"""
        self.sigma = daily_volatility * np.sqrt(self.dt)
        self.mu = 2 * self.sigma
    
    def estimated_transactions_per_step(self):
        """Return the expected number of transactions per time step"""
        return self.lambda_param


def calculate_indicators(df, params):
    """Calculate all indicators needed for the strategy"""
    # Basic indicators using ta library
    rsi_indicator = RSIIndicator(df['close'], window=params.rsi_period)
    df['rsi'] = rsi_indicator.rsi()
    
    atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=params.atr_period)
    df['atr'] = atr_indicator.average_true_range()
    
    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(window=params.volume_window).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Calculate log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate market spread if available (high-low or estimated from volatility)
    if 'ask' in df.columns and 'bid' in df.columns:
        df['spread'] = df['ask'] - df['bid']
        df['rel_spread'] = df['spread'] / df['close']
    else:
        # Estimate spread from volatility (μ = σ√τ)
        df['volatility'] = df['log_return'].rolling(window=20).std()
        avg_vol = df['volatility'].mean()
        # Set params spread if not already set
        if params.sigma is None:
            params.set_spread_from_volatility(avg_vol)
        df['est_spread'] = 2 * params.sigma
    
    return df


def quantum_oscillator_wavefunction(x, n, sigma):
    """
    Calculate the wave function for a quantum harmonic oscillator in energy state n
    
    Args:
        x: Price position (log price)
        n: Energy level (integer number of transactions)
        sigma: Ground state volatility
        
    Returns:
        Complex-valued wave function amplitude
    """
    # Normalize coordinate
    xi = x / (sigma * np.sqrt(2))
    
    # Calculate Hermite polynomial for state n
    hermite_poly = special.hermite(n)(xi)
    
    # Calculate normalization factor
    norm_factor = 1.0 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))
    
    # Calculate wave function (real part only, for simplicity)
    psi = norm_factor * hermite_poly * np.exp(-xi**2/2)
    
    return psi


def generate_price_distribution(n, params):
    """
    Generate the price distribution for a quantum oscillator in energy state n
    
    Args:
        n: Energy level
        params: QuantumOscillatorParameters instance
        
    Returns:
        Dictionary with distribution data
    """
    # Volatility for energy state n (equation 4 in the paper)
    sigma_n = params.sigma * np.sqrt(2*n + 1)
    
    # Create a range of price values to calculate distribution
    x_values = np.linspace(-5*sigma_n, 5*sigma_n, 1000)
    
    # Calculate wave function for each price value
    psi_values = np.array([quantum_oscillator_wavefunction(x, n, params.sigma) for x in x_values])
    
    # Calculate probability density as |Ψ|²
    prob_density = np.abs(psi_values)**2
    
    # Normalize to ensure the integral = 1
    prob_density = prob_density / np.trapz(prob_density, x_values)
    
    return {
        'x_values': x_values,
        'prob_density': prob_density,
        'sigma_n': sigma_n,
        'energy_level': n
    }


def calculate_price_impact(order_size, daily_volume, params):
    """
    Calculate the price impact of a large order using the square-root law (equation 5)
    
    Args:
        order_size: Size of the order
        daily_volume: Average daily volume
        params: QuantumOscillatorParameters instance
        
    Returns:
        Expected price change
    """
    # Calculate imbalance ratio
    imbalance = order_size / daily_volume
    
    # Use equation 5 from the paper
    impact_factor = np.sqrt(2 / (2*params.lambda_param + 1))
    price_impact = impact_factor * params.sigma * np.sqrt(imbalance)
    
    return price_impact


def calculate_volatility(price_change, time_period, params):
    """
    Calculate the expected volatility given a price change over a time period (equation 8)
    
    Args:
        price_change: Observed price change
        time_period: Time period in days
        params: QuantumOscillatorParameters instance
        
    Returns:
        Expected volatility
    """
    # Calculate base volatility
    base_variance = params.sigma**2
    
    # Calculate additional variance from displacement (equation 8)
    displacement_variance = (price_change**2) / (2 * time_period)
    
    # Total variance
    total_variance = base_variance + displacement_variance
    
    return np.sqrt(total_variance)


def generate_price_path(days, params, initial_price=100, plot=False):
    """
    Generate a price path using the quantum oscillator model
    
    Args:
        days: Number of days to simulate
        params: QuantumOscillatorParameters instance
        initial_price: Starting price
        plot: Whether to plot the result
        
    Returns:
        DataFrame with price path and energy levels
    """
    # Initialize arrays for price and energy levels
    prices = np.zeros(days)
    energy_levels = np.zeros(days)
    log_prices = np.zeros(days)
    
    # Set initial price
    prices[0] = initial_price
    log_prices[0] = np.log(initial_price)
    
    # Generate random energy levels from Poisson distribution
    energy_levels = np.random.poisson(params.lambda_param, days)
    
    # Calculate price changes based on energy levels
    for i in range(1, days):
        # Current energy level determines volatility
        n = energy_levels[i-1]
        sigma_n = params.sigma * np.sqrt(2*n + 1)
        
        # Generate price change from normal distribution with current volatility
        delta_x = np.random.normal(0, sigma_n)
        
        # Update log price
        log_prices[i] = log_prices[i-1] + delta_x
        
        # Convert back to price
        prices[i] = np.exp(log_prices[i])
    
    # Create DataFrame
    df = pd.DataFrame({
        'price': prices,
        'log_price': log_prices,
        'energy_level': energy_levels
    })
    
    # Optionally plot the results
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot price
        ax1.plot(df.index, df['price'])
        ax1.set_ylabel('Price')
        ax1.set_title('Quantum Oscillator Price Path')
        ax1.grid(True, alpha=0.3)
        
        # Plot energy levels
        ax2.plot(df.index, df['energy_level'])
        ax2.set_ylabel('Energy Level (n)')
        ax2.set_xlabel('Days')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('static/results/quantum_oscillator_price_path.png')
        plt.close()
    
    return df


def find_trading_signals(df, i, params):
    """
    Use the quantum oscillator model to find trading signals based on David Orrell's paper
    "A Quantum Oscillator Model of Stock Markets"
    
    Args:
        df: DataFrame with price data
        i: Current index
        params: Model parameters
        
    Returns:
        Dictionary with signal information
    """
    if i < 30:  # Need enough data for signals
        return {'signal': 'neutral', 'confidence': 0, 'energy_level': 0}
    
    # Get recent data for analysis
    window = df.iloc[i-30:i]
    
    # Current price and indicators
    current_price = df.iloc[i]['close']
    rsi = df.iloc[i]['rsi'] if 'rsi' in df.columns else 50
    atr = df.iloc[i]['atr'] if 'atr' in df.columns else current_price * 0.01
    
    # ==== MARKET IMBALANCE CALCULATION (EQUATION 3) ====
    # Calculate volume imbalance as proxy for market imbalance described in the paper
    if 'volume' in df.columns:
        # Use 5-day average volume as the baseline
        avg_volume = df.iloc[i-5:i]['volume'].mean()
        current_volume = df.iloc[i]['volume']
        
        # Volume ratio as indicator of imbalance (higher = more imbalance)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Imbalance direction based on price movement
        price_direction = np.sign(df.iloc[i]['close'] - df.iloc[i-1]['close'])
        
        # Calculate imbalance (ι) as per equation 3 in the paper
        # ι = (Nb - Na) / min(Na, Nb)
        imbalance = price_direction * (volume_ratio - 1.0) if volume_ratio > 1.0 else 0
    else:
        # Fallback if volume not available - estimate from price momentum
        price_change_5day = df.iloc[i]['close'] / df.iloc[i-5]['close'] - 1
        imbalance = price_change_5day * 10  # Scale to reasonable range
    
    # ==== VOLATILITY ESTIMATION (EQUATION 4 & 8) ====
    # Calculate base volatility (sigma) from the spread parameter
    base_vol = params.sigma
    
    # Recent volatility (using paper's definition with lookback)
    recent_returns = df.iloc[i-20:i]['log_return'].dropna()
    if len(recent_returns) > 5:
        recent_vol = recent_returns.std()
    else:
        recent_vol = base_vol
    
    # Calculate price displacement over time period (x in paper)
    price_displacement = np.log(df.iloc[i]['close'] / df.iloc[i-20]['close'])
    
    # Calculate expected volatility using the paper's equation 8
    # σ_x² = σ² + x²/(2T)
    time_period = 20  # 20-day window
    expected_variance = base_vol**2 + (price_displacement**2) / (2 * time_period)
    expected_vol = np.sqrt(expected_variance)
    
    # ==== ENERGY LEVEL ESTIMATION (EQUATION 4) ====
    # Energy level n based on volatility as per equation 4
    # σ_n = σ√(2n + 1)
    # Solving for n: n = ((σ_n/σ)² - 1) / 2
    measured_vol = max(recent_vol, expected_vol)  # Use the higher of the two
    energy_ratio = (measured_vol / base_vol)**2
    estimated_energy = max(0, (energy_ratio - 1) / 2)
    
    # Calculate volatility factor for signal scaling
    vol_factor = np.sqrt(2 * estimated_energy + 1)
    
    # Calculate volatility trend (increasing or decreasing)
    if i > 40:
        # Previous volatility window
        prev_returns = df.iloc[i-40:i-20]['log_return'].dropna()
        if len(prev_returns) > 5:
            prev_vol = prev_returns.std()
            vol_trend = recent_vol / prev_vol - 1  # Positive = increasing volatility
        else:
            vol_trend = 0
    else:
        vol_trend = 0
    
    # ==== PRICE IMPACT ESTIMATION (EQUATION 5) ====
    # Expected price change using square-root law (equation 5)
    # x = ±σ_d√(Q/V_d)
    # For our trading signal, we'll use this to gauge potential move magnitude
    
    # Calculate signal strength based on oscillator energy level and imbalance
    signal_strength = 0
    
    # 1. Component from energy level - higher energy suggests larger price moves
    energy_component = 0.3 * np.sqrt(estimated_energy) * np.sign(imbalance)
    
    # 2. Component from market imbalance - drives direction (buy/sell)
    imbalance_component = 0.4 * np.sign(imbalance) * min(1.0, np.sqrt(abs(imbalance)))
    
    # 3. Component from RSI - mean reversion signal
    if rsi < 30:
        rsi_component = 0.3 * (30 - rsi) / 30  # Positive = buy signal
    elif rsi > 70:
        rsi_component = -0.3 * (rsi - 70) / 30  # Negative = sell signal
    else:
        rsi_component = 0
    
    # 4. Component from volatility trend
    vol_component = 0.2 * vol_trend * np.sign(imbalance)
    
    # Combine components with energy-dependent scaling
    signal_strength = (
        energy_component + 
        imbalance_component + 
        rsi_component +
        vol_component
    )
    
    # Scale by the hbar parameter (higher hbar = more aggressive)
    signal_strength *= (params.hbar / 1.0)
    
    # Determine signal type and confidence
    signal = None
    confidence = 0
    
    # Amplitude of the signal (paper discusses minimum amplitude for meaningful signals)
    amplitude = abs(signal_strength)
    
    # Get the signal threshold parameter (or use default)
    signal_threshold = getattr(params, 'signal_threshold', 0.6)
    
    # Get the minimum amplitude parameter (or use default from paper)
    min_amplitude = getattr(params, 'min_amplitude', 0.02)
    
    # Apply quantum wave function collapse - when measured, we get a discrete signal
    if amplitude >= min_amplitude:
        if signal_strength > 0:
            signal = 'buy'
            # Scale confidence by signal strength but not linearly (quantum probability)
            confidence = min(0.95, signal_threshold + (np.sqrt(amplitude) - np.sqrt(min_amplitude)))
        elif signal_strength < 0:
            signal = 'sell'
            # Scale confidence by signal strength but not linearly (quantum probability)
            confidence = min(0.95, signal_threshold + (np.sqrt(amplitude) - np.sqrt(min_amplitude)))
    else:
        signal = 'neutral'
        confidence = 0.3 * (amplitude / min_amplitude)
    
    # Check for significant energy level transitions (quantum jumps)
    if i > 50:
        # Look for energy transitions over longer period
        long_window = df.iloc[i-50:i-30]['log_return'].dropna()
        if len(long_window) > 5:
            long_vol = long_window.std()
            long_energy_ratio = (long_vol / base_vol)**2
            long_energy = max(0, (long_energy_ratio - 1) / 2)
            
            # Major energy level transition (quantum jump)
            energy_jump = estimated_energy - long_energy
            
            # Big jumps in energy levels can precede significant price moves
            if abs(energy_jump) > params.lambda_param:
                # Direction determined by combination of trend and rsi
                if energy_jump > 0 and (rsi < 50 or df.iloc[i]['close'] > df.iloc[i-1]['close']):
                    signal = 'buy'
                    confidence = max(confidence, min(0.95, 0.5 + 0.1 * energy_jump))
                elif energy_jump > 0:
                    signal = 'sell'
                    confidence = max(confidence, min(0.95, 0.5 + 0.1 * energy_jump))
    
    # ==== FINAL SIGNAL DETERMINATION ====
    # If trading parameters exist, apply them (for improved strategy variants)
    
    # Effective threshold - lower for aggressive strategies with higher energy levels
    effective_threshold = signal_threshold
    if estimated_energy > params.lambda_param * 1.5:
        # Higher energy levels = lower threshold for signals
        effective_threshold = max(0.3, signal_threshold - 0.2)
    
    # Return detailed signal information
    signal_dict = {
        'signal': signal if confidence > effective_threshold else 'neutral',
        'confidence': confidence,
        'energy_level': estimated_energy,
        'volatility_factor': vol_factor,
        'imbalance': imbalance,
        'signal_strength': signal_strength,
        'amplitude': amplitude,
        'rsi': rsi
    }
    
    return signal_dict


def fetch_crypto_data(symbol, timeframe, start_date, end_date=None, exchange_id='binance'):
    """
    Fetch cryptocurrency historical data
    """
    # Initialize exchange
    exchange = getattr(ccxt, exchange_id)()
    
    # Convert dates to timestamps
    since = exchange.parse8601(f"{start_date}T00:00:00Z")
    until = None if end_date is None else exchange.parse8601(f"{end_date}T23:59:59Z")
    
    # Fetch data
    all_candles = []
    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not candles or len(candles) == 0:
                break
            
            all_candles.extend(candles)
            
            # Update since timestamp for next iteration
            since = candles[-1][0] + 1
            
            # Break if we reached the end date
            if until and since >= until:
                break
                
            # Exchange-specific rate limiting
            if exchange_id == 'binance':
                time.sleep(1)  # Respect rate limits
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df


def visualize_quantum_distributions(params, save_path=None):
    """
    Visualize the quantum oscillator probability distributions for different energy levels
    """
    # Energy levels to visualize
    energy_levels = [0, 1, 3, 5]
    labels = ['Ground State (n=0)', 'First Excited (n=1)', 'Third Excited (n=3)', 'Fifth Excited (n=5)']
    
    plt.figure(figsize=(12, 8))
    
    # Plot probability density for each energy level
    for n, label in zip(energy_levels, labels):
        # Get distribution
        dist = generate_price_distribution(n, params)
        
        # Plot
        plt.plot(dist['x_values'], dist['prob_density'], label=f"{label}, σ={dist['sigma_n']:.4f}")
    
    # Add Gaussian for comparison
    x = np.linspace(-5*params.sigma, 5*params.sigma, 1000)
    plt.plot(x, stats.norm.pdf(x, 0, params.sigma), 'k--', label='Gaussian (Classical)')
    
    # Add annotations
    plt.xlabel('Log Price Change', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Quantum Oscillator Probability Distributions at Different Energy Levels', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text explaining energy levels
    annotation = (
        "Energy Levels:\n"
        "- Ground State (n=0): Basic volatility from spread\n"
        "- Excited States (n>0): Additional volatility from transactions\n"
        "Note: Higher energy levels lead to fat-tailed distributions"
    )
    plt.figtext(0.5, 0.01, annotation, ha='center', fontsize=10, bbox=dict(facecolor='ivory', alpha=0.5))
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    # Also visualize price impact and volatility curves
    plt.figure(figsize=(12, 8))
    
    # Plot price impact (square-root law)
    imbalance = np.linspace(0, 1, 100)
    impact = [calculate_price_impact(q, 1, params) for q in imbalance]
    
    plt.subplot(2, 1, 1)
    plt.plot(imbalance, impact)
    plt.xlabel('Order Size / Daily Volume')
    plt.ylabel('Price Impact')
    plt.title('Square-Root Law of Price Impact')
    plt.grid(True, alpha=0.3)
    
    # Plot volatility smile
    price_changes = np.linspace(-0.05, 0.05, 100)
    volatility = [calculate_volatility(x, 1, params) for x in price_changes]
    
    plt.subplot(2, 1, 2)
    plt.plot(price_changes, volatility)
    plt.xlabel('Price Change')
    plt.ylabel('Volatility')
    plt.title('Volatility Smile from Quantum Oscillator Model')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_impact_volatility.png'), dpi=300)
        plt.close()
    else:
        plt.show()


def run_backtest(df, params, initial_capital=10000, plot=True):
    """
    Run a backtest of the Quantum Oscillator strategy
    """
    print(f"Starting Quantum Oscillator backtest with {len(df)} data points...")
    
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
    
    # Prepare for tracking
    trade_log = []
    energy_levels = []
    
    # Set confidence threshold based on parameters
    # More aggressive strategies (higher lambda, higher hbar) should have lower thresholds
    base_confidence_threshold = 0.6
    confidence_threshold = max(0.4, base_confidence_threshold - ((params.lambda_param - 0.25) * 0.3) - ((params.hbar - 1.0) * 0.1))
    
    print(f"Using confidence threshold: {confidence_threshold:.2f} for trade entry")
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
                take_profit = entry_price * (1 + params.take_profit_atr * df.iloc[entry_index]['atr'] / entry_price)
                
                # Check stop loss or take profit
                if current_price <= stop_price or current_price >= take_profit:
                    # Close position
                    trade_result = (current_price - entry_price) / entry_price
                    equity_change = equity[-1] * (1 + trade_result)
                    equity.append(equity_change)
                    
                    # Record trade
                    exit_reason = 'Stop Loss' if current_price <= stop_price else 'Take Profit'
                    trade_data = {
                        'entry_date': df.index[entry_index],
                        'exit_date': current_date,
                        'position': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_price,
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
                take_profit = entry_price * (1 - params.take_profit_atr * df.iloc[entry_index]['atr'] / entry_price)
                
                # Check stop loss or take profit
                if current_price >= stop_price or current_price <= take_profit:
                    # Close position
                    trade_result = (entry_price - current_price) / entry_price
                    equity_change = equity[-1] * (1 + trade_result)
                    equity.append(equity_change)
                    
                    # Record trade
                    exit_reason = 'Stop Loss' if current_price >= stop_price else 'Take Profit'
                    trade_data = {
                        'entry_date': df.index[entry_index],
                        'exit_date': current_date,
                        'position': position_type,
                        'entry_price': entry_price,
                        'exit_price': current_price,
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
            # Get signal from the quantum oscillator model
            signal_data = find_trading_signals(df, i, params)
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            energy_level = signal_data['energy_level']
            
            # Track energy level estimates
            energy_levels.append(energy_level)
            
            # Enter position if signal is strong enough
            if signal == 'buy' and confidence > confidence_threshold:
                # Enter long position
                in_position = True
                position_type = 'long'
                entry_price = current_price
                entry_index = i
                
            elif signal == 'sell' and confidence > confidence_threshold:
                # Enter short position
                in_position = True
                position_type = 'short'
                entry_price = current_price
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
            
            # Plot price
            axs[1].plot(df.index, df['close'])
            axs[1].set_title('Price')
            axs[1].set_ylabel('Price')
            axs[1].grid(True, alpha=0.3)
            
            # Plot energy levels - FIX: Ensure x and y have the same dimensions
            if energy_levels:
                x_energy = df.index[30:30+len(energy_levels)]
                # Check if x and y have same length, adjust if needed
                if len(x_energy) != len(energy_levels):
                    # Make them the same length by using the shorter one
                    min_len = min(len(x_energy), len(energy_levels))
                    x_energy = x_energy[:min_len]
                    energy_plot = energy_levels[:min_len]
                    print(f"Adjusted energy plotting arrays to length {min_len}")
                else:
                    energy_plot = energy_levels
                    
                axs[2].plot(x_energy, energy_plot)
                axs[2].set_title('Estimated Energy Levels')
                axs[2].set_ylabel('Energy Level (n)')
                axs[2].set_xlabel('Date')
                axs[2].grid(True, alpha=0.3)
            
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
            plt.savefig(f"static/results/quantum_oscillator_backtest_{timestamp}.png")
            plt.close()
            
            # Also save the distribution plots
            visualize_quantum_distributions(params, save_path=f"static/results/quantum_oscillator_distributions_{timestamp}.png")
        except Exception as e:
            # If plotting fails, log the error but don't stop the backtest
            print(f"Warning: Failed to create plots: {str(e)}")
            plt.close('all')  # Close any open plots to avoid memory leaks
    
    # Print results
    print("\n=== Quantum Oscillator Backtest Results ===")
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
        trade_df.to_csv(f"static/results/quantum_oscillator_trades_{timestamp}.csv", index=False)
    
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
        'trade_log': trade_log
    }
    
    return results


if __name__ == "__main__":
    """
    Demonstration of the Quantum Oscillator Model for Stock Markets
    
    Key concepts from David Orrell's paper:
    
    1. The model treats stock prices as quantum harmonic oscillators
    2. Energy level corresponds to integer number of transactions
    3. The model is derived by quantizing entropic forces representing buying/selling intentions
    4. It captures the non-Gaussian nature of financial returns
    5. It correctly predicts the square-root law of price impact and volatility smile
    """
    print("\n=============== Quantum Oscillator Model for Stock Markets ===============")
    print("Based on 'A Quantum Oscillator Model of Stock Markets' by David Orrell")
    print("===================================================================\n")
    
    # Initialize parameters
    print("Initializing quantum oscillator parameters...")
    params = QuantumOscillatorParameters()
    
    # Set volatility parameter for demonstration
    params.set_spread_from_volatility(0.01)  # 1% daily volatility
    
    # Generate and visualize distributions
    print("Generating visualizations of quantum oscillator distributions...")
    visualize_quantum_distributions(params, save_path="static/results/quantum_oscillator_distributions.png")
    print("Distributions saved to static/results/quantum_oscillator_distributions.png")
    
    # Generate a price path
    print("Generating a price path simulation...")
    price_path = generate_price_path(252, params, initial_price=100, plot=True)
    print("Price path saved to static/results/quantum_oscillator_price_path.png")
    
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
        print("\nRunning backtest with Quantum Oscillator model...")
        results = run_backtest(df, params, initial_capital=10000, plot=True)
    
    print("\nQuantum Oscillator Model demonstration complete.")
    print("This model captures the following market features:")
    print("1. Non-Gaussian nature of price returns")
    print("2. The square-root law of price impact")
    print("3. The relationship between volatility and price change")
    print("4. The connection between market spread and uncertainty") 