"""
Quantum Anharmonic Oscillator Model of Stock Markets

This module implements a Quantum Anharmonic Oscillator Model for financial markets,
extending the Quantum Oscillator Model by David Orrell with anharmonic terms.

The anharmonic oscillator adds cubic and quartic terms to the potential function,
allowing for asymmetric distributions that better model market skew and kurtosis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special, integrate
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

class QuantumAnharmonicParameters:
    def __init__(self, market_type="crypto"):
        # Quantum parameters
        self.hbar = 1.0       # Financial version of Planck's constant
        self.omega = 1/252    # Base oscillator frequency (typical ~ once per year)
        self.lambda_param = 0.25  # Energy level parameter
        
        # Anharmonic parameters (extensions to the model)
        self.beta = 0.1       # Cubic anharmonic term (controls skew)
        self.gamma = 0.05     # Quartic anharmonic term (controls kurtosis)
        
        # Market spread parameters
        self.use_empirical_spread = True
        self.mu = None        # Market spread (will be calculated from data)
        self.sigma = None     # Ground state volatility
        
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
    
    # Calculate volume relative to moving average
    df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume'].rolling(window=params.volume_window).mean()
    
    # Calculate log returns
    df_copy['log_return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
    
    # Estimate market spread (if empirical spread is used)
    if params.use_empirical_spread and params.mu is None:
        # Use high-low range as a proxy for spread
        avg_range = (df_copy['high'] - df_copy['low']).median() / df_copy['close'].median()
        params.mu = avg_range
        params.sigma = params.mu / 2
    
    return df_copy

def anharmonic_potential(x, params):
    """
    Calculate the anharmonic potential energy function
    
    V(x) = (1/2)mω²x² + βx³ + γx⁴
    
    where:
    - m is the mass
    - ω is the oscillator frequency
    - β is the cubic anharmonic coefficient
    - γ is the quartic anharmonic coefficient
    - x is the price position (log price)
    """
    m = params.calculate_mass()
    omega_squared = params.omega ** 2
    
    # Harmonic term
    harmonic = 0.5 * m * omega_squared * x**2
    
    # Anharmonic terms
    cubic = params.beta * x**3
    quartic = params.gamma * x**4
    
    return harmonic + cubic + quartic

def quantum_anharmonic_wavefunction_numerical(x, n, params):
    """
    Calculate an approximate wave function for a quantum anharmonic oscillator
    in energy state n using perturbation theory
    
    Args:
        x: Price position (log price)
        n: Energy level (integer number of transactions)
        params: QuantumAnharmonicParameters instance
        
    Returns:
        Approximate wave function amplitude
    """
    # Start with the harmonic oscillator wave function
    sigma = params.sigma
    xi = x / (sigma * np.sqrt(2))
    
    # Calculate unperturbed harmonic oscillator wavefunction
    hermite_poly = special.hermite(n)(xi)
    norm_factor = 1.0 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))
    psi_harmonic = norm_factor * hermite_poly * np.exp(-xi**2/2)
    
    # Apply first-order perturbation for anharmonic terms
    # This is a simplified approximation - a full solution would require
    # numerical solution of the Schrödinger equation with anharmonic potential
    
    # Calculate perturbation factors based on anharmonic strength
    beta_factor = params.beta * sigma**3 * (n**0.5) * 0.1
    gamma_factor = params.gamma * sigma**4 * (n + 0.5) * 0.05
    
    # Add skew from cubic term (proportional to x)
    skew_factor = 1.0 + beta_factor * xi
    
    # Add kurtosis from quartic term (proportional to x²)
    kurtosis_factor = 1.0 + gamma_factor * (xi**2 - 1.0)
    
    # Combine the factors for final wavefunction
    psi = psi_harmonic * skew_factor * kurtosis_factor
    
    return psi

def generate_price_distribution(n, params):
    """
    Generate the price distribution for a quantum anharmonic oscillator in energy state n
    
    Args:
        n: Energy level
        params: QuantumAnharmonicParameters instance
        
    Returns:
        Dictionary with distribution data
    """
    # Base volatility adjusted for energy level
    # For anharmonic oscillators, energy levels are no longer equally spaced
    base_sigma = params.sigma
    anharmonic_correction = 1.0 + 0.1 * params.gamma * (n**2)
    sigma_n = base_sigma * np.sqrt(2*n + 1) * anharmonic_correction
    
    # Create a range of price values with more range on positive side if beta > 0
    skew_direction = np.sign(params.beta)
    x_min = -5 * sigma_n
    x_max = 5 * sigma_n + skew_direction * 2 * abs(params.beta) * sigma_n
    x_values = np.linspace(x_min, x_max, 1000)
    
    # Calculate wave function for each price value
    psi_values = np.array([quantum_anharmonic_wavefunction_numerical(x, n, params) for x in x_values])
    
    # Calculate probability density as |Ψ|²
    prob_density = np.abs(psi_values)**2
    
    # Normalize to ensure the integral = 1
    prob_density = prob_density / integrate.trapezoid(prob_density, x_values)
    
    # Calculate statistical moments
    mean = integrate.trapezoid(x_values * prob_density, x_values)
    variance = integrate.trapezoid((x_values - mean)**2 * prob_density, x_values)
    skewness = integrate.trapezoid((x_values - mean)**3 * prob_density, x_values) / variance**1.5
    kurtosis = integrate.trapezoid((x_values - mean)**4 * prob_density, x_values) / variance**2 - 3
    
    return {
        'x_values': x_values,
        'prob_density': prob_density,
        'sigma_n': sigma_n,
        'energy_level': n,
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis
    }

def calculate_price_impact(order_size, daily_volume, params):
    """
    Calculate the price impact of a large order using an extended square-root law
    that accounts for market asymmetry from the anharmonic terms
    
    Args:
        order_size: Size of the order
        daily_volume: Average daily volume
        params: QuantumAnharmonicParameters instance
        
    Returns:
        Expected price change
    """
    # Calculate imbalance ratio
    imbalance = order_size / daily_volume
    
    # Direction of the order (buy = 1, sell = -1)
    direction = np.sign(order_size)
    
    # Base impact from harmonic oscillator model
    impact_factor = np.sqrt(2 / (2*params.lambda_param + 1))
    base_impact = impact_factor * params.sigma * np.sqrt(abs(imbalance))
    
    # Additional impact from anharmonic terms
    # Cubic term causes asymmetry between buys and sells
    cubic_correction = 1.0 + direction * params.beta * 2.0
    
    # Quartic term causes larger impact for bigger orders
    quartic_correction = 1.0 + params.gamma * abs(imbalance)
    
    # Combined impact
    price_impact = direction * base_impact * cubic_correction * quartic_correction
    
    return price_impact

def find_trading_signals(df, i, params):
    """
    Find trading signals based on the quantum anharmonic oscillator model
    
    Args:
        df: DataFrame with price data and indicators
        i: Current index
        params: QuantumAnharmonicParameters instance
        
    Returns:
        Dictionary with signal information
    """
    # Extract relevant data
    price = df.iloc[i]['close']
    prev_price = df.iloc[i-1]['close']
    rsi = df.iloc[i]['rsi']
    atr = df.iloc[i]['atr']
    volume_ratio = df.iloc[i]['volume_ratio']
    
    # Calculate recent returns and volatility
    returns = df.iloc[i-20:i]['log_return'].dropna()
    if len(returns) < 5:
        return {'signal': 'neutral', 'confidence': 0, 'energy_level': 0}
    
    # Calculate statistical moments of recent returns
    recent_skew = stats.skew(returns)
    recent_kurt = stats.kurtosis(returns)
    
    # Estimate current energy level from the volatility
    # Higher volatility = higher energy level
    std_dev = returns.std()
    base_vol = params.sigma
    estimated_energy = max(0, round((std_dev / base_vol)**2 / 2 - 0.5))
    
    # Generate the current quantum state distribution
    dist = generate_price_distribution(estimated_energy, params)
    
    # Calculate expected return from the quantum distribution
    expected_return = dist['mean']
    
    # Calculate signal - MUCH MORE SENSITIVE VERSION
    # Basic signal based on price momentum and direction of the wavefunction
    signal_strength = 0
    
    # Component 1: Price momentum (short-term)
    price_change_1d = price / prev_price - 1
    momentum_signal = np.sign(price_change_1d) * min(1.0, abs(price_change_1d) * 100)
    
    # Component 2: RSI divergence from equilibrium
    rsi_signal = 0
    if rsi < 50:
        # The lower the RSI below 50, the stronger the buy signal
        rsi_signal = (50 - rsi) / 20  # Normalized to roughly -1.0 to 1.0
    else:
        # The higher the RSI above 50, the stronger the sell signal
        rsi_signal = -1 * (rsi - 50) / 20
    
    # Component 3: Anharmonic quantum state expected return
    # Increase importance of anharmonic effects based on params
    anharmonic_impact = params.beta * 10 + params.gamma * 20
    quantum_signal = expected_return * (1 + anharmonic_impact)
    
    # Component 4: Energy level transitions (quantum jumps)
    if i > 30:
        # Look at energy level changes over time
        prev_returns = df.iloc[i-30:i-10]['log_return'].dropna()
        if len(prev_returns) > 5:
            prev_std = prev_returns.std()
            prev_energy = max(0, round((prev_std / base_vol)**2 / 2 - 0.5))
            energy_change = estimated_energy - prev_energy
            
            # Energy level increases tend to precede price movements
            energy_signal = np.sign(energy_change) * min(1.0, abs(energy_change) * 0.5)
        else:
            energy_signal = 0
    else:
        energy_signal = 0
    
    # Component 5: Volume impact
    volume_signal = 0
    if volume_ratio > 1.2:
        # Higher volume reinforces the direction of price change
        volume_signal = np.sign(price_change_1d) * min(1.0, (volume_ratio - 1) * 0.5)
    
    # Weight the components
    signal_strength = (
        momentum_signal * 0.3 +
        rsi_signal * 0.3 +
        quantum_signal * 0.2 +
        energy_signal * 0.1 +
        volume_signal * 0.1
    )
    
    # Adjust with cubic anharmonic parameter (skew)
    # If beta is positive, enhance positive signals; if negative, enhance negative signals
    if params.beta > 0 and signal_strength > 0:
        signal_strength *= (1 + params.beta * 2)
    elif params.beta < 0 and signal_strength < 0:
        signal_strength *= (1 + abs(params.beta) * 2)
        
    # Calculate base confidence from signal strength
    base_confidence = min(0.95, abs(signal_strength) + 0.3)
    
    # Add confidence from energy level - higher energy means more confident signals
    energy_boost = min(0.2, estimated_energy * 0.05)
    
    # Combine for final confidence value
    confidence = min(0.95, base_confidence + energy_boost)
    
    # Apply minimum threshold based on parameters
    # Get threshold from parameters or use default
    threshold = getattr(params, 'signal_threshold', 0.3)
    
    # Determine final signal with much lower threshold
    if signal_strength > threshold / 2:
        return {'signal': 'buy', 'confidence': confidence, 'energy_level': estimated_energy}
    elif signal_strength < -threshold / 2:
        return {'signal': 'sell', 'confidence': confidence, 'energy_level': estimated_energy}
    else:
        return {'signal': 'neutral', 'confidence': confidence, 'energy_level': estimated_energy}

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

def visualize_quantum_distributions(params, save_path=None):
    """
    Visualize the quantum anharmonic oscillator probability distributions 
    for different energy levels
    """
    # Energy levels to visualize
    energy_levels = [0, 1, 3, 5]
    labels = ['Ground State (n=0)', 'First Excited (n=1)', 'Third Excited (n=3)', 'Fifth Excited (n=5)']
    
    plt.figure(figsize=(12, 10))
    
    # Plot probability density for each energy level
    for n, label in zip(energy_levels, labels):
        # Get distribution
        dist = generate_price_distribution(n, params)
        
        # Plot
        plt.plot(dist['x_values'], dist['prob_density'], label=f"{label}, σ={dist['sigma_n']:.4f}")
        
        # Add vertical line at the mean
        plt.axvline(x=dist['mean'], color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=0.5)
    
    # Add Gaussian for comparison
    x = np.linspace(-5*params.sigma, 5*params.sigma, 1000)
    plt.plot(x, stats.norm.pdf(x, 0, params.sigma), 'k--', label='Gaussian (Classical)')
    
    # Add annotations
    plt.xlabel('Log Price Change', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Quantum Anharmonic Oscillator Probability Distributions', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text explaining the distributions
    text = (
        f"Anharmonic Parameters: β={params.beta:.2f} (skew), γ={params.gamma:.2f} (kurtosis)\n"
        "• Ground State: Base volatility from spread\n"
        "• Higher Energy: Added volatility + skew/kurtosis\n"
        "• Vertical lines show mean value (return bias)"
    )
    plt.figtext(0.5, 0.01, text, ha='center', fontsize=10, bbox=dict(facecolor='ivory', alpha=0.5))
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
    
    # Create a second plot for moments vs energy level
    plt.figure(figsize=(12, 8))
    
    # Generate data for more energy levels
    e_levels = list(range(8))
    moments = [generate_price_distribution(n, params) for n in e_levels]
    
    # Plot statistical moments
    plt.subplot(2, 2, 1)
    plt.plot(e_levels, [m['sigma_n'] for m in moments], 'o-')
    plt.title('Volatility vs Energy Level')
    plt.xlabel('Energy Level (n)')
    plt.ylabel('Volatility (σₙ)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(e_levels, [m['mean'] for m in moments], 'o-')
    plt.title('Mean Return vs Energy Level')
    plt.xlabel('Energy Level (n)')
    plt.ylabel('Expected Return')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(e_levels, [m['skewness'] for m in moments], 'o-')
    plt.title('Skewness vs Energy Level')
    plt.xlabel('Energy Level (n)')
    plt.ylabel('Skewness')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(e_levels, [m['kurtosis'] for m in moments], 'o-')
    plt.title('Excess Kurtosis vs Energy Level')
    plt.xlabel('Energy Level (n)')
    plt.ylabel('Excess Kurtosis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save moments plot
    if save_path:
        moments_path = save_path.replace('.png', '_moments.png')
        plt.savefig(moments_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def run_backtest(df, params, initial_capital=10000, plot=True):
    """
    Run a backtest of the Quantum Anharmonic Oscillator strategy
    """
    try:
        print(f"Starting Quantum Anharmonic Oscillator backtest with {len(df)} data points...")
        
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
        
        # Set confidence threshold based on parameters - MUCH LOWER THRESHOLD
        # Get threshold from parameters or use default with a 0.35 minimum
        confidence_threshold = max(0.35, getattr(params, 'signal_threshold', 0.5) - 0.15)
        
        print(f"Starting backtest loop through {len(df)-30} bars...")
        # Loop through data (start from index 30 to have enough data for indicators)
        for i in range(30, len(df)):
            try:
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
                    try:
                        # Get signal from the quantum anharmonic oscillator model
                        signal_data = find_trading_signals(df, i, params)
                        signal = signal_data['signal']
                        confidence = signal_data['confidence']
                        energy_level = signal_data['energy_level']
                        
                        # Track energy level estimates
                        energy_levels.append(energy_level)
                        
                        # Enter position if signal is strong enough - USES LOWER THRESHOLD NOW
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
                    except Exception as signal_error:
                        print(f"Error finding signal at bar {i}: {signal_error}")
                        # Continue with next bar
                
                # If no trade, equity remains the same
                if len(equity) <= i - 29:
                    equity.append(equity[-1])
                    
            except Exception as bar_error:
                print(f"Error processing bar {i}: {bar_error}")
                # Continue with next bar
                if len(equity) <= i - 29:
                    equity.append(equity[-1] if equity else initial_capital)
        
        print("Backtest loop completed.")
        
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
        winning_trades = sum(1 for t in trades if t > 0) if trades else 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t for t in trades if t > 0]) * 100 if winning_trades > 0 else 0
        avg_loss = np.mean([t for t in trades if t < 0]) * 100 if (total_trades - winning_trades) > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(t for t in trades if t > 0) if trades else 0
        gross_loss = abs(sum(t for t in trades if t < 0)) if trades else 1  # Avoid division by zero
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate max drawdown
        running_max = initial_capital
        drawdown = 0
        max_drawdown = 0
        
        for e in equity:
            if e > running_max:
                running_max = e
            
            drawdown = (running_max - e) / running_max if running_max > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Final equity and return
        final_equity = equity[-1] if equity else initial_capital
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Create trade DataFrame for analysis
        trade_df = pd.DataFrame(trade_log)
        
        try:
            # Plot results if requested
            if plot and len(equity) > 0:
                # Create a figure with subplots
                fig, axs = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
                
                # Plot equity curve
                axs[0].plot(df.index[29:29+len(equity)], equity)
                axs[0].set_title('Equity Curve')
                axs[0].set_ylabel('Equity')
                axs[0].grid(True, alpha=0.3)
                
                # Plot price
                axs[1].plot(df.index, df['close'])
                axs[1].set_title('Price')
                axs[1].set_ylabel('Price')
                axs[1].grid(True, alpha=0.3)
                
                # Plot energy levels
                if energy_levels:
                    axs[2].plot(df.index[30:30+len(energy_levels)], energy_levels)
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
                plt.savefig(f"static/results/quantum_anharmonic_backtest_{timestamp}.png")
                plt.close()
                
                # Also save the distribution plots
                try:
                    visualize_quantum_distributions(params, save_path=f"static/results/quantum_anharmonic_distributions_{timestamp}.png")
                except Exception as viz_error:
                    print(f"Error generating distributions: {viz_error}")
        except Exception as plot_error:
            print(f"Error generating plots: {plot_error}")
        
        # Print results
        print("\n=== Quantum Anharmonic Oscillator Backtest Results ===")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Win: {avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        
        # Save trade log
        try:
            if not trade_df.empty:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file = f"static/results/quantum_anharmonic_trades_{timestamp}.csv"
                trade_df.to_csv(csv_file, index=False)
                print(f"Trade log saved to {csv_file}")
        except Exception as save_error:
            print(f"Error saving trade log: {save_error}")
        
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
    
    except Exception as e:
        import traceback
        print(f"ERROR in Quantum Anharmonic Oscillator backtest: {e}")
        traceback.print_exc()
        # Return a minimal result set to avoid breaking the application
        return {
            'equity': [initial_capital],
            'total_return': 0,
            'total_trades': 0,
            'win_rate': 0,
            'average_win': 0,
            'average_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'trade_log': [],
            'error': str(e)
        }


if __name__ == "__main__":
    """
    Demonstration of the Quantum Anharmonic Oscillator Model for Stock Markets
    
    Key concepts:
    
    1. The model treats stock prices as quantum anharmonic oscillators
    2. Adds cubic (β) and quartic (γ) terms to model return skewness and kurtosis
    3. Energy level corresponds to market transaction/activity level
    4. Captures market asymmetry and fat-tail distributions better than harmonic oscillator
    """
    print("\n=============== Quantum Anharmonic Oscillator Model ===============")
    print("Extending the Quantum Oscillator with anharmonic terms for better market modeling")
    print("===================================================================\n")
    
    # Initialize parameters
    print("Initializing quantum anharmonic oscillator parameters...")
    params = QuantumAnharmonicParameters()
    
    # Set volatility parameter for demonstration
    params.set_spread_from_volatility(0.01)  # 1% daily volatility
    
    # Generate and visualize distributions
    print("Generating visualizations of quantum anharmonic oscillator distributions...")
    visualize_quantum_distributions(params, save_path="static/results/quantum_anharmonic_distributions.png")
    print("Distributions saved to static/results/quantum_anharmonic_distributions.png")
    
    print("\nQuantum Anharmonic Oscillator Model demonstration complete.")
    print("This model extends the quantum oscillator with:")
    print("1. Cubic term (β) to model return skewness/asymmetry")
    print("2. Quartic term (γ) to model excess kurtosis/fat tails")
    print("3. Better captures market asymmetry and complex statistical properties")
    print("4. Provides enhanced accuracy for option pricing and risk analysis") 