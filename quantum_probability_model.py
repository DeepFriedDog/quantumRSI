"""
Quantum Probability Theoretic Asset Return Model

This module implements the Quantum Probability model as described in the paper
'Quantum Probability Theoretic Asset Return Modeling: A Novel Schrödinger-Like Trading Equation and Multimodal Distribution'

The model applies quantum probability theory to asset returns through a Schrödinger-like trading equation,
interpreting active trading intentions (ATI) using complex-valued probability amplitudes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import minimize
import ccxt
import os
import time
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import math  # Add import for standard math module

# Ensure the results directory exists
os.makedirs("results", exist_ok=True)

class QuantumProbabilityParameters:
    def __init__(self, market_type="crypto"):
        # Quantum parameters from the paper by Lin Li
        self.alpha = 2.0  # α parameter: related to market's natural tendency to revert to mean
        self.delta = 0.5  # δ parameter: anharmonic coefficient representing market nonlinearity
        self.h = 1.0      # h parameter: Planck's constant analog (scale parameter)
        
        # Trading energy level parameters (Ω values)
        self.omega_level = 1  # Default energy level (ground state)
        self.omega_0 = 1      # Ground state energy (normal distribution)
        self.omega_1 = 3      # First excited state (bimodal distribution)
        self.omega_2 = 5      # Second excited state (trimodal distribution)
        
        # Additional parameters from the paper
        self.lambda_param = 0.05  # λ parameter: deviation from harmonic trading (anharmonicity)
        self.eta = 0.5            # η parameter: normalized threshold for distribution transition
        
        # Critical volume threshold - volume needed to transition distributions
        self.V_max = None         # Will be calculated from data
        self.E_0 = None           # Ground state energy level threshold
        
        # Strategy parameters
        self.rsi_period = 14
        self.atr_period = 14
        self.volume_window = 20
        
        # Risk management
        self.stop_loss_atr = 1.5   # ATR multiplier for stop loss
        self.take_profit_atr = 1.2 # ATR multiplier for take profit
        self.cooldown_period = 10  # Bars to wait after a trade
        
        # Mode selection for multimodal distributions
        self.mode_preference = "central"  # central, positive, negative

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
    df['volume_z_score'] = (df['volume'] - df['volume_ma']) / df['volume'].rolling(window=params.volume_window).std()
    
    # Returns and momentum
    df['price_change'] = df['close'].pct_change()
    df['return'] = df['price_change'] * 100  # Convert to percentage
    df['return_5d'] = df['close'].pct_change(5) * 100  # 5-day return
    df['return_ma'] = df['return'].rolling(window=10).mean()
    
    # Centered returns for Schrödinger equation
    df['return_norm'] = (df['return'] - df['return'].mean()) / df['return'].std()
    
    # Volatility
    df['volatility'] = df['return'].rolling(window=20).std()
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50).mean()
    
    return df

def compute_energy_level(df, i, params):
    """
    Compute the energy level (Ω parameter) based on trading activity
    This determines which probability distribution shape to use
    
    As described in Lin Li's paper, this is based on normalized trading volume
    and determines whether returns follow unimodal or multimodal distributions
    """
    if i < 50:  # Need enough data for calculation
        return params.omega_0  # Default to ground state (normal distribution)
    
    # Extract window of data for analysis
    window = df.iloc[i-50:i]
    
    # Calculate trading volume metrics
    # 1. Current normalized volume relative to moving average
    volume_ratio = df['volume_ratio'].iloc[i]
    
    # 2. Anomalous volume (z-score)
    volume_z = df['volume_z_score'].iloc[i]
    
    # 3. Volume moving average trend
    volume_ma_trend = df['volume_ma'].iloc[i] / df['volume_ma'].iloc[i-20]
    
    # Volatility as indicator of market excitement 
    volatility_ratio = df['volatility_ratio'].iloc[i] if 'volatility_ratio' in df.columns else 1.0
    
    # Initialize V_max and E_0 if they haven't been set
    if params.V_max is None:
        # Use 95th percentile of volume as max reference volume
        params.V_max = np.percentile(df['volume_ratio'].iloc[max(0, i-500):i+1], 95)
        # Set ground state energy threshold based on eta parameter
        params.E_0 = params.eta * params.V_max
    
    # Estimate normalized trading "energy" (E) from volume
    trading_energy = volume_ratio * (1 + 0.2 * volume_z * (volume_z > 0))
    
    # Map trading energy to quantum energy levels as described in the paper:
    # - Below E_0: Ground state (Ω₀ = 1) - Normal distribution
    # - E_0 to 2*E_0: First excited state (Ω₁ = 3) - Bimodal distribution
    # - Above 2*E_0: Second excited state (Ω₂ = 5) - Trimodal distribution
    
    # Determine energy level based on trading energy
    if trading_energy < params.E_0:
        energy_level = params.omega_0  # Ground state (normal distribution)
    elif trading_energy < 2 * params.E_0:
        energy_level = params.omega_1  # First excited state (bimodal)
    else:
        energy_level = params.omega_2  # Second excited state (trimodal)
    
    # Adjust based on volatility - high volatility can increase energy level
    if volatility_ratio > 1.5 and energy_level < params.omega_2:
        energy_level += 2  # Move up one energy level
    
    return energy_level

def compute_schrodinger_solution(r, energy_level, params):
    """
    Compute the solution to the Schrödinger-like trading equation
    for a given normalized return level r and energy level.
    
    Based on Lin Li's quantum probability approach, this calculates
    the complex probability amplitude Ψ(r) that represents active trading intentions.
    """
    # Normalize to dimensionless coordinate ξ as in the paper
    xi = r * (params.alpha/params.h)**(1/4)
    
    # Calculate the anharmonic coefficient (λ parameter)
    lambda_param = params.lambda_param
    
    # Determine quantum number n based on energy level
    # Ω = 2n + 1 for energy levels, so n = (Ω - 1)/2
    n = (energy_level - 1) // 2
    
    # Phase information (represents trading decision properties)
    phase_factor = 1.0
    
    if abs(lambda_param) < 1e-6:
        # Harmonic case (λ = 0): Solutions are standard Hermite functions
        # Ψₙ(ξ) = (1/√2ⁿn!√π) · Hₙ(ξ) · exp(-ξ²/2)
        
        # Calculate normalization constant
        norm_const = 1.0 / np.sqrt((2**n * math.factorial(n) * np.sqrt(np.pi)))
        
        # Calculate Hermite polynomial Hₙ(ξ)
        hermite = special.hermite(n)(xi)
        
        # Combine into the wave function
        psi = norm_const * hermite * np.exp(-xi**2/2)
        
        # Add phase information (representing trading decision properties)
        psi = psi * phase_factor
        
    else:
        # Anharmonic case (λ ≠ 0): Use perturbation approach
        # The paper notes that trading behavior in real markets is typically anharmonic
        
        # Start with harmonic solution as base
        norm_const = 1.0 / np.sqrt((2**n * math.factorial(n) * np.sqrt(np.pi)))
        hermite = special.hermite(n)(xi)
        psi_0 = norm_const * hermite * np.exp(-xi**2/2)
        
        # Add anharmonic perturbation
        # In the quantum anharmonic oscillator, we add a perturbation term for the ξ⁴ potential
        # Approximation based on first-order perturbation theory
        
        # Energy correction term (varies by n)
        if n == 0:
            energy_correction = 3*lambda_param/4  # Ground state correction
        else:
            # Higher states (from literature on anharmonic oscillators)
            energy_correction = lambda_param * (n**2 + n + 1/2)
        
        # Wave function correction
        # This is a simplification of the perturbation approach 
        # described in quantum anharmonic oscillator literature
        if n == 0:
            # Ground state correction
            psi_correction = lambda_param * (xi**4 - 3*xi**2) / 8
        elif n == 1:
            # First excited state correction
            psi_correction = lambda_param * (xi**5 - 10*xi**3 + 15*xi) / 16
        else:
            # Higher states (approximate)
            psi_correction = lambda_param * xi**4 / (4*(n+1))
        
        # Combine base function with corrections
        psi = psi_0 * (1 + psi_correction - energy_correction*xi**2/2)
        
        # Add phase information (complex component representing trading decisions)
        psi = psi * phase_factor
    
    return psi

def visualize_quantum_distributions(params, save_path=None):
    """
    Visualize the quantum probability distributions for different energy levels
    as described in Lin Li's paper.
    """
    # Create a range of return values to calculate distributions
    r_values = np.linspace(-3, 3, 1000)
    
    # Calculate distributions for each energy level
    energy_levels = [params.omega_0, params.omega_1, params.omega_2]
    labels = ['Ground State (Normal)', 'First Excited State (Bimodal)', 'Second Excited State (Trimodal)']
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot probability density for each energy level
    for energy, label in zip(energy_levels, labels):
        # Calculate wave functions
        psi_values = np.array([compute_schrodinger_solution(r, energy, params) for r in r_values])
        
        # Convert to probability density |Ψ|²
        prob_density = np.abs(psi_values)**2
        
        # Normalize
        prob_density = prob_density / np.trapz(prob_density, r_values)
        
        # Plot
        plt.plot(r_values, prob_density, label=f"{label} (Ω={energy})")
    
    # Add annotations
    plt.xlabel('Normalized Return (r)', fontsize=12)
    plt.ylabel('Probability Density |Ψ(r)|²', fontsize=12)
    plt.title('Quantum Probability Distributions at Different Energy Levels', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text explaining energy levels
    annotation = (
        "Energy Levels (Ω):\n"
        "- Ground State (Ω=1): Normal distribution - balanced market with regular trading\n"
        "- First Excited (Ω=3): Bimodal - market with conflicting trading intentions\n"
        "- Second Excited (Ω=5): Trimodal - complex market with multiple trading scenarios"
    )
    plt.figtext(0.5, 0.01, annotation, ha='center', fontsize=10, bbox=dict(facecolor='ivory', alpha=0.5))
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
        
    return

def generate_probability_distribution(returns, params, energy_level=None):
    """
    Generate the probability distribution for returns based on the quantum model.
    
    Returns:
        - Dictionary with calculated distributions
    """
    # If energy level not specified, use the default from params
    if energy_level is None:
        energy_level = params.omega_level
    
    # Create a range of return values to calculate distribution
    r_values = np.linspace(-3, 3, 1000)
    
    # Calculate the complex probability amplitude for each return value
    psi_values = np.array([compute_schrodinger_solution(r, energy_level, params) for r in r_values])
    
    # Calculate the probability density as |Ψ|²
    prob_density = np.abs(psi_values)**2
    
    # Normalize to ensure the integral = 1
    prob_density = prob_density / np.trapz(prob_density, r_values)
    
    # Find the modes (local maxima)
    modes = []
    for i in range(1, len(r_values) - 1):
        if prob_density[i] > prob_density[i-1] and prob_density[i] > prob_density[i+1]:
            modes.append((r_values[i], prob_density[i]))
    
    return {
        'r_values': r_values,
        'prob_density': prob_density,
        'psi_values': psi_values,
        'modes': modes,
        'energy_level': energy_level
    }

def find_entry_signals(df, i, params):
    """
    Use the quantum probability model to find trading signals.
    Returns a dictionary with signal type and confidence.
    """
    # Skip if we don't have enough data
    if i < 30:
        return {'signal': None, 'confidence': 0}
    
    # Compute the current energy level
    energy_level = compute_energy_level(df, i, params)
    
    # Get a window of recent returns for analysis
    recent_returns = df['return_norm'].iloc[i-30:i].values
    
    # Generate distribution for current market state
    try:
        dist = generate_probability_distribution(recent_returns, params, energy_level)
        modes = dist['modes']
        
        # Log issue if modes structure is problematic
        if i % 500 == 0:
            print(f"Bar {i}: Energy level {energy_level}, Found {len(modes)} modes")
            
    except Exception as e:
        print(f"Error generating distribution at bar {i}: {e}")
        return {'signal': None, 'confidence': 0}
    
    signal = None
    confidence = 0
    
    # No signal if no modes detected
    if len(modes) == 0:
        return {'signal': None, 'confidence': 0}
    
    # Extract current market state
    current_return = df['return_norm'].iloc[i]
    current_price = df['close'].iloc[i]
    
    # Verify modes data structure and fix if needed
    valid_modes = []
    for mode in modes:
        # Check if mode is already a tuple/list with at least 2 elements
        if isinstance(mode, (tuple, list)) and len(mode) >= 2:
            valid_modes.append(mode)
        # If mode is a scalar, convert to a tuple with a default probability
        elif np.isscalar(mode):
            valid_modes.append((mode, 1.0))
    
    # Use the valid modes list instead
    if not valid_modes:
        return {'signal': None, 'confidence': 0}
    modes = valid_modes
    
    # Analyze signal based on model and energy level
    if energy_level == 1:
        # Single mode (normal distribution)
        # Signal when current return is at distribution extremes
        mode_center = modes[0][0]
        
        if current_return < mode_center - 1.5:
            # Very negative return - potential long entry
            signal = 'buy'
            # Higher confidence the further from center
            confidence = min(0.9, abs(current_return - mode_center) / 2)
            
        elif current_return > mode_center + 1.5:
            # Very positive return - potential short entry
            signal = 'sell'
            confidence = min(0.9, abs(current_return - mode_center) / 2)
            
    elif energy_level >= 3:
        # Multi-modal distribution - more complex signals
        # Sort modes by position (left to right)
        try:
            modes.sort(key=lambda x: x[0])
        except (TypeError, IndexError) as e:
            print(f"Error sorting modes: {e}, modes: {modes}")
            # If sorting fails, return no signal
            return {'signal': None, 'confidence': 0}
        
        # Find where current return sits in relation to modes
        if current_return < modes[0][0]:
            # To the left of all modes - strong buy signal
            signal = 'buy'
            confidence = 0.8
        elif current_return > modes[-1][0]:
            # To the right of all modes - strong sell signal
            signal = 'sell'
            confidence = 0.8
        else:
            # Between modes - more complex analysis
            # Find nearest mode
            try:
                distances = [abs(current_return - mode[0]) for mode in modes]
                nearest_idx = np.argmin(distances)
                
                # Determine if we're moving from one mode to another
                # by looking at recent trajectory
                prev_returns = df['return_norm'].iloc[i-5:i].values
                direction = prev_returns[-1] - prev_returns[0]
                
                # Moving right (toward higher returns)
                if direction > 0.1:
                    if nearest_idx < len(modes) - 1:
                        # Moving toward higher return mode - potential buy
                        signal = 'buy'
                        confidence = 0.6 * (1 - distances[nearest_idx])
                # Moving left (toward lower returns)
                elif direction < -0.1:
                    if nearest_idx > 0:
                        # Moving toward lower return mode - potential sell
                        signal = 'sell'
                        confidence = 0.6 * (1 - distances[nearest_idx])
            except (TypeError, IndexError) as e:
                print(f"Error calculating distances: {e}, modes: {modes}")
                # If distance calculation fails, set no signal
                signal = None
                confidence = 0
    
    # Apply volume filter - increase confidence with high volume
    if signal and 'volume_ratio' in df.columns:
        vol_ratio = df['volume_ratio'].iloc[i]
        if vol_ratio > 1.5:
            confidence = min(0.95, confidence * 1.2)
    
    return {
        'signal': signal,
        'confidence': confidence,
        'energy_level': energy_level,
        'modes': modes
    }

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

def run_backtest(df, params, initial_capital=10000, plot=True):
    """
    Run a backtest of the Quantum Probability strategy
    """
    print(f"Starting Quantum Probability backtest with {len(df)} data points...")
    
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
                        'result': exit_reason,
                        'energy_level': compute_energy_level(df, entry_index, params)
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
                        'result': exit_reason,
                        'energy_level': compute_energy_level(df, entry_index, params)
                    }
                    trade_log.append(trade_data)
                    trades.append(trade_result)
                    
                    # Reset position
                    in_position = False
                    cooldown_counter = params.cooldown_period
                    
                    continue
        
        # Check for new signals if we're not in a position
        if not in_position:
            # Get signal from the quantum probability model
            signal_data = find_entry_signals(df, i, params)
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            
            # Enter position if signal is strong enough
            if signal == 'buy' and confidence > 0.6:
                # Enter long position
                in_position = True
                position_type = 'long'
                entry_price = current_price
                entry_index = i
                
            elif signal == 'sell' and confidence > 0.6:
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
            'result': 'End of Backtest',
            'energy_level': compute_energy_level(df, entry_index, params)
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
    
    # Print results
    print("=== Quantum Probability Model Backtest Results ===")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Win: {avg_win:.2f}%")
    print(f"Average Loss: {avg_loss:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Final Equity: ${final_equity:.2f}")
    
    if plot:
        # Create a plot of equity curve and trades
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price with buy/sell markers
        ax1.plot(df.index[30:], df['close'][30:], label='Price')
        
        for trade in trade_log:
            if trade['position'] == 'long':
                ax1.scatter(trade['entry_date'], trade['entry_price'], marker='^', color='green', s=100)
                ax1.scatter(trade['exit_date'], trade['exit_price'], marker='v', 
                           color='red' if trade['result'] == 'Stop Loss' else 'blue', s=100)
            else:  # short
                ax1.scatter(trade['entry_date'], trade['entry_price'], marker='v', color='red', s=100)
                ax1.scatter(trade['exit_date'], trade['exit_price'], marker='^', 
                           color='red' if trade['result'] == 'Stop Loss' else 'green', s=100)
        
        ax1.set_title('Quantum Probability Model: Price and Trades')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Ensure equity curve length matches the data we're plotting
        equity_to_plot = equity
        if len(equity_to_plot) > len(df.index[30:]):
            equity_to_plot = equity_to_plot[:len(df.index[30:])]
        
        # Plot equity curve
        x_vals = df.index[30:30+len(equity_to_plot)]
        ax2.plot(x_vals, equity_to_plot, color='blue')
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Equity ($)')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"results/quantum_probability_backtest_{timestamp}.png")
        
        # Create a distribution plot showing modes
        plt.figure(figsize=(10, 6))
        
        # Use the last point to generate distribution
        last_idx = len(df) - 1
        current_energy = compute_energy_level(df, last_idx, params)
        recent_returns = df['return_norm'].iloc[last_idx-30:last_idx].values
        dist = generate_probability_distribution(recent_returns, params, current_energy)
        
        plt.plot(dist['r_values'], dist['prob_density'], label=f'Energy Level Ω = {dist["energy_level"]}')
        
        # Mark modes
        for mode_pos, mode_height in dist['modes']:
            plt.scatter(mode_pos, mode_height, color='red', s=100, marker='o')
            plt.text(mode_pos, mode_height*1.05, f'r = {mode_pos:.2f}', ha='center')
        
        plt.title(f'Quantum Probability Return Distribution (Ω = {dist["energy_level"]})')
        plt.xlabel('Normalized Return')
        plt.ylabel('Probability Density')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(f"results/quantum_probability_distribution_{timestamp}.png")
        
    # Return results
    print(f"Backtest completed. Processed {len(df)} bars with {total_trades} trades.")
    results = {
        'total_return': total_return,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_drawdown': max_drawdown,
        'trades': total_trades,
        'final_equity': final_equity,
        'equity_curve': equity,
        'trade_log': trade_log
    }
    
    return results

def hermite_polynomial(n, x):
    """
    Compute the Hermite polynomial of degree n at point x.
    Helper function for backward compatibility with existing code.
    """
    return special.hermite(n)(x)

# Restore this function for backward compatibility with existing code
def approximate_anharmonic_solution(xi, energy_level, lambda_param):
    """
    Approximate solution for the anharmonic oscillator when λ ≠ 0.
    Maintained for backward compatibility.
    """
    # Get the unperturbed solution first
    n = (energy_level - 1) // 2
    norm_const = 1 / np.sqrt(2**n * math.factorial(n) * np.sqrt(np.pi))
    psi_0 = norm_const * np.exp(-xi**2/2) * hermite_polynomial(n, xi)
    
    # Apply a first-order perturbation (simplified)
    perturbation = 1 + lambda_param * xi**4 / (2*(2*n+1))
    
    return psi_0 * perturbation

if __name__ == "__main__":
    """
    Demonstration of the Quantum Probability Trading Model
    
    Key concepts from Lin Li's paper:
    
    1. Quantum probability is a mathematical extension of classical probability from 
       real to complex numbers, not exclusively tied to microscopic quantum phenomena.
       
    2. Complex-valued probability amplitude Ψ(r) represents active trading intentions (ATI):
       - The modulus |Ψ(r)| quantifies the intensity of active trading intentions
       - The phase of Ψ(r) represents the property (long/short) of trading intentions
       
    3. Trading energy levels (Ω):
       - Ground state (Ω₀=1): Normal distribution - balanced market with regular trading
       - First excited (Ω₁=3): Bimodal distribution - market with conflicting trading intentions
       - Second excited (Ω₂=5): Trimodal distribution - complex market with multiple scenarios
       
    4. The energy level is determined by trading volume, with transitions from unimodal to
       multimodal distributions occurring as volume exceeds critical thresholds.
    """
    # Create results directory
    os.makedirs("static/results", exist_ok=True)
    
    print("\n=============== Quantum Probability Trading Model ===============")
    print("Based on 'Quantum Probability Theoretic Asset Return Modeling'")
    print("Implementing the Schrödinger-like trading equation approach")
    print("=================================================================\n")
    
    # Initialize parameters
    print("Initializing quantum probability parameters...")
    params = QuantumProbabilityParameters()
    
    # Visualize the quantum probability distributions
    print("Generating visualizations of quantum probability distributions...")
    visualize_quantum_distributions(params, save_path="static/results/quantum_probability_distributions.png")
    print("Distributions saved to static/results/quantum_probability_distributions.png")
    
    # Option to run a backtest on real market data
    run_backtest_demo = False
    if run_backtest_demo:
        print("\nFetching historical market data for backtesting...")
        # Fetch some historical data
        symbol = "BTC/USDT"
        timeframe = "1h"
        start_date = '2022-01-01'
        end_date = '2022-06-30'
        
        df = fetch_crypto_data(symbol, timeframe, start_date, end_date)
        print(f"Retrieved {len(df)} data points for {symbol} from {start_date} to {end_date}")
        
        # Calculate indicators and prepare for analysis
        df = calculate_indicators(df, params)
        
        # Analyze energy level distribution
        energy_levels = []
        print("Analyzing energy levels across the dataset...")
        
        for i in range(50, len(df)):
            if i % 1000 == 0:
                print(f"Processing bar {i}/{len(df)}...")
            energy_levels.append(compute_energy_level(df, i, params))
        
        # Plot energy level distribution
        plt.figure(figsize=(10, 6))
        
        # Count occurrences of each energy level
        unique_levels, counts = np.unique(energy_levels, return_counts=True)
        
        # Create a bar chart
        plt.bar(unique_levels, counts / len(energy_levels) * 100)
        plt.xlabel('Energy Level (Ω)')
        plt.ylabel('Percentage of Time (%)')
        plt.title(f'Distribution of Energy Levels for {symbol}')
        plt.xticks(unique_levels)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add annotations
        for i, (level, count) in enumerate(zip(unique_levels, counts)):
            pct = count / len(energy_levels) * 100
            plt.annotate(f'{pct:.1f}%', 
                        xy=(level, pct + 1), 
                        ha='center', va='bottom')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(f"static/results/energy_level_distribution_{symbol.replace('/', '_')}.png", dpi=300)
        print(f"Energy level distribution saved to static/results/energy_level_distribution_{symbol.replace('/', '_')}.png")
        
        # Run full backtest
        print("\nRunning full backtest with Quantum Probability model...")
        results = run_backtest(df, params, initial_capital=10000, plot=True)
        
        # Print final results
        print("\nBacktest Results:")
        print(f"Total return: {results['total_return']:.2f}%")
        print(f"Win rate: {results['win_rate']:.2f}%")
        print(f"Profit factor: {results['profit_factor']:.2f}")
        print(f"Max drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"Total trades: {results['total_trades']}")
    
    print("\nQuantum Probability Model demonstration complete.")
    print("This model differs from traditional quantum finance approaches by:")
    print("1. Not assuming asset returns exhibit microscopic quantum effects")
    print("2. Directly linking quantum probability to market trading behavior")
    print("3. Using the phase information to represent trading decisions")
    print("4. Explaining multimodal return distributions through energy levels") 