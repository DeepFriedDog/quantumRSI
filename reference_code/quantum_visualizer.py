#!/usr/bin/env python
# Quantum RSI Visualizer
# Displays quantum mechanics components of the trading model

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.ticker as ticker
import argparse
import ccxt
from datetime import datetime, timedelta
import os

# Import our utility functions from the live trader
from quantum_rsi_live_trader import (
    calculate_rsi,
    normalize_rsi,
    calculate_potential,
    tunneling_probability
)

# Constants for quantum calculations
HBAR = 1.0  # Reduced Planck constant
MASS = 1.0  # Particle mass
XMIN, XMAX = -2.0, 2.0  # Space bounds
NX = 1000  # Number of spatial grid points
DX = (XMAX - XMIN) / NX  # Spatial step
X = np.linspace(XMIN, XMAX, NX)  # Spatial grid

# Params for time evolution
TMIN, TMAX = 0, 10.0  # Time bounds
NT = 100  # Number of time steps
DT = (TMAX - TMIN) / NT  # Time step
T = np.linspace(TMIN, TMAX, NT)  # Time grid

def gaussian_wavepacket(x, x0, sigma, k0=0):
    """Initialize a Gaussian wavepacket centered at x0 with width sigma and momentum k0."""
    return np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x) / (sigma * np.sqrt(2 * np.pi))

def normalize_wavefunction(psi):
    """Normalize the wavefunction to ensure total probability is 1."""
    norm = np.sqrt(np.sum(np.abs(psi)**2) * DX)
    return psi / norm

def time_evolution(psi_0, v0=1.0, a=0.5):
    """Solve the time-dependent Schrödinger equation for the double-well potential."""
    # Hamiltonian components
    def V(x):
        return v0 * (x**2 - a**2)**2
    
    # Kinetic energy operator (2nd derivative) using finite difference
    def laplacian(psi):
        return (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / DX**2
    
    # Time-dependent Schrödinger equation
    def schrodinger_eq(t, psi_flat):
        psi = psi_flat.reshape(2, -1)  # Reshape to complex form (real, imag)
        psi_complex = psi[0] + 1j * psi[1]
        
        # Calculate the new psi
        laplacian_psi = laplacian(psi_complex)
        V_psi = V(X) * psi_complex
        
        # i*hbar*dpsi/dt = -hbar^2/(2m) * laplacian(psi) + V*psi
        dpsi = -1j * (-HBAR**2 / (2 * MASS) * laplacian_psi + V_psi) / HBAR
        
        # Split complex values for the ODE solver
        return np.vstack([np.real(dpsi), np.imag(dpsi)]).flatten()
    
    # Initial condition with real and imaginary parts separated
    psi_0_flat = np.vstack([np.real(psi_0), np.imag(psi_0)]).flatten()
    
    # Solve the ODE
    sol = solve_ivp(
        schrodinger_eq, 
        [TMIN, TMAX], 
        psi_0_flat, 
        method='RK45', 
        t_eval=T,
        rtol=1e-6,
        atol=1e-8
    )
    
    # Process the solution back to complex form
    psi_t = np.zeros((NT, NX), dtype=np.complex128)
    for i, t in enumerate(range(NT)):
        if i < len(sol.t):
            psi_real_imag = sol.y[:, i].reshape(2, -1)
            psi_t[i] = psi_real_imag[0] + 1j * psi_real_imag[1]
            # Re-normalize at each step to handle numerical errors
            psi_t[i] = normalize_wavefunction(psi_t[i])
    
    return psi_t

def fetch_crypto_data(symbol="BTC/USDT", timeframe='1h', limit=500):
    """Fetch cryptocurrency OHLCV data."""
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        print(f"Fetched {len(df)} bars of data for {symbol}")
        return df
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def visualize_quantum_rsi_static(market_data=None, v0=1.0, a=0.5, rsi_period=14, save_path=None):
    """Create a static visualization of the quantum RSI model."""
    if market_data is None:
        # If no data provided, fetch some
        market_data = fetch_crypto_data()
    
    if market_data.empty:
        print("No market data available. Using simulated data.")
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        close = np.sin(np.linspace(0, 8*np.pi, 100)) * 10 + np.random.normal(0, 1, 100) + 100
        market_data = pd.DataFrame({'close': close}, index=dates)
    
    # Calculate RSI
    market_data['rsi'] = calculate_rsi(market_data['close'], period=rsi_period)
    market_data = market_data.dropna()
    
    # Calculate normalized RSI (-1 to 1) and potential
    market_data['norm_rsi'] = market_data['rsi'].apply(normalize_rsi)
    
    # Calculate potential for all x values
    potential = np.array([calculate_potential(xi, v0, a) for xi in X])
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 12))
    grid = plt.GridSpec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Price and RSI History
    ax1 = fig.add_subplot(grid[0, :])
    ax1.plot(market_data.index, market_data['close'], 'b-', label='Price')
    ax1.set_ylabel('Price', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Price History and RSI')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for RSI
    ax1_rsi = ax1.twinx()
    ax1_rsi.plot(market_data.index, market_data['rsi'], 'r-', label='RSI')
    ax1_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax1_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax1_rsi.set_ylabel('RSI', color='r')
    ax1_rsi.tick_params(axis='y', labelcolor='r')
    ax1_rsi.set_ylim(0, 100)
    
    # Plot 2: Double-Well Potential
    ax2 = fig.add_subplot(grid[1, :2])
    ax2.plot(X, potential, 'k-', linewidth=2)
    ax2.set_xlabel('Normalized RSI Position')
    ax2.set_ylabel('Potential Energy')
    ax2.set_title(f'Double-Well Potential (v0={v0}, a={a})')
    ax2.grid(True, alpha=0.3)
    
    # Mark important areas in the potential
    ax2.axvline(x=-a, color='g', linestyle='--', alpha=0.5, label=f'RSI={30}')
    ax2.axvline(x=a, color='r', linestyle='--', alpha=0.5, label=f'RSI={70}')
    ax2.axhline(y=v0*a**4, color='b', linestyle=':', alpha=0.5, label='Barrier Height')
    
    # Add the last known RSI position
    last_rsi_pos = market_data['norm_rsi'].iloc[-1]
    last_potential = calculate_potential(last_rsi_pos, v0, a)
    ax2.plot(last_rsi_pos, last_potential, 'ro', markersize=8, label='Current RSI')
    
    # Plot 3: Wavefunction and Probability Density
    # For simplicity, we'll use a Gaussian wavepacket centered at the last RSI position
    psi_0 = gaussian_wavepacket(X, last_rsi_pos, 0.2)
    psi_0 = normalize_wavefunction(psi_0)
    
    # Evolve the wavefunction
    psi_t = time_evolution(psi_0, v0, a)
    
    # Plot current wavefunction (real and imaginary parts)
    ax3 = fig.add_subplot(grid[1, 2])
    ax3.plot(X, np.real(psi_0), 'b-', label='Real')
    ax3.plot(X, np.imag(psi_0), 'g-', label='Imaginary')
    ax3.plot(X, np.abs(psi_0)**2, 'r-', label='Probability')
    ax3.set_xlabel('Normalized RSI Position')
    ax3.set_ylabel('Wavefunction')
    ax3.set_title('Initial Quantum State')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Tunneling probability vs RSI
    ax4 = fig.add_subplot(grid[2, 0])
    rsi_values = np.linspace(0, 100, 100)
    norm_rsi_values = [normalize_rsi(r) for r in rsi_values]
    
    # Calculate momentum proxy from recent RSI changes
    recent_momentum = market_data['rsi'].diff(3).iloc[-1]
    
    tunnel_probs = [tunneling_probability(r, recent_momentum, v0, a, MASS, HBAR) for r in rsi_values]
    ax4.plot(rsi_values, tunnel_probs, 'k-', linewidth=2)
    ax4.set_xlabel('RSI Value')
    ax4.set_ylabel('Tunneling Probability')
    ax4.set_title(f'Tunneling Probability vs RSI (momentum={recent_momentum:.2f})')
    ax4.grid(True, alpha=0.3)
    
    # Mark current RSI and its tunneling probability
    current_rsi = market_data['rsi'].iloc[-1]
    current_tunnel_prob = tunneling_probability(current_rsi, recent_momentum, v0, a, MASS, HBAR)
    ax4.plot(current_rsi, current_tunnel_prob, 'ro', markersize=8, 
             label=f'Current RSI: {current_rsi:.2f}, P(tunnel)={current_tunnel_prob:.4f}')
    ax4.legend()
    
    # Plot 5: Probability evolution over time
    ax5 = fig.add_subplot(grid[2, 1:])
    
    # Compute probability densities from wavefunctions
    prob_densities = np.abs(psi_t)**2
    
    # Create a 2D plot (time vs. position)
    time_mesh, pos_mesh = np.meshgrid(T, X)
    c = ax5.pcolormesh(pos_mesh, time_mesh, prob_densities.T, shading='auto', cmap='viridis')
    ax5.set_xlabel('Normalized RSI Position')
    ax5.set_ylabel('Time')
    ax5.set_title('Probability Density Evolution')
    fig.colorbar(c, ax=ax5, label='Probability Density')
    
    # Mark the wells at x = ±a
    ax5.axvline(x=-a, color='w', linestyle='--', alpha=0.5)
    ax5.axvline(x=a, color='w', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def animate_quantum_evolution(market_data=None, v0=1.0, a=0.5, save_path=None):
    """Create an animation of quantum state evolution based on market data."""
    if market_data is None:
        # If no data provided, fetch some
        market_data = fetch_crypto_data()
    
    if market_data.empty:
        print("No market data available. Using simulated data.")
        # Create synthetic data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        close = np.sin(np.linspace(0, 8*np.pi, 100)) * 10 + np.random.normal(0, 1, 100) + 100
        market_data = pd.DataFrame({'close': close}, index=dates)
    
    # Calculate RSI
    market_data['rsi'] = calculate_rsi(market_data['close'])
    market_data = market_data.dropna()
    
    # Calculate normalized RSI (-1 to 1)
    market_data['norm_rsi'] = market_data['rsi'].apply(normalize_rsi)
    
    # Calculate potential for all x values
    potential = np.array([calculate_potential(xi, v0, a) for xi in X])
    
    # Set up the animation figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Quantum RSI Model Evolution", fontsize=16)
    
    # Plot the potential in the top subplot
    ax1.plot(X, potential, 'k-', linewidth=2)
    ax1.set_ylim(0, v0*a**4*2)  # Adjust y limit based on barrier height
    ax1.set_xlabel('Normalized RSI Position')
    ax1.set_ylabel('Potential Energy')
    ax1.set_title(f'Double-Well Potential (v0={v0}, a={a})')
    ax1.grid(True, alpha=0.3)
    
    # Mark important areas in the potential
    ax1.axvline(x=-a, color='g', linestyle='--', alpha=0.5, label=f'RSI={30}')
    ax1.axvline(x=a, color='r', linestyle='--', alpha=0.5, label=f'RSI={70}')
    ax1.axhline(y=v0*a**4, color='b', linestyle=':', alpha=0.5, label='Barrier Height')
    ax1.legend()
    
    # Initial wavefunction in the bottom subplot
    last_rsi_pos = market_data['norm_rsi'].iloc[0]
    psi_0 = gaussian_wavepacket(X, last_rsi_pos, 0.2)
    psi_0 = normalize_wavefunction(psi_0)
    
    # Plot initial probability
    line_psi_real, = ax2.plot(X, np.real(psi_0), 'b-', label='Real')
    line_psi_imag, = ax2.plot(X, np.imag(psi_0), 'g-', label='Imaginary')
    line_prob, = ax2.plot(X, np.abs(psi_0)**2, 'r-', label='Probability')
    
    # Plot current RSI position
    rsi_point, = ax1.plot([last_rsi_pos], [calculate_potential(last_rsi_pos, v0, a)], 'ro', markersize=8)
    
    # Configure the probability plot
    ax2.set_xlim(XMIN, XMAX)
    ax2.set_ylim(-0.5, 0.5)  # Adjust as needed
    ax2.set_xlabel('Normalized RSI Position')
    ax2.set_ylabel('Wavefunction / Probability')
    ax2.set_title('Quantum State Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add text for current time and RSI
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    rsi_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes)
    tunnel_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
    
    # Evolve wavefunction for each frame
    all_psi = []
    for i in range(min(len(market_data), 100)):  # Limit to 100 frames for performance
        rsi_pos = market_data['norm_rsi'].iloc[i]
        
        # Create a new wavefunction centered at the current RSI position
        if i == 0 or i % 10 == 0:  # Re-center occasionally
            psi = gaussian_wavepacket(X, rsi_pos, 0.2)
            psi = normalize_wavefunction(psi)
        else:
            # Evolve from previous wavefunction
            psi_evolved = time_evolution(all_psi[-1][-1], v0, a)
            psi = psi_evolved[-1]  # Take the last time step
        
        all_psi.append(psi_evolved if i % 10 == 0 else [psi])
    
    # Animation update function
    def update(frame):
        index = min(frame, len(market_data) - 1)
        date = market_data.index[index]
        rsi_value = market_data['rsi'].iloc[index]
        norm_rsi = market_data['norm_rsi'].iloc[index]
        
        # Calculate tunneling probability
        momentum = market_data['rsi'].diff(3).iloc[index] if index >= 3 else 0
        tunnel_prob = tunneling_probability(rsi_value, momentum, v0, a, MASS, HBAR)
        
        # Update the wavefunction and probability plots
        psi = all_psi[min(index, len(all_psi)-1)][-1]
        line_psi_real.set_ydata(np.real(psi))
        line_psi_imag.set_ydata(np.imag(psi))
        line_prob.set_ydata(np.abs(psi)**2)
        
        # Update RSI point in potential
        rsi_point.set_xdata([norm_rsi])
        rsi_point.set_ydata([calculate_potential(norm_rsi, v0, a)])
        
        # Update text elements
        time_text.set_text(f'Date: {date.strftime("%Y-%m-%d %H:%M")}')
        rsi_text.set_text(f'RSI: {rsi_value:.2f} (Normalized: {norm_rsi:.2f})')
        tunnel_text.set_text(f'Tunneling Probability: {tunnel_prob:.4f}')
        
        return line_psi_real, line_psi_imag, line_prob, rsi_point, time_text, rsi_text, tunnel_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=min(len(market_data), 100), interval=200, blit=True)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        ani.save(save_path, writer='pillow', fps=10, dpi=100)
        print(f"Animation saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Quantum RSI Model Visualizer')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair to analyze')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe for data')
    parser.add_argument('--limit', type=int, default=500, help='Number of candles to fetch')
    parser.add_argument('--v0', type=float, default=1.5, help='Potential strength parameter')
    parser.add_argument('--a', type=float, default=0.5, help='Well position parameter')
    parser.add_argument('--animate', action='store_true', help='Create animation of quantum evolution')
    parser.add_argument('--save', type=str, help='Save the visualization to file')
    
    args = parser.parse_args()
    
    # Fetch data
    market_data = fetch_crypto_data(args.symbol, args.timeframe, args.limit)
    
    if args.animate:
        # Create animation
        animate_quantum_evolution(
            market_data,
            v0=args.v0,
            a=args.a, 
            save_path=args.save if args.save else None
        )
    else:
        # Create static visualization
        save_path = args.save if args.save else None
        if save_path and not save_path.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            save_path += '.png'
        
        visualize_quantum_rsi_static(
            market_data,
            v0=args.v0,
            a=args.a,
            save_path=save_path
        )

if __name__ == "__main__":
    main() 