"""
Test script for the Quantum RSI implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add numpy bool8 patch
np.bool8 = np.bool_ if not hasattr(np, 'bool8') else np.bool8
print("NumPy patched to add np.bool8 alias")

# Create sample data
def generate_sample_data(days=100):
    """Generate sample price data with volume for testing"""
    # Create date range
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    
    # Generate synthetic price data (random walk with bias)
    seed = 42
    np.random.seed(seed)
    
    # Start price
    price = 100
    prices = [price]
    
    # Generate prices
    for i in range(1, days):
        # Random walk with upward bias
        change = np.random.normal(0.05, 1.0)  # Mean 0.05, std 1.0
        price *= (1 + change / 100)
        prices.append(price)
    
    # Generate OHLC data
    opens = prices.copy()
    closes = []
    highs = []
    lows = []
    volumes = []
    
    for i in range(days):
        # Current price
        current = opens[i]
        
        # Random intraday move
        move = current * np.random.uniform(-0.02, 0.02)
        high_move = current * np.random.uniform(0, 0.02)
        low_move = current * np.random.uniform(-0.02, 0)
        
        # OHLC
        close = current + move
        high = max(current, close) + high_move
        low = min(current, close) + low_move
        
        # Append
        closes.append(close)
        highs.append(high)
        lows.append(low)
        
        # Volume (higher in trending periods)
        vol_factor = 1.0 + 0.5 * abs(move) / current
        volume = np.random.uniform(100000, 500000) * vol_factor
        volumes.append(volume)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    df.index = dates
    
    return df

# Import quantum RSI functions
from quantum_rsi_optimized import QuantumRSIParameters, calculate_indicators, compute_barrier_height, compute_tunneling_probability

# Generate sample data
df = generate_sample_data(days=200)

# Initialize parameters
params = QuantumRSIParameters()

# Calculate indicators
try:
    df = calculate_indicators(df, params)
    print("Successfully calculated indicators")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show a few RSI values
    print("\nSample RSI Values:")
    print(df['rsi'].tail().to_string())
    
    # Plot key indicators
    plt.figure(figsize=(14, 10))
    
    # Price chart
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['close'], label='Price')
    plt.title('Price Chart')
    plt.legend()
    
    # RSI
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['rsi'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.title('RSI Indicator')
    plt.legend()
    
    # Tunneling Probability
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['tunneling_probability'], label='Tunneling Probability')
    plt.title('Quantum Tunneling Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('quantum_rsi_test.png')
    plt.close()
    
    print("\nSuccessfully generated plot: quantum_rsi_test.png")
    
except Exception as e:
    print(f"Error calculating indicators: {e}") 