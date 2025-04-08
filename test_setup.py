"""
Test script to verify our Quantum RSI setup is working correctly
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

def run_tests():
    print("\n=== Running Quantum RSI Setup Tests ===\n")
    
    # Test 1: Verify NumPy bool8 patch
    print("Test 1: NumPy bool8 patch")
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
        print("  ✓ Applied NumPy bool8 patch")
    else:
        print("  ✓ NumPy bool8 already exists")
    
    # Verify it works
    try:
        test_array = np.array([True, False], dtype=np.bool8)
        print("  ✓ Successfully created array with bool8 dtype")
    except Exception as e:
        print(f"  ✗ Failed to create array with bool8 dtype: {str(e)}")
        return False
    
    # Test 2: Verify TA package imports
    print("\nTest 2: TA package imports")
    try:
        from ta.momentum import RSIIndicator
        from ta.volatility import AverageTrueRange
        print("  ✓ Successfully imported RSIIndicator and AverageTrueRange from ta package")
    except Exception as e:
        print(f"  ✗ Failed to import from ta package: {str(e)}")
        return False
    
    # Test 3: Import Quantum RSI modules
    print("\nTest 3: Quantum RSI module imports")
    try:
        from quantum_rsi_optimized import (
            QuantumRSIParameters, 
            calculate_indicators,
            maxwell_demon_filter
        )
        print("  ✓ Successfully imported Quantum RSI modules")
    except Exception as e:
        print(f"  ✗ Failed to import Quantum RSI modules: {str(e)}")
        return False
    
    # Test 4: Create sample data and calculate indicators
    print("\nTest 4: Calculate indicators with sample data")
    try:
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=100)
        data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(102, 5, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
        
        # Create parameters
        params = QuantumRSIParameters()
        
        # Calculate indicators
        result_df = calculate_indicators(data, params)
        
        if 'rsi' in result_df.columns and 'tunneling_probability' in result_df.columns:
            print("  ✓ Successfully calculated RSI and tunneling probability")
        else:
            print(f"  ✗ Failed to calculate expected indicators. Available columns: {result_df.columns}")
            return False
        
        # Check sample values
        print(f"  Sample RSI value: {result_df['rsi'].iloc[-1]:.2f}")
        print(f"  Sample tunneling probability: {result_df['tunneling_probability'].iloc[-1]:.4f}")
        
    except Exception as e:
        print(f"  ✗ Failed to calculate indicators: {str(e)}")
        traceback.print_exc()
        return False
    
    print("\n=== All tests passed successfully! ===")
    return True

if __name__ == "__main__":
    import traceback
    
    try:
        success = run_tests()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"Tests failed with error: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 