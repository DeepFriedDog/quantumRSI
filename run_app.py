"""
Helper script to run the app with NumPy bool8 patch applied
"""

import os
import sys
import numpy as np

# Add NumPy bool8 alias if it doesn't exist
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
    print("NumPy patched to add np.bool8 alias")
else:
    print("NumPy bool8 already exists")

# Set matplotlib backend to non-interactive to avoid warnings
import matplotlib
matplotlib.use('Agg')

# Verify the patch was applied
try:
    bool_array = np.array([True, False], dtype=np.bool8)
    print("NumPy bool8 patch verified successfully")
except AttributeError:
    print("ERROR: NumPy bool8 patch failed!")
    sys.exit(1)

# Import the app after patching NumPy
# This ensures all modules using NumPy will have the patch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting Quantum RSI Backtesting App...")
print("Access the web interface at http://localhost:5000")

try:
    # Run the app
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)
except Exception as e:
    print(f"ERROR: Failed to start the app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 