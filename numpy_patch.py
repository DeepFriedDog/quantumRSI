"""
NumPy compatibility patch for Bokeh.

This module adds the np.bool8 alias if it doesn't exist, to ensure compatibility
with newer NumPy versions while using Bokeh.
"""

import numpy as np

# Add bool8 alias if it doesn't exist (compatibility with newer NumPy versions)
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
    print("NumPy patched to add np.bool8 alias")

# Add this file to the site-packages directory to auto-patch on import 