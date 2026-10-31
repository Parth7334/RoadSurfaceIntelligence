import sys
import platform

import numpy as np
import pandas as pd
import matplotlib
from PIL import Image

print("=" * 45)
print("ROAD SURFACE INTELLIGENCE")
print("Stage 1: Environment Check")
print("=" * 45)

print("Python version:", sys.version.split()[0])
print("Operating system:", platform.system(), platform.machine())
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Pillow version:", Image.__version__)

print("\nEnvironment setup successful!")
