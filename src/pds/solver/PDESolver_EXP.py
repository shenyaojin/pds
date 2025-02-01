# Explicit PDE solver made by Shenyao

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

def solver_explicit(A, b):
    return 0