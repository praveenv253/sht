#!/usr/bin/env python3

from .sht import sht, isht
from . import utils
from . import grids

# Try importing the plot module. This will fail if mayavi isn't installed.
# But requiring mayavi is too much overhead for the other sht modules. The plot
# module will work automatically if mayavi is installed.
try:
    from . import plot
except ImportError:
    pass
