#!/usr/bin/env python3

from .sht import sht, isht
from . import utils
from . import grids

# Try importing the plot module. This will fail if mayavi isn't installed.
# But requiring mayavi is too much overhead for the other sht modules. The plot
# module will work automatically if mayavi is installed.

# FIXME This creates a problem with scripts that need to import sht, but
# which are run on a machine that don't have access to an X-server. These
# scripts crash with "cannot connect to X server". We need to give scripts the
# option to not import the plot module. Until we figure out how to do that,
# this will be commented out.
#try:
#    from . import plot
#except ImportError:
#    pass
