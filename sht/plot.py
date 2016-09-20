#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
from mayavi import mlab
#import matplotlib.pyplot as plt


def scatter(x, y, z, c, s):
    """
    Creates a scatter plot of points (depicted as spheres) in 3D.

    Parameters
    ----------
    x, y, z: np.ndarray
        Locations of the points in 3D coordinates.
    c: np.ndarray
        Colors of the points
    s: np.ndarray
        Sizes of the points (i.e. radii of the spheres)
    """

    pts = mlab.quiver3d(x, y, z, s, s, s, scalars=c, mode="sphere",
                        scale_factor=0.25)
    pts.glyph.color_mode = "color_by_scalar"
    pts.glyph.glyph_source.glyph_source.center = [0, 0, 0]
