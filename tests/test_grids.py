#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
from sht.grids import standard_grid, get_cartesian_grid

def test_grids():
    L = 10
    thetas, phis = standard_grid(L)

    # Can't really test much here
    assert thetas.size == L
    assert phis.size == L**2

    grid = get_cartesian_grid(thetas, phis)
    assert grid.shape == (L**2, 3)
