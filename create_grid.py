#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
from scipy import io

from grids import standard_grid
from utils import l_to_lm

if __name__ == '__main__':
    L = 20
    thetas, phis = standard_grid(L)
    thetas = l_to_lm(thetas)
    io.savemat('grid_L%d.mat' % L, {'thetas': thetas, 'phis': phis})
