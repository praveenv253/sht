#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

from sht import isht
from sht.grids import standard_grid
from sht.utils import l_to_lm
from test_utils import get_transform_matrix


def test_isht():
    L = 20
    thetas, phis = standard_grid(L)

    #np.set_printoptions(linewidth=400)

    # Test signal defined in spherical harmonic domain
    #l = 2
    #m = -2
    #flm = np.zeros(L**2)
    #flm[l**2 + l + m] = 1
    flm = np.random.randn(L**2, 2)

    # Test isht
    intermediates = {}
    f = isht(flm, thetas, phis, intermediates)
    io.savemat('f.mat', {'f': f})
    #print(f)
    #print()

    # Compare with transform matrix
    ylms = get_transform_matrix(thetas, phis, L)
    f_true = np.dot(ylms.T, flm)
    io.savemat('f_true.mat', {'f': f_true})
    #print(f_true)
    #print()

    #print(np.real(f) / np.real(f_true))
    #print(np.imag(f) / np.imag(f_true))
    assert np.all(np.abs(f - f_true) < 1e-12)

    #plt.plot(f)
    #plt.plot(f_true)
    #plt.show()
