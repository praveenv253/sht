#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import scipy.special as spl
from scipy import io

from sht import sht
from sht.grids import standard_grid
from sht.utils import l_to_lm
from test_utils import get_transform_matrix


def test_sht():
    L = 20
    thetas, phis = standard_grid(L)

    #np.set_printoptions(linewidth=400, precision=4)

    # Test signal defined in spherical harmonic domain, and then converted to
    # spatial domain
    #l = 1
    #m = 1
    #flm = np.zeros(L**2)
    #flm[l**2 + l + m] = 1
    flm = np.random.randn(L**2, 2)
    #print(flm)

    # Compare with transform matrix
    ylms = get_transform_matrix(thetas, phis, L)
    f_true = np.dot(ylms.T, flm)
    #io.savemat('f_true.mat', {'f': f_true})
    #print(f_true)

    intermediates = {}
    flm_recovered = sht(f_true, thetas, phis, intermediates)

    #print()
    #print(flm_recovered)

    #print(flm_recovered / flm)
    #print(np.real(flm_recovered) / np.real(flm))
    #print(np.imag(flm_recovered) / np.imag(flm))
    assert np.all(np.abs(flm_recovered - flm) < 1e-12)
