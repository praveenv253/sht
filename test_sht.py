#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spl
from scipy import io

from sht import sht
from grids import standard_grid
from utils import l_to_lm
from test_utils import get_transform_matrix


if __name__ == '__main__':
    L = 3
    thetas, phis = standard_grid(L)

    np.set_printoptions(linewidth=400, precision=4)

    # Test signal defined in spherical harmonic domain, and then converted to
    # spatial domain
    l = 1
    m = 1
    flm = np.zeros(L**2)
    flm[l**2 + l + m] = 1
    #flm = np.random.randn(L**2)
    print(flm)

    # Compare with transform matrix
    ylms = get_transform_matrix(thetas, phis, L)
    f_true = np.dot(ylms.T, flm)
    #io.savemat('f_true.mat', {'f': f_true})
    print(f_true)

    intermediates = {}
    flm_recovered = sht(f_true, thetas, phis, intermediates)
    print(flm_recovered)

    #print(np.real(f) / np.real(f_true))
    #print(np.imag(f) / np.imag(f_true))
    print(np.abs(flm_recovered - flm))

    #plt.plot(f)
    #plt.plot(f_true)
    #plt.show()
