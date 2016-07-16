#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spl
from scipy import io

from sht import isht
from grids import standard_grid
from utils import l_to_lm


def get_transform_matrix(thetas, phis, num_l=20):
    """
    Transform matrix, for comparison
    """
    ylms = np.empty((num_l ** 2, num_l**2), dtype=complex)
    thetas = l_to_lm(thetas)
    for l in range(num_l):
        for m in range(-l, l+1):
            ylms[l**2 + (l+m), :] = spl.sph_harm(m, l, phis, thetas)
    return ylms


if __name__ == '__main__':
    L = 20
    thetas, phis = standard_grid(L)

    # Test signal defined in spherical harmonic domain
    l = 1
    m = -1
    flm = np.zeros(L**2)
    flm[l**2 + l + m] = 1

    # Test isht
    intermediates = {}
    f = isht(flm, thetas, phis, intermediates)
    io.savemat('f.mat', {'f': f})
    #print(f)

    # Compare with transform matrix
    ylms = get_transform_matrix(thetas, phis, L)
    f_true = np.dot(ylms.T, flm)
    io.savemat('f_true.mat', {'f': f_true})
    #print(f_true)

    #print(np.real(f) / np.real(f_true))
    #print(np.imag(f) / np.imag(f_true))
    print(np.abs(f - f_true) < 1e-10)

    #plt.plot(f)
    #plt.plot(f_true)
    #plt.show()
