#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import scipy.special as spl

from sht.utils import l_to_lm


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
