#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from sht import isht
from sht.grids import standard_grid
from sht.utils import l_to_lm, argsort_thetaphi_wrt_theta

def test_argsort():
    L = 10
    thetas, phis = standard_grid(L)

    x = np.zeros(L)
    x[1] = 1
    x = l_to_lm(x)
    y = isht(x, thetas, phis).real

    i = argsort_thetaphi_wrt_theta(thetas)
    assert np.all(np.abs(np.sort(y)[::-1] - y[i]) < 1e-15)
