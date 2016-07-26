#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from sht import isht
from grids import standard_grid
from utils import l_to_lm, argsort_thetaphi_wrt_theta

if __name__ == '__main__':
    L = 10
    thetas, phis = standard_grid(L)

    x = np.zeros(L)
    x[1] = 1
    x = l_to_lm(x)
    y = isht(x, thetas, phis).real

    plt.plot(y)

    i = argsort_thetaphi_wrt_theta(thetas)

    plt.plot(y[i])
    plt.show()
