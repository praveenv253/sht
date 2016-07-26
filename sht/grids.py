#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np


def standard_grid(L):
    thetas = []
    for i in range(1, L+1):
        if i % 2 == 0:
            theta = np.pi * (i - 1) / (2 * L - 1)
        else:
            theta = np.pi * (2 * L - i) / (2 * L - 1)
        thetas.append(theta)
    thetas = np.array(thetas)

    phis = []
    for i in range(L):
        for j in range(-i, i+1):
            phis.append(2 * np.pi * j / (2*i + 1))
    phis = np.array(phis)

    return thetas, phis
