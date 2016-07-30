#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np

from sht import isht
from sht.grids import standard_grid
from sht.utils import l_to_lm, argsort_thetaphi_wrt_theta

def test_l_to_lm():
    L = 10
    x = np.random.randn(L)

    # Test without fill_zeros
    y = l_to_lm(x)
    for i in range(L):
        assert np.array_equal(y[i**2:(i+1)**2], x[i] * np.ones(2 * i + 1))

    # Test with fill_zeros
    z = l_to_lm(x, fill_zeros=True)
    l = np.arange(L)
    assert np.array_equal(z[l**2 + l], x)
    z = np.delete(z, l**2 + l)
    assert np.array_equal(z, np.zeros(L**2 - L))

    # Test with axis
    p = 5
    q = 7
    x = np.random.randn(p, q)
    y = l_to_lm(x)
    assert y.shape == (p, q**2)
    y = l_to_lm(x, axis=0)
    assert y.shape == (p**2, q)
    y = l_to_lm(x, axis=-2)
    assert y.shape == (p**2, q)
    y = l_to_lm(x, axis=1)
    assert y.shape == (p, q**2)


def test_argsort():
    L = 10
    thetas, phis = standard_grid(L)

    x = np.zeros(L)
    x[1] = 1
    x = l_to_lm(x)
    y = isht(x, thetas, phis).real

    i = argsort_thetaphi_wrt_theta(thetas)
    assert np.all(np.abs(np.sort(y)[::-1] - y[i]) < 1e-15)
