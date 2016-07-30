#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
from sht.utils import l_to_lm

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
