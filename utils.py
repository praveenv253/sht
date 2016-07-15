#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np


def l_to_lm(xl, axis=-1, fill_zeros=False):
    """
    Converts an array x(l) to an aggregate array x(l, m).
    By default, it repeats x(l) for all x(l, m). If fill_zeros is True, then it
    fills only x(l, 0) with x(l). The values of x for the remaining m are set
    to zero.
    """
    if axis < 0:
        axis = xl.ndim + axis
    num_l = xl.shape[axis]
    # The axis corresponding to l will need num_l**2 elements
    xlm_shape = tuple(num_l**2 if i == axis else xl.shape[i]
                      for i in range(xl.ndim))
    # TODO There has to be a neater way of doing this! Check numpy source code
    # if required.
    indices_lhs = [slice(None, None, None), ] * xl.ndim
    indices_rhs = [slice(None, None, None), ] * xl.ndim
    if fill_zeros:
        xlm = np.zeros(xlm_shape)
        for i in range(num_l):
            indices_lhs[axis] = i**2 + i
            indices_rhs[axis] = i
            xlm[indices_lhs] = xl[indices_rhs]
    else:
        xlm = np.empty(xlm_shape)
        for i in range(num_l):
            # Define the rhs index to be a List to prevent dimension reduction:
            # this is required for proper broadcasting later
            indices_lhs[axis] = slice(i**2, (i+1)**2, None)
            indices_rhs[axis] = [i, ]
            xlm[indices_lhs] = xl[indices_rhs]
    return xlm
