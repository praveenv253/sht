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


def argsort_thetaphi_wrt_theta(thetas):
    """
    For a given thetas vector, return indices that will convert the usual
    ordering of the L**2-length vector (in order of increasing number of phi-
    points) into the order corresponding to monotonically increasing theta
    (from 0 to pi).
    This is useful for plotting the L**2-length signal in 1 dimension, so that
    the signal can still be partially interpreted as increasing in theta over
    the x-axis.
    """
    L = thetas.size
    theta_sort_indices = np.argsort(thetas)
    result = np.empty(L**2, dtype=int)
    count = 0
    for i in theta_sort_indices:
        indices = np.arange(2 * i + 1)
        result[count + indices] = i ** 2 + indices
        count += 2 * i + 1
    return result


def get_cartesian_grid(thetas, phis):
    """
    Converts a list of thetas and phis into a cartesian grid.
    """
    thetas = l_to_lm(thetas)
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    return np.vstack((x, y, z)).T
