#!/usr/bin/env python3

"""
Module providing forward and inverse Spherical Harmonic Transforms.

The algorithm followed is that given in the paper:
Zubair Khalid, Rodney A. Kennedy, Jason D. McEwen, ``An Optimal-Dimensionality
Sampling Scheme on the Sphere with Fast Spherical Harmonic Transforms'', IEEE
Transactions on Signal Processing, vol. 62, no. 17, pp. 4597-4610, Sept.1, 2014
DOI: http://dx.doi.org/10.1109/TSP.2014.2337278
arXiv: http://arxiv.org/abs/1403.4661 [cs.IT]
"""

from __future__ import print_function, division

import numpy as np
import scipy.special as spl
import scipy.linalg as la


def _compute_P(thetas):
    """Computes all Pm, to be used for intermediate computations."""
    L = thetas.size
    P = []  # List of all Pm's
    for m in range(L):
        ls = np.arange(m, L)
        Pm = spl.sph_harm(m, ls[np.newaxis, :], 0, thetas[:, np.newaxis])
        P.append(2 * np.pi * Pm)
    return P


def sht(f, thetas, phis, intermediates=None):
    """
    Computes the spherical harmonic transform of f, for the grid specified by
    thetas and phis. This grid must conform to a specific format.

    Currently, f can be at most two dimensional. The first dimension will be
    transformed.
    """
    f = f.copy().astype(complex)    # Shouldn't corrupt the original
    L = thetas.size

    # Check intermediates for P, and compute it if absent
    if intermediates is None:   # Caller not using intermediates
        P = _compute_P(thetas)
    elif 'P' in intermediates:  # Caller provided P
        P = intermediates['P']
    else:                       # Caller wants P
        P = _compute_P(thetas)
        intermediates['P'] = P

    # Compute and store the LU factors of P[m]'s, so that computing the sht
    # multiple times is inexpensive
    if intermediates is None:
        Pm_factors = [la.lu_factor(P[m][m:, :]) for m in range(L)]
    elif 'Pm_factors' in intermediates:
        Pm_factors = intermediates['Pm_factors']
    else:
        Pm_factors = [la.lu_factor(P[m][m:, :]) for m in range(L)]
        intermediates['Pm_factors'] = Pm_factors

    # Initialize g: for L=4, it looks like this when complete:
    #  0  *  *  *  *  *  *
    #  0  1  *  *  *  * -1
    #  0  1  2  *  * -2 -1
    #  0  1  2  3 -3 -2 -1
    # The numbers here indicate the value of m. A * indicates an unused space.
    # The l'th row is the FFT of the ring corresponding to theta[l].
    # The m'th column (excluding the unused entries) is essentially gm.
    # Thus, gm is valid only when |m| <= l, and is best indexed from -m to m.
    g = np.zeros((L, 2 * L - 1) + f.shape[1:], dtype=complex)

    # Intialize result vector
    flm = np.zeros(f.shape, dtype=complex)

    for m in reversed(range(L)):
        # Update g by computing gm
        # Perform (2m+1)-point FFT of the m'th phi-ring
        # The sampling of f is from -m to m, whereas for the FFT, we need it to
        # be from 0 to 2m+1. Hence the ifftshift.
        temp = np.fft.fft(np.fft.ifftshift(f[m**2:(m+1)**2], axes=0),
                          axis=0) * 2 * np.pi / (2*m+1)
        # Add this to the main matrix g
        g[m, :m+1] = temp[:m+1]
        g[m, (2*L-1-m):] = temp[m+1:]

        # Solve for fm and fm_neg
        fm = la.lu_solve(Pm_factors[m], g[m:, m])
        if m > 0:
            fm_neg = (-1)**m * la.lu_solve(Pm_factors[m], g[m:, -m])

        # Store results
        ls = np.arange(m, L)
        flm[ls**2 + ls + m] = fm
        if m > 0:
            flm[ls**2 + ls - m] = fm_neg

        # Compute gm for the *other* thetas
        gm = np.einsum('i...,ki->k...', fm, P[m][:m, :])
        if m > 0:
            gm_neg = np.einsum('i...,ki->k...', fm_neg, (-1)**m * P[m][:m, :])

        for k in range(m):
            # Note: we won't enter this loop if m==0
            # Extend dimensions of phi for proper broadcasting with g
            ext_indices = ((slice(k**2, (k+1)**2),)
                           + (None,) * (len(f.shape) - 1))
            f_tilde = ((np.exp(1j * m * phis[ext_indices]) * gm[[k]]
                        + np.exp(-1j * m * phis[ext_indices]) * gm_neg[[k]])
                       / (2 * np.pi))
            f[k**2:(k+1)**2] -= f_tilde

    return flm


def isht(flm, thetas, phis, intermediates=None):
    """
    Computes the inverse spherical harmonic transform.
    """
    L = thetas.size

    # Check intermediates for P, and compute it if absent
    if intermediates is None:   # Caller not using intermediates
        P = _compute_P(thetas)
    elif 'P' in intermediates:  # Caller provided P
        P = intermediates['P']
    else:                       # Caller wants P
        P = _compute_P(thetas)
        intermediates['P'] = P

    # Initialize return vector
    f = np.zeros(flm.shape, dtype=complex)

    for m in range(L):
        ls = np.arange(m, L)
        gm = np.einsum('i...,ki->k...', flm[ls**2 + ls + m], P[m])
        gm_neg = np.einsum('i...,ki->k...', flm[ls**2 + ls - m],
                           (-1)**m * P[m])
        for k in range(L):
            # Extend dimensions of phi for proper broadcasting with g
            ext_indices = ((slice(k**2, (k+1)**2),)
                           + (None,) * (len(f.shape) - 1))
            if m == 0:
                f_tilde = gm[[k]] / (2 * np.pi)
            else:
                f_tilde = ((np.exp(-1j * m * phis[ext_indices]) * gm_neg[[k]]
                            + np.exp(1j * m * phis[ext_indices]) * gm[[k]])
                           / (2 * np.pi))
            f[k**2:(k+1)**2] += f_tilde

    return f
