r"""
Ways of generating correlated noise
-----------------------------------

This module provides various methods to generate correlated noise in a simulation of an active
Rouse polymer. A valid correlation matrix :math:`C_{ij} \in [0, 1]` must be positive definite and
contains 1's on the diagonal. A covariance matrix is related to the underlying correlation matrix
via :math:`\langle \eta_i \eta_j \rangle = 2 \sqrt{D_i} \sqrt{D_j} C_{ij}`.
"""

import numpy as np
from numba import njit

def gaussian_correlation(N, length, s0, s0_prime, max):
    """ Generate a N x N covariance matrix, C, with a circular Guassian
    localized around (s0, s0_prime).

    Note: I had to do some sketchy rescalings at the end to ensure that the matrix is PD.
    Would not recommend this approach.

    Parameters
    ----------
    N : int
        Number of monomers in chain
    length : float
        std of Gaussian (Assume variance in s and s_prime are identical)
    s0 : int
        location of one locus along chain
    s0_prime : int
        location of another locus along chain
    max : float
        strength of correlation (if max <0 then the loci will be anti-correlated)

    Note: C[i, j] = correlation between s=i and s_prime=j. So C[s0, s0_prime] = max.
    """
    if np.abs(max) > 1.0:
        raise ValueError('Correlation coefficient cannot be greater than 1 or less than -1.')
    x = np.arange(0, N)
    y = np.arange(0, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    C = max * np.exp(-((X - s0)**2 + (Y - s0_prime)**2)/(2 * length**2))
    C = C + C.T #make symmetric
    #make 1s on diagonal
    for i in range(N):
        C[i, i] = 1.0
    #make positive definite
    C = C.T @ C
    #rescale to be valid correlation matrix
    C /= np.abs(np.max(C))
    return C

def psd_gaussian_correlation(N, length, A):
    """ Generate a N x N correlation matrix with elements Ae^(-(s-s')^2/2L^2),
    where L is `length`.
    Note all elements on each diagonal of the matrix will be the same.

    PROBLEM: This matrix is extremely ill conditioned for large N due to
    the correlation matrix being largely flat for small s-s'.
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.822.3231&rep=rep1&type=pdf
    """

    if np.abs(A) > 1.0:
        raise ValueError('Correlation coefficient cannot be > 1 or < -1.')
    x = np.arange(0, N)
    y = np.arange(0, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    C = A * np.exp(-(X - Y)**2/(2 * length**2))
    return C

def exponential_correlation(N, length, A):
    """ Generate a N x N correlation matrix with elements Ae^(-|s-s'|/L).
    Note all elements on each diagonal of the matrix will be the same.

    This matrix is better conditioned.
    """
    if np.abs(A) > 1.0:
        raise ValueError('Correlation coefficient cannot be > 1 or < -1.')
    x = np.arange(0, N)
    y = np.arange(0, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    C = A * np.exp(-np.abs(X - Y)/length)
    return C

def random_factors_correlation(N, k):
    """ Generate a random correlation matrix using the random factors method.

    Parameters
    ----------
    N : int
        Dimension of correlation matrix
    k : int
        Number of random factors :math:`k \in (1, N)`
    """
    if k >= N:
        raise ValueError('The number of random factors, k, must be < N')
    W = np.random.randn(N, k)
    S = W @ W.T + np.diag(np.random.rand(N))
    S = np.diag(1./np.sqrt(np.diag(S))) @ S @ np.diag(1./np.sqrt(np.diag(S)))
    return S

def generate_correlations(identity_mat, rhos, d=3):
    """ Generate correlated noise via process in which each of N monomers is
    assigned k binary identities, each of type -1 or 1, where noise acts on monomers of type 1
    differently than on monomers of type -1.

    In particular, type 1 variables will all be correlated with correlation coefficient rho, and
    type -1 variables will all be correlated with correlation coefficient rho. type 1 / type -1
    variables will be anti-correlated with coefficient -rho.

    Parameters
    ----------
    identity_mat: (k, N) array-like
        kth row contains 1s or 0s to assign monomers of type 1 vs type 0 for the kth feature
    rhos : (k,) array-like
        Correlation coefficient associated with kth feature

    Returns
    -------
    noise: (N, d) array-like
        noise matrix
    corr : (N, N) array-like
        correlation matrix
    """

    k, N = identity_mat.shape
    identity_mat = identity_mat.astype(bool)
    rhos = np.sqrt(rhos) #correlation between 1 and x
    #N x 3 noise matrix
    noise = np.zeros((N, d))
    for i in range(k):
        #select out entries of noise matrix that correspond to type 1
        num_type1 = np.sum(identity_mat[i, :])
        num_type0 = np.sum(~identity_mat[i, :])
        x = np.random.randn(1, d)
        y = np.random.randn(num_type1, d)
        z = np.random.randn(num_type0, d)
        #correlate type 1 beads with x
        noise[identity_mat[i, :], :] += (rhos[i] * x + np.sqrt(1 - rhos[i]**2)*y)
        #anti-correlate type -1 beads with x
        noise[~identity_mat[i, :], :] += (-rhos[i] * x + np.sqrt(1 - rhos[i]**2)*z)
    return noise

@njit
def generate_correlations_vars(identity_mat, rhos, stds, d=3):
    """ Generate correlated noise via process in which each of N monomers is
    assigned k identities, each of type -1, 0, or 1. type 1 beads are correlated with each other,
    type -1 beads are correlated with each other, and type 1/-1 are anti-correlated. type 0 beads
    are uncorrelated with all other beads.

    Parameters
    ----------
    identity_mat: (k, N) array-like
        kth row contains 1s, 0s, or -1s to assign monomers of type 1, type 0, or type -1 for the
        kth feature
    rhos : (k,) array-like
        Correlation coefficient associated with kth feature
    stds : (N,) array-like
        Standard deviations of the N beads (i.e. temperatures)

    Returns
    -------
    noise: (N, d) array-like
        noise matrix
    corr : (N, N) array-like
        correlation matrix
    """

    k, N = identity_mat.shape
    rhos = np.sqrt(rhos) #correlation between 1 and x
    stds = stds.reshape((N, 1)) / np.sqrt(k)
    #N x 3 noise matrix
    noise = np.zeros((N, d))
    for i in range(k):
        # select out entries of noise matrix that correspond to each type
        num_type1 = np.sum(identity_mat[i, :] == 1)
        num_type_neg1 = np.sum(identity_mat[i, :] == -1)
        num_type0 = np.sum(identity_mat[i, :] == 0)
        x = np.random.randn(1, d)
        y = np.random.randn(num_type1, d)
        z = np.random.randn(num_type_neg1, d)
        w = np.random.randn(num_type0, d)
        #correlate type 1 beads with x
        noise[identity_mat[i, :] == 1, :] += stds[identity_mat[i, :] == 1] * (rhos[i] * x + np.sqrt(
            1 - rhos[i]**2)*y)
        #anti-correlate type -1 beads with x
        noise[identity_mat[i, :] == -1, :] += stds[identity_mat[i, :] == -1] * (-rhos[i] * x +
                                                                             np.sqrt(1 - rhos[i]**2)*z)
        #draw uncorrelated variables for type 0
        noise[identity_mat[i, :] == 0, :] += stds[identity_mat[i, :] == 0] * w
    return noise

def covariance_from_noise(identity_mat, rhos, stds, niter=1000, **kwargs):
    """ Compute the covariance matrix expected from the noise generation process
    in `generate_correlation_vars`. Draw `niter` samples of the noise and calculate
    the average :math:`\langle \eta_i \eta_j \rangle`."""

    k, N = identity_mat.shape
    #sum of (eta_i * eta_j)
    covariance = np.zeros((N, N))
    for i in range(niter):
        #N x 1 example noise matrix
        noise = generate_correlations_vars(identity_mat, rhos, stds, d=1, **kwargs)
        #outer product gives eta_i eta_j
        covariance += np.outer(noise, noise)
    #obtain average by dividing by number of replicates
    avg_covariance = covariance / niter
    return avg_covariance

