""" Ways to generate example correlation matrices. """

import numpy as np

def gaussian_correlation(N, length, s0, s0_prime, max):
    """ Generate a N x N covariance matrix, C, with a circular Guassian
    localized around (s0, s0_prime).

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
    """ Generate a N x N correlation matrix with elements Ae^(-(s-s')^2/2L^2).
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
    """
    if k >= N:
        raise ValueError('The number of random factors, k, must be < N')
    W = np.random.randn(N, k)
    S = W @ W.T + np.diag(np.random.rand(N))
    S = np.diag(1./np.sqrt(np.diag(S))) @ S @ np.diag(1./np.sqrt(np.diag(S)))
    return S