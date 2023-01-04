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
from ..analysis.files import extract_cov

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
        Standard deviations of the N monomers (i.e. :math:`\sqrt{2 D dt}`)

    Returns
    -------
    noise: (N, d) array-like
        noise matrix

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

def covariance_from_identities(simdir, **kwargs):
    """ For a given simulation, extract the monomer identities (type 1, 0, or -1) and diffusion
    coefficients from saved files and then reconstruct the covariance matrix of the noise."""

    D, idmat, rhos = extract_cov(simdir, **kwargs)
    N = len(D)
    rho = rhos[0]
    corr = np.outer(idmat, idmat)
    corr *= rho
    corr[np.diag_indices(N)] = 1.0 #diagonal of correlation matrix is 1
    cov = np.sqrt(np.outer(D, D)) * corr
    return cov
