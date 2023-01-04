r"""
Initializaiton routines for BD simulations
------------------------------------------

Different initialization routines for the positions of N monomers of a Rouse chain.
In the absence of any global forces, we initialize the Rouse chain in a free-draining equilibrium
where the first monomer is centered at the origin. Within a confinement, we
keep redrawing positions of successive monomers until they are within
the elliptical confinement.

"""
import numpy as np
from numba import njit

@njit
def free_draining(N, bhat):
    """ Initialize monomers in a free draining equilibrium.

    Parameters
    ----------
    N : int
        number of monomers
    bhat : float
        square root of mean squared bond extension
    """
    # initial position
    x0 = bhat / np.sqrt(3) * np.random.randn(N, 3)
    # for jit, we unroll ``x0 = np.cumsum(x0, axis=0)``
    for i in range(1, N):
        x0[i] = x0[i - 1] + x0[i]
    return x0

@njit
def init_conf(N, bhat, rx, ry, rz):
    """ Initialize monomers to be in a (rx, ry, rz) elliptical confinement.

    Parameters
    ----------
    N : int
        number of monomers
    bhat : float
        square root of mean squared bond extension
    rx : float
    ry: float
    rz : float
    """
    # initial position
    x0 = np.zeros((N, 3))
    # x0 = np.cumsum(x0, axis=0)
    for i in range(1, N):
        # 1/sqrt(3) since generating per-coordinate
        x0[i] = x0[i - 1] + bhat / np.sqrt(3) * np.random.randn(3)
        while x0[i, 0] ** 2 / rx ** 2 + x0[i, 1] ** 2 / ry ** 2 + x0[i, 2] ** 2 / rz ** 2 > 1:
            x0[i] = x0[i - 1] + bhat / np.sqrt(3) * np.random.randn(3)
    return x0