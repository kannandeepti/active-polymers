r"""
Initializaiton routines for BD simulations
------------------------------------------

Different initialization routines for the positions of N monomers of a Rouse chain.
In the absence of any global forces, we initialize the Rouse chain in a free-draining equilibrium
where the first mornomer is centered at the origin. With a confinement or self-avoidance, we
keep redrawing positions of successive monomers until they do not overlap and until they are within
the elliptical confinement.

"""
import numpy as np
from numba import njit

@njit
def init_conf(N, bhat, rx, ry, rz):
    """ Initialize beads to be in a confinement.
    This works.
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

def grow_cubic(N, box_size):
    """ Initialize polymer on a cubic lattice of size box_size."""
    pass

@njit
def init_conf_avoid(N, bhat, rx, ry, rz, dsq):
    """ Initialize beads to be in confinement and to not overlap.
    TODO: test. This does not work.
    """
    # initial position
    x0 = np.zeros((N, 3))
    for i in range(1, N):
        # 1/sqrt(3) since generating per-coordinate
        x0[i] = x0[i - 1] + bhat / np.sqrt(3) * np.random.randn(3)
        #keep redrawing new positions until within confinement and no overlap with existing beads
        while (x0[i, 0] ** 2 / rx ** 2 + x0[i, 1] ** 2 / ry ** 2 + x0[i, 2] ** 2 / rz ** 2 > 1
                and np.any(np.sum((x0[i, :] - x0[:i, :])**2, axis=-1) <= dsq)):
            x0[i] = x0[i - 1] + bhat / np.sqrt(3) * np.random.randn(3)
    return x0

@njit
def init_avoid(N, bhat, dsq, atol=1.0e-4):
    """ Initialize beads to not overlap.
    TODO: test. This does not work... Perhaps best to initialize in free draining equilibrium
    and then run some initialization BD steps until monomers are sufficiently spaced apart so as to
    not overlap.
    """
    # initial position
    x0 = np.zeros((N, 3))
    for i in range(1, N):
        # 1/sqrt(3) since generating per-coordinate
        x0[i] = x0[i - 1] + bhat / np.sqrt(3) * np.random.randn(3)
        #keep redrawing new positions until within confinement and no overlap with existing beads
        while np.any(np.abs(np.sum((x0[i, :] - x0[:i, :])**2, axis=-1) - dsq) > atol):
            x0[i] = x0[i - 1] + bhat / np.sqrt(3) * np.random.randn(3)
    return x0