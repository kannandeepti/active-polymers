r"""
Forces involved in BD simulations
---------------------------------

All Rouse polymer simulations involve spring forces between adjacent monomers (
`f_elas_linear_rouse`). Here, we also implement a spherical confinement and self avoidance
via different possible potentials. Combinations of confinement, springs, and self-avoidance
forces are also implemented in the same for loop to make code more efficient.
"""

from numba import njit
import numpy as np

@njit
def f_self_avoidance(x, a, dsq):
    """ Force due to overlap with other monomers via a repulsive LJ potential.

    Parameters
    ----------
    x : (N, 3) array-like
        positions of N monomers in Rouse chain
    a : float
        diameter of monomer
    dsq : float
        square of d = minimum of LJ potential = 2^(1/6)*a (computed once)
    """
    f = np.zeros(x.shape)
    for i in range(x.shape[0]):
        #force on monomer i due to all other monomers j not equal to i
        #For each monomer, compute distances to all other monomers.
        #naive way: relative position from x
        #for j > i, compute distances to monomer i
        for j in range(i + 1, x.shape[0]):
            dist = x[j] - x[i]
            rijsq = np.sum(dist**2)
            if rijsq <= dsq:
                rij = np.sqrt(rijsq)
                unit_rij = dist / rij
                ratio = a / rij
                fji = (24 / rij) * (2 * ratio ** 12 - ratio ** 6)
                f[i] -= fji * unit_rij # force on monomer i due to overlap with monomer j
                f[j] += fji * unit_rij # equal and opposite force on monomer j
    return f

@njit
def f_conf_ellipse(x0, Aex, rx, ry, rz):
    """Compute soft (cubic) force due to elliptical confinement."""
    N, _ = x0.shape
    f = np.zeros(x0.shape)
    for j in range(N):
        conf = x0[j, 0]**2/rx**2 + x0[j, 1]**2/ry**2 + x0[j, 2]**2/rz**2
        if conf > 1:
            conf_u = np.array([
                -x0[j, 0]/rx**2, -x0[j, 1]/ry**2, -x0[j, 2]/rz**2
            ])
            conf_u = conf_u/np.linalg.norm(conf_u)
            # Steph's confinement from
            # https://journals.aps.org/pre/abstract/10.1103/PhysRevE.82.011913
            f[j] += Aex*conf_u*np.power(np.sqrt(conf) - 1, 3)
    return f

@njit
def f_elas_linear_rouse(x0, k_over_xi):
    """Compute spring forces on single, linear rouse polymer."""
    N, _ = x0.shape
    f = np.zeros(x0.shape)
    for j in range(1, N):
        for n in range(3):
            f[j, n] += -k_over_xi*(x0[j, n] - x0[j-1, n])
            f[j-1, n] += -k_over_xi*(x0[j-1, n] - x0[j, n])
    return f

@njit
def f_elas_loops(x0, k_over_xi, relk, K, lamb=0):
    """Compute spring forces on single, linear rouse polymer with additional
    springs located at points in K matrix."""
    N, _ = x0.shape
    f = np.zeros(x0.shape)
    for j in range(1, N):
        f[j, :] += -lamb * x0[j, :]
        for n in range(3):
            f[j, n] += -k_over_xi*(x0[j, n] - x0[j-1, n])
            f[j-1, n] += -k_over_xi*(x0[j-1, n] - x0[j, n])
    # add additional springs at specific locations
    for k in range(len(K)):
        s1 = K[k][0]
        s2 = K[k][1]
        for n in range(3):
            f[s1, n] += -relk * k_over_xi*(x0[s1, n] - x0[s2, n])
            f[s2, n] += -relk * k_over_xi*(x0[s2, n] - x0[s1, n])
    return f

@njit
def f_conf_spring(x0, k_over_xi, Aex, rx, ry, rz):
    """ Compute forces due to springs and confinement
        all in the same for loop."""
    N, _ = x0.shape
    f = np.zeros(x0.shape)
    for i in range(N):
        # SPRING FORCES
        if i >= 1:
            for n in range(3):
                f[i, n] += -k_over_xi * (x0[i, n] - x0[i - 1, n])
                f[i - 1, n] += -k_over_xi * (x0[i - 1, n] - x0[i, n])
        # CONFINEMENT
        conf = x0[i, 0] ** 2 / rx ** 2 + x0[i, 1] ** 2 / ry ** 2 + x0[i, 2] ** 2 / rz ** 2
        if conf > 1:
            conf_u = np.array([
                -x0[i, 0] / rx ** 2, -x0[i, 1] / ry ** 2, -x0[i, 2] / rz ** 2
            ])
            conf_u = conf_u / np.linalg.norm(conf_u)
            f[i] += Aex * conf_u * np.power(np.sqrt(conf) - 1, 3)
    return f

@njit
def f_combined(x0, k_over_xi, Aex, rx, ry, rz, a, dsq):
    """ Compute forces due to springs, confinement, and self-avoidance
    all in the same for loop."""
    N, _ = x0.shape
    f = np.zeros(x0.shape)
    for i in range(N):
        #SPRING FORCES
        if i >= 1:
            for n in range(3):
                f[i, n] += -k_over_xi * (x0[i, n] - x0[i - 1, n])
                f[i - 1, n] += -k_over_xi * (x0[i - 1, n] - x0[i, n])
        #CONFINEMENT
        conf = x0[i, 0] ** 2 / rx ** 2 + x0[i, 1] ** 2 / ry ** 2 + x0[i, 2] ** 2 / rz ** 2
        if conf > 1:
            conf_u = np.array([
                -x0[i, 0] / rx ** 2, -x0[i, 1] / ry ** 2, -x0[i, 2] / rz ** 2
            ])
            conf_u = conf_u / np.linalg.norm(conf_u)
            f[i] += Aex * conf_u * np.power(np.sqrt(conf) - 1, 3)
        #SELF-AVOIDANCE via repulsive LJ potential
        for j in range(i + 1, N):
            dist = x0[j] - x0[i]
            rijsq = np.sum(dist**2)
            if rijsq <= dsq:
                rij = np.sqrt(rijsq)
                unit_rij = dist / rij
                ratio = a / rij
                fji = (24 / rij) * (2 * ratio ** 12 - ratio ** 6)
                f[i] -= fji * unit_rij # force on monomer i due to overlap with monomer j
                f[j] += fji * unit_rij # equal and opposite force on monomer j
    return f

@njit
def f_spring_avoid(x0, k_over_xi, a, dsq):
    """ Compute forces due to springs and self-avoidance
    all in the same for loop."""
    N, _ = x0.shape
    f = np.zeros(x0.shape)
    for i in range(N):
        #SPRING FORCES
        if i >= 1:
            for n in range(3):
                f[i, n] += -k_over_xi * (x0[i, n] - x0[i - 1, n])
                f[i - 1, n] += -k_over_xi * (x0[i - 1, n] - x0[i, n])
        #SELF-AVOIDANCE via repulsive LJ potential
        for j in range(i + 1, N):
            dist = x0[j] - x0[i]
            rijsq = np.sum(dist**2)
            if rijsq <= dsq:
                rij = np.sqrt(rijsq)
                unit_rij = dist / rij
                ratio = a / rij
                fji = (24 / rij) * (2 * ratio ** 12 - ratio ** 6)
                f[i] -= fji * unit_rij # force on monomer i due to overlap with monomer j
                f[j] += fji * unit_rij # equal and opposite force on monomer j
    return f

@njit
def f_spring_avoid_NL(x0, k_over_xi, a, dsq, cl, nl, box_size):
    """ Compute forces due to springs and self-avoidance
    all in the same for loop using neighbor list."""
    N, _ = x0.shape
    f = np.zeros(x0.shape)
    for i in range(N):
        #SPRING FORCES
        if i >= 1:
            for n in range(3):
                f[i, n] += -k_over_xi * (x0[i, n] - x0[i - 1, n])
                f[i - 1, n] += -k_over_xi * (x0[i - 1, n] - x0[i, n])

        #SELF-AVOIDANCE via repulsive LJ potential
        for j in range(cl[i], cl[i+1]):
            jj = nl[j]
            dist = x0[jj] - x0[i]
            dist -= np.rint(dist / box_size) * box_size
            rijsq = dist @ dist
            if rijsq <= dsq:
                try:
                    rij = np.sqrt(rijsq)
                    unit_rij = dist / rij
                    ratio = a / rij
                    fji = (24 / rij) * (2 * ratio ** 12 - ratio ** 6)
                    f[i] -= fji * unit_rij # force on monomer i due to overlap with monomer j
                    f[jj] += fji * unit_rij # equal and opposite force on monomer j
                except:
                    print(rij)
                    print(x0)
    return f

@njit
def f_spring_scr_NL(x0, k_over_xi, ks_over_xi, a, dsq, cl, nl, box_size):
    """ Compute forces due to springs and soft core repulsive potential

    Parameters
    ----------
    ks_over_xi : float
        strength of scr potential divided by friction coefficient
    a : float
        radius of monomer
    dsq : float
        (2a)^2, precomputed

    TODO: debugs
    """
    N, _ = x0.shape
    f = np.zeros(x0.shape)
    for i in range(N):
        #SPRING FORCES
        if i >= 1:
            for n in range(3):
                f[i, n] += -k_over_xi * (x0[i, n] - x0[i - 1, n])
                f[i - 1, n] += -k_over_xi * (x0[i - 1, n] - x0[i, n])
        #SELF-AVOIDANCE via repulsive soft core (harmonic) potential
        for j in range(cl[i], cl[i+1]):
            jj = nl[j]
            dist = x0[i] - x0[jj]
            dist -= np.rint(dist / box_size) * box_size
            rijsq = dist @ dist
            if rijsq <= dsq:
                rij = np.sqrt(rijsq) # | ri - rj |
                unit_rij = dist / rij
                fij = ks_over_xi * (2*a - rij) #always a positive number
                f[i] += fij * unit_rij # force on monomer i due to overlap with monomer j
                f[jj] -= fij * unit_rij # equal and opposite force on monomer j
    return f

@njit
def f_spring_conf_scrNL(x0, k_over_xi, ks_over_xi, a, dsq, Aex, rx, ry, rz, cl, nl, box_size):
    """ Compute forces due to springs and soft core repulsive potential.

    Parameters
    ----------
    ks_over_xi : float
        strength of scr potential divided by friction coefficient
    a : float
        radius of monomer
    dsq : float
        (2a)^2, precomputed
    Aex : float
        Strength of elliptical confinement
    rx : float
        semi-major x-axis of ellipsoid
    ry : float
        semi-major y-axis of ellipsoid
    rz : float
        semi-major z-axis of ellipsoid

    """
    N, _ = x0.shape
    f = np.zeros(x0.shape)
    for i in range(N):
        #SPRING FORCES
        if i >= 1:
            disp = x0[i] - x0[i - 1]
            disp -= np.rint(disp / box_size) * box_size
            rij = np.sqrt(disp @ disp)
            fij = -k_over_xi * (rij - 2*a)
            unit_rij = disp / rij
            f[i] += fij * unit_rij
            f[i-1] -= fij * unit_rij
            #for n in range(3):
            #    f[i, n] += -k_over_xi * (x0[i, n] - x0[i - 1, n])
            #    f[i - 1, n] += -k_over_xi * (x0[i - 1, n] - x0[i, n])
        # CONFINEMENT
        conf = x0[i, 0] ** 2 / rx ** 2 + x0[i, 1] ** 2 / ry ** 2 + x0[i, 2] ** 2 / rz ** 2
        if conf > 1:
            conf_u = np.array([
                -x0[i, 0] / rx ** 2, -x0[i, 1] / ry ** 2, -x0[i, 2] / rz ** 2
            ])
            conf_u = conf_u / np.linalg.norm(conf_u)
            f[i] += Aex * conf_u * np.power(np.sqrt(conf) - 1, 3)
        #SELF-AVOIDANCE via repulsive soft core (harmonic) potential
        for j in range(cl[i], cl[i+1]):
            jj = nl[j]
            dist = x0[i] - x0[jj]
            dist -= np.rint(dist / box_size) * box_size
            rijsq = dist @ dist
            if rijsq <= dsq:
                rij = np.sqrt(rijsq) # | ri - rj |
                unit_rij = dist / rij
                fij = ks_over_xi * (2*a - rij) #always a positive number
                f[i] += fij * unit_rij # force on monomer i due to overlap with monomer j
                f[jj] -= fij * unit_rij # equal and opposite force on monomer j
    return f
