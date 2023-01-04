r"""
Simulations of active Rouse polymers
------------------------------------

We adapt the `wlcsim.bd.rouse <https://github.com/SpakowitzLab/wlcsim>`_ module developed by the
Spakowitz Lab to perform Brownian dynamics simulations of Rouse polymers driven by correlated,
active noise. Simulations are accelerated using just in
time compilation (jit) using the package Numba.

Provides routines for easily parameterizing Rouse polymer simulations with different types of
noise, deterministic forces, and integrators. Below we summarize the routines most useful for
`Goychuk et al. (2022) <https://doi.org/10.1101/2022.12.24.521789>`_
    1. **with_srk1**: Simulate a Rouse polymer with 1st order SRK integrator. Uses a discrete Rouse
       chain to approximate a semiflexible chain with parameters ``(N, L, b,
       D)``", where :math:`N` is the number of monomers, :math:`L` is the physical
       length of the chain (see note below), :math:`b` is the Kuhn length, and
       :math:`D` is an array specifying the diffusion coefficients of the N monomers.
       If the same value is specified for all N monomers, then we recover equilibrium
       Rouse polymer dynamics.
    2. **identity_core_noise_srk2**: Simulate a Rouse polymer subjected to correlated noise using a
        2nd order SRK integrator. Assign monomers identities (type 1, 0, or -1) such that same
        type monomers are correlated and opposite type monomers are anticorrelated. type 0
        monomers do not experience correlated active forces.
    3. **loops_with_srk1**: Simulate a Rouse polymer with additional Hookean springs connecting
        distinct monomers.

Notes
-----
Our ultimate goal is to apply the Rouse model to the long length scale limit of a semiflexible
chain such as chromatin with persistence length :math:`l_p = b / 2`. Most textbooks parameterize a
Rouse chain by the number of Kuhn segments (defined as ``N`` in Doi & Edwards). Here, we use `N`
to mean the number of monomers in the chain, and instead refer to `\hat{N} = L / b` as the number of
Kuhn lengths in the chain (``Nhat`` in the code). Correspondingly, most textbooks choose the mean
squared extension of the harmonic bonds that make up the Rouse polymer backbone to be `b`. Instead,
we coarse-grain the underlying wormlike chain (WLC) by choosing the mean squared bond extension
of the Rouse chain to be  :math:`\hat{b}^2 = L_0 b`, where :math:`L_0 = L / (N-1)` is the amount of
chain connecting adjacent monomers. This choice corresponds to the mean squared end-to-end
distance :math:`\langle R^2 \rangle = 2Ll_p` of the WLC that connects adjacent monomers.

The Langevin equation for a continuous Rouse polymer, where :math:`s` is the continuous
material coordinate along the polymer backbone is

    .. math::
        \xi \frac{d}{dt} \vec{r}(s, t)
        = k \frac{d^2}{ds^2} \vec{r}(s, t) + f^{(B)}

where :math:`k = 3k_BT/b^2` is the "spring constant" for a Rouse polymer with
Kuhn length :math:`b`, :math:`D` is the "diffusivity of a Kuhn length", i.e. :math:`k_bT/\xi`,
:math:`\xi` is the dynamic viscosity, and :math:`f^{(B)}` is the Brownian force. In order to
dicretize the Langevin dynamics, it can be shown that we must choose the diffusion coefficient
of each monomer to be :math:`\hat{D} = D (N/\hat{N})` (``Dhat`` in the
code). Since :math:`\xi` is in units of "viscosity per Kuhn length" and
:math:`(N/\hat{N})` is in units of "number of monomers over number of Kuhn
lengths", this can be thought of as changing units from "viscosity per Kuhn
length" to "viscosity per bead".

"""
from numba import njit
import numpy as np
from .correlations import *
from .forces import *
from .init_beads import *

@njit
def with_srk1(N, L, b, D, h, tmax, t_save=None, t_msd=None, msd_start_time=0.0, Deq=1):
    r"""
    Just-in-time compilable Rouse simulation use 1st order SRK scheme developed by
    A. J. Roberts: https://arxiv.org/abs/1210.0933 (modification of improved Euler
    scheme for SDEs).

    Simulate a Rouse polymer made of N monomers free in solution. Save conformations
    at specified time points (t_save) and calculate the mean squared displacement of all N
    monomers as well as the center of mass MSD at specified time points (t_msd).

    Parameters
    ----------
    N : float
        Number of monomers in the chain.
    L : float
        Length of chain.
    b : float
        Kuhn length of the chain (same units as *L*).
    D : (N,) array_like
        Diffusion coefficient of N monomers. (Units of ``length**2/time``).
        To recapitulate equilibrium Rouse polymer, set D to be the same value for all monomers.
    h : float
        Time step to use for stepping the integrator. Same units as *D*.
    tmax : float
        Total simulation time.
    t_save : (Nt,) array-like
        Time points at which to save output of simulation.
    t_msd : (Nm,) array-like
        Lag times at which to compute MSDs of each monomer + center of mass.
    msd_start_time : float
        Starting time point from which to compute MSD.
    Deq : float
        diffusion coefficient used to compute the spring stiffness, :math:`k/xi =3Deq/b^2`

    Returns
    -------
    X : (Nt, N, 3) array_like of float
        The positions of the *N* monomers at each of the *Nt* time points.
    msds : (Nm, N+1) array_like of float
        Mean squared displacement of the *N* monomers and center of mass at the *Nm* time points

    Notes
    -----
    The Langevin equation for the ith monomers is given by
    .. math::
        \frac{dx(i, t)}{dt} = - \frac{k}{\xi} (x(i, t) - x(i+1, t))
                                  - \frac{k}{\xi} (x(i, t) - x(i-1, t))
                                  + \eta_i(t)
    where :math:`\xi` is the friction coefficient, :math:`k/\xi = 3Deq/b^2` is the spring constant,
    :math:`b` is the Kuhn length of the polymer, :math:`D` is the self-diffusion coefficient of a
    bead, and each spatial component of :math:`\eta_i(t)` is a delta-correlated stationary Gaussian
    process with mean zero and :math:`\langle \eta_i(t) \eta_j(t') \rangle =
    2D_i \delta_{ij} \delta(t-t')`.
    Notice that in practice, :math:`k/\xi = 3D/b^2`, so we do not need to
    include mass units (i.e. there's no dependence on :math:`k_BT`).

    Force calculations, initialization steps, and MSD calculations are in-lined to make code
    as efficient as possible.
    """
    rtol = 1e-5
    # derived parameters
    L0 = L/(N-1)  # length per bead
    bhat = np.sqrt(L0*b)  # mean squared bond length of discrete gaussian chain
    Nhat = L/b  # number of Kuhn lengths in chain
    Dhat = D*N/Nhat  # diffusion coef of a discrete gaussian chain bead
    Deq = Deq * N / Nhat
    #set spring constant to be 3Deq/b^2 where D is the average monomer diffusion coefficient
    k_over_xi = 3*Deq/bhat**2
    # initial position, free draining equilibrium
    x0 = bhat/np.sqrt(3)*np.random.randn(N, 3)
    # for jit, we unroll ``x0 = np.cumsum(x0, axis=0)``
    for i in range(1, N):
        x0[i] = x0[i-1] + x0[i]

    msds = np.zeros((1, N+1))
    if t_msd is not None:
        print('Setting up msd calculation')
        #at each msd save point, msds of N monomers + center of mass
        msds = np.zeros((len(t_msd), N+1))
        msd_i = 0
        msd_start_ind = int(msd_start_time / h)
        if msd_start_time == 0:
            msd_start_pos = x0.copy() #N x 3
        msd_inds = np.rint((msd_start_time + t_msd) / h)

    if t_save is None:
        t_save = np.linspace(0.0, tmax, 101)

    x = np.zeros(t_save.shape + x0.shape)
    save_i = 0
    save_inds = np.rint(t_save / h)
    if 0 == t_save[save_i]:
        x[0] = x0
        save_i += 1
    ntimesteps = int(tmax // h) + 1  # including 0th time step
    # -1 or 1, p=1/2
    S = 2 * (np.random.rand(ntimesteps) < 0.5) - 1
    # at each step i, we use data (x,t)[i-1] to create (x,t)[i]
    # in order to make it easy to pull into a new functin later, we'll call
    # t[i-1] "t0", old x (x[i-1]) "x0", and t[i]-t[i-1] "h".
    for i in range(1, ntimesteps):
        dW = np.random.randn(*x0.shape)
        # D = sigma^2/2 ==> sigma = np.sqrt(2*D)
        Fbrown = (np.sqrt(2*Dhat/h) * (dW - S[i]).T).T
        # estimate for slope at interval start
        f = np.zeros(x0.shape)
        #loop over N monomers
        for j in range(1, N):
            #loop over 3 dimensions
            for n in range(3):
                f[j, n] += -k_over_xi * (x0[j, n] - x0[j-1, n])
                f[j-1, n] += -k_over_xi * (x0[j-1, n] - x0[j, n])
        K1 = f + Fbrown
        Fbrown = (np.sqrt(2*Dhat/h) * (dW + S[i]).T).T
        # estimate for slope at interval end
        x1 = x0 + h*K1
        f = np.zeros(x0.shape)
        for j in range(1, N):
            for n in range(3):
                f[j, n] += -k_over_xi*(x1[j, n] - x1[j-1, n])
                f[j-1, n] += -k_over_xi*(x1[j-1, n] - x1[j, n])
        K2 = f + Fbrown
        x0 = x0 + h * (K1 + K2)/2
        if t_msd is not None:
            if i == msd_start_ind:
                msd_start_pos = x0.copy()
            if i == msd_inds[msd_i]:
                #calculate msds, increment msd save index
                mean = np.zeros(x0[0].shape)
                for j in range(N):
                    diff = x0[j] - msd_start_pos[j]
                    mean += diff
                    msds[msd_i, j] = diff @ diff
                #center of mass msd
                diff = mean / N
                msds[msd_i, -1] = diff @ diff
                msd_i += 1
        if i == save_inds[save_i]:
            x[save_i] = x0
            save_i += 1
    return x, msds

@njit
def loops_with_srk1(N, L, b, D, h, tmax, K, relk=1.0, lamb=0.0,
                    t_save=None, t_msd=None, msd_start_time=0.0, Deq=1):
    r"""
    Simulate a Rouse polymer with additional harmonic bonds coupling distinct regions along the
    chain. Here, the forces are not in-lined for code clarity.

    Parameters
    ----------
    K : (m, 2) array-like[int]
        indicies of monomers that are connected by m additional springs to add to the Rouse chain
    relk : float
        Ratio of spring stiffness of additional bonds relative to polymer connectivity.
    """
    rtol = 1e-5
    # derived parameters
    L0 = L/(N-1)  # length per bead
    bhat = np.sqrt(L0*b)  # mean squared bond length of discrete gaussian chain
    Nhat = L/b  # number of Kuhn lengths in chain
    Dhat = D*N/Nhat  # diffusion coef of a discrete gaussian chain bead
    Deq = Deq * N / Nhat
    #set spring constant to be 3D/b^2 where D is the diffusion coefficient of the coldest bead
    k_over_xi = 3*Deq/bhat**2
    # initial position, free draining equilibrium
    x0 = bhat/np.sqrt(3)*np.random.randn(N, 3)
    # for jit, we unroll ``x0 = np.cumsum(x0, axis=0)``
    for i in range(1, N):
        x0[i] = x0[i-1] + x0[i]

    msds = np.zeros((1, N+1))
    if t_msd is not None:
        print('Setting up msd calculation')
        #at each msd save point, msds of N monomers + center of mass
        msds = np.zeros((len(t_msd), N+1))
        msd_i = 0
        msd_start_ind = int(msd_start_time / h)
        if msd_start_time == 0:
            msd_start_pos = x0.copy()  # N x 3
        msd_inds = np.rint((msd_start_time + t_msd) / h)

    if t_save is None:
        t_save = np.linspace(0.0, tmax, 101)

    x = np.zeros(t_save.shape + x0.shape)
    save_i = 0
    save_inds = np.rint(t_save / h)
    if 0 == t_save[save_i]:
        x[0] = x0
        save_i += 1
    ntimesteps = int(tmax // h) + 1  # including 0th time step
    # -1 or 1, p=1/2
    S = 2 * (np.random.rand(ntimesteps) < 0.5) - 1
    # at each step i, we use data (x,t)[i-1] to create (x,t)[i]
    # in order to make it easy to pull into a new functin later, we'll call
    # t[i-1] "t0", old x (x[i-1]) "x0", and t[i]-t[i-1] "h".
    for i in range(1, ntimesteps):
        dW = np.random.randn(*x0.shape)
        # D = sigma^2/2 ==> sigma = np.sqrt(2*D)
        Fbrown = (np.sqrt(2*Dhat/h) * (dW - S[i]).T).T
        # estimate for slope at interval start
        f = f_elas_loops(x0, k_over_xi, relk, K, lamb)
        K1 = f + Fbrown
        Fbrown = (np.sqrt(2*Dhat/h) * (dW + S[i]).T).T
        # estimate for slope at interval end
        x1 = x0 + h*K1
        f = f_elas_loops(x1, k_over_xi, relk, K, lamb)
        K2 = f + Fbrown
        x0 = x0 + h * (K1 + K2)/2
        if t_msd is not None:
            if i == msd_start_ind:
                msd_start_pos = x0.copy()
            if i == msd_inds[msd_i]:
                #calculate msds, increment msd save index
                mean = np.zeros(x0[0].shape)
                for j in range(N):
                    diff = x0[j] - msd_start_pos[j]
                    mean += diff
                    msds[msd_i, j] = diff @ diff
                #center of mass msd
                diff = mean / N
                msds[msd_i, -1] = diff @ diff
                msd_i += 1
        if i == save_inds[save_i]:
            x[save_i] = x0
            save_i += 1
    return x, msds


@njit
def identity_core_noise_srk2(N, L, b, D, h, tmax, t_save, mat, rhos,
                             t_msd=None, msd_start_time=None, Deq=1):
    """ BD simulation with correlated noise using SRK 2 integrator. Instead of
     inputting a correlation matrix from which multivariate Gaussian noise can be drawn,
     this function takes a matrix of monomer identities `mat` and correlation coefficients
     `rhos` to generate correlated noise directly.
     See correlations.generate_correlations_vars() for details.

    Parameters
    ----------
    identity_mat: (k, N) array-like
        kth row contains 1s, 0s, or -1s to assign monomers of type 1, type 0, or type -1 for the
        kth feature
    rhos : (k,) array-like
        Correlation coefficient associated with kth feature

    Notes
    -----
    The Langevin equation for the ith bead,
    .. math::
        \frac{dx(i, t)}{dt} = - \frac{k}{\xi} (x(i, t) - x(i+1, t))
                                  - \frac{k}{\xi} (x(i, t) - x(i-1, t))
                                  + \eta_i(t)
    is the same as above except the noise term is modified such that and each spatial component of
    :math:`\eta_i(t)` is a delta-correlated stationary Gaussian
    process with mean zero and
    :math:`\langle \eta_i(t) \eta_j(t') / \rangle = 2 \sqrt{D_i D_j} C_{ij} \delta(t-t')`,
    where :math:`C_{ij} \in [-1, 1]` is the correlation coefficient of monomer i and monomer j.
    """
    rtol = 1e-5
    # derived parameters
    L0 = L / (N - 1)  # length per bead
    bhat = np.sqrt(L0 * b)  # mean squared bond length of discrete gaussian chain
    Nhat = L / b  # number of Kuhn lengths in chain
    Dhat = D * N / Nhat  # diffusion coef of a discrete gaussian chain bead
    Deq = Deq * N / Nhat
    # set spring constant to be 3D/b^2 where D is the diffusion coefficient of the coldest bead
    k_over_xi = 3 * Deq / bhat ** 2
    #initial position
    # initial position, sqrt(3) since generating per-coordinate
    x0 = bhat / np.sqrt(3) * np.random.randn(N, 3)
    # for jit, we unroll ``x0 = np.cumsum(x0, axis=0)``
    for i in range(1, N):
        x0[i] = x0[i - 1] + x0[i]
    x = np.zeros(t_save.shape + x0.shape)
    if t_msd is not None:
        #at each msd save point, msds of N monomers + center of mass
        msds = np.zeros((len(t_msd), N+1))
        msd_inds = np.rint(t_msd / h)
        msd_i = 0
        if msd_start_time is None:
            msd_start_ind = 0
            msd_start_pos = x0.copy() #N x 3
        else:
            msd_start_ind = int(msd_start_time // h)
    save_i = 0
    save_inds = np.rint(t_save / h)
    if 0 == t_save[save_i]:
        x[0] = x0
        save_i += 1
    #standard deviation of noise 2Dh
    sigma = np.sqrt(2 * Dhat * h)
    ntimesteps = int(tmax // h) + 1 #including 0th time step

    for i in range(1, ntimesteps):
        #correlated noise matrix (N x 3)
        noise = generate_correlations_vars(mat, rhos, sigma)
        # force at position a
        Fa = f_elas_linear_rouse(x0, k_over_xi)
        x1 = x0 + h * Fa + noise
        # force at position b
        noise = generate_correlations_vars(mat, rhos, sigma)
        Fb = f_elas_linear_rouse(x1, k_over_xi)
        x0 = x0 + 0.5 * (Fa + Fb) * h + noise
        if t_msd is not None:
            if i == msd_start_ind:
                msd_start_pos = x0.copy()
            if i == msd_inds[msd_i]:
                #calculate msds, increment msd save index
                mean = np.zeros(x0[0].shape)
                for j in range(N):
                    diff = x0[j] - msd_start_pos[j]
                    mean += diff
                    msds[msd_i, j] = diff @ diff
                #center of mass msd
                diff = mean / N
                msds[msd_i, -1] = diff @ diff
                msd_i += 1
        if i == save_inds[save_i]:
            x[save_i] = x0
            save_i += 1
    return x, msds

@njit
def conf_identity_core_noise_srk2(N, L, b, D, h, tmax, t_save,
                             Aex, rx, ry, rz, mat, rhos, Deq=1):
    """ Same as `identity_core_noise_srk2` except within a confinement. Here, online MSD
    calculations have yet to be implemented.

    Parameters
    ----------
    Aex : float
        Strength of elliptical confinement
    rx : float
        semi-major x-axis of ellipsoid
    ry : float
        semi-major y-axis of ellipsoid
    rz : float
        semi-major z-axis of ellipsoid
    identity_mat: (k, N) array-like
        kth row contains 1s, 0s, or -1s to assign monomers of type 1, type 0, or type -1 for the
        kth feature
    rhos : (k,) array-like
        Correlation coefficient associated with kth feature

    Returns
    -------
    x : (Nt, N, 3) array_like of float
        The positions of the *N* monomers at each of the *Nt* time points.
    """
    rtol = 1e-5
    # derived parameters
    L0 = L / (N - 1)  # length per bead
    bhat = np.sqrt(L0 * b)  # mean squared bond length of discrete gaussian chain
    Nhat = L / b  # number of Kuhn lengths in chain
    Dhat = D * N / Nhat  # diffusion coef of a discrete gaussian chain bead
    Deq = Deq * N / Nhat
    # set spring constant to be 3D/b^2 where D is the diffusion coefficient of the coldest bead
    k_over_xi = 3 * Deq / bhat ** 2
    #initial position
    x0 = init_conf(N, bhat, rx, ry, rz)
    x = np.zeros(t_save.shape + x0.shape)
    save_i = 0
    if 0 == t_save[save_i]:
        x[0] = x0
        save_i += 1
    #standard deviation of noise 2Dh
    sigma = np.sqrt(2 * Dhat * h)
    ntimesteps = int(tmax // h) + 1 #including 0th time step

    for i in range(1, ntimesteps):
        #correlated noise matrix (N x 3)
        noise = generate_correlations_vars(mat, rhos, sigma)
        # force at position a
        Fa = f_conf_spring(x0, k_over_xi, Aex, rx, ry, rz)
        x1 = x0 + h * Fa + noise
        # force at position b
        noise = generate_correlations_vars(mat, rhos, sigma)
        Fb = f_conf_spring(x1, k_over_xi, Aex, rx, ry, rz)
        x0 = x0 + 0.5 * (Fa + Fb) * h + noise
        if np.abs(i*h - t_save[save_i]) < rtol * np.abs(t_save[save_i]):
            x[save_i] = x0
            save_i += 1
    return x



