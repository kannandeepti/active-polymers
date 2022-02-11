r"""
Simulations of Active Rouse polymers (based on wlcsim.bd.rouse)
---------------------------------------------------------------

We adapt the wlcsim.bd.rouse module developed by the Spakowitz Lab
to perform Brownian dynamics simulations of active Rouse polymers.
At equilibrium, all monomers of the chain experience the same magnitude
of stochastic kicks, i.e. are at uniform temperature. Out of equilibrum,
activity along the polymer (i.e. action of motor proteins and enzymes
that walk along the DNA) can induce stronger kicks along certain regions
of the chain, which can be modeled as a local increase in the effective
temperature of the monomer. More generally, activity at different regions
of the polymer could be correlated, for example, due to an enhancer regulating
two promoters, or a master transcription factor turning on multiple genes.

Here we explore a minimal model in which activity is encoded via correlations
in athermal flucutations experienced by monomers of a Rouse chain.
The act_pol.analyze module then allows one to compute the mean distance
between all pairs of monomers, as well as contact probabilities. The act_pol.msd module
can be used to study dynamics of such correlated active systems, such as the relaxation of the
monomer mean squared displacement (MSD) and center of mass MSD.

Contains routines for easily parameterizing Rouse polymer simulations and some
simple pre-built example simulators:
    1. **with_integrator**: Example Rouse simulator. Uses a discrete Rouse
       chain to approximate a semiflexible chain with parameters ``(N, L, b,
       D)``", where :math:`N` is the number of beads, :math:`L` is the physical
       length of the chain (see note below), :math:`b` is the Kuhn length, and
       :math:`D` is the diffusivity of a single Kuhn length of the polymer.
    2. **recommended_dt**: Given ``(N, L, b, D)``, compute the recommended
       maximum step size for a Rouse polymer simulation.
    3. **measured_D_to_rouse**: compute the diffusivity-per-Kuhn length
       (:math:`D`, required as input for simulations) from a log-log linear
       regression fit to an MSD with slope 1/2 (the usual way to measure
       diffusivity of a polymer bead).
.. note::
    While a Rouse chain does not technically have a well-defined length in the
    traditional sense of arc-length (due to its fractal nature), since our
    ultimate goal is the modeling of semiflexible polymers like DNA, we
    typically think of Rouse polymers as long length scale approximations of an
    underlying semiflexible chain with persistence length :math:`l_p = b / 2`.
    For an explicit derivation that all semiflexible chains can be
    well-approximated by the Rouse model on long enough length scales, see the
    third chapter of Doi & Edwards."
For details on Rouse polymer theory, see the book "Theory of Polymer Dynamics"
by Doi & Edwards.
Notes
-----
There are various ways to parameterize Rouse polymers. In this module, we use
the convention that :math:`N` (``N`` in the code) is the number of beads of a
Rouse polymer (contrary to e.g. Doi & Edwards, where ``N`` is the number of
Kuhn lengths in the polymer). We instead use :math:`\hat{N}` (``Nhat`` in the
code) for the number of Kuhn lengths in the polymer.
.. warning:
    This is different from the convention in :mod:`wlcsim.analytical.rouse`,
    where :math:`N` is the number of Kuhn lengths in the polymer, to match the
    usual conventions in e.g. Doi & Edwards.
:math:`D` is the "diffusivity of a Kuhn length", i.e. :math:`k_bT/\xi`, where
:math:`\xi` is the dynamic viscosity of the medium in question, as typically
found in the Langevin equation for a continuous Rouse polymer:
    .. math::
        \xi \frac{d}{dt} \vec{r}(n, t)
        = k \frac{d^2}{dn^2} \vec{r}(n, t) + f^{(B)}
where :math:`k = 3k_BT/b^2` is the "spring constant" for a Rouse polymer with
Kuhn length :math:`b`, and :math:`n` is the continuous coordinate specifying
our location along the chain.
In order to recreate the diffusion of this continuous chain with our discrete
chain of beads, it can be shown that we must choose the diffusion coefficient
of each bead should be set to :math:`\hat{D} = D (N/\hat{N})` (``Dhat`` in the
code). Since :math:`\xi` is in units of "viscosity per Kuhn length" and
:math:`(N/\hat{N})` is in units of "number of beads over number of Kuhn
lengths", this can be thought of as changing units from "viscosity per Kuhn
length" to "viscosity per bead".
Some books use :math:`b` for the mean (squared) bond length between beads in
the discrete gaussian chain. We instead use :math:`b` to be the real Kuhn
length of the polymer, so that the mean squared bond length :math:`\hat{b}^2`
is instead given by ``L0*b``, where ``L0 = L/(N-1)`` is the amount of polymer
"length" represented by the space between two beads.
The units chosen here make the most sense if you don't actually consider the
polymer to be a "real" Rouse polymer, but instead an e.g. semiflexible chain
with a real "length" whose long distance statistics are being captured by the
simulation.
.. figure:: rouse-regimes.svg
    The analytical result for the MSD of a discrete, (free-draining) Rouse
    polymer diffusing with no confinement. Because a discrete polymer is
    composed of finitely many beads, there exists a minimum time scale below
    which the system will appear to behave as though each bead was diffusing
    independently (:math:`\approx t^1`). At slightly longer time scales, the
    beads will begin to feel the effects of the springs connecting them
    together. Namely, as each bead attempts to diffuse in a given direction, it
    will always feel a restoring force from the rest of the polymer "dragging"
    it back, slowing the MSD to scale instead like :math:`t^{1/2}`. Finally at
    time scales longer than the "Rouse time" of the polymer, the entire polymer
    can be thought of as one large "effective" bead with a diffusivity
    :math:`N` times smaller than the individual beads, and will again exhibit
    regular diffusion (:math:`t^1` scaling).
    It is worthwhile to note that the case of the continuous Rouse polymer,
    there is no shortest time scale (because of its fractal nature, wherein at
    each time point, the Rouse chain's spatial configuration traces out a
    Brownian walk). In this case, the :math:`t^{1/2}` behavior simply stretches
    out to arbitrarily short time scales.
    For a more detailed discussion of the scaling behavior of Rouse MSD's, see
    the `measured_D_to_rouse` function.
To compare the results of simulations in this module to the
:mod:`rouse.analytical.rouse` module, you can use code like:
>>> plt.plot(t_save, sim_msd)
>>> real_msd = wlcsim.analytical.rouse.rouse_mid_msd(
...     t_save, b, Nhat, D, num_modes=int(N/2)
... )
>>> plt.plot(t_save, real_msd)
We can also plot simply lines in log-log space that describe the behavior
within each of the three regimes we describe above:
>>> plt.plot(t_save, 6*Dhat*t_save)
>>> plt.plot(t_save, 6*(Dhat/N)*t_save)  # or 6*(D/Nhat)*t_save
>>> # constant prefactor determined empirically...
>>> # no, using the value of 6*(Dhat/N)*t_R doesn't work...
>>> plt.plot(t_save, 1.9544100*b*np.sqrt(D)*np.sqrt(t_save))
where cutting off the number of modes corresponds to ensuring that the rouse
behavior only continues down to the length scale of a single "bead", thus
matching the simulation at arbitrarily short time scales. (Notice that we abide
by the warning above. Our ``Nhat`` is :mod:`wlcsim.analytical.rouse`'s ``N``).
"""
#from .runge_kutta import rk4_thermal_lena

from numba import njit
import numpy as np
from correlations import generate_correlations_vars
from forces import *
from neighbors import NeighborList

def recommended_dt(N, L, b, D):
    r"""
    Recommended "dt" for use with ``rouse*jit`` family of functions.
    Our srk1 scheme is accurate as long as :math:`\Delta t` is less than the
    transition time where the MSD goes from high k behavior (:math:`t^1`) to
    Rouse behavior (:math:`t^{1/2}`).  This is exactly the time required to
    diffuse a Kuhn length, so we just need to ensure that :math:`\Delta t <
    \frac{\hat{b}^2}{6\hat{D}}` in 3D. The "crossover" from fast-k to
    rouse-like behavior takes about one order of magnitude in time, and it was
    determined empirically that decreasing the time step beyond that point
    doesn't seem to make the MSD any more accurate.
    Note that it can be shown that *no* scheme can accurately reproduce the
    bead behavior using larger time steps than the time required to diffuse a
    distance of one Kuhn length.
    Notes
    -----
    Recall (doi & edwards, eq 4.25) that the first mode's relaxation time is
    :math:`\tau_1 = \frac{\xi N^2}{k \pi^2 }` (see Doi & Edwards, Eq. 4.25),
    and the :math:`p`\th mode is :math:`\tau_p = \tau_1/p^2` (this is the
    exponential falloff rate of the :math:`p`\th mode's correlation function).
    The accuracy of these values was also used when empirically determining the
    recommended ``dt``.
    """
    Nhat = L/b
    L0 = L/(N-1)
    Dhat = D*N/Nhat
    bhat = np.sqrt(L0*b)
    return (1/10)*bhat**2/(6*Dhat)


def measured_D_to_rouse(Dapp, d, N, bhat=None, regime='rouse'):
    r"""
    Get the Rouse monomer diffusion coefficient from the "apparent" anomolous
    diffusion coefficient `Dapp` extracted from subdiffusive dynamics of a locus
    on a polymer.

    In general, a discrete Rouse monomer MSD will have three regimes. On time
    scales long enough that the whole polymer diffuses as a large effective
    particle, the MSD is of the form :math:`6 D/\hat{N} t`, where, as is true
    throughout this module, :math:`D` is the diffusivity of a Kuhn length, and
    :math:`\hat{N}` is the number of Kuhn lengths in the polymer.
    This is true down to the full chain's relaxation time (the relaxation time
    of the 0th Rouse mode, also known as the "Rouse time") :math:`t_R = N^2 b^2
    / D`. For times shorter than this, the MSD will scale as :math:`t^{1/2}`.
    Imposing continuity of the MSD, this means that the MSD will behave as
    :math:`\kappa_0 D/\hat{N} (t_R t)^{1/2}`. While one could in principle
    compute :math:`\kappa_0` using :math:`\lim_{t\to 0}` of the analytical
    Rouse MSD result, it was more simple to just determine it empirically. We
    rewrite this MSD as :math:`\kappa b D^{-1/2} t^{1/2}`, and find by
    comparing to the analytical Rouse theory that :math:`\kappa =
    1.9544100(4)`.
    Eventually (at extremely short times), discrete polymers will eventually
    revert to a MSD that scales as :math:`t^1`. This cross-over time/length
    scale defines a "number of segments" :math:`\tilde{N}`, where the
    diffusivity of a segment of length :math:`\tilde{L} = L/(\tilde{N} - 1)`
    matches the time it takes stress to propagate a distance :math:`\tilde{L}`
    along the polymer. Said in other words, for polymer lengths smaller than
    :math:`\tilde{L}`, the diffusivity of the segment outruns the stress
    communication time between two neighboring segments.
    The diffusivity at these extremely short times will look like :math:`6 D
    (\tilde{N} / \hat{N}) t`. In order to simulate this behavior exactly, one
    can simply use a polymer with :math:`\tilde{N}` beads, then all three
    length scales of the MSD behavior will match.
    This three-regime MSD behavior can be very easily visualized in log-log
    space, where it is simply the continuous function defined by the following
    three lines. From shortest to longest time scales, :math:`\log{t}
    + \log{6D(\tilde{N}/\hat{N})}`, :math:`(1/2)\log{t} + \log{\kappa b
    D^{1/2}}`, and :math:`\log{t} + \log{6D\hat{N}}`. The lines
    (in log-log space), have slopes 1, 1/2, and 1, respectively, and their
    "offsets" (y-intercept terms with the log removed) are typically referred
    to as :math:`D_\text{app}`.
    The simulations in this module use the diffusivity of a single Kuhn length
    as input (i.e. plain old :math:`D` is expected as input) but typically,
    measurements of diffusing polymers are done below the Rouse time, and for
    long enough polymers (like a mammalian chromosome), it may be difficult to
    impossible to measure unconfined diffusion at time scales beyond the Rouse
    time. This means you often can't just look at the diffusivity of the whole
    polymer and multiply by :math:`\hat{N}` to get the diffusivity of a Kuhn
    length. And since there's not really a principled way in general to guess
    :math:`\tilde{N}`, you usually can't use the short-time behavior to get
    :math:`D` either, even supposing you manage to measure short enough times
    to see the cross-over back to :math:`t^1` scaling. This means that usually
    the only way one can extract :math:`D` is by measuring
    :math:`D_\text{app}`, since we can use the fact that
    :math:`D_\text{app} = \kappa b D^{1/2}` (the :math:`\kappa` value
    quoted above only works for 3-d motion. I haven't bothered to check if it
    scales linearly with the number of dimensions or like :math:`\sqrt{d}` for
    :math:`d`-dimensional motion, but this shouldn't be too hard if you are
    interested).
    In short, this function gives you :math:`D` given :math:`D_\text{app}`.
    Parameters
    ----------
    D_app : float
        The measured :math:`D_\text{app}` based on the y-intercept of the MSD
        in log-log space. This can be computed as
        ``np.exp(scipy.stats.linregress(np.log(t), np.log(msd))[1])``,
        assuming that :math:`t` is purely in the regime where the MSD scales
        like :math:`t^{1/2}`.
    Nhat : int
        number of beads in the polymer
    d : int
        number of dimensions of the diffusion (e.g. 3 in 3-d space)
    Returns
    -------
    D_rouse : float
        the diffusivity of a Kuhn length of the polymer
    """
    if d != 3:
        raise ValueError("We've only calculated kappa for d=3")
    kappa = 1.9544100 #this is just (12/pi)**(1/2)
    return (Dapp/(kappa*bhat))**2

@njit
def with_srk1(N, L, b, D, h, tmax, t_save=None, t_msd=None, msd_start_time=None, Deq=1):
    r"""
    Just-in-time compilable Rouse simulation use 1st order SRK scheme developed by
    A. J. Roberts: https://arxiv.org/abs/1210.0933 (modification of improved Euler
    scheme for SDEs).

    By in-lining the integrator and using numba, we are able to get over 1000x
    speedup. For example, a simulation with parameters ``N=101,L=100,b=1,D=1``
    takes about 2 min to run on my laptop when ``t=np.linspace(0,
    1e5, 1e7+1)``.
    It was determined empirically that adding an extra parameter ``t_save``
    controlling which time points to keep does not slow function down.

    Simulate a Rouse polymer made of N beads free in solution.

    Parameters
    ----------
    N : float
        Number of beads in the chain.
    L : float
        Length of chain.
    b : float
        Kuhn length of the chain (same units as *L*).
    D : (N,) array_like
        Diffusion coefficient of N monomers. (Units of ``length**2/time``). In order to
        compute *D* from a single-locus MSD, use `measured_D_to_rouse`.
        To recapitulate Rouse polymer, set D to be the same value for all monomers.
    h : float
        Time step to use for stepping the integrator. Same units as *D*. Use
        `~.bd.recommended_dt` to compute the optimal *dt*.
    tmax : float
        Total simulation time.
    t_save : (Nt,) array-like
        Time points at which to save output of simulation.
    t_msd : (Nm,) array-like
        Lag times at which to compute MSDs of each monomer + center of mass.
    msd_start_time : float
        Starting time point from which to compute MSD.
    x0 : (N, 3) array_like of float, optional
        The initial locations of each bead. If not specified, defaults to
        initializing from the free-draining equilibrium, with the first bead at
        the origin.

    Returns
    -------
    (Nt, N, 3) array_like of float
        The positions of the *N* beads at each of the *Nt* time points.
    Notes
    -----
    The polymer beads can be described by the discrete Rouse equations
    .. math::
        \xi \frac{dx(i, t)}{dt} = - k (x(i, t) - x(i+1, t))
                                  - k (x(i, t) - x(i-1, t))
                                  + R(t)
    where :math:`\xi = k_BT/D`, :math:`k = 3k_BT/b^2`, :math:`b` is the Kuhn
    length of the polymer, :math:`D` is the self-diffusion coefficient of a
    bead, and :math:`R(t)/\xi` is a delta-correlated stationary Gaussian
    process with mean zero and :math:`\langle R(t) R(t')/\xi^2 \rangle =
    2DI\delta(t-t')`.
    Notice that in practice, :math:`k/\xi = 3D/b^2`, so we do not need to
    include mass units (i.e. there's no dependence on :math:`k_BT`).
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

    if t_msd is not None:
        print('Setting up msd calculation')
        #at each msd save point, msds of N monomers + center of mass
        msds = np.zeros((len(t_msd), N+1))
        msd_inds = np.rint(t_msd / h)
        msd_i = 0
        if msd_start_time is None:
            msd_start_ind = 0
            msd_start_pos = x0.copy() #N x 3
        else:
            msd_start_ind = int(msd_start_time // h)

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
        #loop over N beads
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
def jit_confined_srk1(N, L, b, D, Aex, rx, ry, rz, t, t_save=None, Deq=1):
    """
    Add an elliptical confinement.

    Energy is like cubed of distance
    outside of ellipsoid, pointing normally back in. times some factor Aex
    controlling its strength.

    TODO: change input t to step size h and test.

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
    # initial position
    x0 = np.zeros((N, 3))
    # x0 = np.cumsum(x0, axis=0)
    for i in range(1, N):
        # 1/sqrt(3) since generating per-coordinate
        x0[i] = x0[i-1] + bhat/np.sqrt(3)*np.random.randn(3)
        while x0[i, 0]**2/rx**2 + x0[i, 1]**2/ry**2 + x0[i, 2]**2/rz**2 > 1:
            x0[i] = x0[i-1] + bhat/np.sqrt(3)*np.random.randn(3)
    if t_save is None:
        t_save = t
    x = np.zeros(t_save.shape + x0.shape)
    dts = np.diff(t)
    # -1 or 1, p=1/2
    S = 2*(np.random.rand(len(t)) < 0.5) - 1
    save_i = 0
    if 0 == t_save[save_i]:
        x[0] = x0
        save_i += 1
    # at each step i, we use data (x,t)[i-1] to create (x,t)[i]
    # in order to make it easy to pull into a new functin later, we'll call
    # t[i-1] "t0", old x (x[i-1]) "x0", and t[i]-t[i-1] "h".
    for i in range(1, len(t)):
        h = dts[i-1]
        dW = np.random.randn(*x0.shape)
        # D = sigma^2/2 ==> sigma = np.sqrt(2*D)
        #Fbrown = np.sqrt(2*Dhat/h)*(dW - S[i])
        Fbrown = (np.sqrt(2 * Dhat / h) * (dW - S[i]).T).T
        # estimate for slope at interval start
        f = np.zeros(x0.shape)
        j = 0
        conf = x0[j, 0]**2/rx**2 + x0[j, 1]**2/ry**2 + x0[j, 2]**2/rz**2
        if conf > 1:
            conf_u = np.array([
                -x0[j, 0]/rx**2, -x0[j, 1]/ry**2, -x0[j, 2]/rz**2
            ])
            conf_u = conf_u/np.linalg.norm(conf_u)
            f[j] += Aex*conf_u*np.power(np.sqrt(conf) - 1, 3)
        for j in range(1, N):
            conf = x0[j, 0]**2/rx**2 + x0[j, 1]**2/ry**2 + x0[j, 2]**2/rz**2
            if conf > 1:
                conf_u = np.array([
                    -x0[j, 0]/rx**2, -x0[j, 1]/ry**2, -x0[j, 2]/rz**2
                ])
                conf_u = conf_u/np.linalg.norm(conf_u)
                f[j] += Aex*conf_u*np.power(np.sqrt(conf) - 1, 3)
            for n in range(3):
                f[j, n] += -k_over_xi*(x0[j, n] - x0[j-1, n])
                f[j-1, n] += -k_over_xi*(x0[j-1, n] - x0[j, n])
        K1 = f + Fbrown
        #Fbrown = np.sqrt(2*Dhat/h)*(dW + S[i])
        Fbrown = (np.sqrt(2 * Dhat / h) * (dW + S[i]).T).T
        # estimate for slope at interval end
        x1 = x0 + h*K1
        f = np.zeros(x1.shape)
        j = 0
        conf = x1[j, 0]**2/rx**2 + x1[j, 1]**2/ry**2 + x1[j, 2]**2/rz**2
        if conf > 1:
            conf_u = np.array([
                -x1[j, 0]/rx**2, -x1[j, 1]/ry**2, -x1[j, 2]/rz**2
            ])
            conf_u = conf_u/np.linalg.norm(conf_u)
            f[j] += Aex*conf_u*np.power(np.sqrt(conf) - 1, 3)
        for j in range(1, N):
            conf = x1[j, 0]**2/rx**2 + x1[j, 1]**2/ry**2 + x1[j, 2]**2/rz**2
            if conf > 1:
                conf_u = np.array([
                    -x1[j, 0]/rx**2, -x1[j, 1]/ry**2, -x1[j, 2]/rz**2
                ])
                conf_u = conf_u/np.linalg.norm(conf_u)
                f[j] += Aex*conf_u*np.power(np.sqrt(conf) - 1, 3)
            for n in range(3):
                f[j, n] += -k_over_xi*(x1[j, n] - x1[j-1, n])
                f[j-1, n] += -k_over_xi*(x1[j-1, n] - x1[j, n])
        K2 = f + Fbrown
        x0 = x0 + h * (K1 + K2)/2
        if np.abs(t[i] - t_save[save_i]) < rtol*np.abs(t_save[save_i]):
            x[save_i] = x0
            save_i += 1
    return x

@njit
def correlated_noise(N, L, b, D, C, h, tmax, t_save, Deq=1):
    """ Add correlation matrix for noise. Draw multivariate noise by computing Cholesky
    decomposition of correlation matrix and multiplying the lower triangle matrix by the (N x 3)
    matrix of standard normal random variables. This still uses the SRK1 scheme from above.

    Parameters
    ----------
    D : array-like (N,)
        diffusivity of each monomer (> 0)
    C : array-like (N, N)
        correlation matrix (values from -1 to 1), must be symmetric
    h : float
        time step
    tmax : float
        maximum simulation time
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
    # initial position, sqrt(3) since generating per-coordinate
    x0 = bhat / np.sqrt(3) * np.random.randn(N, 3)
    # for jit, we unroll ``x0 = np.cumsum(x0, axis=0)``
    for i in range(1, N):
        x0[i] = x0[i - 1] + x0[i]
    x = np.zeros(t_save.shape + x0.shape)
    save_i = 0
    if 0 == t_save[save_i]:
        x[0] = x0
        save_i += 1
    #standard deviation of noise
    sigma = np.sqrt(2 * Dhat / h)
    #covariance matrix for noise: diag(std) @ C @ diag(std), i.e. cov(x,y) = corr(x,y)*sigma_x*sigma_y
    cov = C * np.outer(sigma, sigma)
    cky = np.linalg.cholesky(cov) #N x N matrix
    # at each step i, we use data (x,t)[i-1] to create (x,t)[i]
    # in order to make it easy to pull into a new functin later, we'll call
    # t[i-1] "t0", old x (x[i-1]) "x0", and t[i]-t[i-1] "h".
    ntimesteps = int(tmax // h) + 1 #including 0th time step
    # -1 or 1, p=1/2
    S = 2 * (np.random.rand(ntimesteps) < 0.5) - 1
    for i in range(1, ntimesteps):
        #correlated noise matrix (N x 3)
        dW = np.random.randn(*x0.shape)
        # D = sigma^2/2 ==> sigma = np.sqrt(2*D)
        Fbrown = np.dot(cky, dW - S[i])
        # estimate for slope at interval start
        f = np.zeros(x0.shape)
        # loop over N beads
        for j in range(1, N):
            # loop over 3 dimensions
            for n in range(3):
                f[j, n] += -k_over_xi * (x0[j, n] - x0[j - 1, n])
                f[j - 1, n] += -k_over_xi * (x0[j - 1, n] - x0[j, n])
        K1 = f + Fbrown
        Fbrown = np.dot(cky, dW + S[i])
        # estimate for slope at interval end
        x1 = x0 + h * K1
        f = np.zeros(x0.shape)
        for j in range(1, N):
            for n in range(3):
                f[j, n] += -k_over_xi * (x1[j, n] - x1[j - 1, n])
                f[j - 1, n] += -k_over_xi * (x1[j - 1, n] - x1[j, n])
        K2 = f + Fbrown
        x0 = x0 + h * (K1 + K2) / 2
        if np.abs(i*h - t_save[save_i]) < rtol * np.abs(t_save[save_i]):
            x[save_i] = x0
            save_i += 1
    return x

@njit
def correlated_noise_srk2(N, L, b, D, C, h, tmax, t_save, Deq=1):
    """ Same as correlated_noise function except with a 2nd order
    SRK scheme as opposed to the Roberts first order scheme."""
    rtol = 1e-5
    # derived parameters
    L0 = L / (N - 1)  # length per bead
    bhat = np.sqrt(L0 * b)  # mean squared bond length of discrete gaussian chain
    Nhat = L / b  # number of Kuhn lengths in chain
    Dhat = D * N / Nhat  # diffusion coef of a discrete gaussian chain bead
    Deq = Deq * N / Nhat
    # set spring constant to be 3D/b^2 where D is the diffusion coefficient of the coldest bead
    k_over_xi = 3 * Deq / bhat ** 2
    # initial position, sqrt(3) since generating per-coordinate
    x0 = bhat / np.sqrt(3) * np.random.randn(N, 3)
    # for jit, we unroll ``x0 = np.cumsum(x0, axis=0)``
    for i in range(1, N):
        x0[i] = x0[i - 1] + x0[i]
    x = np.zeros(t_save.shape + x0.shape)
    save_i = 0
    if 0 == t_save[save_i]:
        x[0] = x0
        save_i += 1
    #standard deviation of noise 2Dh
    sigma = np.sqrt(2 * Dhat * h)
    #covariance matrix for noise: diag(std) @ C @ diag(std), i.e. cov(x,y) = corr(x,y)*sigma_x*sigma_y
    cov = C * np.outer(sigma, sigma)
    cky = np.linalg.cholesky(cov) #N x N matrix
    ntimesteps = int(tmax // h) + 1 #including 0th time step

    for i in range(1, ntimesteps):
        #correlated noise matrix (N x 3)
        noise = np.dot(cky, np.random.randn(*x0.shape))
        # force at position a
        Fa = f_elas_linear_rouse(x0, k_over_xi)
        x1 = x0 + h * Fa + noise
        # force at position b
        noise = np.dot(cky, np.random.randn(*x0.shape))
        Fb = f_elas_linear_rouse(x1, k_over_xi)
        x0 = x0 + 0.5 * (Fa + Fb) * h + noise
        if np.abs(i*h - t_save[save_i]) < rtol * np.abs(t_save[save_i]):
            x[save_i] = x0
            save_i += 1
    return x

@njit
def identity_core_noise_srk2(N, L, b, D, h, tmax, t_save, mat, rhos,
                             t_msd=None, msd_start_time=None, Deq=1):
    """ BD simulation with correlated noise using SRK 2 integrator. Instead of
     inputting a correlation matrix directly, this function instead takes a matrix of monomer
     identities `mat` and correlation coefficients `rhos` to generate correlated noise directly.
     See correlations.generate_correlated_noise_vars() for details.

    Parameters
    ----------
    identity_mat: (k, N) array-like
        kth row contains 1s, 0s, or -1s to assign monomers of type 1, type 0, or type -1 for the
        kth feature
    rhos : (k,) array-like
        Correlation coefficient associated with kth feature
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
    """ Same as `identity_core_noise_srk2` except within a confinement.

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

@njit
def scr_avoidNL_srk2(N, L, b, D, a, h, tmax, t_save=None, Deq=1):
    """ Simulate a self-avoiding Rouse polymer via a soft core (harmonic)
    repulsive potential. With h=0.001, this works! Roughly 10 pairs of monomers
    may be overlapping at any given time but overlaps decay in a matter of a few time steps.

    Parameters
    ----------
    a : float
        Radius of monomer

    TODO: use @jit(parallel=True) to parallelize neighborlist construction and
    force computation.
    """
    d = 2.0 * a #minimum of interaction potential (diameter)
    dsq = d ** 2
    rtol = 1e-5
    # derived parameters
    L0 = L / (N - 1)  # length per bead
    bhat = np.sqrt(L0 * b)  # mean squared bond length of discrete gaussian chain
    Nhat = L / b  # number of Kuhn lengths in chain
    Rg = Nhat * b ** 2 / 6 #estimated radius of gyration of the chain
    box_size = 3 * Rg #length of edge of simulation domain
    Dhat = D * N / Nhat  # diffusion coef of a discrete gaussian chain bead
    Deq = Deq * N / Nhat
    # set spring constant to be 3D/b^2 where D is the equilibrium diffusion coefficient
    k_over_xi = 3 * Deq / bhat ** 2
    ks_over_xi = 100.0 * Deq / a ** 2 #copying parameter chosen by Weber & Fry
    #initial position
    x0 = bhat / np.sqrt(3) * np.random.randn(N, 3)
    for i in range(1, N):
        x0[i] = x0[i - 1] + x0[i]
    #build neighbor list
    neighlist = NeighborList(d, 0.5*d, box_size)
    cl, nl = neighlist.updateNL(x0)

    x = np.zeros(t_save.shape + x0.shape)
    # setup for saving only requested time points
    save_i = 0
    if 0 == t_save[save_i]:
        x[0] = x0
        save_i += 1
    # standard deviation of noise 2Dh
    sigma = np.sqrt(2 * Dhat * h)
    ntimesteps = int(tmax // h) + 1  # including 0th time step

    # at each step i, we use data (x,t)[i-1] to create (x,t)[i]
    # in order to make it easy to pull into a new function later, we'll call
    # t[i-1] "t0", old x (x[i-1]) "x0", and t[i]-t[i-1] "h".
    for i in range(1, ntimesteps):
        if neighlist.checkNL(x0):
            cl, nl = neighlist.updateNL(x0)
        # noise matrix (N x 3)
        noise = (sigma * np.random.randn(*x0.shape).T).T
        # force at position a
        Fa = f_spring_scr_NL(x0, k_over_xi, ks_over_xi, a, dsq, cl, nl, box_size)
        x1 = x0 + h * Fa + noise
        # force at position b
        noise = (sigma * np.random.randn(*x0.shape).T).T
        Fb = f_spring_scr_NL(x1, k_over_xi, ks_over_xi, a, dsq, cl, nl, box_size)
        x0 = x0 + 0.5 * (Fa + Fb) * h + noise
        if np.abs(i * h - t_save[save_i]) < rtol * np.abs(t_save[save_i]):
            x[save_i] = x0
            save_i += 1
    return x

@njit
def jit_avoidNL_srk2(N, L, b, D, a, h, tmax, t_save=None, Deq=1):
    """ Simulate a self-avoiding Rouse polymer via a repulsive WCA potential.

    TODO: debug. Even for h=2e-5 this has numerical problems...
    """
    d = 2.0 ** (1 / 6) * a
    dsq = d ** 2
    rtol = 1e-5
    # derived parameters
    L0 = L / (N - 1)  # length per bead
    bhat = np.sqrt(L0 * b)  # mean squared bond length of discrete gaussian chain
    Nhat = L / b  # number of Kuhn lengths in chain
    Rg = Nhat * b ** 2 / 6 #estimated radius of gyration of the chain
    box_size = 3 * Rg #length of edge of simulation domain
    Dhat = D * N / Nhat  # diffusion coef of a discrete gaussian chain bead
    Deq = Deq * N / Nhat
    # set spring constant to be 3D/b^2 where D is the equilibrium diffusion coefficient
    k_over_xi = 3 * Deq / bhat ** 2
    #initial position
    x0 = bhat / np.sqrt(3) * np.random.randn(N, 3)
    for i in range(1, N):
        x0[i] = x0[i - 1] + x0[i]
    #build neighbor list
    neighlist = NeighborList(d, 0.1*d, box_size)
    cl, nl = neighlist.updateNL(x0)
    # keep track of how often NL is updated
    nlupdates = [0]

    x = np.zeros(t_save.shape + x0.shape)
    # setup for saving only requested time points
    save_i = 0
    if 0 == t_save[save_i]:
        x[0] = x0
        save_i += 1
    # standard deviation of noise 2Dh
    sigma = np.sqrt(2 * Dhat * h)
    ntimesteps = int(tmax // h) + 1  # including 0th time step

    # at each step i, we use data (x,t)[i-1] to create (x,t)[i]
    # in order to make it easy to pull into a new function later, we'll call
    # t[i-1] "t0", old x (x[i-1]) "x0", and t[i]-t[i-1] "h".
    for i in range(1, ntimesteps):
        if neighlist.checkNL(x0):
            cl, nl = neighlist.updateNL(x0)
            nlupdates.append(i*h)
        # noise matrix (N x 3)
        noise = (sigma * np.random.randn(*x0.shape).T).T
        # force at position a
        #Fa = f_spring_avoid(x0, k_over_xi, a, dsq)
        Fa = f_spring_avoid_NL(x0, k_over_xi, a, dsq, cl, nl, box_size)
        x1 = x0 + h * Fa + noise
        # force at position b
        noise = (sigma * np.random.randn(*x0.shape).T).T
        Fb = f_spring_avoid_NL(x1, k_over_xi, a, dsq, cl, nl, box_size)
        #Fb = f_spring_avoid(x1, k_over_xi, a, dsq)
        x0 = x0 + 0.5 * (Fa + Fb) * h + noise
        if np.abs(i * h - t_save[save_i]) < rtol * np.abs(t_save[save_i]):
            x[save_i] = x0
            save_i += 1
    return x, nlupdates

@njit
def jit_conf_avoid(N, L, b, D, a, Aex, rx, ry, rz, t, t_save=None, Deq=1):
    """ Clean version of BD code with forces and initialization in separate functions.
    NOTE: this function does not yet work with srk_1 scheme. TODO: debug.
    """
    d = 2.0 ** (1 / 6) * a
    dsq = d ** 2
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
    x0 = init_conf_avoid(N, bhat, rx, ry, rz, dsq)
    # pre-alloc output
    if t_save is None:
        t_save = t
    x = np.zeros(t_save.shape + x0.shape)
    # setup for saving only requested time points
    save_i = 0
    if 0 == t_save[save_i]:
        x[0] = x0
        save_i += 1
    # at each step i, we use data (x,t)[i-1] to create (x,t)[i]
    # in order to make it easy to pull into a new function later, we'll call
    # t[i-1] "t0", old x (x[i-1]) "x0", and t[i]-t[i-1] "h".
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        dW = np.random.randn(*x0.shape)
        # -1 or 1, p=1/2
        S = 2 * (np.random.rand() < 0.5) - 1
        # D = sigma^2/2 ==> sigma = np.sqrt(2*D)
        Fbrown = (np.sqrt(2 * Dhat / h) * (dW - S).T).T
        # estimate for slope at interval start
        K1 = f_combined(x0, k_over_xi, Aex, rx, ry, rz, a, dsq) + Fbrown
        Fbrown = (np.sqrt(2 * Dhat / h) * (dW + S).T).T
        x1 = x0 + h * K1
        # estimate for slope at interval end
        K2 = f_combined(x0, k_over_xi, Aex, rx, ry, rz, a, dsq) + Fbrown
        # average the slope estimates
        x0 = x0 + h * (K1 + K2) / 2
        if np.abs(t[i] - t_save[save_i]) < rtol * np.abs(t_save[save_i]):
            x[save_i] = x0
            save_i += 1
    return x



