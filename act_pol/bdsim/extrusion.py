""" Additional class to govern 1D dynamics of extruders on DNA. """

from numba import njit
import numpy as np
from .forces import *

def extrusion_parameters(p, sigma, mean_loop_size, b=1, D=1, g=2):
    """ Convert real parameters to simulation units. For a polymer with diffusion coefficient of D
    nm^2/s and a Kuhn length of b nm.

    Parameters
    ----------
    p : float
        probability of being in looped state (time_looped / (time_unlooped + time_looped))
    sigma : float
        ratio of rouse time to mean_time_looped
    mean_loop_size : int
        average loop size (aka processivity) in number of kuhn lengths
    g : int
        1 or 2 for 1 or 2 sided extrusion
    """
    #D = np.pi * Dapp**2 / (12 * b**2) #in nm^2 / second
    #time to equilibrate a loop of length N in seconds -- for loop of length 25 monomers,
    # this is 21 s
    rouse_time = (mean_loop_size ** 2) * (b ** 2) / (3 * np.pi ** 2 * D)
    print(f'Rouse time for loop of size {mean_loop_size}: {rouse_time}')
    mean_time_looped = rouse_time / sigma #in seconds
    mean_time_unlooped = ((1 - p)/p) * mean_time_looped
    vextrude = mean_loop_size / mean_time_looped
    vextrude /= g
    return mean_time_looped, mean_time_unlooped, vextrude

@njit
def loops_with_srk1(N, L, b, D, h, tmax, s1, s2, relk=1.0, lamb=0.0,
                    t_save=None, t_msd=None, msd_start_time=0.0, Deq=1):
    r"""
    Simulate a Rouse polymer with additional harmonic bonds coupling distinct regions along the
    chain.

    Parameters
    ----------
    K : (m, 2) array-like[int]
        Bead indices of m additional springs to add to the Rouse chain
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
    
    K = [[s1, s2]]
    msds = np.zeros((1, N + 1))
    mscd = np.zeros((1,))
    if t_msd is not None:
        print('Setting up msd calculation')
        # at each msd save point, msds of N monomers + center of mass
        msds = np.zeros((len(t_msd), N + 1))
        mscd = np.zeros((len(t_msd),))
        msd_i = 0
        msd_start_ind = int(msd_start_time / h)
        if msd_start_time == 0:
            mscd_start = x0[s1] - x0[s2] # (3,)
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
                mscd_start = x0[s1] - x0[s2]  # (3,)
            if i == msd_inds[msd_i]:
                #calculate mean squared change in distance
                newdist = x0[s1] - x0[s2]
                scd = newdist - mscd_start
                mscd[msd_i] = scd @ scd
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
    return x, msds, mscd

@njit
def loop_extrusion(N, L, b, D, h, tmax,
                   mean_time_looped, mean_time_unlooped, vextrude, s1, s2,
                   relk=1.0, lamb=0.0, t_save=None, t_msd=None, msd_start_time=0.0, Deq=1):
    r"""
    Simulate a Rouse polymer with a single extruder that can bind for mean time looped,
    stay unbound for mean_time_unlooped, and extrude loops at a rate 2*vextrude.

    Parameters
    ----------
    K : (m, 2) array-like[int]
        Bead indices of m additional springs to add to the Rouse chain
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

    if t_save is None:
        t_save = np.linspace(0.0, tmax, 101)

    msds = np.zeros((1, N + 1))
    mscd = np.zeros((1,))
    if t_msd is not None:
        print('Setting up msd calculation')
        # at each msd save point, msds of N monomers + center of mass
        msds = np.zeros((len(t_msd), N + 1))
        mscd = np.zeros((len(t_msd),))
        msd_i = 0
        msd_start_ind = int(msd_start_time / h)
        if msd_start_time == 0:
            mscd_start = x0[s1] - x0[s2] # (3,)
            msd_start_pos = x0.copy()  # N x 3
        msd_inds = np.rint((msd_start_time + t_msd) / h)

    x = np.zeros(t_save.shape + x0.shape)
    save_i = 0
    save_inds = np.rint(t_save / h)
    if 0 == t_save[save_i]:
        x[0] = x0
        save_i += 1
    ntimesteps = int(tmax // h) + 1  # including 0th time step
    # -1 or 1, p=1/2
    S = 2 * (np.random.rand(ntimesteps) < 0.5) - 1

    #begin in an unlooped state
    extruding = False
    K = [[s1, s2]]
    time_unlooped = np.random.exponential(scale=mean_time_unlooped)
    #time steps unlooped cannot be 0, need to be at least 1 so it starts to extrude
    timesteps_unlooped = max(np.rint(time_unlooped / h), 1)
    timesteps_looped = 0
    pextrude = vextrude * h

    for i in range(1, ntimesteps):
        """ UPDATE POSITIONS """
        if extruding:
            u = np.random.rand()
            if u < pextrude:
                #two sided extrusion with possibility of one sided at CTCF boundary
                if K[0][0] > s1:
                    K[0][0] -= 1
                if K[0][1] < s2:
                    K[0][1] += 1
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
        else:
            dW = np.random.randn(*x0.shape)
            # D = sigma^2/2 ==> sigma = np.sqrt(2*D)
            Fbrown = (np.sqrt(2 * Dhat / h) * (dW - S[i]).T).T
            # estimate for slope at interval start
            f = f_elas_linear_rouse(x0, k_over_xi)
            K1 = f + Fbrown
            Fbrown = (np.sqrt(2 * Dhat / h) * (dW + S[i]).T).T
            # estimate for slope at interval end
            x1 = x0 + h * K1
            f = f_elas_linear_rouse(x1, k_over_xi)
            K2 = f + Fbrown
            x0 = x0 + h * (K1 + K2) / 2

        """ END EXTRUSION """
        if extruding and i == timesteps_looped:
            extruding = False
            time_unlooped = np.random.exponential(scale=mean_time_unlooped)
            timesteps_unlooped  = max(np.rint(time_unlooped / h), 1) + i

        """ START EXTRUSION """
        if not extruding and i == timesteps_unlooped:
            extruding = True
            time_looped = np.random.exponential(scale=mean_time_looped)
            timesteps_looped = max(np.rint(time_looped / h), 1) + i
            #choose a random place to bind the cohesin
            sleft = np.random.randint(s1, s2)
            K = [[sleft, sleft]]

        """ OUTPUT MSDs, MSCDs, and SNAPSHOTS"""
        if t_msd is not None:
            if i == msd_start_ind:
                msd_start_pos = x0.copy()
                mscd_start = x0[s1] - x0[s2]  # (3,)
            if i == msd_inds[msd_i]:
                #calculate mean squared change in distance
                newdist = x0[s1] - x0[s2]
                scd = newdist - mscd_start
                mscd[msd_i] = scd @ scd
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
    return x, msds, mscd
