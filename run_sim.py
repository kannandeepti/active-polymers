""" Script to run brownian dynamics simulations of active polymer."""
import numpy as np
from bd import *
import pandas as pd
from pathlib import Path
from correlations import *
from numba import njit
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import pdist, squareform
import time

#sun a sample Brownian dynamics simulation
def test_bd_sim(i, N=101, L=100, b=1, D=1):
    """ Test Brownian dynamics simulation for a random set of parameters
    A single simulation of a single chain with dt = 0.01 and 10^5 time steps
    finished in just 2 minutes.
    """
    h = 0.001
    tmax = 1.0e4
    #t = np.linspace(0, 1e5, int(1e7) + 1)
    #print(f'Simulation time step: {t[1] - t[0]}')
    #save 100 conformations
    Nhat = N / b
    # at times >> t_R, polymer should diffuse with Dg = D/N
    rouse_time = (Nhat ** 2) * (b ** 2) / (3 * np.pi ** 2 * D)
    print(f'Rouse time: {rouse_time}')
    msd_start = 0.0
    t_save = np.logspace(-2, 4, 50)
    t_msd = t_save
    X, msd = with_srk1(N, L, b, np.tile(D, N), h, tmax, t_save=t_save, t_msd=t_msd)
    return X, msd, t_save

def test_init_avoid(N=101, L=100, b=1, D=1, a=0.2):
    """ Test initialization of particles that do not overlap."""
    d = 2.0 ** (1 / 6) * a
    dsq = d ** 2
    rtol = 1e-5
    # derived parameters
    L0 = L / (N - 1)  # length per bead
    bhat = np.sqrt(L0 * b)
    x0 = init_avoid(N, bhat, dsq)
    distsq = pdist(x0, metric='sqeuclidean')
    assert(np.all(distsq >= dsq))
    return x0


def test_correlated_noise(N=101, L=100, b=1, D=1):
    """ Test Brownian dynamics simulation for a random set of parameters
    A single simulation of a single chain with dt = 0.01 and 10^5 time steps
    finished in just 2 minutes.
    """
    t = np.linspace(0, 1e2, int(1e4) + 1)
    h = np.diff(t)[0]
    tmax = t[-1]
    print(f'Simulation time step: {h}')
    #save 100 conformations
    t_save = np.linspace(0, 1e2, 100 + 1)
    C = np.eye(N) #correlation matrix of identity
    X = correlated_noise_srk2(N, L, b, np.tile(D, N), C, h, tmax, t_save)
    return X, t_save

def test_identity_correlated_noise(N=101, L=100, b=1, D=1):
    """ Test Brownian dynamics simulation for a random set of parameters
    A single simulation of a single chain with dt = 0.01 and 10^5 time steps
    finished in just 2 minutes.
    """
    t = np.linspace(0, 1e2, int(1e4) + 1)
    h = np.diff(t)[0]
    tmax = t[-1]
    print(f'Simulation time step: {h}')
    #save 100 conformations
    t_save = np.linspace(0, 1e2, 100 + 1)
    mat = np.zeros((1, N))
    rhos = np.zeros(1)
    X = identity_core_noise_srk2(N, L, b, np.tile(D, N), h, tmax, t_save, mat, rhos)
    return X, t_save

def test_bd_sim_confined(N=101, L=100, b=1, D=1):
    """ Test Brownian dynamics simulation for a random set of parameters
    A single simulation of a single chain with dt = 0.01 and 10^5 time steps
    finished in just 2 minutes.
    """
    dt = recommended_dt(N, L, b, D)
    print(f'Maximum recommended time step: {dt}')
    t = np.linspace(0, 1e5, int(1e7) + 1)
    print(f'Simulation time step: {t[1] - t[0]}')
    #save 100 conformations
    t_save = np.linspace(0, 1e5, 100 + 1)
    #radius of gyration is 16 --- confine it in smaller than this space
    X = jit_confined_srk1(N, L, b, np.tile(D, N), 1.0, 10.0, 10.0, 10.0, t, t_save=t_save)
    return X, t_save

def test_bd_clean(N=101, L=100, b=1, D=1.0, a=0.5, tmax=350.0, h=0.001):
    """ TODO: debug"""
    D = np.tile(D, N)
    t_save = np.linspace(0, tmax, 100 + 1)
    t_msd = np.logspace(-2, 2, 10)
    X, msd = scr_avoidNL_srk2(N, L, b, D, a, h, tmax, t_save=t_save,
                                    t_msd=t_msd)
    return X, msd, t_save

def run_correlated(i, N, L, b, D, filedir, length=10, 
                  s0=30, s0_prime=70, max=0.9):
    """ Run one simulation of a length L chain ith N beads, 
    Kuhn length b, and array of diffusion coefficients D. Use a Gaussian
    correlation matrix. """

    file = Path(filedir)/f'tape{i}.csv'
    try:
        file.parent.mkdir(parents=True)
    except:
        if file.parent.is_dir() is False:
            # if the parent directory does not exist and mkdir still failed, re-raise an exception
            raise
    tmax = 1e5
    h = 0.01
    print(f'Simulation time step: {h}')
    C = gaussian_correlation(N, length, s0, s0_prime, max)
    #save 100 conformations
    t_save = np.linspace(0, 1e5, 200 + 1)
    X = correlated_noise(N, L, b, D, C, h, tmax, t_save)
    dfs = []
    for i in range(X.shape[0]):
        df = pd.DataFrame(X[i, :, :])
        df['t'] = t_save[i]
        df['D'] = D #save diffusivities of beads
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df.set_index(['t'], inplace=True)
    df.to_csv(file)

def run_identity_correlated(i, N, L, b, D, filedir, mat, rhos, confined,
                           Aex=5.0, rx=5.0, ry=5.0, rz=5.0):
    """ Run one simulation of a length L chain ith N beads, 
    Kuhn length b, and array of diffusion coefficients D. Use a Gaussian
    correlation matrix. """

    file = Path(filedir)/f'tape{i}.csv'
    msd_file = Path(filedir)/f'msds{i}.csv'
    try:
        file.parent.mkdir(parents=True)
    except:
        if file.parent.is_dir() is False:
            # if the parent directory does not exist and mkdir still failed, re-raise an exception
            raise
    tmax = 1e5
    h = 0.001
    t_save = np.linspace(350.0, tmax, 1000 + 1)
    t_msd = np.logspace(-2, 5, 100)
    if confined:
        X = conf_identity_core_noise_srk2(N, L, b, D, h, tmax, t_save, Aex, rx,
                                          ry, rz, mat, rhos)
    else:
        X, msd = identity_core_noise_srk2(N, L, b, D, h, tmax, t_save, mat, rhos,
                                    t_msd=t_msd)
    dfs = []
    for i in range(X.shape[0]):
        df = pd.DataFrame(X[i, :, :])
        df['t'] = t_save[i]
        df['D'] = D #save diffusivities of beads
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df.set_index(['t'], inplace=True)
    df.to_csv(file)
    df = pd.DataFrame(msd)
    df['t_msd'] = t_msd
    df.to_csv(msd_file)

def run_msd(i, N, L, b, D, a=0.5, filedir, h=None, tmax=None):
    """ Run one simulation of a length L chain with N beads,
    Kuhn length b, and array of diffusion coefficients D."""
    file = Path(filedir)/f'tape{i}.csv'
    msd_file = Path(filedir)/f'msds{i}.csv'
    try:
        file.parent.mkdir(parents=True)
    except:
        if file.parent.is_dir() is False:
            # if the parent directory does not exist and mkdir still failed, re-raise an exception
            raise
    if h is None:
        h = 0.001
    if tmax is None:
        tmax = 1.0e4
    t_save = np.linspace(350.0, tmax, 100 + 1)
    t_msd = np.logspace(-2, 5, 100)
    X, msd = scr_avoidNL_srk2(N, L, b, D, a, h, tmax, t_save=t_save,
                              t_msd=t_msd)
    #X, msd = with_srk1(N, L, b, D, h, tmax, t_save=t_save, t_msd=t_msd)
    dfs = []
    for i in range(X.shape[0]):
        df = pd.DataFrame(X[i, :, :])
        df['t'] = t_save[i]
        df['D'] = D #save diffusivities of beads
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df.set_index(['t'], inplace=True)
    df.to_csv(file)
    df = pd.DataFrame(msd)
    df['t_msd'] = t_msd
    df.to_csv(msd_file)

def run(i, N, L, b, D, filedir, t=None, confined=False, 
        Aex=5.0, rx=5.0, ry=5.0, rz=5.0):
    """ Run one simulation of a length L chain with N beads,
    Kuhn length b, and array of diffusion coefficients D."""
    file = Path(filedir)/f'tape{i}.csv'
    try:
        file.parent.mkdir(parents=True)
    except:
        if file.parent.is_dir() is False:
            # if the parent directory does not exist and mkdir still failed, re-raise an exception
            raise
    if t is None:
        t = np.linspace(0, 1e5, int(1e7) + 1)
    print(f'Running simulation {filedir.name}')
    #save 100 conformations
    t_save = np.linspace(0, 1e5, 200 + 1)
    if confined:
        X = jit_confined_srk1(N, L, b, D, Aex, rx, ry, rz, t, t_save)
    else:
        X = with_srk1(N, L, b, D, t, t_save)
    dfs = []
    for i in range(X.shape[0]):
        df = pd.DataFrame(X[i, :, :])
        df['t'] = t_save[i]
        df['D'] = D #save diffusivities of beads
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df.set_index(['t'], inplace=True)
    df.to_csv(file)

if __name__ == '__main__':
    N = 101
    L = 100
    b = 1
    D = np.tile(1, N)
    #D = np.tile(0.25, N)
    #D[10:30] = 1.75
    #D[50:80] = 1.75
    filedir = Path('csvs/corr_msd1')
    """
    tic = time.perf_counter()
    X, t_save = test_identity_correlated_noise()
    toc = time.perf_counter()
    print(f'Ran simulation in {(toc - tic):0.4f}s')
    #define cosine wave of temperature activity with amplitude 5 times equilibrium temperature
    #period of wave is 25, max is 11, min is 1
    #B = 2 * np.pi / 25
    #D = 0.9 * np.cos(B * np.arange(0, N)) + 1
    #reduce time step by order of magnitude due to higher diffusivity
    #t = np.linspace(0, 1e5, int(1e8) + 1)
    #D[int(N//2)] = 10 #one hot bead
    #alternating hot and cold regions
    
    D[0:33] = 5.0
    D[66:] = 5.0
    mat = np.zeros((1, N))
    mat[0, 0:11] = 1.0
    mat[0, 22:33] = -1.0
    mat[0, 33:44] = 1.0
    mat[0, 55:66] = -1.0
    mat[0, 66:77] = 1.0
    mat[0, 88:] = - 1.0 
    rhos = np.array([0.5])
    file = Path(filedir)/'idmat.csv'
    try:
        file.parent.mkdir(parents=True)
    except:
        if file.parent.is_dir() is False:
            # if the parent directory does not exist and mkdir still failed, re-raise an exception
            raise
    #save identity matrix for future reference
    df = pd.DataFrame(mat)
    df.to_csv(file)
    print(f'Running simulation {filedir.name}')
    func = partial(run_identity_correlated, N=N, L=L, b=b, D=D,
                   filedir=filedir, mat=mat, rhos=rhos, confined=False)
    tic = time.perf_counter()
    pool_size = 16
    N = 96
    with Pool(pool_size) as p:
        result = p.map(func, np.arange(0, N))
    toc = time.perf_counter()
    print(f'Ran simulation in {(toc - tic):0.4f}s')
    """

