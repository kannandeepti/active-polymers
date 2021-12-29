""" Script to run brownian dynamics simulations of active polymer."""
import numpy as np
from bd import *
import pandas as pd
from pathlib import Path
from correlations import *
from numba import njit
from multiprocessing import Pool
from functools import partial
import time

#sun a sample Brownian dynamics simulation
def test_bd_sim(i, N=101, L=100, b=1, D=1):
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
    X = with_srk1(N, L, b, np.tile(D, N), t, t_save)
    return X, t_save

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

def test_bd_clean(N=11, L=10, b=1, D=1.0, a=1.0):
    """ TODO: debug"""
    D = np.tile(D, N)
    t = np.linspace(0, 1e2, int(1e4) + 1)
    t_save = np.linspace(0, 1e2, 100 + 1)
    X = jit_conf_avoid(N, L, b, D, a, 5.0, 8.0, 8.0, 8.0, t, t_save)
    return X, t_save

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
    print(f'Simulation time step: {t[1] - t[0]}')
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
    #D[10:30] = 10.0
    #D[50:80] = 10.0
    """
    filedir = Path('csvs/Deq1_anticorr30_70')
    func = partial(run_correlated, N=N, L=L, b=b, D=D, filedir=filedir, max=-0.9)
    tic = time.perf_counter()
    pool_size = 16
    N = 96
    with Pool(pool_size) as p:
        result = p.map(func, np.arange(0, N))
    toc = time.perf_counter()
    print(f'Ran simulation in {(toc - tic):0.4f}s')
    """

