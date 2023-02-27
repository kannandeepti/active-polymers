""" Script to run brownian dynamics simulations of active polymer."""
import numpy as np
from act_pol.bdsim.bd import *
from act_pol.bdsim.extrusion import *
import pandas as pd
from pathlib import Path
from act_pol.bdsim.correlations import *
from numba import njit
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import pdist, squareform
import time

def test_bd_sim(i, N=101, L=100, b=1, D=1):
    """ Test Brownian dynamics simulation of a simple Rouse polymer.
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

def test_bd_loops(N=101, L=100, b=1, D=1):
    """ Test Rouse simulation with additional Hookean springs connecting distinct monomers."""
    h = 0.001
    tmax = 350.0
    K = np.array([[25, 75]])
    t_save = np.linspace(0, tmax, 100 + 1)
    t_msd = np.logspace(-2, 2, 10)
    X, msd = loops_with_srk1(N, L, b, np.tile(D, N), h, tmax, K,
                             t_save=t_save, t_msd=t_msd, msd_start_time=1.0)
    return X, msd, t_save

def test_bd_extrusion(N=101, L=100, b=1, D=10, s1=25, s2=75):
    """ Test Rouse simulation with a single loop extruding factor."""
    print(f'Recommended dt: {recommended_dt(N, L, b, D)}')
    p = 1.0 #always looped
    sigma = 1.0
    mean_loop_size = 20
    mean_time_looped, mean_time_unlooped, vextrude = extrusion_parameters(p, sigma,
                                                                          mean_loop_size, b=b, D=D)
    h = 0.001
    tmax = 100.0
    t_save = np.linspace(0, tmax, 100 + 1)
    t_msd = np.logspace(-2, 2, 10)
    X, msd, mscd = loop_extrusion(N, L, b, np.tile(D, N), h, tmax,
                   mean_time_looped, mean_time_unlooped, vextrude, s1, s2,
                   t_save=t_save, t_msd=t_msd, msd_start_time=1.0)
    return X, msd, mscd


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
    """ Test BD simulations with inputted correlation matrix.
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
    """ Test BD simulation with correlated noise generated from a matrix of monomer identities
    and correlation coefficients.
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
    """ Test BD simulation with confinement.
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

def test_corr_D(N=101, L=100, b=1, tmax=350.0, h=0.001):
    """ TODO: debug"""
    D = np.ones(N)
    rhos = np.zeros((1, N))
    rhos[0, 0:33] = 0.5
    rhos[0, 66:] = -0.5
    t_save = np.linspace(0, tmax, 100 + 1)
    t_msd = np.logspace(-2, 2, 10)
    X, msd = correlated_amplitudes_srk2(N, L, b, rhos, D, h, tmax, t_save=t_save,
                                    t_msd=t_msd)
    return X, msd, t_save

def run_loops(i, N, L, b, D, filedir, s1, s2, relk=0.01, lamb=0.1,
             desired_tmax=1e3, DT=35.0, msd_start_time=0.0):
    """ Run a single simulation of a Rouse polymer with loops."""
    file = Path(filedir) / f'tape{i}.csv'
    msd_file = Path(filedir) / f'msds{i}.csv'
    mscd_file = Path(filedir) / f'mscd{i}.csv'
    try:
        file.parent.mkdir(parents=True)
    except:
        if file.parent.is_dir() is False:
            # if the parent directory does not exist and mkdir still failed, re-raise an exception
            raise
    h = 0.001
    tmax = desired_tmax + msd_start_time + h
    t_save = DT * np.arange(0, np.floor(tmax / DT) + 1)
    t_msd = np.logspace(-1, 3, 100)
    X, msd, mscd = loops_with_srk1(N, L, b, D, h, tmax, s1, s2, relk=relk, lamb=lamb,
                             t_save=t_save, t_msd=t_msd,
                             msd_start_time=msd_start_time, Deq=np.mean(D))
    dfs = []
    for i in range(X.shape[0]):
        df = pd.DataFrame(X[i, :, :])
        df['t'] = t_save[i]
        df['D'] = D  # save diffusivities of beads
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df.set_index(['t'], inplace=True)
    df.to_csv(file)
    df = pd.DataFrame(msd)
    df['t_msd'] = t_msd
    df.to_csv(msd_file)
    df = pd.DataFrame(mscd)
    df['t_msd'] = t_msd
    df.to_csv(mscd_file)

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
    t_save = np.linspace(350.0, tmax, 100 + 1)
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

def run_correlated_diffusion(i, N, L, b, rhos, meanD, stdD, filedir,
                             confined=False, Aex=5.0, rx=5.0, ry=5.0, rz=5.0):
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

    h = 0.001
    msd_start_time = 0.0
    tmax = 1e5 + msd_start_time + h
    t_save = 350.0 * np.arange(0, np.floor(tmax / 350.0) + 1)
    t_msd = np.logspace(-2, 5, 100)
    X, D, msd = correlated_diffusion_srk2(N, L, b, rhos, meanD, stdD, h, tmax, t_save=t_save,
                                          t_msd=t_msd)
    dfs = []
    for i in range(X.shape[0]):
        df = pd.DataFrame(X[i, :, :])
        df['t'] = t_save[i]
        df['D'] = D[i] #save diffusivities of beads
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df.set_index(['t'], inplace=True)
    df.to_csv(file)
    df = pd.DataFrame(msd)
    df['t_msd'] = t_msd
    df.to_csv(msd_file)

def run_correlated_amplitudes(i, N, L, b, rhos, D, filedir,
                             confined=False, Aex=5.0, rx=5.0, ry=5.0, rz=5.0):
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

    h = 0.001
    msd_start_time = 0.0
    tmax = 1e5 + msd_start_time + h
    t_save = 350.0 * np.arange(0, np.floor(tmax / 350.0) + 1)
    t_msd = np.logspace(-2, 5, 100)
    X, msd = correlated_amplitudes_srk2(N, L, b, rhos, D, h, tmax, t_save=t_save,
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

def run_msd(i, N, L, b, D, a, filedir, h=None, tmax=None, mat=None, rhos=None):
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
    msd_start_time = 1.0
    if h is None:
        h = 0.001
    if tmax is None:
        tmax = 1.0e4 + msd_start_time + h
    t_save = 350.0 * np.arange(0, np.floor(tmax / 350.0) + 1)
    t_msd = np.logspace(-2, 4, 86)
    X, msd = scr_avoidNL_srk2(N, L, b, D, a, h, tmax, t_save=t_save,
                              t_msd=t_msd,  mat=mat, rhos=rhos)
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

def run(i, N, L, b, D, filedir, h=None, tmax=None, confined=False, 
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
    print(f'Running simulation {filedir.name}')
    if h is None:
        h = 0.001
    if tmax is None:
        tmax = 1.0e4 + h
    t_save = 350.0 * np.arange(0, np.floor(tmax / 350.0) + 1)
    if confined:
        X = jit_confined_srk1(N, L, b, D, h, Aex, rx, ry, rz, t, t_save)
    else:
        X, _ = with_srk1(N, L, b, D, h, tmax, t_save)
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
    B = 2 * np.pi / 25
    D = 0.5 * np.cos(B * np.arange(0, N)) + 1
    filedir = Path('csvs/cos_3x')
    print(f'Running simulation {filedir.name}')
    func = partial(run, N=N, L=L, b=b, D=D, h=0.01,
                   filedir=filedir)
    tic = time.perf_counter()
    pool_size = 16
    N = 480
    with Pool(pool_size) as p:
        result = p.map(func, np.arange(0, N))
    toc = time.perf_counter()
    print(f'Ran simulation in {(toc - tic):0.4f}s')



