"""
Running the simulation code
---------------------------
Example script on how to use the act_pol.bdsim.bd module to run Brownian Dynamics simulations of
Rouse polymers.

"""
import numpy as np
from act_pol.bdsim.bd import *
from act_pol.bdsim.extrusion import *
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import time

def run(i, N, L, b, D, filedir, h=None, tmax=None):
    """ Run a single simulation of a Rouse polymer and save the output into
    filedir/tape{i}.csv.

    Parameters
    ----------
    i : int
        replicate of simulation
    N : float
        Number of monomers in the chain.
    L : float
        Length of chain.
    b : float
        Kuhn length of the chain (same units as *L*).
    D : (N,) array_like
        Diffusion coefficient of N monomers. (Units of ``length**2/time``).
    filedir: str or Path
        directory in which to save output of simulation
    h : float
        Time step to use for stepping the integrator. Same units as *D*.
    tmax : float
        Total simulation time.

    """
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
    #350.0 is slightly larger than the Rouse time for this particular chain
    # (see act_pol.analysis.rouse.terminal_relaxation())
    t_save = 350.0 * np.arange(0, np.floor(tmax / 350.0) + 1)
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

def run_identity_correlated(i, N, L, b, D, filedir, mat, rhos, h=None, tmax=None,
                            confined=False, Aex=5.0, rx=5.0, ry=5.0, rz=5.0):
    """ Run  a simulation of a Rouse polymer with monomer-dependent diffusion coefficients AND
    correlated noise. Correlations are generated by assigning each monomer an identity (1, 0,
    or -1) in `mat[k, :]` and demanding that type 1 monomers are correlated, type -1 monomers are
    correlated, but type 1 and type -1 monomers are anticorrelated with associated correlation
    coefficient provided in `rhos[k]`.

    Parameters
    ----------
    i : int
        replicate of simulation
    N : float
        Number of monomers in the chain.
    L : float
        Length of chain.
    b : float
        Kuhn length of the chain (same units as *L*).
    D : (N,) array_like
        Diffusion coefficient of N monomers. (Units of ``length**2/time``).
    filedir: str or Path
        directory in which to save output of simulation
    mat: (k, N) array-like
        kth row contains 1s, 0s, or -1s to assign monomers of type 1, type 0, or type -1 for the
        kth feature
    rhos : (k,) array-like
        Correlation coefficient associated with kth feature
    h : float
        Time step to use for stepping the integrator. Same units as *D*.
    tmax : float
        Total simulation time.
    confined : bool
        whether or not to put polymer in a confinement. If True, supply confinement params.
        Defaults to False.

    """

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
        tmax = 1e5
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

def simulation_for_fig2A():
    """ Simulation code used to produce the simulation snapshot in Figure 1A of
    https://doi.org/10.1101/2022.12.24.521789. """
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
    N = 96
    with Pool(pool_size) as p:
        result = p.map(func, np.arange(0, N))
    toc = time.perf_counter()
    print(f'Ran simulation in {(toc - tic):0.4f}s')



