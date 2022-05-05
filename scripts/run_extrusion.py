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
import sys


def run_extrusion(i, N, L, b, D, p, sigma, mean_loop_size, filedir,
                  s1=25, s2=75, h=0.001, desired_tmax=1e4, DT=35.0):
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
    msd_start_time = 0.0
    tmax = desired_tmax + msd_start_time + h
    t_save = DT * np.arange(0, np.floor(tmax / DT) + 1)
    t_msd = np.logspace(-2, 4, 86)
    tmax = 100.0
    t_save = np.linspace(0, tmax, 100 + 1)
    t_msd = np.logspace(-2, 2, 10)
    mean_time_looped, mean_time_unlooped, vextrude = extrusion_parameters(p, sigma,
                                                                          mean_loop_size, b=b,
                                                                          D=D[0])
    X, msd, mscd = loop_extrusion(N, L, b, D, h, tmax,
                                  mean_time_looped, mean_time_unlooped, vextrude, s1, s2,
                                  t_save=t_save, t_msd=t_msd, msd_start_time=1.0)

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


if __name__ == '__main__':
    N = 101
    L = 100
    b = 1
    D = 10 * np.ones(N)
    p = sys.argv[1]
    sigma = sys.argv[2]
    mean_loop_size = sys.argv[3]
    filedir = Path('csvs/extrusion_25_75')
    print(f'Running simulation {filedir.name}')
    func = partial(run_extrusion, N=N, L=L, b=b, D=D, p=p, sigma=sigma, mean_loop_size=mean_loop_size,
                   filedir=filedir)
    tic = time.perf_counter()
    func(0)
    # pool_size = 16
    # N = 96
    # with Pool(pool_size) as p:
    #    result = p.map(func, np.arange(0, N))
    toc = time.perf_counter()
    print(f'Ran simulation in {(toc - tic):0.4f}s')