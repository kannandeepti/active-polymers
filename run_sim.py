""" Script to run brownian dynamics simulations of active polymer."""
import numpy as np
from bd import recommended_dt, with_srk1
import pandas as pd
from pathlib import Path
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

def run(i, N, L, b, D, filedir):
    """ Run one simulation of a length L chain with N beads,
    Kuhn length b, and array of diffusion coefficients D."""
    file = Path(filedir)/f'tape{i}.csv'
    try:
        file.parent.mkdir(parents=True)
    except:
        if file.parent.is_dir() is False:
            # if the parent directory does not exist and mkdir still failed, re-raise an exception
            raise
    t = np.linspace(0, 1e5, int(1e7) + 1)
    print(f'Simulation time step: {t[1] - t[0]}')
    #save 100 conformations
    t_save = np.linspace(0, 1e5, 200 + 1)
    X = with_srk1(N, L, b, D, t, t_save)
    dfs = []
    for i in range(X.shape[0]):
        df = pd.DataFrame(X[i, :, :])
        df['t'] = t_save[i]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df.set_index(['t'], inplace=True)
    df.to_csv(file)

if __name__ == '__main__':
    N = 101
    L = 100
    b = 1
    D = np.tile(1, N)
    filedir = Path('csvs/bdeq')
    func = partial(run, N=N, L=L, b=b, D=D, filedir=filedir)
    tic = time.perf_counter()
    pool_size = 8
    N = 16
    with Pool(pool_size) as p:
        result = p.map(func, np.arange(0, N))
    toc = time.perf_counter()
    print(f'Ran {N} simulations in {(toc - tic):0.4f}s')

