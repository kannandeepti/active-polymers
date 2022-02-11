#!/usr/bin/env python3
""" Script to pull down and analyze simulation data from remote."""

import numpy as np
from pathlib import Path
import pandas as pd

def process_sim(file):
    df = pd.read_csv(file)
    dfg = df.groupby('t')
    t_save = []
    X = []
    for t, mat in dfg:
        t_save.append(t)
        X.append(mat.to_numpy()[:, 1:])
    t_save = np.array(t_save)
    X = np.array(X)
    return X, t_save

def extract_cov(simdir, tape=0):
    """ Extract list of D's and identity matrix from files in simulation directory
    `simdir' from tape file."""
    simdir = Path(simdir)
    simname = simdir.name
    df = pd.read_csv(simdir / f'tape{tape}.csv')
    # extract temperatures
    D = np.array(df[df['t'] == 350.0].D)
    mat = np.zeros((1, 101))
    rhos = np.array([0.5])  # TODO: save rhos to file as well
    if (simdir / 'idmat.csv').is_file():
        df = pd.read_csv(simdir / 'idmat.csv')
        mat = np.array(df)
        mat = mat[0, 1:].reshape((1, 101))
    return D, mat, rhos

def to_XYZ(X, temps, filepath, filename, frames=None, L=100, b=1):
    """" Convert the last frame of the simulation trajectory into an
    XYZ file readable by OVITO visualization software.

    TODO: export multiple frames to visualize simulation trajectory.
    """
    nframes, N, dim = X.shape
    filepath = Path(filepath)
    particle_radius = np.sqrt(L * b / (N - 1))/2
    cmap = mpl.cm.get_cmap('coolwarm')
    #if all the beads are at the same temperature, assume they are all cold
    if len(np.unique(temps)) == 1:
        colors = [mpl.colors.to_rgb(cmap(0.0)) for n in range(N)]
    else:
        #assign color based on temperature
        norm = mpl.colors.Normalize(vmin=np.min(temps), vmax=np.max(temps))
        colors = [mpl.colors.to_rgb(cmap(norm(d))) for d in temps]
    assert (len(colors) == N)
    with open(filepath/f'{filename}.txt', 'w') as f:
        f.write(f'{N}\n\n')
        for i in range(N):
            f.write(f'{i} {X[-1, i, 0]} {X[-1, i, 1]} {X[-1, i, 2]} {particle_radius} {temps[i]} {colors[i][0]} {colors[i][1]} {colors[i][2]}\n')
    """ TODO: Write topology and atom info in LAMMPS data format so as to not have to add modifiers in GUI application."""
    """
    topology = np.zeros((N-1, 4))
    topology[:, 0] = np.arange(1, N)
    topology[:, 1] = np.tile(1, N-1)
    topology[:, 2:] = np.array([np.arange(n, n+2) for n in range(N - 1)])
    with open(filepath/f'topology_{filename}.txt', 'w') as f:
        f.write('LAMMPS Description\n\n')
        f.write(f'{N-1} bonds\n\n')
        f.write('Bonds\n\n')
        for i in range(N-1):
            f.write(f'{topology[i, 0]} {topology[i, 1]} {topology[i, 2]} {topology[i, 3]}\n')
    """


def convert_to_xyz(simdir, tape, L=100, b=1, **kwargs):
    """ For specified tape (trajectory) within a given simulation directory,
    write the last frame to an XYZ file, specifying temperature, etc."""
    simdir = Path(simdir)
    simname = simdir.name
    df = pd.read_csv(simdir / f'tape{tape}.csv')
    # extract temperatures
    D = np.array(df[df['t'] == 0.0].D)
    X, t_save = process_sim(Path(simdir / f'tape{tape}.csv'))
    nframes, N, dim = X.shape
    to_XYZ(X, D, simdir, f'tape{tape}_frame{nframes-1}', **kwargs)