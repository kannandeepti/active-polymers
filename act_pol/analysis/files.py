#!/usr/bin/env python3
""" Script to pull down and analyze simulation data from remote."""

import numpy as np
from pathlib import Path
import subprocess
import pandas as pd
import matplotlib as mpl
import glob
import sysrsync

REMOTE = 'dkannan@eofe8:mit.edu:~/git-remotes/active-polymers/csvs/extrusion'
LOCAL = '/Users/deepti/Documents/MIT/Research/git-remotes/active-polymers/csvs'

def pull_down_extrusion_data(remote=REMOTE, local=LOCAL):
    sysrsync.run(source=remote, destination=local, options=['-a', '-r', '-v'])

def pull_down_data(simdirs, remote='dkannan@eofe8.mit.edu:~/git-remotes/active-polymers/csvs',
                   local='csvs'):
    """ Script to transfer output of simulation on remote cluster to local directory.

    TODO: debug

    Parameters
    ----------
    remote : str or Path
        path to directory containing simulation output on remote

    """

    remote = Path(remote)
    local = Path(local)
    for sim in simdirs:
        remotesim = remote / sim
        localsim = local / sim
        localsim.mkdir() #this will raise a FileExistsError if directory already exists
        subprocess.run(f'rsync {remotesim}/*.csv {localsim}/')

def get_ntraj(simdir):
    """ Counts the number of files of the form tape*.csv -- aka number of independent simulation
    trajectories that were run for this parameter set specified by simdir."""
    tapes = glob.glob(str(simdir) + '/tape*.csv')
    return len(tapes)

def process_sim(file):
    df = pd.read_csv(file)
    dfg = df.groupby('t')
    t_save = []
    D = []
    X = []
    for t, mat in dfg:
        t_save.append(t)
        if 'D' in mat.columns:
            D.append(mat['D'].to_numpy())
        #column 0 is time, columns 1-3 are x,y,z, and then column 4 is D
        X.append(mat.to_numpy()[:, 1:4])
    t_save = np.array(t_save)
    X = np.array(X)
    return X, t_save, D

def extract_cov(simdir, tape=0, time=350.0):
    """ Extract list of D's and identity matrix from files in simulation directory
    `simdir' from tape file."""
    simdir = Path(simdir)
    simname = simdir.name
    df = pd.read_csv(simdir / f'tape{tape}.csv')
    # extract temperatures
    D = np.array(df[df['t'] == time].D)
    mat = np.zeros((1, 101))
    rhos = np.array([0.5])  # TODO: save rhos to file as well
    if (simdir / 'idmat.csv').is_file():
        df = pd.read_csv(simdir / 'idmat.csv')
        mat = np.array(df)
        if 'rho' in df.columns:
            rhos = np.array(df['rho'])
            mat = mat[0, 1:-1].reshape((1, 101))
        else:
            mat = mat[0, 1:].reshape((1, 101))
    return D, mat, rhos

def extract_rhomat(simdir):
    """ Extract contents of rhomat.csv in simulation directory."""
    simdir = Path(simdir)
    simname = simdir.name
    if (simdir / 'rhomat.csv').is_file():
        df = pd.read_csv(simdir / 'rhomat.csv')
        mat = np.array(df)
        return mat
    else:
        raise ValueError(f'There is no file called rhomat.csv in {simdir}')

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