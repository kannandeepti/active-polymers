r"""
File I/O
--------
Script to extract simulation trajectory from files outputted by simulation (of the form
'simdir/tape[int].csv') and convert to file formats useful for post processing analysis,
including visualization by the OVITO software.

"""

import numpy as np
from pathlib import Path
import subprocess
import pandas as pd
import matplotlib as mpl
import glob

def get_ntraj(simdir):
    """ Counts the number of files of the form tape*.csv -- aka number of independent simulation
    trajectories that were run for this parameter set specified by `simdir`."""
    tapes = glob.glob(str(simdir) + '/tape*.csv')
    return len(tapes)

def process_sim(file):
    """ Extracts a file outputted by functions in scripts/run_sim.py and creates a numpy.ndarray
    with the entire simulation trajectory.

    Parameters
    ----------
    file : str or Path to csv file
       file should have 5 columns with t, x, y, z, and D (diffusion coefficient of N monomers)

    Returns
    -------
    X : np.ndarray[float64] (Nt, N, 3)
        positions of all N monomers for each time point of simulation trajectory. This array is
        used in many of the post processing analysis functions.
    t_save : np.ndarray[float64 (Nt,)
        time points at which monomer positions were saved

    """
    df = pd.read_csv(file)
    dfg = df.groupby('t')
    t_save = []
    X = []
    for t, mat in dfg:
        t_save.append(t)
        #column 0 is time, columns 1-3 are x,y,z, and then column 4 is D
        X.append(mat.to_numpy()[:, 1:4])
    t_save = np.array(t_save)
    X = np.array(X)
    return X, t_save

def to_XYZ(X, temps, filepath, filename, L=100, b=1):
    """" Convert the last frame of the simulation trajectory into an
    XYZ file readable by OVITO visualization software.

    TODO: export multiple frames to visualize simulation trajectory as a movie.

    Parameters
    ----------
    X : array-like (Nt, N, 3)
        positions of all N monomers for each time point of simulation trajectory. This array is
        used in many of the post processing analysis functions.
    temps : array-like (N,)
        temperatures or scalar activities of the N monomers (for coloring monomers on red/blue
        scale)
    filepath : str or Path
        path of where to write out the XYZ file
    filename : str
        name of file to write out
    L : float
        length of polymer (to calculate monomer radius)
    b: float
        Kuhn length of polymer (to calculate monomer radius)

    Notes
    -----
    Since a Rouse polymer does not feature volume exclusion, the monomers have no finite size.
    Here we arbitrarily choose the particle radius to be half of the average bond extension,
    i.e. the average end-to-end distance of the underlying WLC that makes up a monomer of the
    Rouse chain.

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


def convert_to_xyz(simdir, tape, **kwargs):
    """ For specified tape within a given simulation directory,
    write the last frame to an XYZ file, and extract the activities of the monomers to pass to
    to_XYZ().

    Parameters
    ----------
    simdir : str or Path
        path to simulation directory contain tape[int].csv files
    tape : int
        id of simulation from which to extract snapshot

    """
    simdir = Path(simdir)
    df = pd.read_csv(simdir / f'tape{tape}.csv')
    # extract temperatures
    D = np.array(df[df['t'] == 0.0].D)
    X, t_save = process_sim(Path(simdir / f'tape{tape}.csv'))
    nframes, N, dim = X.shape
    to_XYZ(X, D, simdir, f'tape{tape}_frame{nframes-1}', **kwargs)