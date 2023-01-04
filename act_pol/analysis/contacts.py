r"""
Constructing simulated contact maps
-----------------------------------
This module can be used to construct maps illustrating the mean separation between beads i and j
for all pairs of monomers on the chain. A contact map can then be constructed by thresholding the
mean separation map to count contacts within a certain capture radius. Functions are also available
to plot the distribution of distances between any two beads.
"""

import numpy as np
from .rouse import gaussian_Ploop
from .files import *
from .analyze import draw_power_law_triangle

from matplotlib import pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
import seaborn as sns
import cmasher as cmr
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
sns.set()

import deepti_utils
from deepti_utils.plotting import *

params = {'axes.edgecolor': 'black',
                  'axes.facecolor':'white',
                  'axes.grid': False,
                  'axes.titlesize': 18,
                  'axes.titlepad' : 12,
                  'axes.labelsize': 18,
                  'legend.fontsize': 18,
                  'text.usetex': True,
                  'xtick.labelsize': 18,
                  'ytick.labelsize': 18,
                  'figure.figsize': [5.67, 4.76],
                  'font.family': 'serif',
                  'font.serif' : ["Computer Modern Roman"],
                  'mathtext.fontset': 'cm',
                  'xtick.bottom':True,
                  'xtick.top': False,
                  'xtick.direction': 'out',
                  'xtick.major.pad': 3,
                  'xtick.major.size': 4,
                  'xtick.minor.bottom': False,
                  'xtick.major.width': 1.0,
                  'ytick.left':True,
                  'ytick.right':False,
                  'ytick.direction':'out',
                  'ytick.major.pad': 3,
                  'ytick.major.size': 4,
                  'ytick.major.width': 1.0,
                  'ytick.minor.right':False,
                  'lines.linewidth':2}
plt.rcParams.update(params)
textwidth = 6.5

cmap_relative = cmr.iceburn
cmap_distance = 'magma'
cmap_temps = 'coolwarm'
cmap_contacts = "YlOrRd"

snapshot_DT = 350.0 #Rouse time for the polymer, i.e. time between uncorrelated snapshots in a
# simulation trajectory

def two_point_msd(simdir, ntraj=None, N=101, relative=None, squared=False):
    """Compute matrix with mean squared distance (msd) between all pairs of beads on a polymer at
    a particular time point. If a second simulation directory is specified, then will compute a
    second msd matrix.

    Extracts snapshots every `snapshot_DT` time steps from each simulation trajectory
    after throwing out the first

    Parameters
    ----------
    simdir : Path or str
        path to simulation results
    ntraj : int
        number of independent simulations in this parameter set. Default is to count
        number of tape*.csv files in directory.
    N : int
        number of monomers in chain
    relative : Path or str
        path to results of a baseline simulation to compare to. Default is None.
    squared : bool
        whether to compute mean squared equiclidean distance (using 'sqeuclidean' metric
        in scipy.spatial.distance.pdist)

    Returns
    -------
    Rg_squared : float
        squared radius of gyration of the polymer (defined as sum of all entries in msd
        matrix / 2N^2)
    average_dist : np.ndarray[float64] (N, N)
        N x N matrix containing pairwise mean squared distance between all pairs of monomers
    nreplicates : int
        total number of simulation snapshots used to construct ensemble average

    """
    simdir = Path(simdir)
    average_dist = np.zeros((N, N))
    eq_dist = np.zeros((N, N))
    metric = 'euclidean'
    if squared:
        metric = 'sqeuclidean'
    if ntraj is None:
        ntraj = get_ntraj(simdir)
    #ignore first half of tape (steady state) and then take time slices every 10 save points
    #to sample equilibrium structures
    nreplicates = 0
    for tape in simdir.glob('tape*.csv'):
        j = tape.name[:-4][4:]
        X, t_save = process_sim(tape)
        DT = np.diff(t_save)[0] #time between save points
        nrousetimes = int(np.ceil(snapshot_DT/ DT)) #number of frames that make up a rouse time
        ntimes, _, _ = X.shape
        #nreplicates = ntraj * (ntimes - 1)
        nreplicates += len(range(nrousetimes, ntimes, nrousetimes))
        if relative:
            Xeq, _ = process_sim(Path(relative) / f'tape{j}.csv')
        for i in range(nrousetimes, ntimes, nrousetimes):
            #for temperature modulations
            dist = pdist(X[i, :, 0:3], metric=metric)
            Y = squareform(dist)
            average_dist += Y
            #for equilibrium case
            if relative:
                dist = pdist(Xeq[i, :, 0:3], metric=metric)
                eq_dist += squareform(dist)

    average_dist = average_dist / nreplicates
    if squared:
        Rg_squared = np.sum(average_dist) / (2 * N ** 2)
    else:
        Rg_squared = np.sum(average_dist ** 2) / (2 * N ** 2)

    if relative:
        eq_dist = eq_dist / nreplicates
        return Rg_squared, average_dist, eq_dist, nreplicates

    return Rg_squared, average_dist, nreplicates

def compute_relative_change(dist, eqdist):
    """ Compute matrix where entries specify the percent change in mean squared distance
    relative to some reference matrix, `eqdist`. """
    dist1 = dist
    eqdist1 = eqdist
    ind = np.diag_indices(dist.shape[0])
    dist1[ind] = 1.0
    eqdist1[ind] = 1.0
    rel_dist = (dist1 - eqdist1)/eqdist1 * 100.0
    return rel_dist

def plot_msd_map(dist, simdir, relative=False, squared=False):
    """ Plot seaborn heatmap where entry (i, j) is the mean distance between beads i and j.
    This version does not include a color bar showing the activity of the monomers.

    Parameters
    ----------
    dist : array-like (N, N)
        atrix containing pairwise mean squared distances between N monomers
    simdir : str or Path
        path to simulation directory containing raw data from which `dist` was computed
    relative : bool
        whether `dist` contains the difference between msds of `simdir` and a reference (eq)
    squared : bool
        whether `dist` contains mean squared distances as opposed to mean distances
    """
    simdir = Path(simdir)
    fig, ax = plt.subplots()
    if relative:
        res = sns.heatmap(dist, xticklabels=25,
                    yticklabels=25, cmap=cmr.iceburn, center=0.0, square=True, ax=ax)
        if squared:
            ax.set_title(r'MSD relative to uniform temperature')
        else:
            ax.set_title(r'$\langle r_{ij} \rangle  - \langle r_{ij} \rangle_{eq}$')
    else:
        res = sns.heatmap(dist, xticklabels=25, yticklabels=25,
                    cmap='magma', square=True, ax=ax)
        if squared:
            ax.set_title(r'Mean squared distance $\langle\Vert \vec{r}_i - \vec{r}_j \Vert^2\rangle$')
        else:
            ax.set_title(r'Mean distance $\langle\Vert \vec{r}_i - \vec{r}_j \Vert\rangle$')
    ax.set_xlabel(r'Bead $i$')
    ax.set_ylabel(r'Bead $j$')
    # make frame visible
    for _, spine in res.spines.items():
        spine.set_visible(True)
    fig.tight_layout()
    if relative:
        plt.savefig(f'plots/two_point_msd_{simdir.name}_relative.pdf')
    else:
        plt.savefig(f'plots/two_point_msd_{simdir.name}.pdf')

def heatmap_divider(mat, temps, simname, relative=False, relative_change=False,
                    width=5, lines=None, marks=None, nreplicates=None, **kwargs):
    """ Plot matrix of mean separations between monomers (i,j) with color bars
     showing activity of beads.

     Parameters
     ----------
     mat : array-like (N, N)
        matrix containing pairwise mean squared distances between N monomers
     temps : array (N,)
        activities or temperatures of the N monomers to be made into a colorbar on x and y axes
     simname : str
        text tag for simulation from which data was computed
     relative : bool
        whether `mat` contains the difference between msds of `simname` and a reference (eq)
     relative_change : bool
        whether `mat` contains the percent change between msds of `simname` and a reference
     width : int
        size of colorbar in pixels (main plot will be N pixels by N pixels)

     """
    if mat is None:
        mat = np.random.randn((101, 101))
    if nreplicates:
        nreps = f', n={nreplicates}'
    else:
        nreps = ''
    D = np.ones((width, 101))
    for i in range(width):
        D[i, :] = temps
    fig = plt.figure()
    ax = plt.subplot(111)
    divider = make_axes_locatable(ax)
    ax_bottom = divider.append_axes("bottom", size=f"{width}%", pad=0.0)
    ax_left = divider.append_axes("left", size=f"{width}%", pad=0.0)
    cax = divider.append_axes("right", size=f"{width}%", pad=0.1)
    ax_bottom.set_xticks([0, 25, 50, 75, 100])
    ax_bottom.set_yticks([])
    ax_left.set_yticks([0, 25, 50, 75, 100])
    ax_left.set_xticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    if relative or relative_change:
        im = ax.imshow(mat, norm=colors.CenteredNorm(), cmap=cmap_relative)
    else:
        im = ax.imshow(mat, cmap=cmap_distance, **kwargs)
    ax_bottom.imshow(D, cmap='coolwarm', vmin=0.25, vmax=1.75)
    ax_left.imshow(D.T, cmap='coolwarm', vmin=0.25, vmax=1.75)
    fig.colorbar(im, cax=cax)
    if lines:
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.vlines(lines, ymin, ymax, linestyles='dashed', color='white', lw=0.5)
        ax.hlines(lines, xmin, xmax, linestyles='dashed', color='white', lw=0.5)
    if marks:
        for mark in marks:
            x, y = mark
            ax.plot(x, y, "X", color='white')
    if relative:
        title = (r'$\langle r_{ij} \rangle  - \langle r_{ij} \rangle_{eq}$')
    elif relative_change:
        title = (r'Relative \% change in $\langle r_{ij} \rangle$')
    else:
        title = (r'Mean distance $\langle r_{ij} \rangle$')
    ax.set_title(title + nreps)
    ax_bottom.set_xlabel(r'Bead $i$')
    ax_left.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    if relative or relative_change:
        plt.savefig(f'plots/two_point_msd_{simname}_relative.pdf')
    else:
        plt.savefig(f'plots/two_point_msd_{simname}.pdf')
    plt.show()

def mdmap_abs_rel(dist, eqdist, temps, simname, relative=False, width=5, **kwargs):
    """ Plots the relative change in mean squared separation above the diagonal and the
    absolute mean squared separation below the diagonal, akin to Fig. 1B.

    Note: the 2 plots were generated separately using relative = True or False and then
    stitched together using Affinity Designer.

    TODO: plot both halves of the diagonal in one go using masking.

    """
    ind = np.diag_indices(dist.shape[0])
    dist[ind] = 1.0
    eqdist[ind] = 1.0
    rel_dist = (dist - eqdist)/eqdist * 100.0
    half_rel = np.triu(rel_dist, k=0)
    half_abs = np.tril(dist, k=1)
    D = np.ones((width, 101))
    for i in range(width):
        D[i, :] = temps
    fig = plt.figure()
    ax = plt.subplot(111)
    divider = make_axes_locatable(ax)
    ax_bottom = divider.append_axes("bottom", size=f"{width}%", pad=0.0)
    ax_left = divider.append_axes("left", size=f"{width}%", pad=0.0)
    cax = divider.append_axes("right", size=f"{width}%", pad=0.1)
    cax2 = divider.append_axes("top", size=f"{width}%", pad=0.1)
    ax_bottom.set_xticks([0, 25, 50, 75, 100])
    ax_bottom.set_yticks([])
    ax_left.set_yticks([0, 25, 50, 75, 100])
    ax_left.set_xticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    if relative:
        im = ax.imshow(dist[::-1, :], cmap=cmap_distance, **kwargs)
        #plot the relative MSD in front
        im2 = ax.imshow(rel_dist[::-1, :], norm=colors.CenteredNorm(), cmap=cmap_relative, **kwargs)
    else:
        im2 = ax.imshow(rel_dist[::-1, :], norm=colors.CenteredNorm(), cmap=cmap_relative, **kwargs)
        #plot the absolute MSD in front
        im = ax.imshow(dist[::-1, :], cmap=cmap_distance, **kwargs)
    ax_bottom.imshow(D, cmap='coolwarm', vmin=D.min(), vmax=D.max())
    ax_left.imshow(D.T[::-1, :], cmap='coolwarm', vmin=D.min(), vmax=D.max())
    cbar = fig.colorbar(im, cax=cax, label=r'MSD $\langle \Delta r^2_{ij} \rangle$')
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal', ticks=[-25, 0, 25])
    cax2.xaxis.set_ticks_position('top')
    cbar2.set_ticklabels(['-25\%', '', '25\%'])
    #ax.set_title(r'$\langle r_{ij} \rangle  - \langle r_{ij} \rangle_{eq}$')
    cax2.set_title('relative change in MSD')
    ax_bottom.set_xlabel(r'Bead $i$')
    ax_left.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    plt.savefig(f'plots/two_point_msd_{simname}_rel_front.pdf')

def contact_probability(a, simdir, N=101, eq_contacts=None):
    """Compute a matrix of contact probabilities between all pairs of monomers, where a contact
    is defined as a pairwise mean separation below `a`.

    Parameters
    ----------
    a : float
        capture radius
    simdir : str or Path
        path to simulation directory containing trajectory files
    N : float
        number of monomers

    Returns
    -------
    counts : array-like (N, N)
        raw integer counts of contacts between pairs of monomers
    contacts : array-like (N, N)
        matrix of contact probabilities (counts / nreplicates)
    nreplicates : int
        total number of simulation snapshots used to compute ensemble average
    """
    simdir = Path(simdir)
    average_dist = np.zeros((N, N))
    #counts(i, j) = number of times monomer i and j loop within contact radius a
    counts = np.zeros((N, N))
    metric = 'euclidean'
    #ignore first half of tape (steady state) and then take time slices every snapshot_DT time
    # points
    #to sample equilibrium structures
    nreplicates = 0
    for tape in simdir.glob('tape*.csv'):
        j = tape.name[:-4][4:]
        X, t_save = process_sim(simdir / f'tape{j}.csv')
        DT = np.diff(t_save)[0]  # time between save points
        nrousetimes = int(np.ceil(snapshot_DT / DT))  # number of frames that make up a rouse time
        ntimes, _, _ = X.shape
        nreplicates += len(range(nrousetimes, ntimes, nrousetimes))
        if eq_contacts:
            Xeq, _ = process_sim(Path('csvs/bdeq1') / f'tape{j}.csv')
        for i in range(nrousetimes, ntimes, nrousetimes):
            #for temperature modulations
            dist = pdist(X[i, :, :], metric=metric)
            Y = squareform(dist)
            counts += (Y < a)
            average_dist += Y
    contacts = counts / nreplicates
    return counts, contacts, nreplicates

def plot_contact_map(contacts, simname, a, width=5, tag='', vmin=None, vmax=None,
                     **kwargs):
    """ Plot contact map where a contact is defined as a pairwise mean squared separation
    below the capture radius, a.

    Parameters
    ----------
    contacts : array-like (N, N)
        matrix of pairwise contact probabilities between all N monomers
    simname : str
        name of simulation from which `contacts` was computed
    a : float
        capture radius used to threshold mean separation and count a contact
    width : int
        size of colorbar in percent of plot size
    tag : str
        additional string to add to plot file name after simname

    """
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=f"{width}%", pad=0.1)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    #define minimum contact probability to be half of the lowest nonzero entry in contacts
    contacts[contacts == 0] = np.min(contacts[contacts > 0])/2.0
    if vmin is None and vmax is None:
        lognorm = LogNorm(vmin=contacts.min(), vmax=contacts.max())
    else:
        lognorm = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(contacts, norm=lognorm, cmap=cmap_contacts, **kwargs)
    fig.colorbar(im, cax=cax)
    ax.set_title(f'Contact Map $P(\\Vert \\vec{{r}}_i - \\vec{{r}}_j \\Vert < {a})$')
    ax.set_xlabel(r'Bead $i$')
    ax.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    plt.savefig(f'plots/contact_map_{simname}{tag}.pdf')
    plt.show()

def plot_contact_map_temps(contacts, temps, simname, width=5, a=1, tag=None,
                           nreplicates=None, **kwargs):
    """ Plot contact map with a colorbar showing the activities of all monomers on bottom
    and left axes. """

    if tag is None:
        tag = ''
    if nreplicates:
        nreps = f', n={nreplicates}'
    else:
        nreps = ''
    # define minimum contact probability to be half of the lowest nonzero entry in contacts
    contacts[contacts == 0] = np.min(contacts[contacts > 0])/2.0
    lognorm = LogNorm(vmin=contacts.min(), vmax=contacts.max())
    D = np.ones((width, 101))
    for i in range(width):
        D[i, :] = temps
    fig = plt.figure()
    ax = plt.subplot(111)
    divider = make_axes_locatable(ax)
    ax_bottom = divider.append_axes("bottom", size=f"{width}%", pad=0.0)
    ax_left = divider.append_axes("left", size=f"{width}%", pad=0.0)
    cax = divider.append_axes("right", size=f"{width}%", pad=0.1)
    ax_bottom.set_xticks([0, 25, 50, 75, 100])
    ax_bottom.set_yticks([])
    ax_left.set_yticks([0, 25, 50, 75, 100])
    ax_left.set_xticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(contacts, norm=lognorm, cmap=cmap_contacts, **kwargs)
    ax_bottom.imshow(D, cmap='coolwarm', vmin=0.25, vmax=1.75)
    ax_left.imshow(D.T, cmap='coolwarm', vmin=0.25, vmax=1.75)
    fig.colorbar(im, cax=cax)
    ax.set_title(f'Contact Map $P(\\Vert \\vec{{r}}_i - \\vec{{r}}_j \\Vert < {a})$' + nreps)
    ax_bottom.set_xlabel(r'Bead $i$')
    ax_left.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    plt.savefig(f'plots/contact_map_{simname}_temps{tag}.pdf')
    plt.show()

def compute_ploop(counts, nreplicates):
    """ Computing contact probability as a function of distance from diagonal.

    Parameters
    ----------
    counts : array-like (N, N)
        raw integer counts of contacts between pairs of monomers
    nreplicates : int
        total number of simulation snapshots used to compute ensemble average

    Returns
    -------
    Ploop : array-like (N,)
        contact probabilities (averaged over diagonal of contact map)
    sdistances : array-like (N,)
        distances along chain in Kuhn lengths
    nreplicates : int

    """
    N, _ = counts.shape
    sdistances = np.arange(0.0, N)
    Ploop = np.zeros_like(sdistances)
    for i in range(N):
        if i == 0:
            Ploop[i] += np.trace(counts, offset=0)
        else:
            Ploop[i] += np.trace(counts, offset=i)
            Ploop[i] += np.trace(counts, offset=-i)
    #P(loop | s) = P(loop, s) / P(s)
    # number of entries of contact matrix where monomers are a distance s apart
    p_of_s = (np.arange(1, N + 1))[::-1]
    p_of_s[1:] *= 2
    normalization = nreplicates * p_of_s
    Ploop = Ploop / normalization
    return Ploop, sdistances, nreplicates

def plot_ploop(sdistances, Ploop, nreplicates, descriptor, a=1, b=1):
    """ Plot looping probability as a function of distance along the chain in Kuhn lengths
    and compare to analytical looping probability of a Gaussian chain. """
    analytical_ploop = [gaussian_Ploop(a, n, b) for n in sdistances[1:]]
    fig, ax = plt.subplots()
    ax.plot(sdistances, Ploop, label=f'Simulation estimate (n={nreplicates})')
    ax.plot(sdistances[1:], analytical_ploop, label='Theory')
    corner = draw_power_law_triangle(-3/2, [1, -0.5], 0.5, 'up', base=10,
                            hypotenuse_only=False, ax=ax)
    ax.text(12.0, 0.07, r'$s^{-3/2}$')
    plt.yscale('log')
    plt.xscale('log')
    ax.set_xlabel('Loop size')
    ax.set_ylabel(f'Looping probability')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/ploop_{descriptor}.pdf')
    plt.show()

def vary_contact_radius(sims, radii):
    """ For each simulation in sims, plot a contact map for different capture radii to understand
    the effect of `a` on contact map."""
    for sim in sims:
        simpath = Path(f'csvs/{sim}')
        df = pd.read_csv(simpath / 'tape0.csv')
        # extract temperatures
        Ds = np.array(df[df['t'] == 0.0].D)
        for a in radii:
            counts, contacts = contact_probability(a, f'csvs/{sim}', 96)
            plot_contact_map_temps(contacts, Ds, sim, a=a, tag=f'_a{a}')

def distance_distribution(ind1, ind2, simdir):
    """ Plot distribution of distances between monomers `ind1` and `ind2`
    from simulation trajectories in `simdir`. """
    simdir = Path(simdir)
    distances = []
    nreplicates = 0
    for tape in simdir.glob('tape*.csv'):
        j = tape.name[:-4][4:]
        X, t_save = process_sim(simdir / f'tape{j}.csv')
        ntimes, _, _ = X.shape
        DT = np.diff(t_save)[0] #time between save points
        nrousetimes = int(np.ceil(snapshot_DT / DT)) #number of frames that make up a rouse time
        nreplicates += len(range(nrousetimes, ntimes, nrousetimes))
        for i in range(nrousetimes, ntimes, nrousetimes):
            #for temperature modulations
            distance = X[i, ind1, :] - X[i, ind2, :]
            dist = np.sqrt(distance @ distance)
            distances.append(dist)
    return np.array(distances)

def squared_distance_distribution(ind1, ind2, simdir):
    """ Plot distribution of squared distances between monomers `ind1` and `ind2`
        from simulation trajectories in `simdir`. """
    simdir = Path(simdir)
    distances = []
    nreplicates = 0
    for tape in simdir.glob('tape*.csv'):
        j = tape.name[:-4][4:]
        X, t_save = process_sim(simdir / f'tape{j}.csv')
        ntimes, _, _ = X.shape
        DT = np.diff(t_save)[0] #time between save points
        nrousetimes = int(np.ceil(snapshot_DT / DT)) #number of frames that make up a rouse time
        nreplicates += len(range(nrousetimes, ntimes, nrousetimes))
        for i in range(nrousetimes, ntimes, nrousetimes):
            #for temperature modulations
            distance = X[i, ind1, :] - X[i, ind2, :]
            dist = distance @ distance
            distances.append(dist)
    return np.array(distances)

def plot_distance_distribution(i, j, simdir1, simdir2, label1, label2, mat):
    """ Plot distribution of distances between monomers `i` and `j` from two
    separate simulations in `simdir1` and `simdir2` which are labeled with
    `label1` and `label2`, respectively. """

    simdir1 = Path(simdir1)
    simname = simdir1.name
    simdir2 = Path(simdir2)
    distances1 = distance_distribution(i, j, simdir1)
    distances2 = distance_distribution(i, j, simdir2)
    print((np.mean(distances1) - np.mean(distances2))/np.mean(distances1))
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(distances1, bins='auto', density=True, histtype='step', label=label1)
    n, bins, patches = ax.hist(distances2, bins='auto', density=True, histtype='step', label=label2)
    plt.legend()
    plt.xlabel(r'$r_{ij}$')
    type1 = int(mat[0, i])
    type2 = int(mat[0, j])
    plt.title(f'Distance distribution, i={i} (type {type1}), j={j} (type {type2})')
    fig.tight_layout()
    plt.savefig(f'plots/distance_distribution_{i}_{j}_{simname}.pdf')
    plt.show()