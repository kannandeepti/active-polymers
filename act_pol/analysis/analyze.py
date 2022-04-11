r"""
Computing structural properties from simulation data
----------------------------------------------------
This module can be used to analyze steady state structures from the simulation data produced
by the act_pol.bdsim module. Functions are available to calculate/plot the end to end distance,
radius of gyration, and distance from center of mass. For contact maps,
see act_pol.analysis.contacts. For dynamical observables, see act_pol.analysis.msd.

"""

import numpy as np
from .rouse import linear_mid_msd, end2end_distance_gauss, gaussian_Ploop
from ..bdsim.correlations import *
from .files import *

import matplotlib as mpl
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
cmap_contacts = "Reds"

"""
b = 30.0 * 10**(-9) #Kuhn length of approximately 30 nm
k_B = 1.38064852 * 10**(-23) #Boltzmann constant
Teq = 310.15 #human body temperature in Kelvin
eta = 10 * 8.9 * 10**(-4) #10 times the dynamic viscosity of water
R = 15 * 10**(-9) #radius of monomer bead (diamater is a Kuhn length)
xi = 6 * np.pi * eta * R #friction/drag coefficient
N = 20000 #number of monomers in chain (if each bead is 5 kb, then a chromosome of 100,000,000 bp implies 20000 beads)
L = 0.034 #length of a chromosome in meters that is 100,000,000 bp long assuming 0.34 nm/bp
Ntest = N/100
L = 0.034/100
D = k_B * Teq / xi

dt = recommended_dt(N, L, b, D)
print(f'Time step: {dt}')
"""

def end_to_end_distance_squared_vs_time(X, t_save, b=1, L=100):
    """ End to end distance is <R_N(t) - R_0(t)>, i.e. the norm of the position vector
    of last bead minus position vector of first beed. """
    end_to_end_squared = np.sum((X[:, -1, :] - X[:, 0, :])**2, axis=-1)
    fig, ax = plt.subplots()
    ax.plot(t_save, end_to_end_squared, label='simulation')
    Nhat = L/b
    ax.plot(t_save, np.tile(Nhat * b**2, len(t_save)), label='theory')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle R^2 \rangle$')
    plt.legend()
    fig.tight_layout()
    plt.show()

def end_to_end_distance_vs_Rmax(X, t_save, b=1, L=100, N=101):
    """ End to end distance is <R_N(t) - R_0(t)>, i.e. the norm of the position vector
        of last bead minus position vector of first beed.
        TODO: debug?
        """
    #take mean over time after steady state
    end_to_end = np.linalg.norm((X[-1, :, :] - X[-1, 0, :]), axis=-1)
    print(end_to_end.shape)
    #end_to_end = np.mean(end_to_end, axis=0)
    #print(end_to_end.shape)
    fig, ax = plt.subplots()
    Nhat = L/b #number of kuhn lengths
    L0 = L/(N-1)  # length per bead
    r = np.linspace(0, 100, 1000)
    #ax.plot(r, end_to_end, label='simulation')
    analytical_r2 = end2end_distance_gauss(r, b, Nhat, L)
    ax.plot(r, analytical_r2, label='theory')
    ax.set_xlabel(r'$R_{max}$')
    ax.set_ylabel(r'$\langle R^2 \rangle$')
    plt.legend()
    #plt.xscale('log')
    #plt.yscale('log')
    fig.tight_layout()
    plt.show()

def average_end2end_distance_vs_Rmax(simdir, b=1, L=100, N=101, ntraj=96):
    """ End to end distance is <R_N(t) - R_0(t)>, i.e. the norm of the position vector
        of last bead minus position vector of first bead. """
    simdir = Path(simdir)
    Nhat = L / b
    Rmax = [n * b for n in range(0, N)]
    end2end = [n**(1/2) * b for n in range(0, N)]
    end_to_end_v_Rmax = np.zeros(N,)
    for j in range(ntraj):
        X, t_save = process_sim(simdir / f'tape{j}.csv')
        DT = np.diff(t_save)[0]  # time between save points
        nrousetimes = int(np.ceil(350. / DT))  # number of frames that make up a rouse time
        ntimes, _, _ = X.shape
        nreplicates = ntraj * len(range(nrousetimes, ntimes, nrousetimes))
        if relative:
            Xeq, _ = process_sim(Path(relative) / f'tape{j}.csv')
        for i in range(nrousetimes, ntimes, nrousetimes):
            end_to_end_squared = np.sum((X[i, :, :] - X[i, 0, :])**2, axis=-1)
            end_to_end_v_Rmax += np.sqrt(end_to_end_squared)
    end_to_end_v_Rmax /= nreplicates
    nu = stats.linregress(np.log(Rmax[1:]), np.log(end_to_end_v_Rmax[1:]))[0]
    fig, ax = plt.subplots()
    ax.plot(Rmax, end_to_end_v_Rmax, label=f'$bN^{{{nu:.3f}}}$, n={nreplicates}')
    ax.plot(Rmax, end2end, label=r'$bN^{1/2}$')
    ax.set_xlabel(r'$R_{max}$')
    ax.set_ylabel(r'$\sqrt{\langle R^2 \rangle}$')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    fig.tight_layout()
    plt.savefig('plots/end2end_screq.pdf')
    plt.show()

def average_R2_vs_time(simdir, b=1, D=1, L=100, N=101, ntraj=16):
    simdir = Path(simdir)
    Nhat = L/b
    fig, ax = plt.subplots()
    palette = sns.cubehelix_palette(n_colors=ntraj)
    ord = np.random.permutation(len(palette))
    X, t_save = process_sim(simdir / f'tape0.csv')
    average_R2 = np.zeros((len(t_save),))
    for i in range(ntraj):
        X, t_save = process_sim(simdir/f'tape{i}.csv')
        end_to_end_squared = np.sum((X[:, -1, :] - X[:, 0, :])**2, axis=-1)
        average_R2 += end_to_end_squared
        #ax.plot(t_save, end_to_end_squared, color=palette[ord[i]], alpha=0.4)
    average_R2 /= ntraj
    ax.plot(t_save, np.sqrt(average_R2), 'k-', label=f'simulation average (N={ntraj})')
    meanR = np.mean(np.sqrt(average_R2[2:]))
    nu = np.log(meanR)/np.log(Nhat)
    print(f'Average end to end distance: {meanR}')
    ax.plot(t_save, np.tile(meanR, len(t_save)), label=f'$bN^{{{nu:.3f}}}$')
    ax.plot(t_save, np.tile(np.sqrt(Nhat) * b, len(t_save)), label=r'$bN^{1/2}$')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\sqrt{\langle R^2 \rangle}$')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/end_to_end_distance_vs_time_{simdir.name}.pdf')
    plt.show()

def plot_correlation(C, name, title):
    fig, ax = plt.subplots()
    sns.heatmap(C, xticklabels=25, yticklabels=25, cmap='viridis', square=True, linewidths=0, ax=ax)
    ax.set_title(f'{title}')
    ax.set_xlabel(r'Bead $i$')
    ax.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    plt.savefig(f'plots/correlation_{name}.pdf')

def plot_cov_from_corr(mat, rhos, D, name,
                       title=r'$\langle \eta_i \eta_j \rangle = 2\sqrt{D_i}\sqrt{D_j}C_{ij}$',
                       N=101, L=100, b=1):
    Nhat = L / b
    Dhat = D * N / Nhat
    stds = np.sqrt(2 * Dhat)
    cov = covariance_from_noise(mat, rhos, stds, niter=100000)
    fig, ax = plt.subplots()
    res = sns.heatmap(cov, xticklabels=25, yticklabels=25, cmap='viridis', square=True,
                     linewidths=0, ax=ax)
    # make frame visible
    for _, spine in res.spines.items():
        spine.set_visible(True)
    ax.set_title(title)
    ax.set_xlabel(r'Bead $i$')
    ax.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    plt.savefig(f'plots/idcov_{name}.pdf')
    plt.show()

def plot_msd_from_center(eqdist, step_dist, conf_step_dist, N=101):
    """ For all three simulation replicates, plot the MSD from the center
    bead in the chain."""
    #_, eqdist = two_point_msd('csvs/bdeq1', 96)
    #_, step_dist = two_point_msd('csvs/step_7x', 96)
    #_, conf_step_dist = two_point_msd('csvs/conf_step_7x', 96)
    fig, ax = plt.subplots()
    ax.plot(np.arange(50, N), eqdist[int(N / 2), int(N / 2):], label=r'$T_{\rm eq}$')
    ax.plot(np.arange(50, N), step_dist[int(N / 2), int(N / 2):], label=r'$[0.25 - 1.75]T_{\rm eq}$')
    ax.plot(np.arange(50, N), conf_step_dist[int(N / 2), int(N / 2):], label=r'$[0.25 - 1.75]T_{\rm eq}$, confined')
    ax.legend()
    ax.set_xlabel('Beads $i > 50$')
    ax.set_ylabel(r'$\langle \Vert \vec{r}_i - \vec{r}_{50} \Vert \rangle$')
    ax.set_title('Distance from center')
    fig.tight_layout()
    plt.savefig(f'plots/msd_from_center_step.pdf')

def radius_of_gyration(simdir, ntraj=96):
    """ Radius of gyration is defined as the mean distance of all beads to center of mass."""
    Rg = []
    for j in range(ntraj):
        X, t_save = process_sim((Path(simdir) / f'tape{j}.csv'))
        DT = np.diff(t_save)[0]  # time between save points
        nrousetimes = int(np.ceil(350. / DT))  # number of frames that make up a rouse time
        ntimes, _, _ = X.shape
        for i in range(nrousetimes, ntimes, nrousetimes):
            com = np.mean(X[i, :, :], axis=0)
            Rg.append(np.sum((X[i, :, :] - com)**2, axis=-1).mean())
    return np.array(Rg).mean()

def distance_from_com(simdir, ntraj=96, N=101):
    B = 2 * np.pi / 25
    D = 0.9 * np.cos(B * np.arange(0, N)) + 1
    hot_beads = np.zeros((N,), dtype=bool)
    hot_beads[D == 1.9] = True
    print(hot_beads.sum())
    cold_beads = np.zeros((N,), dtype=bool)
    cold_beads[D == D.min()] = True
    print(cold_beads.sum())
    hot_bead_distances = []
    cold_bead_distances = []
    for j in range(ntraj):
        X, t_save = process_sim((Path(simdir) / f'tape{j}.csv'))
        ntimes, _, _ = X.shape
        for i in range(int(ntimes // 2), ntimes, 5):
            com = np.mean(X[i, :, :], axis=0)
            distances_to_com = np.sum((X[i, :, :] - com) ** 2, axis=-1)
            hot_bead_distances += list(distances_to_com[hot_beads])
            cold_bead_distances += list(distances_to_com[cold_beads])
    fig, ax = plt.subplots()
    ax.hist(hot_bead_distances, bins='auto', label='hot', density=True, color='red', histtype='step')
    ax.hist(cold_bead_distances, bins='auto', label='cold', density=True, color='blue', histtype='step')
    ax.set_xlabel('Distance to COM')
    ax.set_title('Radial distribution function')
    ax.legend()
    fig.tight_layout()
    plt.savefig('plots/radial_distribution_cosine_conf.pdf')
    return hot_bead_distances, cold_bead_distances

def step_radial_distribution(sim='conf_step_7x', ntraj=96, N=101):
    simpath = Path(f'csvs/{sim}')
    df = pd.read_csv(simpath / 'tape0.csv')
    # extract temperatures
    D = np.array(df[df['t'] == 0.0].D)
    hot_beads = np.zeros((N,), dtype=bool)
    hot_beads[D == D.max()] = True
    print(hot_beads.sum())
    cold_beads = np.zeros((N,), dtype=bool)
    cold_beads[D == D.min()] = True
    print(cold_beads.sum())
    hot_bead_distances = []
    cold_bead_distances = []
    for j in range(ntraj):
        X, t_save = process_sim((simpath / f'tape{j}.csv'))
        ntimes, _, _ = X.shape
        for i in range(int(ntimes // 2), ntimes, 5):
            com = np.mean(X[i, :, :], axis=0)
            distances_to_com = np.sum((X[i, :, :] - com) ** 2, axis=-1)
            hot_bead_distances += list(distances_to_com[hot_beads])
            cold_bead_distances += list(distances_to_com[cold_beads])
    fig, ax = plt.subplots()
    ax.hist(hot_bead_distances, bins='auto', label='hot', density=True, color='red',
            histtype='step')
    ax.hist(cold_bead_distances, bins='auto', label='cold', density=True, color='blue',
            histtype='step')
    ax.set_xlabel('Distance to COM')
    ax.set_title('Radial distribution function')
    ax.legend()
    fig.tight_layout()
    plt.savefig('plots/radial_distribution_cosine_conf.pdf')

def plot_chain(simdir, ntraj=96, mfig=None, **kwargs):
    """ Plot a random chain from the last time point of one of these simulations
    using Mayavi.

    TODO: test. Mayavi is not installing properly on my computer,
    so we use olivo to visualize snapshots instead. """
    if mfig is None:
        mfig = mlab.figure()
    j = np.random.randint(0, ntraj)
    X, t_save = process_sim(Path(simdir) / f'tape{j}.csv')
    #take the last time slice -- 1 x N x 3 matrix
    positions = X[-1, :, :]
    print(positions.shape)
    N = positions.shape[0]
    D = np.tile(1, N)
    cmap = mpl.cm.get_cmap('coolwarm')
    colors = np.tile(mpl.colors.to_rgb(cmap(0.0)), N)
    if simdir.name == 'cosine':
        B = 2 * np.pi / 25
        D = 5 * np.cos(B * np.arange(0, N)) + 6
        norm = mpl.colors.Normalize(vmin=np.min(D), vmax=np.max(D))
        cmap = mpl.cm.get_cmap('coolwarm')
        colors = [mpl.colors.to_rgb(cmap(norm(d))) for d in D]
        assert(len(colors) == N)
    for i in range(positions.shape[0]):
        mlab.points3d(positions[i, 0], positions[i, 1], positions[i, 2], scale_factor=5, figure=mfig,
                      color=colors[i], **kwargs)
    return mfig

def draw_power_law_triangle(alpha, x0, width, orientation, base=10,
                            hypotenuse_only=False, **kwargs):
    """Draw a triangle showing the best-fit power-law on a log-log scale.

    Parameters
    ----------
    alpha : float
        the power-law slope being demonstrated
    x0 : (2,) array_like
        the "left tip" of the power law triangle, where the hypotenuse starts
        (in log units, to be consistent with draw_triangle)
    width : float
        horizontal size in number of major log ticks (default base-10)
    orientation : string
        'up' or 'down', control which way the triangle's right angle "points"
    base : float
        scale "width" for non-base 10

    Returns
    -------
    corner : (2,) np.array
        coordinates of the right-angled corner of the triangle
    """
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = 'k'
    x0, y0 = [base**x for x in x0]
    x1 = x0*base**width
    y1 = y0*(x1/x0)**alpha
    plt.plot([x0, x1], [y0, y1], 'k')
    if (alpha >= 0 and orientation == 'up') \
    or (alpha < 0 and orientation == 'down'):
        if hypotenuse_only:
            plt.plot([x0, x1], [y0, y1], color=color, **kwargs)
        else:
            plt.plot([x0, x1], [y1, y1], color=color, **kwargs)
            plt.plot([x0, x0], [y0, y1], color=color, **kwargs)
        # plt.plot lines have nice rounded caps
        # plt.hlines(y1, x0, x1, **kwargs)
        # plt.vlines(x0, y0, y1, **kwargs)
        corner = [x0, y1]
    elif (alpha >= 0 and orientation == 'down') \
    or (alpha < 0 and orientation == 'up'):
        if hypotenuse_only:
            plt.plot([x0, x1], [y0, y1], color=color, **kwargs)
        else:
            plt.plot([x0, x1], [y0, y0], color=color, **kwargs)
            plt.plot([x1, x1], [y0, y1], color=color, **kwargs)
        # plt.hlines(y0, x0, x1, **kwargs)
        # plt.vlines(x1, y0, y1, **kwargs)
        corner = [x1, y0]
    else:
        raise ValueError(r"Need $\alpha\in\mathbb{R} and orientation\in{'up', 'down'}")
    return corner

def analyze_idcorrs(N=101):
    """ Generate two-point MSD plots, contact maps, plots of covariance matrices for the
    simulations run using the procedure that generates correlations by assigning identities to
    monomers."""
    rhos = np.array([0.5])
    """
    #simulation `1feat_rho.5_sameT`
    D = np.tile(1.0, N)
    mat = np.zeros((1, N))
    mat[0, 0:25] = 1.0
    mat[0, 50:75] = -1.0
    plot_cov_from_corr(mat, rhos, D, '1feat_rho.5_sameT', 'Covariance matrix')
    Rgs, avedist, eqdist = two_point_msd('csvs/1feat_rho.5_sameT', ntraj=96)
    counts, Ploop, sdistances, nreplicates = contact_probability(1.0, 'csvs/1feat_rho.5_sameT', 96)

    #simulation `1feat_rho.5_altT`
    D = np.tile(1.0, N)
    D[0:20] = 5.0
    D[40:60] = 5.0
    D[80:] = 5.0
    mat = np.zeros((1, N))
    mat[0, 40:60] = 1.0
    mat[0, 80:] = 1.0
    plot_cov_from_corr(mat, rhos, D, '1feat_rho.5_altT', 'Covariance matrix')
    Rgs, avedist, eqdist = two_point_msd('csvs/1feat_rho.5_altT', ntraj=96)
    counts, Ploop, sdistances, nreplicates = contact_probability(1.0, 'csvs/1feat_rho.5_altT', 96)

    #simulation `1feat_rho.5_altT_alt0'
    D = np.tile(1.0, N)
    D[0:33] = 5.0
    D[66:] = 5.0
    mat = np.zeros((1, N))
    mat[0, 0:11] = 1.0
    mat[0, 22:33] = -1.0
    mat[0, 33:44] = 1.0
    mat[0, 55:66] = -1.0
    mat[0, 66:77] = 1.0
    mat[0, 88:] = -1.0
    plot_cov_from_corr(mat, rhos, D, '1feat_rho.5_altT_alt0', 'Covariance matrix')
    Rgs, avedist, eqdist = two_point_msd('csvs/1feat_rho.5_altT_alt0', ntraj=96)
    counts, Ploop, sdistances, nreplicates = contact_probability(1.0, 'csvs/1feat_rho.5_altT_alt0',
                                                                 96)
    """
    #simulation `conf1_rho.5_altT`
    D = np.tile(1.0, N)
    D[0:20] = 5.0
    D[40:60] = 5.0
    D[80:] = 5.0
    mat = np.zeros((1, N))
    mat[0, 40:60] = 1.0
    mat[0, 80:] = 1.0
    plot_cov_from_corr(mat, rhos, D, 'conf1_rho.5_altT', 'Covariance matrix')
    Rgs, avedist, eqdist = two_point_msd('csvs/conf1_rho.5_altT', ntraj=96)
    counts, Ploop, sdistances, nreplicates = contact_probability(1.0, 'csvs/conf1_rho.5_altT', 96)

    #simulation `conf`_rho.5_altT_alt0`
    D = np.tile(1.0, N)
    D[0:33] = 5.0
    D[66:] = 5.0
    mat = np.zeros((1, N))
    mat[0, 0:11] = 1.0
    mat[0, 22:33] = -1.0
    mat[0, 33:44] = 1.0
    mat[0, 55:66] = -1.0
    mat[0, 66:77] = 1.0
    mat[0, 88:] = -1.0
    plot_cov_from_corr(mat, rhos, D, '1feat_rho.5_altT_alt0', 'Covariance matrix')
    Rgs, avedist, eqdist = two_point_msd('csvs/1feat_rho.5_altT_alt0', ntraj=96)
    counts, Ploop, sdistances, nreplicates = contact_probability(1.0, 'csvs/1feat_rho.5_altT_alt0',
                                                                 96)

def analyze_step_sims(sims=['step_7x']):
    """ Plot mean distance map, contact map (relative and absolute) for the step function
    simulations."""
    for sim in sims:
        simpath = Path(f'csvs/{sim}')
        df = pd.read_csv(simpath / 'tape0.csv')
        # extract temperatures
        Ds = np.array(df[df['t'] == 0.0].D)
        if 'conf' in sim:
            Rgsquared, dist, eqdist = two_point_msd(f'csvs/{sim}', 96,
                                                  relative='csvs/bdeq_conf_Aex5_R5', squared=False)
        else:
            Rgsquared, dist, eqdist = two_point_msd(f'csvs/{sim}', 96, relative='csvs/bdeq1',
                                                    squared=False)
        #print(f'Radius gyration of {sim}: {Rgsquared}')
        #mdmap_abs_rel(dist, eqdist, Ds, sim, relative=True)
        #heatmap_divider(dist, Ds, sim, relative=False)
        #heatmap_divider(dist - eqdist, Ds, sim, relative=True)
        counts, contacts = contact_probability(1.0, f'csvs/{sim}', 96)
        plot_contact_map_temps(contacts, Ds, sim)

def analyze_sims(sims=['conf1_alt0_altT',
                       'conf1_alt0_sameT', 'conf1_altT']):
    """ Plot mean distance map, contact map, and covariance matrix for each simulation in sims."""
    rhos = np.array([0.5])
    for sim in sims:
        simpath = Path(f'csvs/{sim}')
        df = pd.read_csv(simpath/'tape0.csv')
        #extract temperatures
        Ds = np.array(df[df['t']==0.0].D)
        #extract covariances
        mat = np.zeros((1, 101))
        if (simpath/'idmat.csv').is_file():
            df = pd.read_csv(simpath/'idmat.csv')
            mat = np.array(df)
            mat = mat[0, 1:].reshape((1, 101))
            print(mat.shape)
        elif 'alt0' in sim:
            mat[0, 0:20] = -1.0
            mat[0, 40:60] = 1.0
            mat[0, 80:] = 1.0
        """
        if 'alt' in sim:
            plot_cov_from_corr(mat, rhos, Ds, sim, r'$\langle \eta_i \eta_j \rangle$')
        if sim == 'conf1_altT':
            Rgs, dist = two_point_msd(f'csvs/{sim}', 96)
            heatmap_divider(dist, Ds, sim, relative=False)
        if sim == 'conf1_alt0_altT':
            Rgs, dist, eqdist = two_point_msd(f'csvs/{sim}', 96, relative='csvs/conf1_altT')
            heatmap_divider(dist, Ds, sim, relative=False)
            heatmap_divider(dist - eqdist, Ds, sim, relative=True)
        if sim == 'conf1_alt0_sameT':
            Rgs, dist, eqdist = two_point_msd(f'csvs/{sim}', 96, relative='csvs/bdeq_conf_Aex5_R5')
            heatmap_divider(dist, Ds, sim, relative=False)
            heatmap_divider(dist - eqdist, Ds, sim, relative=True)
        """

        counts, contacts = contact_probability(1.0, f'csvs/{sim}', 96)
        plot_contact_map_temps(contacts, Ds, sim)



