"""
Dynamical observables
---------------------

Script to efficiently calculate mean squared displacements of individual
monomers or of the center of mass of the polymer over time. """

import numpy as np
import scipy as sp
import pandas as pd
from numba import njit, jit
from pathlib import Path
import matplotlib as mpl
from matplotlib import pyplot as plt
from .rouse import linear_mid_msd
from .files import *
from .analyze import draw_power_law_triangle, plot_cov_from_corr

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
                  'font.size' : 18,
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

@jit(nopython=True)
def select_hot_cold_subsets(X, D):
    """ Return a subset of X with M rows, according to
    M hot monomers, and another subset of X with N rows, according to N cold monomers."""
    num_t, num_beads, d = X.shape
    num_hot = np.sum(D == D.max())
    num_cold = np.sum(D == D.min())
    Xhot = np.zeros((num_t, num_hot, d))
    Xcold = np.zeros((num_t, num_cold, d))
    hotind = 0
    coldind = 0
    for i in range(num_beads):
        if D[i] == D.max():
            Xhot[:, hotind, :] = X[:, i, :]
            hotind += 1
        if D[i] == D.min():
            Xcold[:, coldind, :] = X[:, i, :]
            coldind += 1
    return Xhot, Xcold

@jit(nopython=True)
def get_bead_msd(Xhot, Xcold):
    """Return mean of hot monomer MSDs and cold monomer MSDs."""
    num_t, num_hot, d = Xhot.shape
    hot_msd = np.zeros((num_t - 1,))
    cold_msd = np.zeros((num_t - 1,))
    count = np.zeros((num_t - 1,))
    for i in range(num_t - 1):
        for j in range(i, num_t - 1):
            diff = Xhot[j] - Xhot[i]
            hot_msd[j-i] += np.mean(np.sum(diff * diff, axis=-1))
            diff = Xcold[j] - Xcold[i]
            cold_msd[j-i] += np.mean(np.sum(diff * diff, axis=-1))
            count[j-i] += 1
    return hot_msd, cold_msd, count

@jit(nopython=True)
def get_com_msd(X, k=None):
    """time averagerd mean squared displacement of the polymer center of mass
    for a single simulation trajectory X."""
    num_t, num_beads, d = X.shape
    if k is None:
        k = max(0, num_beads/2 - 1)
    k = int(k)
    ta_msd = np.zeros((num_t,))
    com_msd = np.zeros((num_t,))
    count = np.zeros((num_t,))
    for i in range(num_t - 1):
        for j in range(i, num_t - 1):
            com_diff = (np.sum(X[j], axis=0) - np.sum(X[i], axis=0))/num_beads
            com_msd[j-i] = com_diff @ com_diff
            ta_msd[j-i] += (X[j, k] - X[i, k])@(X[j, k] - X[i, k])
            count[j-i] += 1
    return ta_msd, com_msd, count

def hot_cold_monomer_msd(simdir, ntraj=96):
    simdir = Path(simdir)
    hot_msd_ave = np.zeros((1000,))
    cold_msd_ave = np.zeros((1000,))
    df = pd.read_csv(simdir / 'tape0.csv')
    # extract temperatures
    D = np.array(df[df['t'] == 350.0].D)
    for i in range(ntraj):
        X, t_save = process_sim(simdir/f'tape{i}.csv')
        Xcold = X[:, D == D.min(), :]
        Xhot = X[:, D == D.max(), :]
        hot_msd, cold_msd, count = get_bead_msd(Xhot, Xcold)
        hot_msd_ave += hot_msd / count
        cold_msd_ave += cold_msd / count
    hot_msd_ave /= ntraj
    cold_msd_ave /= ntraj
    return hot_msd_ave, cold_msd_ave

def com_time_ave_rouse_msd(simdir, b, D, N, ntraj=96):
    """ Time averaged, ensemble averaged MSD of polymer center of mass. """
    simdir = Path(simdir)
    simname = simdir.name
    ta_msd_ave = np.zeros((1001,))
    com_msd_ave = np.zeros((1001,))
    for i in range(ntraj):
        X, t_save = process_sim(simdir/f'tape{i}.csv')
        ta_msd, com_msd, count = get_com_msd(X)
        ta_msd_ave += ta_msd / count
        com_msd_ave += com_msd / count
    ta_msd_ave /= ntraj
    com_msd_ave /= ntraj
    return ta_msd_ave, com_msd_ave

def plot_com_msd(com_msd_ave, com_msd_ave_neq, simdir='csvs/bdeq_msd', D=1, N=101):
    """ Plot ensemble averaged time averaged MSD of center of mass against theory"""
    fig, ax = plt.subplots()
    X, t_save = process_sim(Path(simdir) / f'tape0.csv')
    ax.plot(t_save, 6 * D * t_save / N, 'k-', label=r'$6Dt/N$')
    ax.plot(t_save, com_msd_ave, 'b--', label='equilibrium')
    ax.plot(t_save, com_msd_ave_neq, 'r--', label='nonequilibrium')
    ax.set_ylabel('Center of mass MSD')
    ax.set_xlabel('Time')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    fig.tight_layout()
    plt.show()

def rouse_msd(X, t_save, b=1, D=1, L=100, N=101, theory=True):
    """ Compute MSD of individual beads of polymer averaged over all beads."""
    mean_monomer_msd = np.sum((X[:, :, :] - X[0, :, :]) ** 2, axis=-1).mean(axis=-1)
    mid_monomer_msd = np.sum((X[:, int(N / 2) - 1, :] - X[0, int(N / 2) - 1, :]) ** 2, axis=-1)
    fig, ax = plt.subplots()
    times = np.logspace(-4, 3, 1000)
    ax.plot(times, 6 * (D / N) * times, 'r--')
    ax.plot(times, (12 / np.pi * D * times) ** (1 / 2) * b, 'b--')
    ax.plot(times, 6 * D * times, 'g--')
    ax.plot(t_save, mean_monomer_msd, '-', label='simulation, mean')
    ax.plot(t_save, mid_monomer_msd, '-', label='simulation, mid')
    if theory:
        Nhat = L / b
        analytical_msd = linear_mid_msd(times, b, Nhat, D, num_modes=int(N / 2))
        ax.plot(times, analytical_msd, 'k-', label='theory')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean monomer MSD')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    fig.tight_layout()
    plt.show()

def analytical_com_diffusion(simdir):
    """ For a given simulation, extract covariance matrix, compute analytical center of mass
    diffusion coefficient (comD) and return ratio of comD to center of mass diffusion coefficient
    of a Rouse polymer."""

    D, idmat, rhos = extract_cov(simdir)
    N = len(D)
    rho = rhos[0]
    corr = np.outer(idmat, idmat)
    corr *= rho
    corr[np.diag_indices(N)] = 1.0 #diagonal of correlation matrix is 1
    cov = np.sqrt(np.outer(D, D)) * corr #cov(i, j) = 2 sqrt(D_i) sqrt(D_j) C_{ij}
    com_diffusion = cov.sum()/ N**2
    # theoretical prediction for com diffusion coefficient except for numerical prefactors
    comD = corr.sum() / N ** 2
    #center of mass diffusion of equilibrium system minus numerical prefactors
    Dg = 1. / N
    return com_diffusion

def msd_from_files(simdir, ntraj=96, b=1.0, corr=False):
    """ Plot subdiffusive dynamics of Rouse monomers from the MSD files
    written to disc during the simulation."""
    simdir = Path(simdir)
    simname = simdir.name
    df = pd.read_csv(simdir / 'tape0.csv')
    # extract temperatures
    D, mat, rhos = extract_cov(simdir)
    D = np.array(df[df['t'] == 350.0].D)
    if (simdir / 'idmat.csv').is_file():
        plot_cov_from_corr(mat, rhos, D, simname)
        mat = mat.reshape((101,))
    N = len(D)
    df = pd.read_csv(simdir / f'msds0.csv')
    ntmsd = len(df['t_msd'])
    cold_msd_ave = np.zeros((ntmsd,)) #+1 if corr = True
    hot_msd_ave = np.zeros((ntmsd,)) #-1 if corr=True
    com_msd_ave = np.zeros((ntmsd,))
    for i in range(ntraj):
        df = pd.read_csv(simdir/f'msds{i}.csv')
        df = df.drop('Unnamed: 0', axis=1)
        com_msd_ave += df.iloc[:, N]
        mondf = df.iloc[:, 0:N]
        if corr:
            cold_msd_ave += mondf.iloc[:, mat == -1.0].mean(axis=1)
            hot_msd_ave += mondf.iloc[:, mat == 0.0].mean(axis=1)
        else:
            cold_msd_ave += mondf.iloc[:, D == D.min()].mean(axis=1)
            hot_msd_ave += mondf.iloc[:, D == D.max()].mean(axis=1)
    cold_msd_ave /= ntraj
    hot_msd_ave /= ntraj
    com_msd_ave /= ntraj
    fig, ax = plt.subplots()
    t_msd = np.array(df['t_msd'])

    #color three regimes
    rouse_time = (N ** 2) * (b ** 2) / (3 * np.pi ** 2 * np.mean(D))
    rouse_time = 10 ** 3
    #transition from initial diffusive to subdiffusive regime
    t1 = b ** 2 / (3 * np.mean(D) * np.pi)
    t1 = 0.1
    #t1 = 10.0
    analytical_msd = linear_mid_msd(t_msd, 1.0, N, np.mean(D), num_modes=int(N / 2))
    ax.plot(t_msd, analytical_msd, 'k-', label=r'Rouse, $T_{\rm eq}$')
    comD = analytical_com_diffusion(simdir)
    #ax.plot(t_msd, 6 * comD * t_msd, 'k--', label=r'Theory ($6D_{COM} t$)')
    ax.set_xlim(t_msd[0], t_msd[-1])
    ax.set_ylim(bottom=10 ** (-3), top=10 ** (4.5))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.fill_between(t_msd, ymin, ymax,
                    where=((t_msd >= xmin) & (t_msd <= t1)),
                    color='#deebf7', alpha=0.5)
    ax.fill_between(t_msd, ymin, ymax,
                    where=((t_msd >= t1) & (t_msd <= rouse_time)),
                    color='#fee6ce', alpha=0.5)
    ax.fill_between(t_msd, ymin, ymax,
                    where=((t_msd >= rouse_time) & (t_msd <= xmax)),
                    color='#e5f5e0', alpha=0.5)
    if corr:
        ax.plot(t_msd, cold_msd_ave, 'b-', label='-1')
        ax.plot(t_msd, hot_msd_ave, 'r-', label='0')
    else:
        if D.min() == D.max():
            mon_msd_ave = (cold_msd_ave + hot_msd_ave)/2
            ax.plot(t_msd, mon_msd_ave, 'b--', label=r'simulation, $T_{\rm eq}$')
        else:
            ax.plot(t_msd[:-1], cold_msd_ave[:-1], 'b-', lw=3,
                    label=r'cold, $T = 0.25T_{\rm eq}$')
            ax.plot(t_msd[:-1], hot_msd_ave[:-1], 'r-', lw=3,
                    label=r'hot, $T = 1.75T_{\rm eq}$')
    slope, intercept, r, p, se = sp.stats.linregress(t_msd[:-1], com_msd_ave[:-1])
    print(f'Slope of center of mass MSD: {slope}')
    print(f'Ratio of equilibrium com MSD to noneq com MSD: {(6 * np.mean(D) / N) / slope}')
    #ax.plot(t_msd[:-1], com_msd_ave[:-1], 'g.-', label=f'com, $D'
    #                                            f'={(slope / (6.0 * np.mean(D) / N)):.2f}D_G$')

    corner = draw_power_law_triangle(1.0, [-2, -0.5], 0.75, 'up', hypotenuse_only=True,
    color = 'grey')
    ax.text(10 ** (-1.75), 10 ** (0.3), r'$\tau^{1}$')
    corner = draw_power_law_triangle(0.5, [0, 1], 2, 'up', hypotenuse_only=True, color='grey')
    ax.text(5, 60, r'$\tau^{1/2}$')
    corner = draw_power_law_triangle(1.0, [3.5, 3], 2, 'up', hypotenuse_only=True, color='grey')
    ax.text(10 ** (4), 7000, r'$\tau^{1}$')

    """
    ax.fill_between(t_msd, ymin, ymax,
                       where=((t_msd >= xmin) & (t_msd <= t1)),
                       color=[0.96, 0.95, 0.95])
    ax.fill_between(t_msd, ymin, ymax,
                       where=((t_msd >= t1) & (t_msd <= rouse_time)),
                       color=[0.99, 0.99, 0.99])
    ax.fill_between(t_msd, ymin, ymax,
                       where=((t_msd >= rouse_time) & (t_msd <= xmax)),
                       color=[0.9, 0.9, 0.91])
    """
    ax.set_xlabel('Time')
    ax.set_ylabel('MSD')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/monomer_msd_{simname}.pdf')
    plt.show()
    return cold_msd_ave, hot_msd_ave, com_msd_ave

def ensemble_ave_rouse_msd(simdir, b=1, D=1, L=100, N=101, ntraj=96):
    simdir = Path(simdir)
    simname = simdir.name
    Nhat = L / b
    fig, ax = plt.subplots()
    mid_monomer_msds = np.zeros((201,))
    mean_monomer_msds = np.zeros((201,))
    for i in range(ntraj):
        X, t_save = process_sim(simdir / f'tape{i}.csv')
        mid_bead_msd = np.sum((X[:, int(N / 2), :] - X[0, int(N / 2), :]) ** 2, axis=-1)
        mean_beads_msd = np.sum((X[:, :, :] - X[0, :, :]) ** 2, axis=-1).mean(axis=-1)
        mid_monomer_msds += mid_bead_msd
        mean_monomer_msds += mean_beads_msd
        # if simname == 'mid_hot_bead':
        #    ax.plot(t_save, hot_bead_msd, color=palette[ord[i]], alpha=0.4)
        # elif simname == 'bdeq':

        #    ax.plot(t_save, mean_beads_msd, color=palette[ord[i]], alpha=0.4)
    times = np.logspace(-3, 5)
    ax.plot(t_save, mid_monomer_msds / ntraj, 'ro', label=f'mid bead (N={ntraj})')
    ax.plot(t_save, mean_monomer_msds / ntraj, 'bo', label=f'mean bead (N={ntraj})')
    analytical_msd = linear_mid_msd(t_save, b, Nhat, D, num_modes=int(N / 2))
    ax.plot(t_save, analytical_msd, 'k-', label=r'theory, $T_{\rm eq}$')
    ax.plot(times, 6 * (D / N) * times, 'r--')
    ax.plot(times, (12 / np.pi * D * times) ** (1 / 2) * b, 'b--')
    ax.plot(times, 6 * D * times, 'g--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean monomer MSD')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/monomer_msd_{simname}.pdf')
    return fig, ax

def plot_analytical_rouse_msd(N, b, D):
    """ Plot three regimes of Rouse monomer MSD."""
    Nhat = N / b
    # at times >> t_R, polymer should diffuse with Dg = D/N
    rouse_time = (Nhat ** 2) * (b ** 2) / (3 * np.pi ** 2 * D)
    times = np.logspace(-3, 5)
    fig, ax = plt.subplots()
    analytical_msd = linear_mid_msd(times, b, Nhat, D, num_modes=int(N / 2))
    ax.plot(times, analytical_msd, 'k-', lw=4, label=r'Rouse, $T_{\rm eq} = 1$')
    #ax.plot(times, 6 * (D / N) * times, 'r--', label=r'$6Dt/N$')
    #ax.plot(times, (12 / np.pi * D * times) ** (1 / 2) * b, 'b--', label=r'$(12Db^2 t/\pi)^{1/2}$')
    #ax.plot(times, 6 * D * times, 'g--', label=r'$6Dt$')
    corner = draw_power_law_triangle(1.0, [-3, -1.5], 1.5, 'up', hypotenuse_only=True,
                                     color='#3182bd')
    ax.text(10**(-2.5), 10**(-0.5), r'$\tau^{1}$')
    corner = draw_power_law_triangle(0.5, [0, 1], 2, 'up', hypotenuse_only=True, color='#e6550d')
    ax.text(5, 60, r'$\tau^{1/2}$')
    corner = draw_power_law_triangle(1.0, [3.5, 3], 2, 'up', hypotenuse_only=True, color='#31a354')
    ax.text(10 ** (4), 7000, r'$\tau^{1}$')
    rouse_time = (N ** 2) * (b ** 2) / (3 * np.pi ** 2 * D)
    rouse_time = 10**3
    # transition from initial diffusive to subdiffusive regime
    t1 = b ** 2 / (3 * D * np.pi)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(bottom=10 ** (-3.5), top=10 ** (4.5))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.fill_between(times, ymin, ymax,
                    where=((times >= xmin) & (times <= t1)),
                    color='#deebf7')
    ax.fill_between(times, ymin, ymax,
                    where=((times >= t1) & (times <= rouse_time)),
                    color='#fee6ce')
    ax.fill_between(times, ymin, ymax,
                    where=((times >= rouse_time) & (times <= xmax)),
                    color='#e5f5e0')

    ax.set_xlabel('Time')
    ax.set_ylabel('Mean monomer MSD')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    fig.tight_layout()
    plt.savefig('plots/rouse_msd_3regimes.pdf')
    plt.show()

def plot_monomer_msd(simdir, hot_msd, cold_msd, D=1, N=101, b=1):
    fig, ax = plt.subplots()
    X, t_save = process_sim(Path(simdir) / f'tape0.csv')
    analytical_msd = linear_mid_msd(t_save, b, N / b, D, num_modes=int(N / 2))
    ax.plot(t_save, analytical_msd, 'k-', label=r'theory')
    ax.plot(t_save[:-1], hot_msd, 'b--', label='hot')
    ax.plot(t_save[:-1], cold_msd, 'r--', label='cold')
    ax.set_ylabel('Monomer MSD')
    ax.set_xlabel('Time')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    fig.tight_layout()
    plt.show()