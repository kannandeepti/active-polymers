""" Script to run brownian dynamics simulations of active polymer."""
import numpy as np
from bd import recommended_dt, with_srk1
from rouse import linear_mid_msd, end2end_distance_gauss
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from multiprocessing import Pool
import time
sns.set()

params = {'axes.edgecolor': 'black',
                  'axes.facecolor':'white',
                  'axes.grid': False,
                  'axes.titlesize': 11,
                  'axes.labelsize': 11,
                  'legend.fontsize': 9,
                  'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'figure.figsize': [4.7, 3.36],
                  'font.family': 'sans-serif',
                  'mathtext.fontset': 'cm',
                  'xtick.bottom':True,
                  'xtick.top': False,
                  'xtick.direction': 'out',
                  'xtick.major.pad': 3,
                  'xtick.major.size': 4,
                  'xtick.minor.bottom': False,
                  'xtick.major.width': 0.2,
                  'ytick.left':True,
                  'ytick.right':False,
                  'ytick.direction':'out',
                  'ytick.major.pad': 3,
                  'ytick.major.size': 4,
                  'ytick.major.width': 0.3,
                  'ytick.minor.right':False,
                  'lines.linewidth':2}
plt.rcParams.update(params)
sns.set()
textwidth = 6.5

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
    assert(len(t_save) == X.shape[0])
    return X, t_save

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
        TODO: look at doi and edwards and figure out theory for this
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

def rouse_msd(X, t_save, b=1, D=1, L=100, N=101):
    """ Compute MSD of individual beads of polymer averaged over all beads."""
    mean_monomer_msd = np.sum((X[:, :, :] - X[0, :, :])**2, axis=-1).mean(axis=-1)
    mid_monomer_msd = np.sum((X[:, int(N/2) - 1, :] - X[0, int(N/2) - 1, :]) ** 2, axis=-1)
    fig, ax = plt.subplots()
    ax.plot(t_save, mean_monomer_msd, label='simulation, mean')
    ax.plot(t_save, mid_monomer_msd, label='simulation, mid')
    Nhat = L/b
    analytical_msd = linear_mid_msd(t_save, b, Nhat, D, num_modes=int(N / 2))
    ax.plot(t_save, analytical_msd, label='theory')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean monomer MSD')
    plt.legend()
    fig.tight_layout()
    plt.show()

def ensemble_ave_rouse_msd(simdir, b=1, D=1, L=100, N=101, ntraj=32):
    simdir = Path(simdir)
    simname = simdir.name
    Nhat = L/b
    fig, ax = plt.subplots()
    palette = sns.cubehelix_palette(n_colors=ntraj)
    ord = np.random.permutation(len(palette))
    hot_monomer_msds = np.zeros((201,))
    cold_monomer_msds = np.zeros((201,))
    mean_monomer_msds = np.zeros((201,))
    for i in range(ntraj):
        X, t_save = process_sim(simdir/f'tape{i}.csv')
        hot_bead_msd = np.sum((X[:, int(N/2), :] - X[0, int(N/2), :]) ** 2, axis=-1)
        cold_beads = np.arange(0, N) != int(N/2)
        cold_beads_msd = np.sum((X[:, cold_beads, :] - X[0, cold_beads, :]) ** 2, axis=-1).mean(axis=-1)
        mean_beads_msd = np.sum((X[:, :, :] - X[0, :, :]) ** 2, axis=-1).mean(axis=-1)
        hot_monomer_msds += hot_bead_msd
        cold_monomer_msds += cold_beads_msd
        mean_monomer_msds += mean_beads_msd
        if simname == 'mid_hot_bead':
            ax.plot(t_save, hot_bead_msd, color=palette[ord[i]], alpha=0.4)
        elif simname == 'bdeq':
            ax.plot(t_save, mean_beads_msd, color=palette[ord[i]], alpha=0.4)

    if simname == 'mid_hot_bead':
        ax.plot(t_save, hot_monomer_msds / ntraj, 'r-', label=f'hot bead (N={ntraj})')
        ax.plot(t_save, cold_monomer_msds / ntraj, 'b--', label=f'cold beads (N={ntraj})')
    elif simname == 'bdeq':
        ax.plot(t_save, mean_monomer_msds / ntraj, 'b-', label=f'simulation average (N={ntraj})')
    analytical_msd = linear_mid_msd(t_save, b, Nhat, D, num_modes=int(N / 2))
    ax.plot(t_save, analytical_msd, 'k-', label=r'theory, $T_{\rm eq}$')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean monomer MSD')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/monomer_msd_{simname}.pdf')
    return fig, ax

def average_R2_vs_time(simdir, b=1, D=1, L=100, N=101, ntraj=16):
    simdir = Path(simdir)
    Nhat = L/b
    fig, ax = plt.subplots()
    palette = sns.cubehelix_palette(n_colors=ntraj)
    ord = np.random.permutation(len(palette))
    average_R2 = np.zeros((201,))
    for i in range(ntraj):
        X, t_save = process_sim(simdir/f'tape{i}.csv')
        end_to_end_squared = np.sum((X[:, -1, :] - X[:, 0, :])**2, axis=-1)
        average_R2 += end_to_end_squared
        ax.plot(t_save, end_to_end_squared, color=palette[ord[i]], alpha=0.4)
    ax.plot(t_save, average_R2/ntraj, 'k-', label=f'simulation average (N={ntraj})')
    ax.plot(t_save, np.tile(Nhat * b ** 2, len(t_save)), label='theory')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle R^2 \rangle$')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/rsquared_vs_time_{simdir.name}.pdf')
    return fig, ax

def two_point_msd(simdir, ntraj, N=101):
    """Compute mean squared distance between two beads on a polymer at
    a particular time point. Plot heatmap"""
    #TODO: average over structures and average over time
    simdir = Path(simdir)
    ntimes = 101
    average_dist = np.zeros((N, N))
    nreplicates = ntraj * (ntimes - int(ntimes//2))
    for j in range(ntraj):
        X, t_save = process_sim(simdir / f'tape{j}.csv')
        for i in range(int(ntimes//2), ntimes):
            dist = pdist(X[i, :, :], metric='euclidean')
            Y = squareform(dist)
            average_dist += Y
    fig, ax = plt.subplots()
    sns.heatmap(average_dist / nreplicates, xticklabels=10, yticklabels=10, ax=ax)
    ax.set_title(r'Mean squared distance $\langle r_i(t) - r_j(t) \rangle$')
    ax.set_xlabel(r'Bead $i$')
    ax.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    plt.savefig(f'plots/two_point_msd_{simdir.name}.pdf')


def end_to_end_distance(simdir, ntraj=16, b=1, L=100, N=101):
    """ End to end distance is <R_N(t) - R_0(t)>, i.e. the norm of the position vector
        of last bead minus position vector of first beed.
        TODO: look at doi and edwards and figure out theory for this
        """
    #take mean over time after steady state
    sqrt_r2 = np.zeros((N,))
    for j in range(ntraj):
        X, t_save = process_sim(Path(simdir) / f'tape{j}.csv')
        end_to_end = np.linalg.norm((X[-1, :, :] - X[-1, 0, :]), axis=-1)
        sqrt_r2 += end_to_end
    #end_to_end = np.mean(end_to_end, axis=0)
    #print(end_to_end.shape)
    fig, ax = plt.subplots()
    Nhat = L/b #number of kuhn lengths
    L0 = L/(N-1)  # length per bead
    r = [n*L0 for n in range(N)]
    ax.plot(r, sqrt_r2/ntraj, label=f'simulation average (N={ntraj})')
    #analytical_r2 = end2end_distance_gauss(r, b, Nhat, L)
    #ax.plot(r, analytical_r2, label='theory')
    ax.set_xlabel(r'$R_{max}$')
    ax.set_ylabel(r'$\langle R^2 \rangle$')
    plt.legend()
    #plt.xscale('log')
    #plt.yscale('log')
    fig.tight_layout()
    plt.show()