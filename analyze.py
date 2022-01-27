""" Script to run brownian dynamics simulations of active polymer."""
import numpy as np
from rouse import linear_mid_msd, end2end_distance_gauss, gaussian_Ploop
from correlations import *
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

def rouse_msd(X, t_save, b=1, D=1, L=100, N=101, theory=True):
    """ Compute MSD of individual beads of polymer averaged over all beads."""
    mean_monomer_msd = np.sum((X[:, :, :] - X[0, :, :])**2, axis=-1).mean(axis=-1)
    mid_monomer_msd = np.sum((X[:, int(N/2) - 1, :] - X[0, int(N/2) - 1, :]) ** 2, axis=-1)
    fig, ax = plt.subplots()
    times = np.logspace(-4, 3, 1000)
    ax.plot(times, 6 * (D / N) * times, 'r--')
    ax.plot(times, (12 / np.pi * D * times) ** (1 / 2) * b, 'b--')
    ax.plot(times, 6 * D * times, 'g--')
    ax.plot(t_save, mean_monomer_msd, '-', label='simulation, mean')
    ax.plot(t_save, mid_monomer_msd, '-', label='simulation, mid')
    if theory:
        Nhat = L/b
        analytical_msd = linear_mid_msd(times, b, Nhat, D, num_modes=int(N / 2))
        ax.plot(times, analytical_msd, 'k-', label='theory')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean monomer MSD')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    fig.tight_layout()
    plt.show()

def ensemble_ave_rouse_msd(simdir, b=1, D=1, L=100, N=101, ntraj=96):
    simdir = Path(simdir)
    simname = simdir.name
    Nhat = L/b
    fig, ax = plt.subplots()
    mid_monomer_msds = np.zeros((201,))
    mean_monomer_msds = np.zeros((201,))
    for i in range(ntraj):
        X, t_save = process_sim(simdir/f'tape{i}.csv')
        mid_bead_msd = np.sum((X[:, int(N/2), :] - X[0, int(N/2), :]) ** 2, axis=-1)
        mean_beads_msd = np.sum((X[:, :, :] - X[0, :, :]) ** 2, axis=-1).mean(axis=-1)
        mid_monomer_msds += mid_bead_msd
        mean_monomer_msds += mean_beads_msd
        #if simname == 'mid_hot_bead':
        #    ax.plot(t_save, hot_bead_msd, color=palette[ord[i]], alpha=0.4)
        #elif simname == 'bdeq':
        
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
    #at times >> t_R, polymer should diffuse with Dg = D/N
    rouse_time = (Nhat**2) * (b**2) / (3 * np.pi**2 * D)
    times =  np.logspace(-3, 5)
    fig, ax = plt.subplots()
    analytical_msd = linear_mid_msd(times, b, Nhat, D, num_modes=int(N / 2))
    ax.plot(times, analytical_msd, 'k-', label=r'theory, $T_{\rm eq} = 1$')
    ax.plot(times, 6 * (D/N) * times, 'r--', label=r'$6Dt/N$')
    ax.plot(times, (12/np.pi * D * times)**(1/2) * b, 'b--', label=r'$(12Db^2 t/\pi)^{1/2}$')
    ax.plot(times, 6 * D * times, 'g--', label=r'$6Dt$')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mean monomer MSD')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    fig.tight_layout()
    plt.savefig('plots/rouse_msd_3regimes.pdf')


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

def plot_correlation(C, name, title):
    fig, ax = plt.subplots()
    sns.heatmap(C, xticklabels=25, yticklabels=25, cmap='viridis', square=True, linewidths=0, ax=ax)
    ax.set_title(f'{title}')
    ax.set_xlabel(r'Bead $i$')
    ax.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    plt.savefig(f'plots/correlation_{name}.pdf')

def plot_cov_from_corr(mat, rhos, D, name, title=r'$\langle \eta_i \eta_j \rangle$', N=101, L=100,
                       b=1):
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

def two_point_msd(simdir, ntraj, N=101, relative=None, squared=False):
    """Compute mean squared distance between two beads on a polymer at
    a particular time point. Plot heatmap"""
    #TODO: average over structures and average over time
    simdir = Path(simdir)
    average_dist = np.zeros((N, N))
    eq_dist = np.zeros((N, N))
    metric = 'euclidean'
    if squared:
        metric = 'sqeuclidean'
    #ignore first half of tape (steady state) and then take time slices every 10 save points
    #to sample equilibrium structures
    for j in range(ntraj):
        X, t_save = process_sim(simdir / f'tape{j}.csv')
        ntimes, _, _ = X.shape
        nreplicates = ntraj * len(range(int(ntimes // 2), ntimes, 5))
        if relative:
            Xeq, _ = process_sim(Path(relative) / f'tape{j}.csv')
        for i in range(int(ntimes//2), ntimes, 5):
            #for temperature modulations
            dist = pdist(X[i, :, :], metric=metric)
            Y = squareform(dist)
            average_dist += Y
            #for equilibrium case
            if relative:
                dist = pdist(Xeq[i, :, :], metric=metric)
                eq_dist += squareform(dist)

    average_dist = average_dist / nreplicates
    if squared:
        Rg_squared = np.sum(average_dist) / (2 * N ** 2)
    else:
        Rg_squared = np.sum(average_dist ** 2) / (2 * N ** 2)

    if relative:
        eq_dist = eq_dist / nreplicates
        return Rg_squared, average_dist, eq_dist

    return Rg_squared, average_dist

def plot_msd_map(dist, simdir, relative=False, squared=False):
    """ Plot heatmap where entry (i, j) is the mean distance between beads i and j. This version
    does not include a bar showing the temperature of the beads or anything."""
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

def heatmap_temps(mat=None, D=None, width=10, **kwargs):
    """ Test heatmap that contains colorbar, and 2 bars on bottom and left specifying
    temperatures of the beads. BUGGY. """
    if mat is None:
        mat = np.random.randn((101, 101))
    if D is None:
        D = np.random.rand(101)
    D = 0.25 * np.ones((width, 101))
    D[:, 30:50] = 1.75
    D[:, 80:] = 1.75
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 3),
                     axes_pad=0.1,
                     )
    #make bottom left axis disappear
    grid[3].axis('off')
    grid[5].axis('off')
    grid[4].set_xticks([0, 25, 50, 75, 100])
    grid[0].set_yticks([0, 25, 50, 75, 100])
    im = grid[1].imshow(mat, cmap='coolwarm', **kwargs)
    grid[4].imshow(D, cmap='coolwarm', vmin=1.0)
    grid[0].imshow(D.T, cmap='coolwarm', vmin=1.0)
    plt.colorbar(im, cax=grid[2])
    #divider = make_axes_locatable(grid[2])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #plt.colorbar(im, cax=cax)
    """
    ax = plt.subplot(111)
    divider = make_axes_locatable(ax)
    ax_bottom = divider.append_axes("bottom", size=f"{width/101}%", pad=0.1, sharex=ax)
    ax_left = divider.append_axes("left", size=f"{width/101}%", pad=0.1, sharey=ax)

    res = sns.heatmap(mat, xticklabels=25, yticklabels=25,
                cmap='magma', square=True, ax=ax, **kwargs)
    ax_bottom.imshow(D, cmap='coolwarm', vmin=1.0)
    #ax_bottom.set_xticks([0, 25, 50, 75, 100])
    #ax_left.imshow(D.T, cmap='coolwarm', vmin=1.0)
    #ax_left.set_yticks([0, 25, 50, 75, 100])

    #temps = sns.heatmap(D, cbar=False, xticklabels=False, yticklabels=False, ax=ax_left,
    #                    cmap='coolwarm', vmin=1.0)
    #temps = sns.heatmap(D, cbar=False, xticklabels=False, yticklabels=False, ax=ax_bottom,
    #                    cmap='coolwarm', vmin=1.0)
    """
    # make frame visible
    #for _, spine in res.spines.items():
    #    spine.set_visible(True)

    grid[1].set_title(r'Mean distance $\langle\Vert \vec{r}_i - \vec{r}_j \Vert\rangle$')
    grid[4].set_xlabel(r'Bead $i$')
    grid[0].set_ylabel(r'Bead $j$')
    fig.tight_layout()
    plt.show()

def heatmap_divider(mat, temps, simname, relative=False, width=5, **kwargs):
    """ Tested and this works!!!! """
    if mat is None:
        mat = np.random.randn((101, 101))

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
    if relative:
        im = ax.imshow(mat, norm=colors.CenteredNorm(), cmap=cmap_relative)
    else:
        im = ax.imshow(mat, cmap=cmap_distance, **kwargs)
    ax_bottom.imshow(D, cmap='coolwarm', vmin=0.25, vmax=1.75)
    ax_left.imshow(D.T, cmap='coolwarm', vmin=0.25, vmax=1.75)
    fig.colorbar(im, cax=cax)
    if relative:
        ax.set_title(r'$\langle r_{ij} \rangle  - \langle r_{ij} \rangle_{eq}$')
        #ax.set_title(r'Relative change in $\langle r_{ij} \rangle$')
    else:
        ax.set_title(r'Mean distance $\langle r_{ij} \rangle$')
    ax_bottom.set_xlabel(r'Bead $i$')
    ax_left.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    if relative:
        plt.savefig(f'plots/two_point_msd_{simname}_relative.pdf')
    else:
        plt.savefig(f'plots/two_point_msd_{simname}.pdf')

def mdmap_abs_rel(dist, eqdist, temps, simname, relative=False, width=5, **kwargs):
    """ Tested and this works!!!! """
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
    im2 = ax.imshow(rel_dist[::-1, :], norm=colors.CenteredNorm(), cmap=cmap_relative, **kwargs)
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

def contact_probability(a, simdir, ntraj, N=101, eq_contacts=None):
    """Compute mean squared distance between two beads on a polymer at
    a particular time point. Plot heatmap"""
    #TODO: average over structures and average over time
    simdir = Path(simdir)
    average_dist = np.zeros((N, N))
    #counts(i, j) = number of times monomer i and j loop within contact radius a
    counts = np.zeros((N, N))
    metric = 'euclidean'
    #ignore first half of tape (steady state) and then take time slices every 10 save points
    #to sample equilibrium structures
    for j in range(ntraj):
        X, t_save = process_sim(simdir / f'tape{j}.csv')
        ntimes, _, _ = X.shape
        nreplicates = ntraj * len(range(int(ntimes // 2), ntimes, 5))
        Xeq, _ = process_sim(Path('csvs/bdeq1') / f'tape{j}.csv')
        for i in range(int(ntimes//2), ntimes, 5):
            #for temperature modulations
            dist = pdist(X[i, :, :], metric=metric)
            Y = squareform(dist)
            counts += (Y < a)
            average_dist += Y
    contacts = counts / nreplicates
    return counts, contacts

def plot_contact_map(contacts, eq_contacts = None):
    #Plot contact map (each entry is probability of looping within radius a)
    fig, ax = plt.subplots()
    if eq_contacts is not None:
        res = sns.heatmap(contacts - eq_contacts, center=0.0, square=True,
                    cmap="vlag", xticklabels=25, yticklabels=25, robust=True, ax=ax)
        ax.set_title(r'Contact map relative to equilibrium')
    else:
        contacts[contacts == 0] = 1e-5
        lognorm = LogNorm(vmin=contacts.min(), vmax=contacts.max())
        res = sns.heatmap(contacts, norm=lognorm,
                    cmap="Reds", square=True, xticklabels=25, yticklabels=25, robust=True, ax=ax)
        ax.set_title(r'Contact Map $P(\Vert \vec{r}_i - \vec{r}_j \Vert < a)$')
    ax.set_xlabel(r'Bead $i$')
    ax.set_ylabel(r'Bead $j$')
    # make frame visible
    for _, spine in res.spines.items():
        spine.set_visible(True)
    fig.tight_layout()
    plt.savefig(f'plots/contact_map_{simdir.name}.pdf')

def plot_contact_map_temps(contacts, temps, simname, width=5, **kwargs):
    contacts[contacts == 0] = 1e-5
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
    ax.set_title(r'Contact Map $P(\Vert \vec{r}_i - \vec{r}_j \Vert < a)$')
    ax_bottom.set_xlabel(r'Bead $i$')
    ax_left.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    plt.savefig(f'plots/contact_map_{simname}_temps.pdf')

def compute_ploop(counts, nreplicates):
    #To compute histogram as a function of s, distance along chain, sum over diagonals
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
    #sdistances is anyway in units of kuhn lengths
    analytical_ploop = [gaussian_Ploop(a, n, b) for n in sdistances[1:]]
    fig, ax = plt.subplots()
    ax.plot(sdistances, Ploop, label=f'Simulation estimate (n={nreplicates})')
    ax.plot(sdistances[1:], analytical_ploop, label='Theory')
    corner = draw_power_law_triangle(-3/2, [1, -0.5], 0.5, 'up', base=10,
                            hypotenuse_only=False)
    ax.text(12.0, 0.07, r'$s^{-3/2}$')
    plt.yscale('log')
    plt.xscale('log')
    ax.set_xlabel('Loop size')
    ax.set_ylabel(f'Looping probability')
    plt.legend()
    fig.tight_layout()
    plt.savefig(f'plots/ploop_{descriptor}.pdf')

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
        ntimes, _, _ = X.shape
        for i in range(int(ntimes // 2), ntimes, 10):
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

def plot_chain(simdir, ntraj=96, mfig=None, **kwargs):
    """ Plot a random chain from the last time point of one of these simulations"""
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
    x0, y0 = [base**x for x in x0]
    x1 = x0*base**width
    y1 = y0*(x1/x0)**alpha
    plt.plot([x0, x1], [y0, y1], 'k')
    if (alpha >= 0 and orientation == 'up') \
    or (alpha < 0 and orientation == 'down'):
        if hypotenuse_only:
            plt.plot([x0, x1], [y0, y1], 'k')
        else:
            plt.plot([x0, x1], [y1, y1], 'k')
            plt.plot([x0, x0], [y0, y1], 'k')
        # plt.plot lines have nice rounded caps
        # plt.hlines(y1, x0, x1, **kwargs)
        # plt.vlines(x0, y0, y1, **kwargs)
        corner = [x0, y1]
    elif (alpha >= 0 and orientation == 'down') \
    or (alpha < 0 and orientation == 'up'):
        if hypotenuse_only:
            plt.plot([x0, x1], [y0, y1], 'k')
        else:
            plt.plot([x0, x1], [y0, y0], 'k')
            plt.plot([x1, x1], [y0, y1], 'k')
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



