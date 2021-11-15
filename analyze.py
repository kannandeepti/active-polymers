""" Script to run brownian dynamics simulations of active polymer."""
import numpy as np
from bd import recommended_dt, with_srk1
from rouse import linear_mid_msd, end2end_distance_gauss, gaussian_Ploop
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
#from mayavi import mlab
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
    X, t_save = process_sim(Path(simdir / f'tape{tape}.csv'))
    nframes, N, dim = X.shape
    D = np.tile(1, N)
    if simname == 'mid_hot_bead':
        D[int(N // 2)] = 10
    if simname == 'cosine1':
        B = 2 * np.pi / 25
        D = 0.9 * np.cos(B * np.arange(0, N)) + 1
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
        #if simname == 'mid_hot_bead':
        #    ax.plot(t_save, hot_bead_msd, color=palette[ord[i]], alpha=0.4)
        #elif simname == 'bdeq':
        #    ax.plot(t_save, mean_beads_msd, color=palette[ord[i]], alpha=0.4)

    if simname == 'mid_hot_bead':
        ax.plot(t_save, hot_monomer_msds / ntraj, 'r-', label=f'hot bead (N={ntraj})')
        ax.plot(t_save, cold_monomer_msds / ntraj, 'b--', label=f'cold beads (N={ntraj})')
    elif simname == 'bdeq1':
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

def two_point_msd(simdir, ntraj, N=101, relative=False, squared=False):
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
        Xeq, _ = process_sim(Path('csvs/bdeq1') / f'tape{j}.csv')
        for i in range(int(ntimes//2), ntimes, 5):
            #for temperature modulations
            dist = pdist(X[i, :, :], metric=metric)
            Y = squareform(dist)
            average_dist += Y
            #for equilibrium case
            dist = pdist(Xeq[i, :, :], metric=metric)
            eq_dist += squareform(dist)
    fig, ax = plt.subplots()
    if relative:
        sns.heatmap((average_dist / nreplicates) - (eq_dist / nreplicates), xticklabels=25,
                    yticklabels=25, cmap='coolwarm', ax=ax)
        if squared:
            ax.set_title(r'MSD relative to uniform temperature')
        else:
            ax.set_title('Mean distance relative to uniform temperature')
    else:
        sns.heatmap(average_dist / nreplicates, xticklabels=25, yticklabels=25,
                    cmap='coolwarm', ax=ax)
        if squared:
            ax.set_title(r'Mean squared distance $\langle\Vert \vec{r}_i(t) - \vec{r}_j(t) \Vert^2\rangle$')
        else:
            ax.set_title(r'Mean distance $\langle\Vert \vec{r}_i(t) - \vec{r}_j(t) \Vert\rangle$')
    ax.set_xlabel(r'Bead $i$')
    ax.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    if relative:
        plt.savefig(f'plots/two_point_msd_{simdir.name}_relative.pdf')
    else:
        plt.savefig(f'plots/two_point_msd_{simdir.name}.pdf')

    average_dist = average_dist / nreplicates
    eq_dist = eq_dist / nreplicates
    if squared:
        Rg_squared = np.sum(average_dist) / (2 * N ** 2)
    else:
        Rg_squared = np.sum(average_dist ** 2) / (2 * N ** 2)
    return Rg_squared, average_dist, eq_dist

def contact_probability(a, simdir, ntraj, N=101, relative=False, squared=False):
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
    #Plot contact map (each entry is probability of looping within radius a)
    fig, ax = plt.subplots()
    sns.heatmap(np.log10(counts), xticklabels=25, yticklabels=25, ax=ax)
    ax.set_title(r'Log contact Map $\langle\Vert \vec{r}_i(t) - \vec{r}_j(t) \Vert\rangle < a$')
    ax.set_xlabel(r'Bead $i$')
    ax.set_ylabel(r'Bead $j$')
    fig.tight_layout()
    plt.savefig(f'plots/contact_map_{simdir.name}.pdf')
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
    return counts, Ploop, sdistances, nreplicates

def plot_ploop(sdistances, Ploop, nreplicates, a=1, b=1, Nhat=100):
    #sdistances is anyway in units of kuhn lengths
    analytical_ploop = [gaussian_Ploop(a, n, b) for n in sdistances[1:]]
    fig, ax = plt.subplots()
    ax.plot(sdistances, Ploop, label=f'Simulation estimate (n={nreplicates})')
    ax.plot(sdistances[1:], analytical_ploop, label='Theory')
    plt.yscale('log')
    plt.xscale('log')
    ax.set_xlabel('Distance along chain (s)')
    ax.set_yaxis(f'Looping probability')
    ax.set_title(f'Looping probability within contact radius a={a}')
    plt.legend()
    fig.tight_layout()
    plt.show()

def plot_msd_from_center(N=101):
    """ For all three simulation replicates, plot the MSD from the center
    bead in the chain."""
    Rgs_cosine, cosine_dist, eq_dist = two_point_msd('csvs/cosine1', 96)
    Rgs_eq = np.sum(eq_dist) / (2 * N ** 2)
    print(f'Radius of gyration (cosine): {Rgs_cosine}')
    print(f'Radius of gyration (eq): {Rgs_eq}')
    Rgs_midhot, mid_dist, _ = two_point_msd('csvs/mid_hot_bead', 96)
    Rgs_midhot1, mid1_dist, _ = two_point_msd('csvs/mid_hot_bead1.9', 96)
    print(f'Radius of gyration (midhot): {Rgs_midhot}')
    fig, ax = plt.subplots()
    ax.plot(np.arange(50, N), eq_dist[int(N / 2), int(N / 2):], label=r'$T_{\rm eq}$')
    ax.plot(np.arange(50, N), cosine_dist[int(N / 2), int(N / 2):], label=r'$1.9 T_{\rm eq}$ (cosine)')
    ax.plot(np.arange(50, N), mid1_dist[int(N / 2), int(N / 2):], label=r'$1.9 T_{\rm eq}$ (single hot bead)')
    ax.plot(np.arange(50, N), mid_dist[int(N / 2), int(N / 2):], label=r'$10 T_{\rm eq}$ (single hot bead)')
    ax.legend()
    ax.set_xlabel('Beads to the right of center (i)')
    ax.set_ylabel('Mean distance')
    fig.tight_layout()
    plt.savefig(f'plots/msd_from_center_allsims.pdf')

def radius_of_gyration(simdir, ntraj=96):
    """ Radius of gyration is defined as the mean distance of all beads to center of mass."""
    Rg = []
    for j in range(ntraj):
        X, t_save = process_sim((Path(simdir) / f'tape{j}.csv'))
        for i in range(int(ntimes // 2), ntimes, 10):
            com = np.mean(X[i, :, :], axis=0)
            Rg.append(np.sum((X[i, :, :] - com)**2, axis=-1).mean())
    return np.array(Rg).mean()

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

