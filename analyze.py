""" Script to run brownian dynamics simulations of active polymer."""
import numpy as np
from bd import recommended_dt, with_srk1
from rouse import linear_mid_msd
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
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
                  'font.family': 'sans-serif',≠≠
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

def process_sim(file):
    df = pd.read_csv(file)
    dfg = df.groupby('t')
    t_save = []
    X = []
    for t, mat in dfg:
        t_save.append(t)
        X.append(mat.to_numpy())
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
        of last bead minus position vector of first beed. """
    #take mean over time after steady state
    end_to_end = np.linalg.norm((X[-1, :, :] - X[-1, 0, :]), axis=-1)
    print(end_to_end.shape)
    #end_to_end = np.mean(end_to_end, axis=0)
    #print(end_to_end.shape)
    fig, ax = plt.subplots()
    L0 = L/(N-1)  # length per bead
    ax.plot([n*L0 for n in range(0, N)], end_to_end, label='simulation')
    ax.set_xlabel(r'$R_{max}$')
    ax.set_ylabel(r'$\langle R^2 \rangle$')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
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

