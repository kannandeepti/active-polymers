""" Script to plot figures in GKCK 2022."""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr
import pandas as pd
from pathlib import Path
import scipy
from deepti_utils.plotting import *

from matplotlib.ticker import EngFormatter
from matplotlib.colors import LogNorm

bp_formatter = EngFormatter('b')
norm = LogNorm(vmax=0.1)

def format_ticks(ax, x=True, y=True, rotate=True):
    if y:
        ax.yaxis.set_major_formatter(bp_formatter)
    if x:
        ax.xaxis.set_major_formatter(bp_formatter)
        ax.xaxis.tick_bottom()
    if rotate:
        ax.tick_params(axis='x',rotation=45)

update_plotting_params('pnas')
textwidth = 6.5
cmap_relative = cmr.iceburn
cmap_distance = 'magma'
cmap_temps = 'coolwarm'
cmap_contacts = "YlOrRd"
red_stroke = '#B30326'
blue_stroke = '#3B4BC0'

def plot_dummy_colorbar(width=5):
    """ Make a dummy plot with coolwarm colorbar for Figure 1A. """
    fig = plt.figure(figsize=[4.0 / cm_in_inch, 4.0 / cm_in_inch])
    ax = plt.subplot(111)
    divider = make_axes_locatable(ax)
    ax_bottom = divider.append_axes("bottom", size=f"{width}%", pad=0.0)
    ax_left = divider.append_axes("left", size=f"{width}%", pad=0.0)
    cax = divider.append_axes("right", size=f"{width}%", pad=0.1)
    fake_data = np.zeros((100, 100))
    im = ax.imshow(fake_data, cmap=cmap_temps, vmin=0.5, vmax=1.5)
    fig.colorbar(im, cax=cax)
    fig.tight_layout()
    plt.savefig(f'plots/coolwarm_colorbar.pdf')
    plt.show()

def plot_contact_map(filename, temps, tag=None, width=5, vmin=None, vmax=None, **kwargs):
    contacts = np.load(filename)
    if tag is None:
        tag = ''
    contacts[contacts == 0] = np.min(contacts[contacts > 0])/2.0
    if vmin is None and vmax is None:
        lognorm = LogNorm(vmin=contacts.min(), vmax=contacts.max())
    else:
        lognorm = LogNorm(vmin=vmin, vmax=vmax)
    N = len(temps)
    D = np.ones((int(width*0.01* N), N))
    for i in range(int(width*0.01* N)):
        D[i, :] = temps
    fig = plt.figure(figsize=[14/cm_in_inch, 14/cm_in_inch])
    ax = plt.subplot(111)
    divider = make_axes_locatable(ax)
    ax_bottom = divider.append_axes("bottom", size=f"{width}%", pad=0.0)
    ax_left = divider.append_axes("left", size=f"{width}%", pad=0.0)
    cax = divider.append_axes("right", size=f"{width}%", pad=0.1)
    ax_bottom.set_xticks([0, 250, 500, 750, 990], minor=False)
    ax_bottom.set_yticks([])
    ax_left.set_yticks([0, 250, 500, 750, 990], minor=False)
    ax_left.set_xticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.matshow(contacts, norm=lognorm, origin="lower", cmap=cmap_contacts, **kwargs)
    ax_bottom.imshow(D, cmap='coolwarm', vmin=0.1, vmax=1.9)
    ax_left.imshow(D.T, cmap='coolwarm', origin="lower", vmin=0.1, vmax=1.9)
    fig.colorbar(im, cax=cax)
    ax.set_title(r'Contact Probability')
    ax_bottom.set_xlabel(r'Monomer $i$')
    ax_left.set_ylabel(r'Monomer $j$')
    fig.tight_layout()
    plt.savefig(f'plots/contact_map_{tag}_temps.pdf')
    plt.show()

def plot_figure1D(exp_map, sim_map, temps, tag=None, width=5, vmin=None, vmax=None, **kwargs):
    contacts = exp_map
    if tag is None:
        tag = ''
    contacts[contacts == 0] = np.min(contacts[contacts > 0])/2.0
    if vmax is None:
        lognorm = LogNorm(vmin=contacts.min(), vmax=contacts.max())
    else:
        lognorm = LogNorm(vmin=vmin, vmax=vmax)
    N = len(temps)
    D = np.ones((int(width*0.01* N), N))
    for i in range(int(width*0.01* N)):
        D[i, :] = temps
    fig = plt.figure(figsize=[14/cm_in_inch, 14/cm_in_inch])
    ax = plt.subplot(111)
    start = 35_000_000
    end = 60_000_000
    im = ax.matshow(
        exp_map,
        norm=norm,
        extent=(start, end, end, start),
        cmap=cmap_contacts
    );
    divider = make_axes_locatable(ax)
    ax_bottom = divider.append_axes("bottom", size=f"{width}%", pad=0.0)
    ax_left = divider.append_axes("left", size=f"{width}%", pad=0.0)
    cax = divider.append_axes("right", size=f"{width}%", pad=0.1)
    ax_bottom.set_xticks([0, 250, 500, 750, 990], minor=False)
    ax_bottom.set_yticks([])
    ax_left.set_yticks([0, 250, 500, 750, 990], minor=False)
    ax_left.set_xticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(contacts, norm=lognorm, origin="lower", cmap=cmap_contacts, **kwargs)
    ax_bottom.imshow(D, cmap='coolwarm', vmin=0.1, vmax=1.9)
    ax_left.imshow(D.T, cmap='coolwarm', origin="lower", vmin=0.1, vmax=1.9)
    fig.colorbar(im, cax=cax)
    ax.set_title(r'Contact Probability')
    ax_bottom.set_xlabel(r'Monomer $i$')
    ax_left.set_ylabel(r'Monomer $j$')
    fig.tight_layout()
    plt.savefig(f'plots/contact_map_{tag}_temps.pdf')
    plt.show()

def plot_correlation_matrix(ids=None, rho=0.5, N=100, width=5, **kwargs):
    if not ids:
        ids = np.ones(N,)
        ids[0:20] = -1
        ids[20:40] = 0.0
        ids[60:80] = 0.0
    D = np.ones((int(width * 0.01 * N), N))
    for i in range(int(width * 0.01 * N)):
        D[i, :] = ids
    #ids is 1 for +, -1 for
    corr = np.outer(ids, ids)
    corr *= rho
    corr[np.diag_indices(N)] = 1.0
    fig, ax = plt.subplots(figsize=([4.5 / cm_in_inch, 4.5 / cm_in_inch]))
    divider = make_axes_locatable(ax)
    ax_bottom = divider.append_axes("bottom", size=f"{width}%", pad=0.0)
    ax_left = divider.append_axes("left", size=f"{width}%", pad=0.0)
    cax = divider.append_axes("right", size=f"{width}%", pad=0.1)
    ax_bottom.set_xticks([0, 20, 40, 60, 80, 100])
    ax_bottom.set_yticks([])
    ax_left.set_yticks([0, 20, 40, 60, 80, 100])
    ax_left.set_xticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap='vanimo', origin="lower", **kwargs)
    ax_bottom.imshow(D, cmap='coolwarm', vmin=-1, vmax=1)
    ax_left.imshow(D.T, cmap='coolwarm', origin="lower", vmin=-1, vmax=1)
    fig.colorbar(im, cax=cax)
    ax.set_title(r'Correlation $C_{nm}$')
    ax_bottom.set_xlabel(r'Monomer $n$')
    ax_left.set_ylabel(r'Monomer $m$')
    fig.tight_layout()
    plt.savefig(f'plots/correlation_matrix.pdf')
    plt.show()

def plot_comp_score(filename, ratios=True, best_fit=False):
    df = pd.read_csv(filename)
    df.drop(columns='Unnamed: 0', inplace=True)
    df.sort_values('activity_ratio', inplace=True)
    hot_cold_ratios = df['activity_ratio']
    compscore = df['comp_score2']
    aspect_ratio = 3.21 / 3.86
    fig, ax = plt.subplots(figsize=[4.5 / cm_in_inch, (4.5 / cm_in_inch)*0.75])
    if not ratios:
        hot_cold_diffs = []
        for hot_cold in hot_cold_ratios:
            x = (hot_cold - 1) / (hot_cold + 1)
            hot_cold_diffs.append(2*x)
        ax.plot(hot_cold_diffs, compscore, 'bo-', markersize=3, label='simulation data')
        # fit line to points and plot
        result = scipy.stats.linregress(hot_cold_diffs[1:5], compscore[1:5])
        print(f'Slope: {result.slope}, intercept: {result.intercept}')
        diffrange = np.linspace(1.0, 2.0, 100)
        #activity difference with comp score of 0.71
        activity_difference = (0.71 - result.intercept) / result.slope
        inferred_ratio = (1 + activity_difference/2) / (1 - activity_difference/2)
        print(f'Activitiy increase with comp=.71: {activity_difference/2}')
        if best_fit:
            ax.plot(diffrange, result.intercept + result.slope * diffrange, 'b-', label=None)
        #plt.xticks(hot_cold_diffs, [f'{diff:.2f}' for diff in hot_cold_diffs])
        plt.ylim(bottom=0.0, top=1.0)
        plt.xlim(0.0, 2.0)
        ax.set_xlabel(r'$(A_A - A_B) / A_0$')
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    else:
        ax.plot(hot_cold_ratios, compscore, '-o', markersize=12, lw=3)
        plt.xticks(hot_cold_ratios, [f'{ratio}' for ratio in hot_cold_ratios])
        ax.set_xlabel(r'$D_A / D_B$')
    ax.set_ylabel('COMP score')
    fig.tight_layout()
    plt.savefig('plots/compartment_scores_bcomps.pdf')

def plot_MSDs(msdfile='data/discrete_diffusion_msd.csv'):
    """ Plot MSD over time in active and inactive regions, along with MSD ratios.
    Full width figure is 360 pt by 126 pt.

    Parameters
    ----------
    msdfile : str or Path
        path to csv file containing columns 'hot_mean', 'cold_mean', 'reference_mean'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(360 / 72, 126/72))
    df = pd.read_csv(Path(msdfile))
    ax1.loglog(df['τs'].values, df['hot_mean'], color=red_stroke, label='active, mean')
    ax1.loglog(df['τs'].values, df['cold_mean'], color=blue_stroke, label='inactive, mean')
    corner = draw_power_law_triangle(0.5, [0.8, 1.0], 2.0, 'up', hypotenuse_only=True, ax=ax1)
    ax1.text(4.5, 30, f'$\Delta t^{{{0.5}}}$')
    ax1.loglog(df['τs'].values, df['reference_mean'], color='black', label='reference mean')
    ax2.plot(df['τs'].values, df['hot_mean'] / df['cold_mean'], color='black')
    ax1.legend(loc='lower right')
    ax1.set_xlabel(r'time $\Delta t$')
    ax2.set_xlabel(r'time $\Delta t$')
    ax1.set_ylabel(r'MSD($\Delta t$) [$b^2$]')
    ax2.set_ylabel(r'MSD ratio')
    #ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax2.set_xscale('log')
    fig.tight_layout()
    plt.savefig('plots/discrete_rouse_msd.pdf')
