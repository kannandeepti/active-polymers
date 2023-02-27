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
import matplotlib.colors as colors

import seaborn as sns

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

def draw_power_law_triangle(alpha, x0, width, orientation, base=10,
                            x0_logscale=True, label=None, hypotenuse_only=False,
                            label_padding=0.1, text_args={}, ax=None,
                            **kwargs):
    """Draw a triangle showing the best-fit power-law on a log-log scale.

    Parameters
    ----------
    alpha : float
        the power-law slope being demonstrated
    x0 : (2,) array_like
        the "left tip" of the power law triangle, where the hypotenuse starts
        (in log units)
    width : float
        horizontal size in number of major log ticks (default base-10)
    orientation : string
        'up' or 'down', control which way the triangle's right angle "points"
    base : float
        scale "width" for non-base 10
    ax : mpl.axes.Axes, optional

    Returns
    -------
    corner : (2,) np.array
        coordinates of the right-angled corhow to get text outline of the
        triangle
    """
    if x0_logscale:
        x0, y0 = [base**x for x in x0]
    else:
        x0, y0 = x0
    if ax is None:
        ax = plt.gca()
    x1 = x0*base**width
    y1 = y0*(x1/x0)**alpha
    ax.plot([x0, x1], [y0, y1], 'k')
    corner = [x0, y0]
    if not hypotenuse_only:
        if (alpha >= 0 and orientation == 'up') \
                or (alpha < 0 and orientation == 'down'):
            ax.plot([x0, x1], [y1, y1], 'k')
            ax.plot([x0, x0], [y0, y1], 'k')
            # plt.plot lines have nice rounded caps
            # plt.hlines(y1, x0, x1, **kwargs)
            # plt.vlines(x0, y0, y1, **kwargs)
            corner = [x0, y1]
        elif (alpha >= 0 and orientation == 'down') \
                or (alpha < 0 and orientation == 'up'):
            ax.plot([x0, x1], [y0, y0], 'k')
            ax.plot([x1, x1], [y0, y1], 'k')
            # plt.hlines(y0, x0, x1, **kwargs)
            # plt.vlines(x1, y0, y1, **kwargs)
            corner = [x1, y0]
        else:
            raise ValueError(r"Need $\alpha\in\mathbb{R} and orientation\in{'up', "
                             r"'down'}")
    if label is not None:
        xlabel = x0*base**(width/2)
        if orientation == 'up':
            ylabel = y1*base**label_padding
        else:
            ylabel = y0*base**(-label_padding)
        ax.text(xlabel, ylabel, label, horizontalalignment='center',
                verticalalignment='center', **text_args)
    return corner

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

def temps_from_ids(activity_ratio=5.974):
    """ Get activity values from monomer IDs. """
    ids = np.load('data/ABidentities_blobel2021_chr2_35Mb_60Mb.npy')
    N = len(ids)
    D = np.ones(N)
    Ddiff = (activity_ratio - 1) / (activity_ratio + 1)
    D[ids == 0] = 1.0 - Ddiff
    D[ids == 1] = 1.0 + Ddiff
    return D

def plot_contact_map(filename, temps=None, tag=None, width=5, vmin=None, vmax=None,
                     ticks=True, colorbar=True, **kwargs):
    contacts = np.load(filename)
    if temps is None:
        temps = temps_from_ids()
    if tag is None:
        tag = ''
    contacts[contacts == 0] = np.min(contacts[contacts > 0])/2.0
    if vmin is None and vmax is None:
        print(contacts.min())
        print(contacts.max())
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
    ax_top = divider.append_axes("top", size=f"{width}%", pad=0.0)
    ax_left = divider.append_axes("left", size=f"{width}%", pad=0.0)
    cax = divider.append_axes("right", size=f"{width}%", pad=0.1)
    ax_top.tick_params(direction='in', bottom=False, top=True, labeltop=True, labelbottom=False,
                       width=1)
    if ticks:
        ax_top.set_xticks([0, 249, 499, 749, 999], minor=False)
        ax_top.set_xticklabels(['1', '250', '500', '750', '1000'])
        ax_left.tick_params(direction='in', width=1)
        ax_left.set_yticks([0, 249, 499, 749, 999], ['1000', '750', '500', '250', '1'], minor=False)
        ax_top.xaxis.set_label_position('top')
        ax_top.set_xlabel(r'Monomer $n$')
        ax_left.set_ylabel(r'Monomer $m$')
    else:
        ax_top.set_xticks([])
        ax_left.set_yticks([])
    ax_top.set_yticks([])
    ax_left.set_xticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.matshow(contacts, norm=lognorm, cmap=cmap_contacts, **kwargs)
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False)
    ax_top.imshow(D, cmap='coolwarm', vmin=0.1, vmax=1.9)
    ax_left.imshow(D[:, ::-1].T, cmap='coolwarm', origin="lower", vmin=0.1, vmax=1.9)
    if colorbar:
        cbar = fig.colorbar(im, cax=cax, label='contact frequency')
        cbar.ax.tick_params(which='both', direction='in', width=1)

    fig.tight_layout()
    plt.savefig(f'plots/contact_map_{tag}_temps.pdf')
    #plt.show()

def plot_std_squared_separation(filename, temps=None, tag=None, width=5, **kwargs):
    std = pd.read_csv(filename).to_numpy()
    if temps is None:
        temps = temps_from_ids()[:-1]
    if tag is None:
        tag = ''
    N = len(temps)
    D = np.ones((int(width*0.01* N), N))
    for i in range(int(width*0.01* N)):
        D[i, :] = temps
    fig = plt.figure(figsize=[14/cm_in_inch, 14/cm_in_inch])
    ax = plt.subplot(111)
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size=f"{width}%", pad=0.0)
    ax_left = divider.append_axes("left", size=f"{width}%", pad=0.0)
    cax = divider.append_axes("right", size=f"{width}%", pad=0.1)
    ax_top.tick_params(direction='in', bottom=False, top=True, labeltop=True, labelbottom=False,
                       width=1)
    ax_top.set_xticks([0, 249, 499, 749, 999], minor=False)
    ax_top.set_xticklabels(['1', '250', '500', '750', '1000'])
    ax_top.set_yticks([])
    ax_left.tick_params(direction='in', width=1)
    ax_left.set_yticks([0, 249, 499, 749, 999], ['1000', '750', '500', '250', '1'], minor=False)
    ax_left.set_xticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    #ax = sns.heatmap(contour_corr, robust=True, center=0, cmap=cmap_relative, cbar_ax=cax,
    #                 cbar_kws={'label' : 'contour alignment'})
    im = ax.imshow(std, cmap='bone', **kwargs)
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False)
    ax_top.imshow(D, cmap='coolwarm', vmin=0.1, vmax=1.9)
    ax_left.imshow(D[:, ::-1].T, cmap='coolwarm', origin="lower", vmin=0.1, vmax=1.9)
    cbar = fig.colorbar(im, cax=cax, label='std/mean separation')
    cax.tick_params(which='both', direction='in', width=1)
    ax_top.xaxis.set_label_position('top')
    ax_top.set_xlabel(r'Monomer $n$')
    ax_left.set_ylabel(r'Monomer $m$')
    fig.tight_layout()
    plt.savefig(f'plots/std_mean_separation_{tag}_temps.pdf')
    #plt.show()

def plot_contour_alignment_heatmap(filename, tag=None):
    """ Plot seaborn heatmap where entry (i, j) is the mean distance between beads i and j.
    This version does not include a color bar showing the activity of the monomers.

    Parameters
    ----------
    filename : str or Path
        path to file containing correlation matrix
    """
    contour_corr = pd.read_csv(filename).to_numpy()[258:318, 258:318]
    fig, ax = plt.subplots(figsize=(14 / cm_in_inch, 14 / cm_in_inch))
    res = sns.heatmap(contour_corr, xticklabels=10, vmin=-0.1, vmax=0.1,
                    yticklabels=10, cmap=cmr.iceburn_r, center=0.0, square=True, ax=ax)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_xlabel(r'Monomer $n$')
    #ax.set_ylabel(r'Monomer $m$')
    # make frame visible
    #for _, spine in res.spines.items():
    #    spine.set_visible(True)
    fig.tight_layout()
    plt.savefig(f'plots/contours_alignment_{tag}.pdf')

def plot_jet_inset(contacts_file, contour_alignment_file, temps=None, tag=None,
                   vmin=10**(-3.5), vmax=1.0, width=5, jet_start=258, jet_end=318, **kwargs):
    """ Plot a small panel showing a zoom in of a jet with contact frequency above
    the diagonal and contour alignment on the other diagonal."""
    contour_corr = pd.read_csv(contour_alignment_file).to_numpy()
    contour_corr = contour_corr[jet_start:jet_end, jet_start:jet_end]
    contacts = np.load(contacts_file)
    contacts = contacts[jet_start:jet_end, jet_start:jet_end]
    if temps is None:
        temps = temps_from_ids()
    temps = temps[jet_start:jet_end]
    if tag is None:
        tag = ''
    contacts[contacts == 0] = np.min(contacts[contacts > 0])/2.0
    if vmin is None and vmax is None:
        print(contacts.min())
        print(contacts.max())
        lognorm = LogNorm(vmin=contacts.min(), vmax=contacts.max())
    else:
        lognorm = LogNorm(vmin=vmin, vmax=vmax)
    N = len(temps)
    D = np.ones((int(width * 0.01 * N), N))
    for i in range(int(width * 0.01 * N)):
        D[i, :] = temps
    fig = plt.figure(figsize=[14 / cm_in_inch, 14 / cm_in_inch])
    ax = plt.subplot(111)
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size=f"{width}%", pad=0.0)
    ax_left = divider.append_axes("left", size=f"{width}%", pad=0.0)
    cax = divider.append_axes("right", size=f"{width}%", pad=0.1)
    ax_top.tick_params(direction='in', bottom=False, top=True, labeltop=True, labelbottom=False,
                       width=1)
    ax_top.set_xticks([12, 32, 52], minor=False)
    ax_top.set_xticklabels(['270', '290', '310'])
    ax_top.set_yticks([])
    ax_left.tick_params(direction='in', width=1)
    ax_left.set_yticks([12, 32, 52], ['270', '290', '310'], minor=False)
    ax_left.set_xticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.matshow(contacts, norm=lognorm, cmap=cmap_contacts, **kwargs)
    ax.tick_params(which='both', bottom=False, top=False, left=False, right=False)
    ax_top.imshow(D, cmap='coolwarm', vmin=0.1, vmax=1.9)
    ax_left.imshow(D[:, ::-1].T, cmap='coolwarm', origin="lower", vmin=0.1, vmax=1.9)
    cbar = fig.colorbar(im, cax=cax, label='contact frequency')
    cbar.ax.tick_params(which='both', direction='in', width=1)
    ax_top.xaxis.set_label_position('top')
    fig.tight_layout()
    plt.savefig(f'plots/jet_inset_{tag}.pdf')


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
    )
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
        print(f'Activity increase with comp=.71: {activity_difference/2}')
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

def plot_comp_score_vol_fraction(filename_active, filename_passive):
    df_active = pd.read_csv(filename_active)
    df_active.drop(columns='Unnamed: 0', inplace=True)
    df_active = df_active[df_active['activity_ratio'] == 5.974]
    df_active = df_active[df_active['t'] == 200000]

    df_sticky = pd.read_csv(filename_passive)
    df_sticky.drop(columns='Unnamed: 0', inplace=True)
    df_sticky = df_sticky[df_sticky['BB_energy'] == 0.4]
    df_sticky = df_sticky[df_sticky['t'] == 200000]
    aspect_ratio = 3.21 / 3.86

    colors = sns.color_palette("Paired", 6)
    fig, ax = plt.subplots(figsize=[4.7 / cm_in_inch, (4.7 / cm_in_inch) * 0.85])
    ax.plot(df_active['volume_fraction'], df_active['AA_cs2'], '-o', color=colors[5], markersize=3,
            lw=1.5, label='A, active')
    ax.plot(df_active['volume_fraction'], df_active['BB_cs2'], '-o', color=colors[1], markersize=3,
            lw=1.5, label='B, active')
    ax.plot(df_sticky['volume_fraction'], df_sticky['AA_cs2'], '--^', color=colors[4], markersize=3,
            lw=1.5, label='A, sticky')
    ax.plot(df_sticky['volume_fraction'], df_sticky['BB_cs2'], '--^', color=colors[0], markersize=3,
            lw=1.5, label='B, sticky')
    plt.legend(loc='lower right')
    ax.set_xlabel(r'volume fraction $\phi$')
    ax.set_ylabel('COMP score')
    fig.tight_layout()
    plt.savefig('plots/compartment_scores_volume_fractions.pdf')

def plot_comp_score_over_time(filename_active, filename_passive):
    df_active = pd.read_csv(filename_active)
    df_active.drop(columns='Unnamed: 0', inplace=True)
    df_active = df_active[df_active['activity_ratio']==5.974]
    df_active = df_active[df_active['volume_fraction']== 0.1172861013333333]
    #extract steady state compartment scores for A and B
    ss_active_A = df_active[df_active['t'] == 200000.0]['AA_cs2'].min()
    ss_active_B = df_active[df_active['t'] == 200000.0]['BB_cs2'].min()
    df_active = df_active[df_active['t'] <= 100000.0]

    df_sticky = pd.read_csv(filename_passive)
    df_sticky.drop(columns='Unnamed: 0', inplace=True)
    df_sticky = df_sticky[df_sticky['BB_energy'] == 0.4]
    df_sticky = df_sticky[df_sticky['volume_fraction'] == 0.1172861013333333]
    ss_sticky_A = df_sticky[df_sticky['t'] == 200000.0]['AA_cs2'].min()
    ss_sticky_B = df_sticky[df_sticky['t'] == 200000.0]['BB_cs2'].min()
    df_sticky = df_sticky[df_sticky['t'] <= 100000.0]
    aspect_ratio = 3.21 / 3.86

    colors = sns.color_palette("Paired", 6)
    fig, ax = plt.subplots(figsize=[162.8/72, 125.4/72])
    ax.plot(df_active['t']*100, df_active['AA_cs2']/ss_active_A, '-o', color=colors[5], markersize=3, lw=2,
            label='A, active')
    ax.plot(df_active['t']*100, df_active['BB_cs2']/ss_active_B, '-o', color=colors[1], markersize=3, lw=2,
            label='B, active')
    ax.plot(df_sticky['t']*100, df_sticky['AA_cs2']/ss_sticky_A, '--^', color=colors[4], markersize=3, lw=2,
            label='A, sticky')
    ax.plot(df_sticky['t']*100, df_sticky['BB_cs2']/ss_sticky_B, '--^', color=colors[0], markersize=3, lw=2,
            label='B, sticky')
    plt.legend(loc='lower right')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlabel('time steps')
    ax.set_ylabel(r'COMP(t) / COMP($\infty$)')
    fig.tight_layout()
    plt.savefig('plots/compartment_scores_over_time.pdf')

def plot_analytical_MSDs(msdfile='data/discrete_diffusion_msd.csv'):
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

def plot_simulated_MSDs(msdfile='data/ens_ave_AB_msds_sticky_BB_0.4.csv'):
    """ Plot MSD over time in active and inactive regions, along with MSD ratios.
    Full width figure is 360 pt by 126 pt.

    Parameters
    ----------
    msdfile : str or Path
        path to csv file containing columns 'hot_mean', 'cold_mean', 'reference_mean'
    """
    #Si Fig 7 is 360 pt by 126 pt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(318.6 / 72, 125.4/72))
    df = pd.read_csv(Path(msdfile))
    ids = np.load('data/ABidentities_blobel2021_chr2_35Mb_60Mb.npy')
    N_A = np.sum(ids)
    N_B = len(ids) - N_A
    ax1.loglog(df['Time'].values[1:] * 100, df['active_MSD'].values[1:], color=red_stroke,
               label='A, mean')
    ax1.loglog(df['Time'].values[1:] * 100, df['inactive_MSD'].values[1:], color=blue_stroke,
               label='B, mean')
    msd_mean = (N_A * df['active_MSD'].values + N_B * df['inactive_MSD'].values) / len(ids)
    ax1.loglog(df['Time'].values[1:] * 100, msd_mean[1:], color='black', label='mean')
    corner = draw_power_law_triangle(0.5, [3.0, 0.6], 2.0, 'up', hypotenuse_only=True, ax=ax1)
    ax1.text(10**(3.5), 20, f'$\Delta t^{{{0.5}}}$')
    ax2.plot(df['Time'].values * 100, df['active_MSD'] / df['inactive_MSD'], color='black')
    times = df['Time'].values * 100
    # active
    res1 = scipy.stats.linregress(np.log10(times[(times >= 1000) & (times < 50000)]),
                                  np.log10(msd_mean[(times >= 1000) & (times < 50000)]))
    alpha = res1.slope
    D = 10**(res1.intercept)
    print(f'MSD with D={D} d^2 / timesteps^{alpha}')
    ax1.legend(loc='lower right')
    ax1.set_xlabel(r'time steps')
    ax2.set_xlabel(r'time steps')
    ax1.set_ylabel(r'MSD [$d^2$]')
    ax2.set_ylabel(r'MSD ratio')
    #ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax2.set_ylim(bottom=1.0)
    ax2.set_xscale('log')
    fig.tight_layout()
    plotname = str(Path(msdfile).name)[:-4] + '.pdf'
    plt.savefig(Path('plots') / plotname)

def plot_contacts_over_time(ntimepoints=8, traj_length=100000):
    DT = traj_length / ntimepoints
    timepoints = np.arange(DT, (ntimepoints + 1) * DT, DT)
    for t in timepoints:
        plot_contact_map(f'data/contact_map_comps_5.974x_t{t}_cutoff2.0.npy',
                         tag=f'comps_5.974x_t{t}', vmin=1e-4, vmax=1.0)
        plot_contact_map(f'data/contact_map_sticky_BB_0.4_t{t}_cutoff2.0.npy',
                         tag=f'sticky_BB_0.4_t{t}', vmin=1e-4, vmax=1.0)

def plot_time_averaged_maps():
    runs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for run in runs:
        savepath = Path('data/time_ave_contact_maps')
        plot_contact_map(savepath / f'contact_map_comps_5.974x_run{run}_cutoff2.0_DT50blocks.npy',
                         vmin=1e-4, vmax=1.0, tag=f'comps_5.97rx_run{run}_sameDT', ticks=False,
                         colorbar=False)
        plot_contact_map(savepath / f'contact_map_sticky_BB_0.4_run{run}_cutoff2.0_DT50blocks.npy',
                         vmin=1e-4, vmax=1.0, tag=f'sticky_BB_0.4_run{run}_sameDT', ticks=False,
                         colorbar=False)
        plot_contact_map(
            savepath / f'contact_map_comps_5.974x_run{run}_cutoff2.0_DT0.0005Trouse.npy',
            vmin=1e-4, vmax=1.0, tag=f'comps_5.97rx_run{run}_DTrouse', ticks=False,
            colorbar=False)
        plot_contact_map(
            savepath / f'contact_map_sticky_BB_0.4_run{run}_cutoff2.0_DT0.0005Trouse.npy',
            vmin=1e-4, vmax=1.0, tag=f'sticky_BB_0.4_run{run}_DTrouse', ticks=False,
            colorbar=False)

def plot_distance_distributions(sim1='comps_5.974x', sim2='sticky_BB_0.4', tag='_steady_state',
                                radial_concentration=False, phi=0.117):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(318.6 / 72, 125.4/72))
    ids = np.load('data/ABidentities_blobel2021_chr2_35Mb_60Mb.npy')
    N = len(ids)
    N_A = np.sum(ids)
    N_B = N - N_A
    phi_A = (N_A / N) * phi
    phi_B = (N_B / N) * phi
    Adists_sim1 = np.load(f'data/distance_distributions/Adistance_distributions_{sim1}.npy')
    Bdists_sim1 = np.load(f'data/distance_distributions/Bdistance_distributions_{sim1}.npy')
    Adists_sim2 = np.load(f'data/distance_distributions/Adistance_distributions_{sim2}.npy')
    Bdists_sim2 = np.load(f'data/distance_distributions/Bdistance_distributions_{sim2}.npy')
    num_sims = len(Adists_sim1) / N_A
    if radial_concentration:
        #shell_volume = (4/3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
        #instead of p(R), plot radial distribution function g(R) = counts / (phi_A * 4pir^2 dr)
        counts, bin_edges = np.histogram(Adists_sim1, bins='auto', density=False)
        ax1.stairs(counts/(num_sims * phi_A * 4*np.pi*bin_edges[1:]**2 * np.diff(bin_edges)),
                   edges=bin_edges,
                   label='A', color=red_stroke)
        counts, bin_edges = np.histogram(Bdists_sim1, bins='auto', density=False)
        ax1.stairs(counts/(num_sims * phi_B * 4 * np.pi * bin_edges[1:] ** 2 * np.diff(bin_edges)),
                   edges=bin_edges,
                   label='B', color=blue_stroke)
        ax1.set_ylabel(r'$g(R)$')
        counts, bin_edges = np.histogram(Adists_sim2, bins='auto', density=False)
        ax2.stairs(counts/(num_sims * phi_A * 4 * np.pi * bin_edges[1:] ** 2 * np.diff(bin_edges)),
                   edges=bin_edges,
                   label='A', color=red_stroke)
        counts, bin_edges = np.histogram(Bdists_sim2, bins='auto', density=False)
        ax2.stairs(counts/(num_sims * phi_B * 4 * np.pi * bin_edges[1:] ** 2 * np.diff(bin_edges)),
                   edges=bin_edges,
                   label='B', color=blue_stroke)
        ax2.set_ylabel(r'$g(R)$')
    else:
        n, bins, patches = ax1.hist(Adists_sim1, bins='auto', density=True, histtype='step',
                                   label='A', color=red_stroke)
        n, bins, patches = ax1.hist(Bdists_sim1, bins='auto', density=True, histtype='step',
                                   label='B', color=blue_stroke)
        ax1.set_ylabel('Density')
        n, bins, patches = ax2.hist(Adists_sim2, bins='auto', density=True, histtype='step',
                                    label='A', color=red_stroke)
        n, bins, patches = ax2.hist(Bdists_sim2, bins='auto', density=True, histtype='step',
                                    label='B', color=blue_stroke)
        ax2.set_ylabel('Density')
    ax1.legend(loc='upper left')
    ax1.set_xlabel(r'$R$')
    ax1.set_title('active model')
    ax2.legend(loc='upper left')
    ax2.set_xlabel(r'$R$')
    ax2.set_title('sticky BB model')
    fig.tight_layout()
    if radial_concentration:
        plt.savefig(f'plots/radial_distance_distributions{tag}.pdf')
    else:
        plt.savefig(f'plots/distance_distributions{tag}.pdf')

def plot_distributions_volume_fractions():
    vol_fractions = [0.25, 0.35, 0.45]
    plot_distance_distributions(f'comps_5.974x', f'sticky_BB_0.4',
                                tag=f'_v0.117', radial_concentration=True)
    for v in vol_fractions:
        plot_distance_distributions(f'comps_5.974x_v{v:.3f}', f'sticky_BB_0.4_v{v:.3f}',
                                    tag=f'_v{v:.3f}', radial_concentration=True, phi=v)

def plot_distributions_over_time():
    timepoints = [125, 250, 375, 500, 625,750, 875, 1000, 12500,
                  25000, 37500, 50000, 62500, 75000, 87500, 100000]
    for t in timepoints:
        plot_distance_distributions(f'comps_5.974x_t{t:.1f}', f'sticky_BB_0.4_t{t:.1f}',
                                    tag=f'_t{t:.1f}', radial_concentration=True)

if __name__ == '__main__':
    plot_distributions_over_time()
    #plot_jet_inset(f'data/contact_map_comps_5.974x_cutoff2.0.npy',
    #               'data/contour_alignment_comps_5.974x.csv')
    #plot_contour_alignment_heatmap('data/contour_alignment_comps_5.974x.csv', tag='jet')
    #plot_comp_score_vol_fraction('data/comp_scores_q25_chr2_blobel_activity_ratios.csv',
    #                                'data/comp_scores_q25_chr2_blobel_stickyBB.csv')
    #plot_comp_score_over_time('data/comp_scores_q25_chr2_blobel_activity_ratios.csv',
    #                          'data/comp_scores_q25_chr2_blobel_stickyBB.csv')
    #plot_std_squared_separation('data/std_over_mean_distances_comps_5.974x.csv',
    #                            tag='comps_5.974x', vmin=0.0, vmax=0.5)
    #plot_std_squared_separation('data/std_over_mean_distances_sticky_BB_0.4.csv',
    #                            tag='sticky_BB_0.4', vmin=0.0, vmax=0.5)
    #plot_contact_map(f'data/contact_map_comps_5.974x_cutoff2.0.npy', tag='comps_5.974x',
    #                 vmin=10**(-3.5), vmax=1.0)
    #plot_contact_map(f'data/contact_map_comps_5.974x_cutoff2.0.npy',
    #                 tag=f'comps_5.974x', vmin=1e-5, vmax=1.0)
    #plot_contact_map(f'data/contact_map_sticky_BB_0.4_cutoff2.0.npy',
    #                 tag=f'sticky_BB_0.4', vmin=1e-5, vmax=1.0)
    #plot_contacts_over_time(10, 1000)


