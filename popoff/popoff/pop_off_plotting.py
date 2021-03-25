import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# import utils_funcs as utils
# import run_functions as rf
# from subsets_analysis import Subsets
# import pickle
# import sklearn.decomposition
from cycler import cycler
import pandas as pd
import math, cmath
from tqdm import tqdm
# import scipy.stats
from Session import Session  # class that holds all data per session
plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))


def plot_df_stats(df, xx, yy, hh, plot_line=True, xticklabels=None, type_scatter='strip', ccolor='grey', aalpha=1):
    """Plot predictions of a pd.Dataframe, with type specified in type_scatter.

    Parameters
    ----------
    df : pd.DataFrame
        df containing predictions
    xx : str
        x axis
    yy : str
        y axis
    hh : str
        hue
    plot_line : bool, default=True
        if True, plot line of means
    xticklabels : list, default=None
        if not None, defines x axis tick labels
    type_scatter : str, dfeault='strip'
        possibiltiies: 'swarm', 'violin', 'strip'
    ccolor : str or tuple, default=;grey
        color`.
    aalpha : float 0<aalpha<1
        transparency

    Returns
    -------
    type
        Description of returned object.

    """
#     yy = yy + '_proj'
    if plot_line and hh is None:
        sns.pointplot(data=df, x=xx, y=yy, color=ccolor, ci='sd', label=None)
    elif plot_line and hh is not None:
        sns.pointplot(data=df, x=xx, y=yy, hue=hh, ci='sd', label=None)
    if type_scatter == 'strip':
        trial_plot_fun = sns.stripplot
    elif type_scatter == 'violin':
        trial_plot_fun = sns.violinplot
    elif type_scatter == 'swarm':
        trial_plot_fun = sns.swarmplot
    if hh is None:
        tmp = trial_plot_fun(x=xx, y=yy, hue=hh, data=df, linewidth=1, label=None, color=ccolor, alpha=aalpha)
    else:
        tmp = trial_plot_fun(x=xx, y=yy, hue=hh, data=df, linewidth=1, label=None, alpha=aalpha)
    if type_scatter == 'violin':
        plt.setp(tmp.collections, alpha=aalpha)
    if xticklabels is not None:
        tmp.set_xticklabels(xticklabels)



def plot_interrupted_trace_simple(ax, time_array, plot_array, llabel='', ccolor='grey',
                                 linest='-', aalpha=1, zero_mean=False, llinewidth=3):
    """Plot plot_array vs time_array, where time_array has 1 gap, which is not plotted.

    Parameters
    ----------
    ax: Axis Handle
        to plot
    time_array: np.array
        of time points,
    plot_array: dict of np.array
        each item is a data set to plot
    llabel : str
        label of these data
    ccolor : str
        color
    linest: str
        linestyle
    aalpha: float 0 < a < 1
        transparency
    zero_mean, bool, default=False
        zero mean data first
    llinewidth; int
        linewidth

    Returns:
    ---------------
        ax: Axis Handle
            return same handle
        """
    assert len(time_array) == len(plot_array)
    breakpoint = np.argmax(np.diff(time_array)) + 1# finds the 0, equivalent to art_gap_start
    if zero_mean:
        plot_array = plot_array - np.mean(plot_array)
    ax.plot(time_array[:breakpoint], plot_array[:breakpoint],  linewidth=llinewidth, linestyle=linest,
                markersize=12, color=ccolor, label=llabel, alpha=aalpha)
    ax.plot(time_array[breakpoint:], plot_array[breakpoint:],  linewidth=llinewidth, linestyle=linest,
                markersize=12, color=ccolor, alpha=aalpha, label=None)
    return ax




def plot_interrupted_trace(ax, time_array, plot_array, llabel='',
                           plot_laser=True, ccolor='grey',
                           plot_groupav=True,
                           plot_errorbar=False, plot_std_area=False, region_list=['s1', 's2'],
                           plot_diff_s1s2=False, freq=5):
    """"Same as plot_interrupted_trace_average_per_mouse(), but customised to plot_array being a dictionary
    of just two s1 and s2 traces. Can plot individual traces & group average. needs checking #TODO 

    Parameters
    ----------
    ax: Axis Handle
        to plot
    time_array: np.array
        of time points,
    plot_array: dict of np.array
        each item is a data set to plot
    llabel : str
        label of these data
    plot_laser : bool
        plot vertical bar during PS
    ccolor : str
        color
    plot_indiv : bool, default=False
        plot individual data sets
    plot_groupav : bool default=True
        plot group average of data sets
    individual_mouse_list : list or None
        if list, only plot these mice (i.e. data set keys)
    plot_errorbar : True,
        plot error bars
    plot_std_area : type
        plot shaded std area
    region_list : list of regions
        to plot
    plot_diff_s1s2 : bool, default=False
        if true, plot s1-s2 difference
    freq : int, default=5
        frequency of time_array (used for laser plot)

    Returns
    -------
    ax: Axis Handle
        return same handle
    average_mean: np .array
        mean over data sets

    """

    breakpoint = np.argmax(np.diff(time_array)) + 1# finds the 0, equivalent to art_gap_start
    time_1 = time_array[:breakpoint]  # time before & after PS
    time_2 = time_array[breakpoint:]
    region_list = np.array(region_list)
    if plot_diff_s1s2:
        assert len(region_list) == 2
    linest = {'s1': '-', 's2': '-'}
    if plot_groupav:
        #         region_hatch = {'s1': '/', 's2': "\ " }
        if plot_diff_s1s2 is False:
            for rr, data in plot_array.items():
                if rr in region_list:
                    assert data.ndim == 2
                    av_mean = data[:, 0]
                    std_means = data[:, 1]
                    if plot_errorbar is False:  # plot gruup means
                        ax.plot(time_1, av_mean[:breakpoint],  linewidth=4, linestyle=linest[rr],
                                        markersize=12, color=ccolor, label=llabel, alpha=0.9)# + f' {rr.upper()}'
                        ax.plot(time_2, av_mean[breakpoint:], linewidth=4, linestyle=linest[rr],
                                    markersize=12, color=ccolor, alpha=0.9, label=None)
                    elif plot_errorbar is True:  # plot group means with error bars
                        ax.errorbar(time_1, av_mean[:breakpoint], yerr=std_means[:breakpoint], linewidth=4, linestyle=linest[rr],
                                        markersize=12, color=ccolor, label=llabel + f' {rr.upper()}', alpha=0.9)
                        ax.errorbar(time_2, av_mean[breakpoint:], yerr=std_means[breakpoint:], linewidth=4, linestyle=linest[rr],
                                    markersize=12, color=ccolor, alpha=0.9, label=None)
                    if plot_std_area:  # plot std area
#                         if len(region_list) == 1:
#                         std_label = f'Std {llabel} {rr.upper()}'
#                         elif len(region_list) == 2:
#                             std_label = f'Group std {rr.upper()}'
                        ax.fill_between(x=time_1, y1=av_mean[:breakpoint] - std_means[:breakpoint],
                                                y2=av_mean[:breakpoint] + std_means[:breakpoint], color=ccolor, alpha=0.1,
                                        label=None)#, hatch=region_hatch[rr])
                        ax.fill_between(x=time_2, y1=av_mean[breakpoint:] - std_means[breakpoint:],
                                       y2=av_mean[breakpoint:] + std_means[breakpoint:], color=ccolor, alpha=0.1,
                                        label=None)#, hatch=region_hatch[rr])
        elif plot_diff_s1s2:
            assert (region_list == np.array(['s1', 's2'])).all() and len(plot_array) == len(average_mean)
            diff_data = plot_array['s1'] - plot_array['s2']
            assert diff_data.ndim == 2 and diff_data.shape[1] == 2
            diff_mean = diff_data[:, 0]
            ax.plot(time_1, diff_mean[:breakpoint],  linewidth=4, linestyle='-',
                        markersize=12, color=ccolor, label=f'{llabel}', alpha=0.9) # S1 - S2 diff.
            ax.plot(time_2, diff_mean[breakpoint:], linewidth=4, linestyle='-',
                        markersize=12, color=ccolor, alpha=0.9, label=None)
    if plot_laser:  # plot laser
        ax.axvspan(xmin=time_1[-1] + 1 / freq, xmax=time_2[0] - 1 / freq, ymin=0.1,
                   ymax=0.9, alpha=0.2, label=None, edgecolor='k', facecolor='red')
    return ax, None


def plot_interrupted_trace_average_per_mouse(ax, time_array, plot_array, llabel='',
            plot_laser=True, ccolor='grey', plot_indiv=False,
            plot_groupav=True, individual_mouse_list=None, plot_errorbar=False,
            plot_std_area=False, region_list=['s1', 's2'],
            plot_diff_s1s2=False, freq=5):
    """"Same as plot_interrupted_trace_simple(), but customised to plot_array being a dictionary
    of individual mouse traces. Can plot individual traces & group average.

    Parameters
    ----------
    ax: Axis Handle
        to plot
    time_array: np.array
        of time points,
    plot_array: dict of np.array
        each item is a data set to plot
    llabel : str
        label of these data
    plot_laser : bool
        plot vertical bar during PS
    ccolor : str
        color
    plot_indiv : bool, default=False
        plot individual data sets
    plot_groupav : bool default=True
        plot group average of data sets
    individual_mouse_list : list or None
        if list, only plot these mice (i.e. data set keys)
    plot_errorbar : True,
        plot error bars
    plot_std_area : type
        plot shaded std area
    region_list : list of regions
        to plot
    plot_diff_s1s2 : bool, default=False
        if true, plot s1-s2 difference
    freq : int, default=5
        frequency of time_array (used for laser plot)

    Returns
    -------
    ax: Axis Handle
        return same handle
    average_mean: np .array
        mean over data sets

    """

    breakpoint = np.argmax(np.diff(time_array)) + 1# finds the 0, equivalent to art_gap_start
    time_1 = time_array[:breakpoint]  # time before & after PS
    time_2 = time_array[breakpoint:]
    mouse_list = list(plot_array.keys())  # all data sets (including _s1 and _s2 )
    region_list = np.array(region_list)
    if plot_diff_s1s2:
        assert len(region_list) == 2
    if individual_mouse_list is None:
        individual_mouse_list = mouse_list
    linest = {'s1': '-', 's2': '-'}
    average_mean = {x: np.zeros(plot_array[mouse_list[0]].shape[0]) for x in region_list}
    all_means = {x: np.zeros((int(len(mouse_list) / 2), plot_array[mouse_list[0]].shape[0])) for x in region_list}
    count_means = {x: 0 for x in region_list}
    for i_m, mouse in enumerate(mouse_list):  # loop through individuals
        reg = mouse[-2:]
        if reg in region_list:  # if in region list
            if plot_array[mouse].ndim == 2:
                plot_mean = plot_array[mouse][:, 0]
            elif plot_array[mouse].ndim == 1:
                plot_mean = plot_array[mouse]
            average_mean[mouse[-2:]] += plot_mean / len(mouse_list) * 2  # save mean (assumes that _s1 and _s2 in mouse_list so factor 2)
            all_means[mouse[-2:]][count_means[reg] ,:] = plot_mean.copy()  # save data for std
            count_means[reg] += 1
            if plot_indiv and mouse in individual_mouse_list and not plot_diff_s1s2:  # plot individual traces
                ax.plot(time_1, plot_mean[:breakpoint],  linewidth=2, linestyle=linest[reg],
                            markersize=12, color=ccolor, label=None, alpha=0.2)
                ax.plot(time_2, plot_mean[breakpoint:],  linewidth=2, linestyle=linest[reg],
                            markersize=12, color=ccolor, alpha=0.2, label=None)
    if plot_groupav:
        #         region_hatch = {'s1': '/', 's2': "\ " }
        if plot_diff_s1s2 is False:
            for rr, av_mean in average_mean.items():
                if rr in region_list:
                    std_means = np.std(all_means[rr], 0)
                    if plot_errorbar is False:  # plot gruup means
                        ax.plot(time_1, av_mean[:breakpoint],  linewidth=4, linestyle=linest[rr],
                                        markersize=12, color=ccolor, label=llabel, alpha=0.9)# + f' {rr.upper()}'
                        ax.plot(time_2, av_mean[breakpoint:], linewidth=4, linestyle=linest[rr],
                                    markersize=12, color=ccolor, alpha=0.9, label=None)
                    elif plot_errorbar is True:  # plot group means with error bars
                        ax.errorbar(time_1, av_mean[:breakpoint], yerr=std_means[:breakpoint], linewidth=4, linestyle=linest[rr],
                                        markersize=12, color=ccolor, label=llabel + f' {rr.upper()}', alpha=0.9)
                        ax.errorbar(time_2, av_mean[breakpoint:], yerr=std_means[breakpoint:], linewidth=4, linestyle=linest[rr],
                                    markersize=12, color=ccolor, alpha=0.9, label=None)
                    if plot_std_area:  # plot std area
#                         if len(region_list) == 1:
#                         std_label = f'Std {llabel} {rr.upper()}'
#                         elif len(region_list) == 2:
#                             std_label = f'Group std {rr.upper()}'
                        ax.fill_between(x=time_1, y1=av_mean[:breakpoint] - std_means[:breakpoint],
                                                y2=av_mean[:breakpoint] + std_means[:breakpoint], color=ccolor, alpha=0.1,
                                        label=None)#, hatch=region_hatch[rr])
                        ax.fill_between(x=time_2, y1=av_mean[breakpoint:] - std_means[breakpoint:],
                                       y2=av_mean[breakpoint:] + std_means[breakpoint:], color=ccolor, alpha=0.1,
                                        label=None)#, hatch=region_hatch[rr])
        elif plot_diff_s1s2:
            assert (region_list == np.array(['s1', 's2'])).all() and len(region_list) == len(average_mean)
            diff_mean = average_mean['s1'] - average_mean['s2']
            ax.plot(time_1, diff_mean[:breakpoint],  linewidth=4, linestyle='-',
                        markersize=12, color=ccolor, label=f'{llabel}', alpha=0.9) # S1 - S2 diff.
            ax.plot(time_2, diff_mean[breakpoint:], linewidth=4, linestyle='-',
                        markersize=12, color=ccolor, alpha=0.9, label=None)
    if len(region_list) == 2:
        assert count_means[region_list[0]] == count_means[region_list[1]]
    if plot_laser:  # plot laser
        ax.axvspan(xmin=time_1[-1] + 1 / freq, xmax=time_2[0] - 1 / freq, ymin=0.1,
                   ymax=0.9, alpha=0.2, label=None, edgecolor='k', facecolor='red')
    return ax, average_mean

