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
import scipy.stats
from Session import Session  # class that holds all data per session
import pop_off_functions as pof
plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))

## Create list with standard colors:
color_dict_stand = {}
for ii, x in enumerate(plt.rcParams['axes.prop_cycle']()):
    color_dict_stand[ii] = x['color']
    if ii > 8:
        break  # after 8 it repeats (for ever)

colors_plot = {'s1': {lick: 0.6 * np.array(color_dict_stand[lick]) for lick in [0, 1]},
               's2': {lick: 1.1 * np.array(color_dict_stand[lick]) for lick in [0, 1]}}
colors_reg = {reg: 0.5 * (colors_plot[reg][0] + colors_plot[reg][1]) for reg in ['s1', 's2']}

color_tt = {'hit': 'green', 'miss': 'grey', 'fp': 'magenta', 'cr': 'brown', 
            'ur_hit': '#7b85d4', 'ar_miss': '#e9d043'}
label_tt = {'hit': 'Hit', 'miss': 'Miss', 'fp': 'FP', 'cr': 'CR',
            'ur_hit': 'UR Hit', 'ar_miss': 'AR Miss'}
linest_reg = {'s1': '-', 's2': '-'}
label_split = {**{0: 'No L.', 1: 'Lick'}, **label_tt}
alpha_reg = {'s1': 0.9, 's2':0.5}

for tt in color_tt.keys():
    colors_plot['s1'][tt] = color_tt[tt]
    colors_plot['s2'][tt] = color_tt[tt]


def plot_df_stats(df, xx, yy, hh, plot_line=True, xticklabels=None, 
                  type_scatter='strip', ccolor='grey', aalpha=1, ax=None):
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
    ax: None 
        ax object
    Returns
    -------
    type
        Description of returned object.

    """
    if ax is None:
        ax = plt.subplot(111)
#     yy = yy + '_proj'
    if plot_line and hh is None:
        sns.pointplot(data=df, x=xx, y=yy, color=ccolor, ci='sd', label=None, ax=ax)
    elif plot_line and hh is not None:
        sns.pointplot(data=df, x=xx, y=yy, hue=hh, ci='sd', label=None, ax=ax)
    if type_scatter == 'strip':
        trial_plot_fun = sns.stripplot
    elif type_scatter == 'violin':
        trial_plot_fun = sns.violinplot
    elif type_scatter == 'swarm':
        trial_plot_fun = sns.swarmplot
    if hh is None:
        tmp = trial_plot_fun(x=xx, y=yy, hue=hh, data=df, linewidth=1, 
                             label=None, color=ccolor, alpha=aalpha, ax=ax)
    else:
        tmp = trial_plot_fun(x=xx, y=yy, hue=hh, data=df, linewidth=1, 
                             label=None, alpha=aalpha, ax=ax)
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
    time_breakpoint = np.argmax(np.diff(time_array)) + 1# finds the 0, equivalent to art_gap_start
    if zero_mean:
        plot_array = plot_array - np.mean(plot_array)
    ax.plot(time_array[:time_breakpoint], plot_array[:time_breakpoint],  linewidth=llinewidth, linestyle=linest,
                markersize=12, color=ccolor, label=llabel, alpha=aalpha)
    ax.plot(time_array[time_breakpoint:], plot_array[time_breakpoint:],  linewidth=llinewidth, linestyle=linest,
                markersize=12, color=ccolor, alpha=aalpha, label=None)
    return ax




def plot_interrupted_trace(ax, time_array, plot_array, llabel='',
                           plot_laser=True, ccolor='grey',
                           plot_groupav=True,
                           plot_errorbar=False, plot_std_area=False, region_list=['s1', 's2'],
                           plot_diff_s1s2=False, freq=30):
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

    time_breakpoint = np.argmax(np.diff(time_array)) + 1# finds the 0, equivalent to art_gap_start
    time_1 = time_array[:time_breakpoint]  # time before & after PS
    time_2 = time_array[time_breakpoint:]
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
                        ax.plot(time_1, av_mean[:time_breakpoint],  linewidth=4, linestyle=linest[rr],
                                        markersize=12, color=ccolor, label=llabel, alpha=0.9)# + f' {rr.upper()}'
                        ax.plot(time_2, av_mean[time_breakpoint:], linewidth=4, linestyle=linest[rr],
                                    markersize=12, color=ccolor, alpha=0.9, label=None)
                    elif plot_errorbar is True:  # plot group means with error bars
                        ax.errorbar(time_1, av_mean[:time_breakpoint], yerr=std_means[:time_breakpoint], linewidth=4, linestyle=linest[rr],
                                        markersize=12, color=ccolor, label=llabel + f' {rr.upper()}', alpha=0.9)
                        ax.errorbar(time_2, av_mean[time_breakpoint:], yerr=std_means[time_breakpoint:], linewidth=4, linestyle=linest[rr],
                                    markersize=12, color=ccolor, alpha=0.9, label=None)
                    if plot_std_area:  # plot std area
#                         if len(region_list) == 1:
#                         std_label = f'Std {llabel} {rr.upper()}'
#                         elif len(region_list) == 2:
#                             std_label = f'Group std {rr.upper()}'
                        ax.fill_between(x=time_1, y1=av_mean[:time_breakpoint] - std_means[:time_breakpoint],
                                                y2=av_mean[:time_breakpoint] + std_means[:time_breakpoint], color=ccolor, alpha=0.1,
                                        label=None)#, hatch=region_hatch[rr])
                        ax.fill_between(x=time_2, y1=av_mean[time_breakpoint:] - std_means[time_breakpoint:],
                                       y2=av_mean[time_breakpoint:] + std_means[time_breakpoint:], color=ccolor, alpha=0.1,
                                        label=None)#, hatch=region_hatch[rr])
        elif plot_diff_s1s2:
            assert (region_list == np.array(['s1', 's2'])).all() and len(plot_array) == len(average_mean)
            diff_data = plot_array['s1'] - plot_array['s2']
            assert diff_data.ndim == 2 and diff_data.shape[1] == 2
            diff_mean = diff_data[:, 0]
            ax.plot(time_1, diff_mean[:time_breakpoint],  linewidth=4, linestyle='-',
                        markersize=12, color=ccolor, label=f'{llabel}', alpha=0.9) # S1 - S2 diff.
            ax.plot(time_2, diff_mean[time_breakpoint:], linewidth=4, linestyle='-',
                        markersize=12, color=ccolor, alpha=0.9, label=None)
    if plot_laser:  # plot laser
        ax.axvspan(xmin=time_1[-1] + 1 / freq, xmax=time_2[0] - 1 / freq, ymin=0.1,
                   ymax=0.9, alpha=0.2, label=None, edgecolor='k', facecolor='red')
    return ax, None


def plot_interrupted_trace_average_per_mouse(ax, time_array, plot_array, llabel='',
            plot_laser=True, ccolor='grey', plot_indiv=False,
            plot_groupav=True, individual_mouse_list=None, plot_errorbar=False,
            plot_std_area=False, region_list=['s1', 's2'],
            plot_diff_s1s2=False, freq=30, running_average_smooth=True, one_sided_window_size=1):
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
    window_size = int(2 * one_sided_window_size + 1)
    time_breakpoint = np.argmax(np.diff(time_array)) + 1# finds the 0, equivalent to art_gap_start
    time_1 = time_array[:time_breakpoint]  # time before & after PS
    time_2 = time_array[time_breakpoint:]
    mouse_list = list(plot_array.keys())  # all data sets (including _s1 and _s2 )
    assert mouse_list[0][:-2] == mouse_list[1][:-2] and mouse_list[0][-2:] == 's1' and mouse_list[1][-2:] == 's2'
    region_list = np.array(region_list)
    if plot_diff_s1s2:
        assert len(region_list) == 2
    if individual_mouse_list is None:
        individual_mouse_list = mouse_list
    linest = {'s1': '-', 's2': '-'}
    average_mean = {x: np.zeros(plot_array[mouse_list[0]].shape[0]) for x in region_list}
    all_means = {x: np.zeros((int(len(mouse_list) / 2), plot_array[mouse_list[0]].shape[0])) for x in region_list}  # mouse_list / 2 beacuse of separate entries for s1 and s2
    count_means = {x: 0 for x in region_list}
    for i_m, mouse in enumerate(mouse_list):  # loop through individuals
        reg = mouse[-2:]
        if reg in region_list:  # if in region list
            if plot_array[mouse].ndim == 2:
                plot_mean = plot_array[mouse][:, 0].copy()
            elif plot_array[mouse].ndim == 1:
                plot_mean = plot_array[mouse].copy()
            if running_average_smooth:
                plot_mean[one_sided_window_size:-one_sided_window_size] = np.convolve(plot_mean, np.ones(window_size), mode='valid') / window_size
            average_mean[reg] += plot_mean / len(mouse_list) * 2  # save mean (assumes that _s1 and _s2 in mouse_list so factor 2)
            all_means[reg][count_means[reg] ,:] = plot_mean.copy()  # save data for std
            count_means[reg] += 1
            if plot_indiv and mouse in individual_mouse_list and not plot_diff_s1s2:  # plot individual traces
                ax.plot(time_1, plot_mean[:time_breakpoint],  linewidth=2, linestyle=linest[reg],
                            markersize=12, color=ccolor, label=None, alpha=0.2)
                ax.plot(time_2, plot_mean[time_breakpoint:],  linewidth=2, linestyle=linest[reg],
                            markersize=12, color=ccolor, alpha=0.2, label=None)
    if plot_groupav:
        #         region_hatch = {'s1': '/', 's2': "\ " }
        if plot_diff_s1s2 is False:
            for rr, av_mean in average_mean.items():
                # av_mean[2:-2] = np.convolve(av_mean, np.ones(window_size), mode='valid') / window_size
                if rr in region_list:
                    std_means = np.std(all_means[rr], 0) / 2
                    if plot_errorbar is False:  # plot group means
                        ax.plot(time_1, av_mean[:time_breakpoint],  linewidth=4, linestyle=linest[rr],
                                        markersize=12, color=ccolor, label=llabel, alpha=0.9)# + f' {rr.upper()}'
                        ax.plot(time_2, av_mean[time_breakpoint:], linewidth=4, linestyle=linest[rr],
                                    markersize=12, color=ccolor, alpha=0.9, label=None)
                    elif plot_errorbar is True:  # plot group means with error bars
                        ax.errorbar(time_1, av_mean[:time_breakpoint], yerr=std_means[:time_breakpoint], linewidth=4, linestyle=linest[rr],
                                        markersize=12, color=ccolor, label=llabel + f' {rr.upper()}', alpha=0.9)
                        ax.errorbar(time_2, av_mean[time_breakpoint:], yerr=std_means[time_breakpoint:], linewidth=4, linestyle=linest[rr],
                                    markersize=12, color=ccolor, alpha=0.9, label=None)
                    if plot_std_area:  # plot std area
#                         if len(region_list) == 1:
#                         std_label = f'Std {llabel} {rr.upper()}'
#                         elif len(region_list) == 2:
#                             std_label = f'Group std {rr.upper()}'
                        ax.fill_between(x=time_1, y1=av_mean[:time_breakpoint] - std_means[:time_breakpoint],
                                                y2=av_mean[:time_breakpoint] + std_means[:time_breakpoint], color=ccolor, alpha=0.1,
                                        label=None)#, hatch=region_hatch[rr])
                        ax.fill_between(x=time_2, y1=av_mean[time_breakpoint:] - std_means[time_breakpoint:],
                                       y2=av_mean[time_breakpoint:] + std_means[time_breakpoint:], color=ccolor, alpha=0.1,
                                        label=None)#, hatch=region_hatch[rr])
        elif plot_diff_s1s2:
            assert (region_list == np.array(['s1', 's2'])).all() and len(region_list) == len(average_mean)
            diff_mean = average_mean['s1'] - average_mean['s2']
            ax.plot(time_1, diff_mean[:time_breakpoint],  linewidth=4, linestyle='-',
                        markersize=12, color=ccolor, label=f'{llabel}', alpha=0.9) # S1 - S2 diff.
            ax.plot(time_2, diff_mean[time_breakpoint:], linewidth=4, linestyle='-',
                        markersize=12, color=ccolor, alpha=0.9, label=None)
    if len(region_list) == 2:
        assert count_means[region_list[0]] == count_means[region_list[1]]
    if plot_laser:  # plot laser
        ax.axvspan(xmin=time_1[-1] + 1 / freq, xmax=time_2[0] - 1 / freq, ymin=0.1,
                   ymax=0.9, alpha=0.2, label=None, edgecolor='k', facecolor='red')
    return ax, average_mean

def plot_behaviour_accuracy_all_mice(sessions={}, metric='accuracy',
                                     n_ps_arr=[5, 10, 20, 30, 40, 50, 150], ax=None,
                                     save_fig=False, save_name='figures/intro_fig/beh_perf.pdf'):
    
    plot_mat_beh = pof.beh_metric(sessions=sessions, metric=metric,
                                  stim_array=n_ps_arr)  # compute accuracy per session
    mouse_list = np.unique([ss.mouse for _, ss in sessions.items()])
    mouse_av_beh = np.zeros((len(mouse_list), plot_mat_beh.shape[1]))  # save average accuracy per mouse
    for i_m, mouse in enumerate(mouse_list):
        tmp_mouse_list = []
        for i_s, ss in sessions.items():
            if ss.mouse == mouse:
                tmp_mouse_list.append(i_s)
        mouse_av_beh[i_m, :] = np.nanmean(plot_mat_beh[np.array(tmp_mouse_list), :], 0)

    ## Plot performance per session in heatmap
    # plt.subplot(1,2 ,1)
    # plt.imshow(plot_mat_beh)
    # plt.colorbar()

    ## Summary plot
    if ax is None:
        ax_perf = plt.subplot(111)
    else:
        ax_perf = ax
    # for i_s in range(plot_mat_beh.shape[0]):  # Plot individual sessions
    #     ax_perf.plot(plot_mat_beh[i_s, :], '-', linewidth=3, markersize=14, color='grey', alpha=0.2)
    for i_m in range(mouse_av_beh.shape[0]):  # plot individual mice
        ax_perf.plot(mouse_av_beh[i_m, :], '.-', linewidth=3, markersize=10, color='grey', alpha=0.13)
    ax_perf.plot(np.nanmean(mouse_av_beh, 0), '.-', linewidth=4, markersize=20, color='black')  # plot average of mouse (almost identical to average of sessions)

    ## Stat test: one-sided wilcoxon signed-rank test to see if greater n_PS
    ## lead to greater accuracy. Group per two n_PS (5 & 10) < (20 & 30) < (40 &50) < 150

    ax_perf.plot([0.5, 2.5], [0.55, 0.55], color='k')
    p_val_low = scipy.stats.wilcoxon(np.mean(mouse_av_beh[:, 0:2], 1), 
                            np.mean(mouse_av_beh[:, 2:4], 1), alternative='less')[1]
    ax_perf.text(x=0.8, y=0.57, s=f'{np.round(p_val_low, 2)}', fontdict={'fontsize': 14})

    ax_perf.plot([2.5, 4.5], [0.64, 0.64], color='k')
    p_val_mid = scipy.stats.wilcoxon(np.mean(mouse_av_beh[:, 2:4], 1), 
                            np.mean(mouse_av_beh[:, 4:6], 1), alternative='less')[1]
    ax_perf.text(x=2.8, y=0.66, s=f'{np.round(p_val_mid, 2)}', fontdict={'fontsize': 14})

    ax_perf.plot([4.5, 6], [0.8, 0.8], color='k')
    p_val_high = scipy.stats.wilcoxon(np.mean(mouse_av_beh[:, 4:6], 1), 
                            mouse_av_beh[:, 6], alternative='less')[1]
    ax_perf.text(x=4.55, y=0.82, s=f'{np.round(p_val_high, 2)}', fontdict={'fontsize': 14})

    ax_perf.set_xticks(np.arange(len(n_ps_arr))); ax_perf.set_xticklabels(n_ps_arr)
    ax_perf.set_ylim([mouse_av_beh.min(), mouse_av_beh.max() + 0.01])
    ax_perf.set_xlabel(r"$n_{PS}$", fontdict={'fontsize': 16}); plt.ylabel('Accuracy', fontdict={'fontsize': 16})
    ax_perf.set_title('Behavioural performance', y=1.05, weight='bold', fontdict={'fontsize': 16})
    ax_perf.spines['top'].set_visible(False)
    ax_perf.spines['right'].set_visible(False)
    if save_fig:
        plt.savefig(save_name, bbox_inches='tight')
    return ax_perf
    

def plot_raster_plots_trial_types_one_session(session):

    fig, ax = plt.subplots(2, 5, figsize=(24, 15), gridspec_kw={'wspace': 0.3})
    # plt.rcParams['figure.figsize']= (24, 15)
    # plt.subplots_adjust(wspace=0.3)

    for i_x, xx in enumerate(['hit', 'miss', 'fp', 'cr']):
        mean_trace = np.mean(session.behaviour_trials[:, np.logical_and(session.outcome == xx, 
                                        session.photostim < 2), :][:, :, session.filter_ps_array][session.s1_bool, :, :], (0, 1))
    #     print(np.sum(mean_trace))
        plot_interrupted_trace_simple(ax[0][0], session.filter_ps_time, mean_trace,
                                        llabel=xx, llinewidth=2, ccolor=color_dict_stand[i_x])
    ax[0][0].legend(); ax[0][0].set_title('Average over all S1 neurons & trials'); 
    ax[0][0].set_xlabel('Time (s)'); ax[0][0].set_ylabel('DF/F')

    for i_x, xx in enumerate(['hit', 'miss', 'fp', 'cr']):
        mean_trace = np.mean(session.behaviour_trials[:, np.logical_and(session.outcome == xx, 
                                        session.photostim < 2), :][:, :, session.filter_ps_array][session.s2_bool, :, :], (0, 1))
    #     print(np.sum(mean_trace))
        plot_interrupted_trace_simple(ax[1][0], session.filter_ps_time, mean_trace,
                                        llabel=xx, llinewidth=2, ccolor=color_dict_stand[i_x])
    ax[1][0].legend(); ax[1][0].set_title('Average over all S2 neurons & trials'); 
    ax[1][0].set_xlabel('Time (s)'); ax[1][0].set_ylabel('DF/F')

    for i_x, xx in enumerate(['hit', 'fp', 'miss', 'cr']):
        im = ax[0][1 + i_x].imshow(np.mean(session.behaviour_trials[:, np.logical_and(session.outcome == xx, 
                                            session.photostim < 2), :][:, :, session.filter_ps_array][session.s1_bool, :, :], 1), 
    #                aspect='auto', vmin=-0.2, vmax=1, cmap='plasma')
                    aspect='auto', vmin=-0.5, vmax=0.5, cmap='PiYG')
        plt.colorbar(im ,ax=ax[0][1 + i_x])
        ax[0][1 + i_x].set_title(xx + ' S1'); 
        
        im = ax[1][1 + i_x].imshow(np.mean(session.behaviour_trials[:, np.logical_and(session.outcome == xx, 
                                            session.photostim < 2), :][:, :, session.filter_ps_array][session.s2_bool, :, :], 1), 
    #                aspect='auto', vmin=-0.2, vmax=1, cmap='plasma')
                    aspect='auto', vmin=-0.5, vmax=0.5, cmap='PiYG')
        plt.colorbar(im, ax=ax[1][1+ i_x])
        ax[1][1 + i_x].set_title(xx + ' S2');
        for ii in [0, 1]:
            ax[ii][1 + i_x].set_ylabel('neuron id')
            ax[ii][1 + i_x].set_xlabel('Time')

    print(session)
    return fig

def plot_mean_traces_per_session(sessions):
    n_cols = 4
    n_rows = int(np.ceil(len(sessions) / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3),
                           gridspec_kw={'wspace': 0.4, 'hspace': 0.4})
    for kk, ss in sessions.items():
        curr_ax = ax[int(np.floor(kk / n_cols))][kk % n_cols]    
        for i_x, xx in enumerate(['hit', 'miss', 'fp', 'cr']):
            mean_trace = np.mean(ss.behaviour_trials[:, np.logical_and(ss.outcome == xx, 
                                            ss.photostim < 2), :][:, :, ss.filter_ps_array], (0, 1))
            plot_interrupted_trace_simple(curr_ax, ss.filter_ps_time, mean_trace,
                                            llabel=xx, llinewidth=2, ccolor=color_dict_stand[i_x])
        if kk == 0:
            curr_ax.legend(); 
        curr_ax.set_title(f'Average {ss.name}');# plt.xlabel('Time (s)'); plt.ylabel('DF/F')

def plot_trial_type_mean_differences_pointplots(sessions, df_differences=None,
                                                save_fig=False, save_name='figures/s1_s2_static_dff_difference_prestim_poststim_1sec.pdf'):
    if df_differences is None:
        df_differences = pof.create_df_differences(sessions=sessions)
    plot_tt_dict = {0: ['hit', 'miss'], 1: ['fp', 'cr'], 2: ['hit', 'fp'], 3: ['miss', 'cr']}
    fig, ax = plt.subplots(2, len(plot_tt_dict), figsize=(13, 7), gridspec_kw={'wspace': 0.25, 'hspace':0.6})
    for i_reg, reg in enumerate(['s1', 's2']):
        for i_comb, tt_comb in plot_tt_dict.items():
            curr_ax = ax[i_reg][i_comb]
            df_plot = df_differences[np.logical_and(df_differences['region'] == reg.upper(),
                                    np.logical_or(df_differences['trial_type'] == tt_comb[0],
                                                df_differences['trial_type'] == tt_comb[1]))]

            for _, ss in sessions.items(): 
                g = sns.pointplot(data=df_plot[df_plot['session'] == ss.signature],  
                            x='trial_type', y='diff_dff',
                            color=colors_reg[reg], ax=curr_ax)
                plt.setp(g.collections, alpha=.8) #for the markers
                plt.setp(g.lines, alpha=.3)       #for the lines
            p_val = scipy.stats.wilcoxon(df_plot[df_plot['trial_type'] == tt_comb[0]]['diff_dff'],
                                df_plot[df_plot['trial_type'] == tt_comb[1]]['diff_dff'],
                                alternative='two-sided')[1]
            curr_ax.set_ylim([df_differences['diff_dff'].min() - 0.005, df_differences['diff_dff'].max() + 0.005])
            if i_comb == 0:
                curr_ax.set_ylabel(r"$\Delta F/F$"); 
            else:
                curr_ax.set_ylabel(''); curr_ax.set_yticklabels('')
            curr_ax.set_title(f'{label_tt[tt_comb[0]]} - {label_tt[tt_comb[1]]}    {reg.upper()}\n P = {np.round(p_val, 4)}', weight='bold')
            curr_ax.set_xticklabels([label_tt[tt_comb[0]], label_tt[tt_comb[1]]]); curr_ax.set_xlabel('')
            curr_ax.spines['top'].set_visible(False)
            curr_ax.spines['right'].set_visible(False)
    curr_ax.text(s='A', x=-8.5, y=0.325, fontdict={'fontsize': 28})
    curr_ax.text(s='B', x=-8.5, y=0.1, fontdict={'fontsize': 28})
    curr_ax.text(s='C', x=-5.95, y=0.325, fontdict={'fontsize': 28})
    curr_ax.text(s='D', x=-5.95, y=0.1, fontdict={'fontsize': 28})
    curr_ax.text(s='E', x=-3.4, y=0.325, fontdict={'fontsize': 28})
    curr_ax.text(s='F', x=-3.4, y=0.1, fontdict={'fontsize': 28})
    curr_ax.text(s='G', x=-0.9, y=0.325, fontdict={'fontsize': 28})
    curr_ax.text(s='H', x=-0.9, y=0.1, fontdict={'fontsize': 28})

    if save_fig:
        plt.savefig(save_name, bbox_inches='tight')


def plot_mean_trace_across_sessions(df_dyn_differences, regions=['s1', 's2'],
                                    trial_types=['hit'], plot_laser=True, ax=None,
                                    text_baseline=False, plot_zero_line=True, save_fig=False, 
                                    save_name='figures/s1_s2_dynamic_dff_difference_prestim_poststim_concl4.pdf'):
    if ax is None:
        ax = plt.subplot(111)
    if plot_zero_line:
        ax.plot([df_dyn_differences['timepoint'].min(), df_dyn_differences['timepoint'].max()], [0, 0], 
                    color='k', linestyle=':', alpha=0.5, linewidth=2)
            
    for i_reg, reg in enumerate(regions):
        for i_tt, tt in enumerate(trial_types):
            df_plot = df_dyn_differences[np.logical_and(df_dyn_differences['region'] == reg.upper(),
                                                        df_dyn_differences['trial_type'] == tt)]
            # if i_reg == 0:
            #     tmpc = '#006600'
            # else:
            #     tmpc = '#4dff4d'
            sns.lineplot(data=df_plot[df_plot['timepoint'] <= 0], x='timepoint', y='diff_dff', linewidth=3,
                        color=color_tt[tt], alpha=alpha_reg[reg], ax=ax, label=label_tt[tt] + f' {reg.upper()}')#, linestyle=linest_reg[reg])
            sns.lineplot(data=df_plot[df_plot['timepoint'] > 0], x='timepoint', y='diff_dff', linewidth=3,
                        color=color_tt[tt], alpha=alpha_reg[reg], ax=ax, label=None)#, linestyle=linest_reg[reg])
    ax.set_ylim([-0.03, 0.05])
    ax.set_title(f'Average {reg.upper()} response', weight='bold');

    ax.set_xlabel('Time (s)'); 

    ax.set_ylabel(r"$\Delta F/F$")
    if text_baseline: # some extra info in left panel
        ax.plot([-2, -2], [0.01, 0.012], color='k')
        ax.plot([0, 0], [0.01, 0.012], color='k')
        ax.plot([-2, 0], [0.012, 0.012], color='k')
        ax.text(s='baseline', x=-2, y=0.0145, fontdict={'fontsize': 10})
    ax.legend(loc='upper right', frameon=False)
                
    if plot_laser:  # plot laser
        plt.axvspan(xmin=0.2, xmax=0.8, ymin=0.05, ymax=0.95, 
                    alpha=0.2, label=None, edgecolor='k', facecolor='red')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if save_fig:
        plt.savefig(save_name, bbox_inches='tight')

def plot_mean_trace_overview_across_sessions(df_dyn_differences, plot_laser=True,
                                             plot_zero_line=True, save_fig=False,
                                             tuple_list_tt=[('hit', 'miss'), ('fp', 'cr'), ('ur_hit', 'ar_miss')]):
    ## Custom
    fig, ax_ov = plt.subplots(2, len(tuple_list_tt), figsize=(5 * len(tuple_list_tt), 7), 
                              gridspec_kw={'wspace': 0.4, 'hspace': 0.4})
    if len(tuple_list_tt) == 1:
        ax_ov = [[ax] for ax in ax_ov]  # for consistent indexing
    for i_reg, reg in enumerate(['s1', 's2']):
        for i_tt, tt_tuple in enumerate(tuple_list_tt):
            plot_mean_trace_across_sessions(df_dyn_differences=df_dyn_differences, trial_types=tt_tuple,
                                            regions=[reg], plot_laser=False, plot_zero_line=plot_zero_line,
                                            ax=ax_ov[i_reg][i_tt], text_baseline=np.logical_and(reg == 's1', tt_tuple == ('hit', 'miss')))

    letter_list = ['A', 'B', 'C', 'D', 'E', 'F']
    i_counter = 0
    for ii in range(len(tuple_list_tt)):
        ax_ov[0][ii].text(s=letter_list[i_counter], x=-4.5, y=0.049, fontdict={'weight': 'normal', 'fontsize': 22})
        i_counter += 1
        ax_ov[1][ii].text(s=letter_list[i_counter], x=-4.5, y=0.049, fontdict={'weight': 'normal', 'fontsize': 22})
        i_counter += 1

    if save_fig:
        plt.savefig('figures/s1_s2_dynamic_dff_difference_prestim_poststim_concl3.pdf', bbox_inches='tight')

def plot_single_session_single_tp_decoding_performance(session, time_frame=1.2, n_splits=4):
    ss_dict = {0: session}  # create dict to be compatible with funciton
    df_prediction_train, df_prediction_test, tmp = pof.train_test_all_sessions(sessions=ss_dict, verbose=1, n_split=n_splits,
                                                trial_times_use=np.array([time_frame]), return_decoder_weights=True,
                                            hitmiss_only=False,# list_test=['dec', 'stim'],
                                            include_autoreward=False, neurons_selection='s1',
                                            C_value=50, train_projected=True, proj_dir='same')

    fig, ax = plt.subplots(2,2, figsize=(7, 6), gridspec_kw={'wspace': 0.3, 'hspace': 0.4})
    
    ## NB: To not plot with hue, set hh=None
    # for mouse in df_prediction_train.keys():
    for mouse in [ss_dict[0].mouse]:
        if 'pred_stim_train' in df_prediction_train[mouse].columns:
            plot_df_stats(df=df_prediction_train[mouse], xx='true_stim_train', yy='pred_stim_train_proj', 
                              hh='true_dec_train', ax=ax[0][0])  # set hh=None or hh='dec_train'
            ax[0][0].set_xlabel('Number of cells PS'); ax[0][0].set_ylabel('Predicted probability of PS'); 
            ax[0][0].set_title('TRAIN - PS', weight='bold')
            ax[0][0].legend([], frameon=False)

            plot_df_stats(df=df_prediction_test[mouse], xx='true_stim_test', yy='pred_stim_test_proj', 
                              hh='true_dec_test', ax=ax[0][1])  # set hh=None or hh='dec_test'
            ax[0][1].set_xlabel('Number of cells PS'); ax[0][1].set_ylabel('Predicted probability of PS'); 
            ax[0][1].set_title('TEST - PS', weight='bold')
            ax[0][1].legend([], frameon=False)
    #         print(f'Accuracy PS: {pof.class_av_mean_accuracy(binary_truth=(df_prediction_test[mouse]["true_stim_test"] > 0).astype("int"), estimate=df_prediction_test[mouse]["pred_stim_test_proj"])[0]}')
            
        if 'pred_dec_train' in df_prediction_train[mouse].columns:  
            plot_df_stats(df=df_prediction_train[mouse], xx='true_dec_train', yy='pred_dec_train_proj', 
                              hh=None, xticklabels=['no lick', 'lick'], ax=ax[1][0])
            ax[1][0].set_xlabel('Decision'); ax[1][0].set_ylabel('Predicted probability of lick'); 
            ax[1][0].set_title('TRAIN - LICK', weight='bold')
            ax[1][0].legend('No lick', 'lick', frameon=False)

            plot_df_stats(df=df_prediction_test[mouse], xx='true_dec_test', yy='pred_dec_test_proj', 
                              hh='true_stim_test', xticklabels=['no lick', 'lick'], type_scatter='strip', ax=ax[1][1])
            ax[1][1].set_xlabel('Decision'); ax[1][1].set_ylabel('Predicted probability of lick'); 
            ax[1][1].set_title('TEST - LICK', weight='bold');
            ax[1][1].legend([], frameon=False)
    #         print(f'Accuracy lick: {pof.class_av_mean_accuracy(binary_truth=df_prediction_test[mouse]["true_dec_test"], estimate=df_prediction_test[mouse]["pred_dec_test_proj"])[0]}')
    # plt.suptitle(f'(Logistic Regression) decoding performance for frame {time_frame}: \n{[str(sessions[x]) for x in range(len(sessions))]}\n' + 
    # #              f'Left column shows Photostimulation (PS) decoding performance, right column shows lick decoding\n' +
    #              f'Top row shows train data; bottom row shows test data, {n_splits} splits\n');

    return (df_prediction_train, df_prediction_test)

def plot_dynamic_decoding_panel(time_array, ps_acc_split, reg='s1', ax=None,
                                smooth_traces=False, one_sided_window_size=1):
    if ax is None:
        ax = plt.subplot(111)
    plot_interrupted_trace_simple(ax=ax, time_array=time_array, 
                                    plot_array=np.zeros_like(time_array) + 0.5, 
                                    ccolor='k', aalpha=0.6, llinewidth=3, linest=':')
    for i_lick, dict_part in ps_acc_split.items():  # PS accuracy split per lick /no lick trials
        plot_interrupted_trace_average_per_mouse(ax=ax, time_array=time_array, plot_array=dict_part, llabel=label_split[i_lick], 
                            ccolor=colors_plot[reg][i_lick], plot_indiv=False, plot_laser=False, #i_lick, 
                            plot_errorbar=False, plot_std_area=True, region_list=[reg],
                            running_average_smooth=smooth_traces, one_sided_window_size=one_sided_window_size)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Accuracy')
    ax.legend(loc='upper left', frameon=False); ax.set_title(f'Dynamic PS decoding in {reg.upper()}', weight='bold')
    return ax

def plot_dynamic_decoding_region_difference_panel(time_array, ps_acc_split, ax=None, p_val_thresh=0.05):
    if ax is None:
        ax = plt.subplot(111)
    plot_interrupted_trace_simple(ax=ax, time_array=time_array, 
                                    plot_array=np.zeros_like(time_array), 
                                ccolor='k', aalpha=0.6, llinewidth=3, linest=':')
    for i_lick, dict_part in ps_acc_split.items():  # PS accuracy split per lick /no lick trials
        plot_interrupted_trace_average_per_mouse(ax=ax, time_array=time_array, plot_array=dict_part, llabel=label_split[i_lick], 
                            ccolor=colors_plot['s1'][i_lick], plot_indiv=False, plot_laser=False, #i_lick, 
                            plot_errorbar=False, plot_std_area=False, region_list=['s1', 's2'], plot_diff_s1s2=True)
        p_vals_wsr = pof.wilcoxon_test(dict_part)
        inds_sign = np.where(p_vals_wsr < p_val_thresh)[0]
    #     ax_diff.scatter(time_array[inds_sign], np.zeros_like(inds_sign) + .17 + (i_lick * 0.015), marker='*', s=60, color=color_dict_stand[i_lick])
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Accuracy difference')
    ax.legend(loc='upper left'); ax.set_title('(S1 - S2) difference mean accuracies ', weight='bold')
    ax.set_ylim([-0.05, 0.2])
    return ax

def plot_dynamic_decoding_two_regions(time_array, ps_acc_split, save_fig=False, yaxis_type='accuracy',
                                      smooth_traces=True, one_sided_window_size=1):
    fig = plt.figure(constrained_layout=False, figsize=(12, 4))
    gs_top = fig.add_gridspec(ncols=2, nrows=1, wspace=0.3, 
                            bottom=0.05, top=0.95, left=0.05, right=0.95)
    ax_acc_ps = {}
    for i_reg, reg in enumerate(['s1', 's2']):
        ax_acc_ps[reg] = fig.add_subplot(gs_top[i_reg])
        _ = plot_dynamic_decoding_panel(time_array=time_array, ps_acc_split=ps_acc_split, 
                                    reg=reg, ax=ax_acc_ps[reg], smooth_traces=smooth_traces, 
                                    one_sided_window_size=one_sided_window_size)

        if yaxis_type == 'accuracy':
            ax_acc_ps[reg].set_ylabel('Prediction accuracy')
        elif yaxis_type == 'prediction':
            ax_acc_ps[reg].set_ylabel('Network prediction P(PS)')
        else:
            print('WARNING: yaxis_type not recognised')
        ax_acc_ps[reg].set_ylim([0.1, 0.9])
        ax_acc_ps[reg].set_xlim([-4, 8.5])
        ax_acc_ps[reg].spines['top'].set_visible(False)
        ax_acc_ps[reg].spines['right'].set_visible(False)

    if save_fig:
        plt.savefig('fourway_dyn_dec.pdf')#, bbox_to_inches='tight')

def plot_dyn_stim_decoding_compiled_summary_figure(ps_acc_split, violin_df_test, time_array, save_fig=False):
    ## PS decoding figure
    fig = plt.figure(constrained_layout=False, figsize=(16, 7))
    gs_top = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1, 1, 1], wspace=0.4, 
                            bottom=0.6, top=1)
    gs_bottom = fig.add_gridspec(ncols=4, nrows=1, width_ratios=[1, 1, 1, 1], wspace=0.4, 
                            bottom=0, top=0.4)  
    
    p_val_thresh = 0.05  # P val for wilcoxon sr test.
    tp_violin = list(violin_df_test.keys())
    
    ## S1 & S2 figures:
    ax_acc_ps = {}
    for i_reg, reg in enumerate(['s1', 's2']):
        ax_acc_ps[reg] = fig.add_subplot(gs_top[i_reg])
        _ = plot_dynamic_decoding_panel(time_array=time_array, ps_acc_split=ps_acc_split, 
                                    reg=reg, ax=ax_acc_ps[reg])
        ax_acc_ps[reg].set_ylim([0.39, 0.8])
        for tp in tp_violin:
            ax_acc_ps[reg].scatter([tp], [.45], marker='^', s=50, color='k')
            ax_acc_ps[reg].text(s=f'{tp}s', x=tp - 0.45, y=0.4)
        
    ## S1/S2 difference fig:
    ax_diff = fig.add_subplot(gs_top[2])
    _ = plot_dynamic_decoding_region_difference_panel(time_array=time_array, ps_acc_split=ps_acc_split, 
                                                      ax=ax_diff, p_val_thresh=p_val_thresh)

    for ax in [ax_acc_ps['s1'], ax_acc_ps['s2'], ax_diff]:
            ax.set_xlim([-4, 8])

    ## Violin plots
    lick_title = {0: 'No Lick', 1: 'Lick'}
    ax_viol = {}
    for i_tp, tp in enumerate(tp_violin):
        for lick in [0, 1]:
            ax_viol[lick + 2 * i_tp] = fig.add_subplot(gs_bottom[lick + 2 * i_tp])
            plot_df = violin_df_test[tp][violin_df_test[tp]['true_dec_test'] == lick]
            viol = sns.violinplot(data=plot_df, x='true_stim_test', y='pred_stim_test',
                        palette=[0.6 * np.array(color_dict_stand[lick]), 1.1 * np.array(color_dict_stand[lick])],
                        hue='region', split=True, inner=None, ax=ax_viol[lick + 2 * i_tp])
            plt.setp(viol.collections, alpha=0.8)
            tmp = sns.pointplot(data=plot_df, x='true_stim_test', y='pred_stim_test',
                        palette=[0.6 * np.array(color_dict_stand[lick]), 1.1 * np.array(color_dict_stand[lick])],
                        hue='region', label=None, linestyles=[linest_reg['s1'], linest_reg['s2']], estimator=np.mean, 
                        ax=ax_viol[lick + 2 * i_tp])
            accuracy_tp_s1 = pof.class_av_mean_accuracy(binary_truth=(plot_df[plot_df['region'] == 'S1']['true_stim_test'] > 0).astype('int'),
                            estimate=plot_df[plot_df['region'] == 'S1']['pred_stim_test'])[0]
            accuracy_tp_s2 = pof.class_av_mean_accuracy(binary_truth=(plot_df[plot_df['region'] == 'S2']['true_stim_test'] > 0).astype('int'),
                            estimate=plot_df[plot_df['region'] == 'S2']['pred_stim_test'])[0]
            
            ax_viol[lick + 2 * i_tp].set_title(f'Time: {tp}s, {lick_title[lick]}, S1 & S2\nAccuracy = {np.round(accuracy_tp_s1, 2)} & {np.round(accuracy_tp_s2, 2)}', weight='bold')
            viol.legend_.remove();
            ax_viol[lick + 2 * i_tp].set_xlabel('# cells PS'); ax_viol[lick + 2 * i_tp].set_ylabel('Decoded P(PS)')
            ax_viol[lick + 2 * i_tp].set_xticklabels([0, 5, 10, 20, 30, 40, 50])
    ## Labels:
    ax_acc_ps['s1'].text(s='A', x=-7, y=0.8, fontdict={'weight': 'normal', 'fontsize': 26})
    ax_acc_ps['s2'].text(s='B', x=-7, y=0.8, fontdict={'weight': 'normal', 'fontsize': 26})
    ax_diff.text(s='C', x=-7.5, y=0.2, fontdict={'weight': 'normal', 'fontsize': 26})
    ax_acc_ps['s1'].text(s='D', x=-7, y=0.14, fontdict={'weight': 'normal', 'fontsize': 26})
    ax_acc_ps['s1'].text(s='E', x=7, y=0.14, fontdict={'weight': 'normal', 'fontsize': 26})
    ax_acc_ps['s1'].text(s='F', x=20, y=0.14, fontdict={'weight': 'normal', 'fontsize': 26})
    ax_acc_ps['s1'].text(s='G', x=33.6, y=0.14, fontdict={'weight': 'normal', 'fontsize': 26})
    
    for ax in {**ax_viol, **{0: ax_diff}, **ax_acc_ps}.values():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if save_fig:
        plt.savefig('figures/decoding_ps.pdf', bbox_inches='tight')


def plot_dyn_lick_decoding_compiled_summary_figure(violin_df_test, time_array, lick_acc, angle_dec,
                                                   region_list=['s1' ,'s2']):
    ## Lick decoding figure
    fig = plt.figure(constrained_layout=False, figsize=(14, 3))
    gs_left = fig.add_gridspec(ncols=1, nrows=1, wspace=0.4, 
                            left=0, right=0.3)
    gs_middle = fig.add_gridspec(ncols=1, nrows=len(violin_df_test), wspace=0.4, 
                            left=0.4, right=0.6, hspace=0.7) 
    gs_right = fig.add_gridspec(ncols=1, nrows=1, wspace=0.4, 
                            left=0.7, right=1)

    tp_violin = list(violin_df_test.keys())

    ## Total average plot
    ax_acc = fig.add_subplot(gs_left[0])
    plot_interrupted_trace_simple(ax=ax_acc, time_array=time_array, plot_array=np.zeros_like(time_array) + 0.5, 
                                ccolor='k', aalpha=0.6, llinewidth=3, linest=':')
    for reg in region_list:
        plot_interrupted_trace_average_per_mouse(ax=ax_acc,  time_array=time_array,plot_array=lick_acc, #individual_mouse_list=['J065_s1', 'J065_s2'],
                            llabel=reg.upper(), ccolor=colors_reg[reg], plot_laser=False,
                            plot_std_area=True, plot_indiv=False, region_list=[reg])
        
    for tp in tp_violin:
        ax_acc.scatter([tp], [.48], marker='^', s=50, color='k')
        ax_acc.text(s=f'{tp}s', x=tp - 0.4, y=0.44)

    ax_acc.set_xlabel('Time (s)'); ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(loc='upper left'); ax_acc.set_title('Dynamic Lick decoding', weight='bold', y=1.13);
    ax_acc.set_ylim([0.43, 0.88])

    ## Violin plot
    for i_tp, tp in enumerate(tp_violin):
        ax_viol = fig.add_subplot(gs_middle[i_tp])
        plot_df = violin_df_test[tp]
        viol = sns.violinplot(data=plot_df, x='true_dec_test', y='pred_dec_test',
                    palette=[0.6 * colors_reg['s1'], 1.1 * colors_reg['s2']],
                    hue='region', split=True, inner=None, bw=0.1, ax=ax_viol)
        plt.setp(viol.collections, alpha=0.8)
        sns.pointplot(data=plot_df, x='true_dec_test', y='pred_dec_test',
                    palette=[0.6 * colors_reg['s1'], 1.1 * colors_reg['s2']],
                    hue='region', label=None, linestyles=[linest_reg['s1'], linest_reg['s2']], ax=ax_viol)
        accuracy_tp_s1 = pof.class_av_mean_accuracy(binary_truth=plot_df[plot_df['region'] == 'S1']['true_dec_test'],
                            estimate=plot_df[plot_df['region'] == 'S1']['pred_dec_test'])[0]
        accuracy_tp_s2 = pof.class_av_mean_accuracy(binary_truth=plot_df[plot_df['region'] == 'S2']['true_dec_test'], 
                            estimate=plot_df[plot_df['region'] == 'S2']['pred_dec_test'])[0]
        ax_viol.set_title(f'Time: {tp}s, S1 & S2\nAccuracy = {np.round(accuracy_tp_s1, 2)} & {np.round(accuracy_tp_s2, 2)}', 
                weight='bold', y=1.04)
        viol.legend_.remove();
        if len(tp_violin) == 2:
            if i_tp == 1:
                ax_viol.set_xlabel('Decision'); ax_viol.set_ylabel('               Decoded')
            else: 
                ax_viol.set_xlabel(''); ax_viol.set_ylabel('P(Lick)             ')
        elif len(tp_violin) == 1:
            ax_viol.set_xlabel('Decision');  ax_viol.set_ylabel('P(Lick)') 
        ax_viol.set_xticklabels(['No lick', 'Lick'])
        
    ## Angle plot
    ax_angle = fig.add_subplot(gs_right[0])
    for reg in region_list:
        ax_angle, mean_angle_traces = plot_interrupted_trace_average_per_mouse(ax=ax_angle, time_array=time_array, plot_array=angle_dec, llabel=None,
                                            plot_errorbar=False, plot_indiv=False, plot_std_area=True, region_list=[reg],
                                            plot_laser=False, 
                                                ccolor=0.5 * (colors_plot[reg][0] + colors_plot[reg][1]))
    ax_angle.set_xlabel('Time (s)'); ax_angle.set_ylabel('Angle (deg)') #ax_angle.legend(loc='upper left');
    ax_angle.set_title('Dissimilarity PS and Lick decoders', weight='bold', y=1.13)

    ## Labels:
    ax_acc.text(s='A', x=-5.5, y=0.92, fontdict={'weight': 'normal', 'fontsize': 28})
    ax_acc.text(s='B', x=9.5, y=0.92, fontdict={'weight': 'normal', 'fontsize': 28})
    ax_acc.text(s='C', x=19, y=0.92, fontdict={'weight': 'normal', 'fontsize': 28})

    ax_acc.set_xlim([-4, 8])
    ax_angle.set_xlim([-4, 8])

    sns.despine()
    # plt.savefig('figures/decoding_licks.pdf', bbox_inches='tight')

def plot_average_ps_traces_per_mouse(sessions, save_fig=False):
    ## Averages of all sessions:
    mouse_list = np.unique([ss.mouse for _, ss in sessions.items()])

    range_ps = 2
    fig, ax = plt.subplots(len(mouse_list), range_ps, figsize=(4 * range_ps, 3 * len(mouse_list)),
                           gridspec_kw={'hspace': 0.4, 'wspace': 0.4})
    
    title_ps = {0: 'No PS', 1: 'Test PS', 2: '150 PS', 3: 'Run legend:'}
    
    av_axes = {}
    i_ax = 0
    color_index = {m: 0 for m in mouse_list}
    legend_tuple = []
    for i_m, mouse in enumerate(mouse_list):
        temp_session_list = [k for k, ss in sessions.items() if ss.mouse == mouse]
        for i_s in temp_session_list:
            ss = sessions[i_s]
            for i_ps in range(range_ps):
                av_axes[i_ax] = ax[i_m][i_ps]
                if i_ps <= 2:
                    plot_interrupted_trace_simple(ax=av_axes[i_ax], time_array=ss.filter_ps_time,
                                                plot_array=np.mean(ss.behaviour_trials[:, ss.photostim == i_ps, :][:, :, ss.filter_ps_array], (0, 1)), 
                                                llabel=f'{mouse}, R{ss.run_number}', ccolor=color_dict_stand[color_index[mouse]], zero_mean=True)
                if i_ps <= 1:
                    av_axes[i_ax].set_ylim([-0.05, 0.05])
                if i_m == 0:
                    av_axes[i_ax].set_title(title_ps[i_ps], weight='bold')
                if i_m == len(mouse_list) - 1:
                    av_axes[i_ax].set_xlabel('Time w.r.t stim. (s)')
                if i_ps == 0:
                    av_axes[i_ax].set_ylabel('Mean ' + r'$\Delta F / F$')
                if i_ps == range_ps - 1 and i_s == temp_session_list[-1]:
                    leg = av_axes[i_ax].legend(bbox_to_anchor=(1.55, 1)) 
                    legend_tuple.append(leg)
                i_ax += 1
            color_index[mouse] += 1
    legend_tuple = tuple(legend_tuple)
    if save_fig:
        plt.savefig('figures/average_all_sessions.pdf', bbox_extra_artists=legend_tuple, bbox_inches='tight')
