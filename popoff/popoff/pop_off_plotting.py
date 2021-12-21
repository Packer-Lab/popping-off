import sys, os, copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar as mpl_colorbar
from matplotlib.ticker import MultipleLocator
# import matplotlib.gridspec as gridspec
from matplotlib import gridspec
import matplotlib.patches
import seaborn as sns
# import utils_funcs as utils
# import run_functions as rf
# from subsets_analysis import Subsets
# import pickle
# import sklearn.decomposition
from cycler import cycler
import pandas as pd
import math, cmath, copy
from tqdm import tqdm
import scipy.stats, scipy.optimize
import statsmodels.api, statsmodels.regression
import sklearn.linear_model
from Session import Session  # class that holds all data per session
import pop_off_functions as pof
# from linear_model import PoolAcrossSessions, LinearModel, MultiSessionModel
from utils.utils_funcs import d_prime

## Set default settings.
plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
## Create list with standard colors:
color_dict_stand = {}
for ii, x in enumerate(plt.rcParams['axes.prop_cycle']()):
    color_dict_stand[ii] = x['color']
    if ii > 8:
        break  # after 8 it repeats (for ever)
color_dict_stand[10] = '#994F00'
color_dict_stand[11] = '#4B0092'

colors_plot = {'s1': {lick: 0.6 * np.array(color_dict_stand[lick]) for lick in [0, 1]},
               's2': {lick: 1.1 * np.array(color_dict_stand[lick]) for lick in [0, 1]}}
colors_reg = {reg: 0.5 * (colors_plot[reg][0] + colors_plot[reg][1]) for reg in ['s1', 's2']}

color_tt = {'hit': '#117733', 'miss': '#882255', 'fp': '#88CCEE', 'cr': '#DDCC77',
            'Hit': '#117733', 'Miss': '#882255', 'FP': '#88CCEE', 'CR': '#DDCC77',
            'urh': '#44AA99', 'arm': '#AA4499', 'spont': '#332288', 'prereward': '#332288', 
            'reward\nonly': '#332288', 'Reward\nonly': '#332288',
            'pre_reward': '#332288', 'Reward': '#332288', 'reward only': '#332288', 'rew. only': '#332288', 'hit&miss': 'k', 
            'fp&cr': 'k', 'photostim': sns.color_palette()[6], 'too_': 'grey',
            'hit_n1': '#b0eac9', 'hit_n2': '#5ab17f', 'hit_n3': '#117733',
            'miss_n1': '#a69098', 'miss_n2': '#985d76', 'miss_n3': '#882255',
            'hit_c1': '#b0eac9', 'hit_c2': '#5ab17f', 'hit_c3': '#117733',
            'miss_c1': '#a69098', 'miss_c2': '#985d76', 'miss_c3': '#882255'
            }  # Tol colorblind colormap https://davidmathlogic.com/colorblind/#%23332288-%23117733-%2300FFD5-%2388CCEE-%23DDCC77-%23CC6677-%23AA4499-%23882255
label_tt = {'hit': 'Hit', 'Hit': 'Hit', 'miss': 'Miss', 'Miss': 'Miss',
            'FP': 'FP', 'fp': 'FP', 'cr': 'CR', 'CR': 'CR', 'too_':  'Too early',
            'hit_n1': 'Hit 5-10', 'hit_n2': 'Hit 20-30', 'hit_n3': 'Hit 40-50',
            'miss_n1': 'Miss 5-10', 'miss_n2': 'Miss 20-30', 'miss_n3': 'Miss 40-50',
            'hit_c1': 'Hit low pop. var.', 'hit_c2': 'Hit mid pop. var.', 'hit_c3': 'Hit high pop. var.',
            'miss_c1': 'Miss low pop. var.', 'miss_c2': 'Miss mid pop. var.', 'miss_c3': 'Miss high pop. var.',
            'urh': 'UR Hit', 'arm': 'AR Miss', 'spont': 'Reward only', 'prereward': 'Reward only',
            'Reward only': 'Reward only', 'reward only': 'Reward only', 'Rew. only': 'Rew. only',
            'reward_only': 'Reward only', 'Reward\nonly': 'Reward\nonly',
            'cr 10 trials': 'CR 10 trials'}
covar_labels = {'mean_pre': 'Pop. mean', 'variance_cell_rates': 'Pop. variance',
                'corr_pre': 'Pop. correlation', 'largest_PC_var': 'Var largest PC',
                'n_PCs_90': 'PCs for 90% var', 'n_PCs_95': 'PCs for 95% var',
                'trial_number': 'Trial number', 'mean_cell_variance': 'Temp. variance',
                'var_cell_variance': 'Meta-variability',
                # 'reward_history': 'Reward history\n(% succes in last 5 trials)'
                'reward_history': 'Rew. history (% hits)'}
linest_reg = {'s1': '-', 's2': '-'}
label_split = {**{0: 'No L.', 1: 'Lick'}, **label_tt}
alpha_reg = {'s1': 0.9, 's2':0.5}

for tt in color_tt.keys():
    colors_plot['s1'][tt] = color_tt[tt]
    colors_plot['s2'][tt] = color_tt[tt]

def despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def naked(ax):
    for ax_name in ['top', 'bottom', 'right', 'left']:
        ax.spines[ax_name].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

def set_fontsize(font_size=12):
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.autolimit_mode'] = 'data' # default: 'data'
    params = {'legend.fontsize': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size}
    plt.rcParams.update(params)
    print(f'Font size is set to {font_size}')

def add_ps_artefact(ax, time_axis, y_min=0, y_max=1):
    ## plot box over artefact
    color_box = color_tt['photostim']
    alpha_box = 0.3
    start_box = time_axis[np.min(np.where(np.isnan(time_axis))[0])- 1] + 1 / 30
    end_box = time_axis[np.max(np.where(np.isnan(time_axis))[0]) + 1] - 1 / 30
    # print(start_box, end_box)
    ax.axvspan(start_box, end_box, ymin=y_min, ymax=y_max,
                alpha=alpha_box, color=color_box)

def equal_xy_lims(ax, start_zero=False):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    max_outer_lim = np.maximum(xlims[1], ylims[1])
    min_inner_lim = np.minimum(xlims[0], ylims[0])

    if start_zero:
        ax.set_xlim([0, max_outer_lim])
        ax.set_ylim([0, max_outer_lim])
    else:
        ax.set_xlim([min_inner_lim, max_outer_lim])
        ax.set_ylim([min_inner_lim, max_outer_lim])

def equal_lims_two_axs(ax1, ax2):

    xlim_1 = ax1.get_xlim()
    xlim_2 = ax2.get_xlim()
    ylim_1 = ax1.get_ylim()
    ylim_2 = ax2.get_ylim()
     
    new_x_min = np.minimum(xlim_1[0], xlim_2[0])
    new_x_max = np.maximum(xlim_1[1], xlim_2[1])
    new_y_min = np.minimum(ylim_1[0], ylim_2[0])
    new_y_max = np.maximum(ylim_1[1], ylim_2[1])

    ax1.set_xlim([new_x_min, new_x_max])
    ax2.set_xlim([new_x_min, new_x_max])
    ax1.set_ylim([new_y_min, new_y_max])
    ax2.set_ylim([new_y_min, new_y_max])

def remove_xticklabels(ax):  # remove labels but keep ticks
    ax.set_xticklabels(['' for x in ax.get_xticklabels()])

def remove_yticklabels(ax):  # remove labels but keep ticks
    ax.set_yticklabels(['' for x in ax.get_yticklabels()])

def remove_both_ticklabels(ax):  # remove labels but keep ticks
    remove_xticklabels(ax)
    remove_yticklabels(ax)

def two_digit_sci_not(x):
    sci_not_spars = np.format_float_scientific(x, precision=1)
    # print(sci_not_spars)
    if sci_not_spars[2] == 'e':  # exactly precision=0 so one shorter
        sci_not_spars = sci_not_spars[0] + sci_not_spars[2:]  # skip dot
    elif sci_not_spars[3] == 'e':  # ceil
        sci_not_spars = str(int(sci_not_spars[0]) + 1) + sci_not_spars[3:]  # skip dot
    else:
        assert False
    return sci_not_spars

def readable_p(p_val):
    if type(p_val) != str:
        p_val = two_digit_sci_not(x=p_val)

    if p_val[2] == 'e':
        assert p_val[:4] == '10e-', p_val
        tmp_exp = int(p_val[-2:])
        p_val = f'1e-{str(tmp_exp - 1).zfill(2)}'

    # assert len(p_val) == 5, f'p_val format not recognised. maybe exp < -99? p val is {p_val}'
    if len(p_val) > 5:
        assert len(p_val) == 6 and p_val[1:3] == 'e-', p_val 
        exponent = p_val[-3:]
        read_p = f'{p_val[0]}x' + r"$10^{{-{tmp}}}$".format(tmp=exponent)  # for curly brackets explanation see https://stackoverflow.com/questions/53781815/superscript-format-in-matplotlib-plot-legend
    
    else:
        if p_val == '1e+00' or p_val == '1e-00':
            read_p = '1.0'
        else:
            assert p_val[2] == '-', f'p value is greater than 1, p val: {p_val}'

            if p_val[-3:] == '-01':
                read_p = f'0.{p_val[0]}'
            elif p_val[-3:] == '-02':
                read_p = f'0.0{p_val[0]}'
            elif p_val[-3:] == '-03':
                read_p = f'0.00{p_val[0]}'
            else:
                if int(p_val[-2:]) < 10:
                    exponent = p_val[-1]
                else:
                    exponent = p_val[-2:]
                read_p = f'{p_val[0]}x' + r"$10^{{-{tmp}}}$".format(tmp=exponent)  # for curly brackets explanation see https://stackoverflow.com/questions/53781815/superscript-format-in-matplotlib-plot-legend
    return read_p

def asterisk_p(p_val, bonf_correction=1):
    if type(p_val) == str:
        p_val = float(p_val)
  
    if p_val < (0.001 / bonf_correction):
        return '***'
    elif p_val < (0.01 / bonf_correction):
        return '**'
    elif p_val < (0.05 / bonf_correction):
        return '*'
    else:
        return 'n.s.'


def translate_session(session_name, number_only=False, capitalize=True):
    session_name = str(session_name)
    dict_translations = {'Mouse J064, run 10': 'Session 1',
                         'Mouse J064, run 11': 'Session 2',
                         'Mouse J064, run 14': 'Session 3',
                         'Mouse RL070, run 28': 'Session 4',
                         'Mouse RL070, run 29': 'Session 5',
                         'Mouse RL117, run 26': 'Session 6',
                         'Mouse RL117, run 29': 'Session 7',
                         'Mouse RL117, run 30': 'Session 8',
                         'Mouse RL123, run 22': 'Session 9',
                         'Mouse RL116, run 32': 'Session 10',
                         'Mouse RL116, run 33': 'Session 11'}

    assert session_name in dict_translations.keys(), f'session name "{session_name}" not recognised!!'
    translation = dict_translations[session_name]
    if capitalize is False:
        translation = translation[0].lower() + translation[1:]
    if number_only:
        translation = translation.split(' ')[-1]
    return translation

def weighted_mean(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def weighted_covariance(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - weighted_mean(x, w)) * (y - weighted_mean(y, w))) / np.sum(w)

def weighted_pearson_corr(x, y, w):
    """Weighted Correlation, adapted from 
    https://stackoverflow.com/questions/38641691/weighted-correlation-coefficient-with-pandas
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    https://files.eric.ed.gov/fulltext/ED585538.pdf
    """
    return weighted_covariance(x, y, w) / np.sqrt(weighted_covariance(x, x, w) * weighted_covariance(y, y, w))


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
            plot_std_area=False, region_list=['s1', 's2'], time_breakpoint=1, time_breakpoint_postnan=None,
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
    # time_breakpoint = np.argmax(np.diff(time_array)) + 1# finds the 0, equivalent to art_gap_start
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

            if np.isnan(plot_mean).sum() > 0:  #nans present if this trial type did not occur for this mouse (eg URH)
                print('skipping', mouse)
                continue
            if running_average_smooth:
                if time_breakpoint_postnan is not None and np.sum(np.isnan(time_array)) > 0:
                    ## filter data points for which time is nan when computing running average
                    plot_mean[:time_breakpoint] = smooth_trace(plot_mean[:time_breakpoint], one_sided_window_size=one_sided_window_size)
                    plot_mean[time_breakpoint:] = smooth_trace(plot_mean[time_breakpoint:], one_sided_window_size=one_sided_window_size)
                else:
                    plot_mean[one_sided_window_size:-one_sided_window_size] = np.convolve(plot_mean, np.ones(window_size), mode='valid') / window_size
            average_mean[reg] += plot_mean #/ len(mouse_list) * 2  # save mean (assumes that _s1 and _s2 in mouse_list so factor 2)
            all_means[reg][count_means[reg] ,:] = plot_mean.copy()  # save data for std
            count_means[reg] += 1
            if plot_indiv and mouse in individual_mouse_list and not plot_diff_s1s2:  # plot individual traces
                ax.plot(time_1, plot_mean[:time_breakpoint],  linewidth=2, linestyle=linest[reg],
                            markersize=12, color=ccolor, label=None, alpha=0.6)
                ax.plot(time_2, plot_mean[time_breakpoint:],  linewidth=2, linestyle=linest[reg],
                            markersize=12, alpha=0.6, label=mouse, color=ccolor)
    # for reg in region_list:
    #     assert count_means[reg] == all_means[reg].shape[0]
    for reg in average_mean.keys():
        average_mean[reg] = average_mean[reg] / count_means[reg]

    if plot_groupav:
        #         region_hatch = {'s1': '/', 's2': "\ " }
        if plot_diff_s1s2 is False:
            for rr, av_mean in average_mean.items():
                # av_mean[2:-2] = np.convolve(av_mean, np.ones(window_size), mode='valid') / window_size
                if rr in region_list:
                    # if rr == 's2':
                    #     print([str(x) + ',' for x in av_mean])
                    std_means = np.std(all_means[rr], 0) / np.sqrt(count_means[rr]) * 1.96  # 95% CI
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

def smooth_trace(trace, one_sided_window_size=3, fix_ends=True):

    window_size = int(2 * one_sided_window_size + 1)
    old_trace = copy.deepcopy(trace)
    trace[one_sided_window_size:-one_sided_window_size] = np.convolve(trace, np.ones(window_size), mode='valid') / window_size

    if fix_ends:
        for i_w in range(one_sided_window_size):
            trace[i_w] = np.mean(old_trace[:(i_w + one_sided_window_size + 1)])
            trace[-(i_w + 1)] = np.mean(old_trace[(-1 * (i_w + one_sided_window_size + 1)):])
    return trace

def transform_reg_sorted_cell_number(session, n_reg_sorted, sort_dict, reg='S1'):
    if reg == 'S1':
        n_reg = sort_dict['s1'][n_reg_sorted]
        reg_arr = np.where(session.s1_bool)[0]
        assert len(reg_arr) == len(sort_dict['s1'])
        n = reg_arr[n_reg]
    elif reg == 'S2':
        n_reg = sort_dict['s2'][n_reg_sorted]
        reg_arr = np.where(session.s2_bool)[0]
        assert len(reg_arr) == len(sort_dict['s2']), f'{len(reg_arr)} and {len(sort_dict["s2"])}'
        n = reg_arr[n_reg]
    return n

def transform_suite2p_cell_number(session, n_suite2p):
    suite2p_arr = session.suite2p_id
    n = np.where(suite2p_arr == n_suite2p)[0]
    assert len(n) == 1, f'Suite2p ID {n_suite2p} not in {session}'
    return int(n[0])

def plot_single_cell_all_trials(session, n=0, start_time=-4, stim_window=0.35,
                                demean=True, y_lim=None, osws=3):

    fig, ax = plt.subplots(2, 3, figsize=(20, 7), gridspec_kw={'hspace': 0.6, 'wspace': 0.4})

    assert False, 'not sure why no normalisation is used here (it is cut off manually)'
    tt_list = ['hit', 'fp', 'miss', 'cr', 'spont']

    start_frame = np.argmin(np.abs(session.filter_ps_time - start_time))  # cut off at start
    stim_frame = np.argmin(np.abs(session.filter_ps_time - stim_window))  # define post-stim response
    pre_stim_frame = np.argmin(np.abs(session.filter_ps_time + stim_window))  # up to stim ( to avoid using stim artefact)

    # (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2, data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
    #     time_ticks, time_tick_labels, start_frame) = normalise_raster_data(session, start_time=start_time, stim_window=stim_window, sorting_method='euclidean',   # also sorting is False so doesn't matter
    #                                                                             sort_tt_list=sort_tt_list, sort_neurons=False)

    time_arr = session.filter_ps_time[start_frame:]

    ## From normalise_rsater function:
        # data_use_mat = session.behaviour_trials[:, session.photostim < 2, :][:, :, start_frame:]  # discarded 150 cells PS & discared pre -4 seconds
        # data_use_mat_norm = data_use_mat - np.mean(data_use_mat[:, :, :(pre_stim_frame - start_frame)], 2)[:, :, None]  # normalize by pre-stim activity per neuron

    for i_tt, tt in enumerate(tt_list):
        curr_ax = ax[(0 if i_tt < 3 else 1)][i_tt % 3]
        if tt == 'spont':
            trial_inds = np.arange(session.pre_rew_trials.shape[1])
            cell_data = session.pre_rew_trials[n, :, :][:, start_frame:]
        else:
            trial_inds = np.where(np.logical_and(session.outcome == tt, session.photostim < 2))[0]
            cell_data = session.behaviour_trials[n, :, :][trial_inds, :][:, start_frame:]
        if demean:
            cell_data = cell_data - np.mean(cell_data[:, :(pre_stim_frame - start_frame)], 1)[:, None]
        for i_trial_n, trial_n in enumerate(trial_inds):
            curr_ax.plot(time_arr, smooth_trace(cell_data[i_trial_n, :], one_sided_window_size=osws),
                          c='grey', alpha=0.8, linewidth=1)
        curr_ax.plot(time_arr, np.mean(cell_data, 0),
                          c=color_tt[tt], alpha=0.9, linewidth=3)
        curr_ax.set_xlabel('Time (s)')
        curr_ax.set_ylabel(r"$\DeltaF/F")
        curr_ax.set_title(f'{tt} trials, N={len(trial_inds)}')
        curr_ax.spines['top'].set_visible(False)
        curr_ax.spines['right'].set_visible(False)
        if y_lim is not None:
            curr_ax.set_ylim(y_lim)

    ax[1][2].axis('off')
    ax[1][2].text(s=f'Traces smoothed with {2 * osws + 1}-frames window', x=0, y=0.5)
    ax[1][2].text(s=f'Cell ID: {n}, session {str(session)}, Suite-2P ID {session.suite2p_id[n]}', x=-0.2, y=0.8, fontdict={'weight': 'bold'})
    ax[1][2].text(s=f'This cell was targeted {int(np.sum(np.mean(session.is_target[n, :, :], 1)))} times', x=0, y=0.2)


def sort_data_matrix(data, session=None, reg=None, sorting_method='euclidean'):
    # print(data.shape)
    if sorting_method == 'correlation':
        sorting = pof.opt_leaf(data, link_metric='correlation')[0]
    elif sorting_method == 'euclidean':
        sorting = pof.opt_leaf(data, link_metric='euclidean')[0]
    elif sorting_method == 'max_pos':
        arg_max_pos = np.argmax(data, 1)
        assert len(arg_max_pos) == data.shape[0]
        sorting = np.argsort(arg_max_pos)
    elif sorting_method == 'abs_max_pos':
        arg_max_pos = np.argmax(np.abs(data), 1)
        assert len(arg_max_pos) == data.shape[0]
        sorting = np.argsort(arg_max_pos)
    elif sorting_method == 'n_targeted':
        neuron_targ = np.mean(session.is_target, (1, 2))
        if reg == 's1':
            assert np.sum(session.s1_bool) == data.shape[0]
            neuron_targ_reg = neuron_targ[session.s1_bool]  # select region
            sorting = np.argsort(neuron_targ_reg)[::-1]
        elif reg == 's2':
            assert np.sum(session.s2_bool) == data.shape[0]
            neuron_targ_reg = neuron_targ[session.s2_bool]  # select region
            sorting = np.argsort(neuron_targ_reg)[::-1]
    elif sorting_method == 'normal':
        return np.arange(data.shape[0])
    elif sorting_method == 'amplitude':
        max_val_arr = np.max(data, 1)
        sorting = np.argsort(max_val_arr)[::-1]
    elif sorting_method == 'sum':
        sum_data = np.sum(data, 1)
        sorting = np.argsort(sum_data)[::-1]
    return sorting

def normalise_raster_data(session, start_time=-2.1, start_baseline_time=-2.1, end_time=4,
                          pre_stim_window=-0.07, post_stim_window=None, filter_150_stim=False,
                          sorting_method='euclidean', sort_tt_list=['hit', 'miss', 'spont'],
                          sort_neurons=True, baseline_by_prestim=True):
    '''overrides session.outcome with ARM and URH types!!'''
    if post_stim_window is None:
        if filter_150_stim:
            post_stim_window = 0.83
        else:
            post_stim_window = 0.35
    start_frame = np.argmin(np.abs(session.filter_ps_time - start_time))  # cut off at start
    end_frame = np.argmin(np.abs(session.filter_ps_time - end_time))
    start_baseline_frame = np.argmin(np.abs(session.filter_ps_time - start_baseline_time))  # start of baselining pre stim eriod
    pre_stim_frame = np.argmin(np.abs(session.filter_ps_time - pre_stim_window))  # end of baselining period pre stim ( to avoid using stim artefact)
    post_stim_frame = np.argmin(np.abs(session.filter_ps_time - post_stim_window))  # define start of post-stim response

    time_axis = session.filter_ps_time[start_frame:end_frame]
    n_time_ticks = int(np.floor((session.filter_ps_time[end_frame] - session.filter_ps_time[start_frame]) / 2) + 1)
    time_ticks = np.arange(n_time_ticks) * 2 * session.frequency
    time_tick_labels = [str(np.round(x)) for x in session.filter_ps_time[start_frame:][time_ticks]]
    # print(start_frame, start_baseline_frame, pre_stim_frame, pre_stim_frame - start_frame, post_stim_frame)

    ## Sort neurons by pearosn corr of post-stim response of sort_tt_list
    if filter_150_stim:  # discard 150 cells if necessary
        data_use_mat = session.behaviour_trials[:, session.photostim < 2, :]
    else:
        data_use_mat = session.behaviour_trials
    data_spont_mat = session.pre_rew_trials

    if baseline_by_prestim:
        data_use_mat_norm = data_use_mat - np.mean(data_use_mat[:, :, start_baseline_frame:pre_stim_frame], 2)[:, :, None]  # normalize by pre-stim activity per neuron
        # data_use_mat_norm = data_use_mat - np.mean(data_use_mat[:, :, start_baseline_frame:pre_stim_frame], (0, 2))[None, :, None]  # normalize by pre-stim activity averaged across neurons
        data_spont_mat_norm = data_spont_mat - np.mean(data_spont_mat[:, :, start_baseline_frame:pre_stim_frame], 2)[:, :, None]
    else:
        print('WARNING: not normalized')
        data_use_mat_norm = data_use_mat #- np.mean(data_use_mat[:, :, start_baseline_frame:pre_stim_frame], 2)[:, :, None]  # normalize by pre-stim activity per neuron
        data_spont_mat_norm = data_spont_mat #- np.mean(data_spont_mat[:, :, start_baseline_frame:pre_stim_frame], 2)[:, :, None]


    data_use_mat_norm = data_use_mat_norm[:, :, start_frame:end_frame]  # discarded pre -4 seconds
    data_spont_mat_norm = data_spont_mat_norm[:, :, start_frame:end_frame]
    # start_baseline_frame = start_baseline_frame - start_frame  # correct for cutting off data at start_frame
    # pre_stim_frame = pre_stim_frame - start_frame
    post_stim_frame = post_stim_frame - start_frame
    # end_frame = end_frame - start_frame


    data_use_mat_norm_s1 = data_use_mat_norm[session.s1_bool, :, :]
    data_use_mat_norm_s2 = data_use_mat_norm[session.s2_bool, :, :]
    outcome_arr = session.outcome
    outcome_arr[session.autorewarded] = 'arm'
    outcome_arr[session.unrewarded_hits] = 'urh'
    if filter_150_stim:
        outcome_arr = outcome_arr[session.photostim < 2]

    if sort_neurons:
        if type(sort_tt_list) == str:
            sort_tt_list = list(sort_tt_list)
        assert type(sort_tt_list) == list

        ## Assemble trial-averaged data from all trial types used for sorting:
        data_sorting_s1 = None
        data_sorting_s2 = None
        for sort_tt in sort_tt_list:
            if sort_tt == 'spont':
                tmp_data_sorting_s1 = data_spont_mat_norm[session.s1_bool, :, :][:, :, post_stim_frame:].mean(1)
                tmp_data_sorting_s2 = data_spont_mat_norm[session.s2_bool, :, :][:, :, post_stim_frame:].mean(1)
            else:
                assert sort_tt in list(outcome_arr), f'trial type {sort_tt} not present in data (for sorting)'
                tmp_data_sorting_s1 = data_use_mat_norm_s1[:, outcome_arr == sort_tt, :][:, :, post_stim_frame:].mean(1)
                tmp_data_sorting_s2 = data_use_mat_norm_s2[:, outcome_arr == sort_tt, :][:, :, post_stim_frame:].mean(1)

            if data_sorting_s1 is None:  # then also s2 is None, because they are filled at the same time
                data_sorting_s1 = tmp_data_sorting_s1.copy()
                data_sorting_s2 = tmp_data_sorting_s2.copy()
            else:
                assert data_sorting_s1.shape[0] == tmp_data_sorting_s1.shape[0] and data_sorting_s2.shape[0] == tmp_data_sorting_s2.shape[0], 'number of neurons not equal ?? '
                data_sorting_s1 = np.hstack((data_sorting_s1, tmp_data_sorting_s1.copy()))  # concatenate
                data_sorting_s2 = np.hstack((data_sorting_s2, tmp_data_sorting_s2.copy()))

        ## Perform sorting:
        ol_neurons_s1 = sort_data_matrix(data_sorting_s1, sorting_method=sorting_method, session=session, reg='s1') # cluster based on averaged (sort_tt) trials, post stim activity
        ol_neurons_s2 = sort_data_matrix(data_sorting_s2, sorting_method=sorting_method, session=session, reg='s2')

        ## Sort data used for this plot:
        data_use_mat_norm_s1 = data_use_mat_norm_s1[ol_neurons_s1, :, :]
        data_use_mat_norm_s2 = data_use_mat_norm_s2[ol_neurons_s2, :, :]
    else:
        ol_neurons_s1 = None
        ol_neurons_s2 = None

    return (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2,
            data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
            time_ticks, time_tick_labels, time_axis)

def plot_single_raster_plot(data_mat, session, ax=None, cax=None, reg='S1', tt='hit', c_lim=0.2,
                            imshow_interpolation='nearest', plot_cbar=False, print_ylabel=False,
                            sort_tt_list='NA', n_trials=None, time_ticks=[], time_tick_labels=[],
                            s1_lim=None, s2_lim=None, plot_targets=True, spec_target_trial=None,
                            ol_neurons_s1=None, ol_neurons_s2=None, plot_yticks=True, transparent_art=False,
                            plot_xlabel=True, n_stim=None, time_axis=None, filter_150_artefact=True,
                            cbar_pad=1.02, target_tt_specific=True):

    if ax is None:
        ax = plt.subplot(111)

    ## Plot artefact
    if tt in ['hit', 'miss']:
        if time_axis is None:
            print('no time axis given to raster')
            zero_tick = 120
            ax.axvspan(zero_tick-2, zero_tick+30*0.5, alpha=1, color=color_tt['photostim'])
        else:
            # time_axis[np.logical_and(time_axis >= -0.07, time_axis < 0.35)] = np.nan
            start_art_frame = np.argmin(np.abs(time_axis + 0.07))
            if filter_150_artefact:
                end_art_frame = np.argmin(np.abs(time_axis - 0.35))
            else:
                end_art_frame = np.argmin(np.abs(time_axis - 0.83))
            if not transparent_art:
                data_mat = copy.deepcopy(data_mat)
                data_mat[:, start_art_frame:end_art_frame] = np.nan
            ax.axvspan(start_art_frame - 0.25, end_art_frame - 0.25, alpha=0.3, color=color_tt['photostim'])

    ## Plot raster plots
    im = ax.imshow(data_mat, aspect='auto', vmin=-c_lim, vmax=c_lim,
                    cmap='BrBG_r', interpolation=imshow_interpolation)

    if plot_cbar:
        if cax is None:
            print('cax is none')
            cbar = plt.colorbar(im, ax=ax).set_label(r"$\Delta F/F$" + ' activity')# \nnormalised per neuron')
        else:
            ## pretty sure shrink & cbar_pad dont work because cax is already defined.
            cbar = plt.colorbar(im, cax=cax, orientation='vertical', shrink=0.5, pad=cbar_pad)
            cbar.set_label(r"$\Delta F/F$", labelpad=3)
            cbar.set_ticks([])
            cbar.ax.text(0.5, -0.01, '-0.2'.replace("-", u"\u2212"), transform=cbar.ax.transAxes, va='top', ha='center')
            cbar.ax.text(0.5, 1.0, '+0.2', transform=cbar.ax.transAxes, va='bottom', ha='center')       
    
    if print_ylabel:
        ax.set_ylabel(f'Neuron ID sorted by {reg}-{sort_tt_list}\npost-stim trial correlation',
                      fontdict={'weight': 'bold'}, loc=('bottom' if n_stim is not None else 'center'))
    if n_stim is None:
        ax.set_title(f'Trial averaged {tt} {reg} (N={n_trials})')
    else:
        ax.set_title(f'{tt} {reg}, n_stim={n_stim} (N={n_trials})')
    if plot_xlabel:
        ax.set_xlabel(f'Time (s)')
    ax.set_xticks(time_ticks)

    ax.set_xticklabels(time_tick_labels)
    if plot_yticks:
        ax.tick_params(axis='y', left='on', which='major')
        ax.yaxis.set_minor_locator(MultipleLocator(2))
    else:
        ax.set_yticks([])
    # ax.tick_params(axis='y', left='on', which='minor', width=0.5)
    if s1_lim is not None and reg == 'S1':
        ax.set_ylim(s1_lim)
    if s2_lim is not None and reg == 'S2':
        ax.set_ylim(s2_lim)

    ## Target indicator
    if plot_targets and tt in ['hit', 'miss']:
        if reg == 'S1':
            reg_bool = session.s1_bool
        elif reg == 'S2':
            reg_bool = session.s2_bool
        assert len(np.unique(session.is_target.mean((0, 1)))) == 1  # same for all time points
        if filter_150_artefact:  # 150 not included
            target_mat = session.is_target[:, session.photostim < 2, :]
        else:
            target_mat = session.is_target
        if spec_target_trial is None: 
            if target_tt_specific:  # get hit/miss specific targets
                if filter_150_artefact:
                    tt_spec_arr = session.outcome[session.photostim < 2] == tt
                else:
                    tt_spec_arr = session.outcome == tt
                target_mat = target_mat[:, tt_spec_arr, :]
            neuron_targ = np.mean(target_mat, (1, 2))
        else:
            neuron_targ = np.mean(target_mat, 2)
            neuron_targ = neuron_targ[:, spec_target_trial]
        neuron_targ_reg = neuron_targ[reg_bool]  # select region
        if reg == 'S1':
            neuron_targ_reg = neuron_targ_reg[ol_neurons_s1]  # sort
        elif reg == 'S2':
            neuron_targ_reg = neuron_targ_reg[ol_neurons_s2]
        divider = make_axes_locatable(ax)
        targ_ax = divider.append_axes('right', size='6%', pad=0.0)
        targ_ax.imshow(neuron_targ_reg[:, None], cmap='Greys', aspect='auto', interpolation='nearest')
        targ_ax.set_xticks([])
        targ_ax.set_yticks([])
        if s1_lim is not None and reg == 'S1':
            targ_ax.set_ylim(s1_lim)
        if s2_lim is not None and reg == 'S2':
            targ_ax.set_ylim(s2_lim)
    return ax


def plot_raster_plots_trial_types_one_session(session, c_lim=0.2, sort_tt_list=['hit'],
                                              plot_averages=False, post_stim_window=0.35,
                                              start_time=-2.1, filter_150_stim=False,
                                              imshow_interpolation='nearest',  # nearest: true pixel values; bilinear: default anti-aliasing
                                              sorting_method='euclidean',
                                              s1_lim=None, s2_lim=None,
                                              show_plot=True,
                                              save_fig=False, save_name=None,
                                              save_folder='/home/tplas/repos/popping-off/figures/raster_plots/'):

    fig, ax = plt.subplots(2, (6 if plot_averages else 5), figsize=(30, 15), gridspec_kw={'wspace': 0.3, 'width_ratios':([1, 1, 1 ,1, 1, 1.2] if plot_averages else [1, 1, 1, 1, 1.2])})

    (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2, data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
        time_ticks, time_tick_labels, time_axis) = normalise_raster_data(session, start_time=start_time, filter_150_stim=filter_150_stim,
                                        sorting_method=sorting_method, sort_tt_list=sort_tt_list, sort_neurons=True)
    sorted_neurons_dict = {'s1': ol_neurons_s1, 's2': ol_neurons_s2}
    reg_names = ['S1' ,'S2']

    ## plot cell-averaged traces
    if plot_averages:
        for i_x, xx in enumerate(['hit', 'miss', 'fp', 'cr']):
            mean_trace = np.mean(data_use_mat_norm_s1[:, outcome_arr == xx, :], (0, 1))  # S1
            plot_interrupted_trace_simple(ax[0][0], time_axis, smooth_trace(mean_trace),
                                            llabel=xx, llinewidth=3, ccolor=color_tt[xx])  # plot all except spont

            mean_trace = np.mean(data_use_mat_norm_s2[:, outcome_arr == xx, :], (0, 1))  # S2
            plot_interrupted_trace_simple(ax[1][0], time_axis, smooth_trace(mean_trace),
                                            llabel=xx, llinewidth=3, ccolor=color_tt[xx])

        for i_ax, reg_bool in enumerate([session.s1_bool, session.s2_bool]):
            mean_trace = np.mean(data_spont_mat_norm[reg_bool, :, :], (0, 1))  # spontaneous
            plot_interrupted_trace_simple(ax[i_ax][0], time_axis, smooth_trace(mean_trace),
                                            llabel='spont', llinewidth=3, ccolor=color_tt['spont'])  # plot spont

            ax[i_ax][0].legend(frameon=False); ax[i_ax][0].set_title(f'Average over all {reg_names[i_ax]} neurons & trials');
            ax[i_ax][0].set_xlabel('Time (s)'); ax[i_ax][0].set_ylabel(r"$\Delta F/F$")
            ax[i_ax][0].set_ylim([-0.2, 0.25])

    ## Plot raster plots
    ax_st = (1 if plot_averages else 0)
    for i_x, xx in enumerate(['hit', 'fp', 'miss', 'cr']):
        data_mat = np.mean(data_use_mat_norm_s1[:, outcome_arr == xx, :], 1)  # S1
        plot_single_raster_plot(data_mat=data_mat, session=session, ax=ax[0][ax_st + i_x], reg='S1', tt=xx, c_lim=c_lim,
                            imshow_interpolation=imshow_interpolation, plot_cbar=False, print_ylabel=(xx == 'hit'),
                            sort_tt_list=sort_tt_list, n_trials=np.sum(outcome_arr == xx), time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                            s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1,
                            ol_neurons_s2=ol_neurons_s2, time_axis=time_axis, filter_150_artefact=filter_150_stim)

        data_mat = np.mean(data_use_mat_norm_s2[:, outcome_arr == xx, :], 1)  # S2
        plot_single_raster_plot(data_mat=data_mat, session=session, ax=ax[1][ax_st + i_x], reg='S2', tt=xx, c_lim=c_lim,
                    imshow_interpolation=imshow_interpolation, plot_cbar=False, print_ylabel=(xx == 'hit'),
                    sort_tt_list=sort_tt_list, n_trials=np.sum(outcome_arr == xx), time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                    s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1,
                    ol_neurons_s2=ol_neurons_s2, time_axis=time_axis, filter_150_artefact=filter_150_stim)

    data_mat = np.mean(data_spont_mat_norm[session.s1_bool, :, :], 1)  # Spont S1
    plot_single_raster_plot(data_mat=data_mat[ol_neurons_s1, :], session=session, ax=ax[0][ax_st + 4], reg='S1', tt='spont', c_lim=c_lim,
                    imshow_interpolation=imshow_interpolation, plot_cbar=True, print_ylabel=False,
                    sort_tt_list=sort_tt_list, n_trials=data_spont_mat_norm.shape[1], time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                    s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1,
                    ol_neurons_s2=ol_neurons_s2, time_axis=time_axis, filter_150_artefact=filter_150_stim)

    data_mat = np.mean(data_spont_mat_norm[session.s2_bool, :, :], 1)  # Spont S2
    plot_single_raster_plot(data_mat=data_mat[ol_neurons_s2, :], session=session, ax=ax[1][ax_st + 4], reg='S2', tt='spont', c_lim=c_lim,
                    imshow_interpolation=imshow_interpolation, plot_cbar=True, print_ylabel=False,
                    sort_tt_list=sort_tt_list, n_trials=data_spont_mat_norm.shape[1], time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                    s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1,
                    ol_neurons_s2=ol_neurons_s2, time_axis=time_axis, filter_150_artefact=filter_150_stim)

    ax[0][2].annotate(s=f'{str(session)}, sorted by {sorting_method} using {imshow_interpolation} interpolation',
                      xy=(0.8, 1.1), xycoords='axes fraction', weight= 'bold', fontsize=14)

    ## save & return
    if save_fig:
        if save_name is None:
            save_name = f'Rasters_{session.signature}_{imshow_interpolation}.pdf'
        plt.savefig(os.path.join(save_folder, save_name), bbox_inches='tight')

    if show_plot is False:
        plt.close()
    return sorted_neurons_dict



def plot_raster_plots_number_stim_one_session(session, c_lim=0.2, sort_tt_list=['hit', 'miss', 'spont'],
                                              plot_averages=False, stim_window=0.35,
                                              start_time=-4,
                                              imshow_interpolation='nearest',  # nearest: true pixel values; bilinear: default anti-aliasing
                                              sorting_method='euclidean',
                                              s1_lim=None, s2_lim=None,
                                              show_plot=True,
                                              save_fig=False, save_name=None,
                                              save_folder='/home/tplas/repos/popping-off/figures/raster_plots/'):
    arr_n_stim = np.array([0, 5, 10, 20, 30, 40, 50, 150])
    assert (np.unique(session.trial_subsets) == arr_n_stim).all()
    arr_n_stim = arr_n_stim[arr_n_stim > 0]
    fig, ax = plt.subplots(4, (len(arr_n_stim) + 1 if plot_averages else len(arr_n_stim)), figsize=(30, 15),
                            gridspec_kw={'wspace': 0.3, 'width_ratios':([1, 1, 1, 1, 1, 1, 1, 1.2] if plot_averages else [1, 1, 1, 1, 1, 1, 1.2])})

    (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2, data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
        time_ticks, time_tick_labels, time_axis) = normalise_raster_data(session, start_time=start_time, stim_window=stim_window, sorting_method=sorting_method,
                                                                           sort_tt_list=sort_tt_list, sort_neurons=True, filter_150_stim=False)
    sorted_neurons_dict = {'s1': ol_neurons_s1, 's2': ol_neurons_s2}
    reg_names = ['S1' ,'S2']

    ## plot cell-averaged traces
    if plot_averages:
        # for i_x, xx in enumerate(['hit', 'miss', 'fp', 'cr']):
        #     mean_trace = np.mean(data_use_mat_norm_s1[:, outcome_arr == xx, :], (0, 1))  # S1
        #     plot_interrupted_trace_simple(ax[0][0], time_axis, smooth_trace(mean_trace),
        #                                     llabel=xx, llinewidth=3, ccolor=color_tt[xx])  # plot all except spont

        #     mean_trace = np.mean(data_use_mat_norm_s2[:, outcome_arr == xx, :], (0, 1))  # S2
        #     plot_interrupted_trace_simple(ax[1][0], time_axis, smooth_trace(mean_trace),
        #                                     llabel=xx, llinewidth=3, ccolor=color_tt[xx])

        # for i_ax, reg_bool in enumerate([session.s1_bool, session.s2_bool]):
        #     mean_trace = np.mean(data_spont_mat_norm[reg_bool, :, :], (0, 1))  # spontaneous
        #     plot_interrupted_trace_simple(ax[i_ax][0], time_axis, smooth_trace(mean_trace),
        #                                     llabel='spont', llinewidth=3, ccolor=color_tt['spont'])  # plot spont

        #     ax[i_ax][0].legend(frameon=False); ax[i_ax][0].set_title(f'Average over all {reg_names[i_ax]} neurons & trials');
        #     ax[i_ax][0].set_xlabel('Time (s)'); ax[i_ax][0].set_ylabel('DF/F')
        #     ax[i_ax][0].set_ylim([-0.2, 0.25])
        assert False, 'average traces not iplemented yet'

    ## Plot raster plots
    ax_st = (1 if plot_averages else 0)
    assert data_use_mat_norm_s1.shape[1] == len(session.trial_subsets)
    assert data_use_mat_norm_s2.shape[1] == len(session.trial_subsets)
    for i_x, xx in enumerate(['hit', 'miss']):
        for i_stim, n_stim in enumerate(arr_n_stim):
            trial_selection = np.logical_and(session.trial_subsets == n_stim, outcome_arr == xx)
            data_mat = np.mean(data_use_mat_norm_s1[:, trial_selection, :], 1)  # S1
            plot_single_raster_plot(data_mat=data_mat, session=session, ax=ax[i_x][ax_st + i_stim], reg='S1', tt=xx, c_lim=c_lim,
                                imshow_interpolation=imshow_interpolation, plot_cbar=(True if i_stim == (len(arr_n_stim) - 1) else False), print_ylabel=(i_stim == 0 and i_x == 1),
                                sort_tt_list=sort_tt_list, n_trials=np.sum(trial_selection), time_ticks=[], time_tick_labels=[],
                                s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1, time_axis=time_axis,
                                ol_neurons_s2=ol_neurons_s2, plot_xlabel=False, plot_yticks=(True if i_stim ==0 else False), n_stim=n_stim, filter_150_artefact=filter_150_stim)

            data_mat = np.mean(data_use_mat_norm_s2[:, trial_selection, :], 1)  # S2
            plot_single_raster_plot(data_mat=data_mat, session=session, ax=ax[i_x + 2][ax_st + i_stim], reg='S2', tt=xx, c_lim=c_lim,
                        imshow_interpolation=imshow_interpolation, plot_cbar=(True if i_stim == (len(arr_n_stim) - 1) else False), print_ylabel=(i_stim == 0 and i_x == 1),
                        sort_tt_list=sort_tt_list, n_trials=np.sum(trial_selection), time_ticks=(time_ticks if i_x == 1 else []),
                        time_tick_labels=(time_tick_labels if i_x == 1 else []), time_axis=time_axis,
                        s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1, filter_150_artefact=filter_150_stim,
                        ol_neurons_s2=ol_neurons_s2, plot_xlabel=(False if i_x == 0 else True), plot_yticks=(True if i_stim == 0 else False), n_stim=n_stim)

    ax[0][1].annotate(s=f'{str(session)}, hit & Miss trials split by number of cells targeted. Sorted by {sorting_method} using {imshow_interpolation} interpolation',
                      xy=(0.4, 1.15), xycoords='axes fraction', weight= 'bold', fontsize=14)

    ## save & return
    if save_fig:
        if save_name is None:
            save_name = f'Rasters_{session.signature}_n-stim_{imshow_interpolation}.pdf'
        plt.savefig(os.path.join(save_folder, save_name), bbox_inches='tight')

    if show_plot is False:
        plt.close()
    return sorted_neurons_dict



def plot_raster_plots_all_trials_one_session(session,  tt_plot='hit', c_lim=0.2, sort_tt_list=['hit'],
                                              stim_window=0.35, start_time=-4,
                                              n_cols=6, reg='S1',
                                              imshow_interpolation='nearest',  # nearest: true pixel values; bilinear: default anti-aliasing
                                              sorting_method='euclidean',
                                              s1_lim=None, s2_lim=None,
                                              show_plot=True,
                                              save_fig=False, save_name=None,
                                              save_folder='/home/tplas/repos/popping-off/figures/raster_plots/individual_trials/'):

    (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2, data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
        time_ticks, time_tick_labels, time_axis) = normalise_raster_data(session, start_time=start_time, stim_window=stim_window, sorting_method=sorting_method, sort_tt_list=sort_tt_list, sort_neurons=True)

    if tt_plot == 'spont':
        abs_trial_arr = []
        n_trials = data_spont_mat_norm.shape[1]
    else: # use session.outcome b/c outcome_arr does not include photostim == 2 and therefore not full length
        abs_trial_arr = np.where(outcome_arr == tt_plot)[0]
        abs_trial_arr = np.where(session.photostim < 2)[0][abs_trial_arr]
        assert len(np.unique(session.outcome[abs_trial_arr])) == 1 and session.outcome[abs_trial_arr][0] == tt_plot
        n_trials = np.sum(outcome_arr == tt_plot)

    n_rows = int(np.ceil(n_trials / n_cols)) + 1  # plus 1 for averages
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 7 * n_rows), gridspec_kw={'wspace': 0.2})

    ## Plot average raster plot
    if reg == 'S1':
        if tt_plot == 'spont':
            data_mat = np.mean(data_spont_mat_norm[session.s1_bool, :, :], 1)  # Spont S2
            data_mat = data_mat[ol_neurons_s1, :]
            tmp_n_trials = data_spont_mat_norm.shape[1]
        else:
            tmp_n_trials = np.sum(outcome_arr == tt_plot)
            data_mat = np.mean(data_use_mat_norm_s1[:, outcome_arr == tt_plot, :], 1)  # S1

        plot_single_raster_plot(data_mat=data_mat, session=session, ax=ax[0][0], reg='S1', tt=tt_plot, c_lim=c_lim,
                            imshow_interpolation=imshow_interpolation, plot_cbar=True, print_ylabel=True,
                            sort_tt_list=sort_tt_list, n_trials=tmp_n_trials, time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                            s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1, time_axis=time_axis,
                            ol_neurons_s2=ol_neurons_s2, filter_150_artefact=filter_150_stim)
    elif reg == 'S2':

        if tt_plot == 'spont':
            data_mat = np.mean(data_spont_mat_norm[session.s2_bool, :, :], 1)  # Spont S2
            data_mat = data_mat[ol_neurons_s2, :]
            tmp_n_trials = data_spont_mat_norm.shape[1]
        else:
            tmp_n_trials = np.sum(outcome_arr == tt_plot)
            data_mat = np.mean(data_use_mat_norm_s2[:, outcome_arr == tt_plot, :], 1)  # S2

        plot_single_raster_plot(data_mat=data_mat, session=session, ax=ax[0][0], reg='S2', tt=tt_plot, c_lim=c_lim,
                    imshow_interpolation=imshow_interpolation, plot_cbar=True, print_ylabel=True,
                    sort_tt_list=sort_tt_list, n_trials=tmp_n_trials, time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                    s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1, time_axis=time_axis,
                    ol_neurons_s2=ol_neurons_s2, filter_150_artefact=filter_150_stim)

    for i_col in range(1, n_cols):
        ax[0][i_col].set_xticks([])
        ax[0][i_col].set_yticks([])
        ax[0][i_col].axis('off')

    ## Plot individual trial raster plots
    for i_trial in range(n_trials):
        i_col = i_trial % n_cols
        i_row = int(np.ceil(i_trial / n_cols + 0.0001))
        if tt_plot != 'spont':
            abs_trial_n = abs_trial_arr[i_trial]
        else:
            abs_trial_n = i_trial
        curr_ax = ax[i_row][i_col]
        if reg == 'S1':
            if tt_plot == 'spont':
                data_mat = data_spont_mat_norm[session.s1_bool, :, :][:, i_trial, :]  # Spont S1
                plot_single_raster_plot(data_mat=data_mat[ol_neurons_s1, :], session=session, ax=curr_ax, reg='S1', tt='spont', c_lim=c_lim,
                        imshow_interpolation=imshow_interpolation, plot_cbar=False, print_ylabel=(i_col == 0),
                        sort_tt_list=sort_tt_list, n_trials=data_spont_mat_norm.shape[1], time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                        s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=False, ol_neurons_s1=ol_neurons_s1, time_axis=time_axis,
                        ol_neurons_s2=ol_neurons_s2, filter_150_artefact=filter_150_stim)

            else:
                data_mat = data_use_mat_norm_s1[:, outcome_arr == tt_plot, :][:, i_trial, :]  # S1
                plot_single_raster_plot(data_mat=data_mat, session=session, ax=curr_ax, reg='S1', tt=tt_plot, c_lim=c_lim,
                                    imshow_interpolation=imshow_interpolation, plot_cbar=False, print_ylabel=(i_col == 0),
                                    sort_tt_list=sort_tt_list, n_trials=np.sum(outcome_arr == tt_plot), time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                                    s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, spec_target_trial=abs_trial_n, ol_neurons_s1=ol_neurons_s1,
                                    ol_neurons_s2=ol_neurons_s2, time_axis=time_axis, filter_150_artefact=filter_150_stim)

        elif reg == 'S2':
            if tt_plot == 'spont':
                data_mat = data_spont_mat_norm[session.s2_bool, :, :][:, i_trial, :]  # Spont S2
                plot_single_raster_plot(data_mat=data_mat[ol_neurons_s2, :], session=session, ax=curr_ax, reg='S2', tt='spont', c_lim=c_lim,
                                            imshow_interpolation=imshow_interpolation, plot_cbar=False, print_ylabel=(i_col == 0),
                                            sort_tt_list=sort_tt_list, n_trials=data_spont_mat_norm.shape[1], time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                                            s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=False, ol_neurons_s1=ol_neurons_s1,
                                            ol_neurons_s2=ol_neurons_s2, time_axis=time_axis, filter_150_artefact=filter_150_stim)
            else:
                data_mat = data_use_mat_norm_s2[:, outcome_arr == tt_plot, :][:, i_trial, :]  # S2
                plot_single_raster_plot(data_mat=data_mat, session=session, ax=curr_ax, reg='S2', tt=tt_plot, c_lim=c_lim,
                            imshow_interpolation=imshow_interpolation, plot_cbar=False, print_ylabel=(i_col == 0),
                            sort_tt_list=sort_tt_list, n_trials=np.sum(outcome_arr == tt_plot), time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                            s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, spec_target_trial=abs_trial_n, ol_neurons_s1=ol_neurons_s1,
                            ol_neurons_s2=ol_neurons_s2, time_axis=time_axis, filter_150_artefact=filter_150_stim)
        curr_ax.set_title(f'Trial #{i_trial, abs_trial_n}')


    if i_col != n_cols - 1:
        for ii_col in range(i_col + 1, n_cols):
            ax[-1][ii_col].set_xticks([])
            ax[-1][ii_col].set_yticks([])
            ax[-1][ii_col].axis('off')


    ax[0][2].annotate(s=f'{str(session)}, sorted by {sorting_method}\nAll {tt_plot} trials are shown with {imshow_interpolation} interpolation', xy=(0.8, 0.5), xycoords='axes fraction', weight= 'bold', fontsize=14)

    ## save & return
    if save_fig:
        if save_name is None:
            save_name = f'Indiv_rasters_{session.signature}_{reg}_{tt_plot}.pdf'
        plt.savefig(os.path.join(save_folder, save_name), bbox_inches='tight')

    if show_plot is False:
        plt.close()


def plot_raster_plots_input_trial_types_one_session(session, ax_dict={'s1': {}, 's2': {}}, c_lim=0.2, sort_tt_list=['hit'],
                                              plot_averages=False, post_stim_window=0.35, cax=None, bool_cb=True,
                                              start_time=-1.1, end_time=2, filter_150_stim=False,
                                              imshow_interpolation='nearest',  # nearest: true pixel values; bilinear: default anti-aliasing
                                              sorting_method='euclidean', cbar_pad=1.02,
                                              s1_lim=None, s2_lim=None):

    (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2, data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
        time_ticks, time_tick_labels, time_axis) = normalise_raster_data(session, start_time=start_time, filter_150_stim=filter_150_stim,
                                        sorting_method=sorting_method, sort_tt_list=sort_tt_list, sort_neurons=True, end_time=end_time)
    sorted_neurons_dict = {'s1': ol_neurons_s1, 's2': ol_neurons_s2}
    reg_names = ['S1' ,'S2']
    assert (time_ticks == [0, 60]).all() and time_tick_labels == ['-1.0', '1.0'], 'hard-coded time tick labels will be incorrect (lines below)'
    time_ticks = [0, 30, 60, 90]
    time_tick_labels = [x.replace("-", u"\u2212") for x in ['-1', '0', '1', '2']]
    
    # return data_use_mat_norm_s1, outcome_arr

    ## Plot raster plots
    for reg, tt_dict in ax_dict.items():
        for xx, ax in tt_dict.items():
            if xx != 'spont':
                if reg == 's1':
                    data_mat = np.mean(data_use_mat_norm_s1[:, outcome_arr == xx, :], 1)  # S1
                    plot_single_raster_plot(data_mat=data_mat, session=session, ax=ax, reg='S1', tt=xx, c_lim=c_lim,
                                        imshow_interpolation=imshow_interpolation, plot_cbar=False, print_ylabel=(xx == 'hit'),
                                        sort_tt_list=sort_tt_list, n_trials=np.sum(outcome_arr == xx), time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                                        s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1,
                                        ol_neurons_s2=ol_neurons_s2, time_axis=time_axis, filter_150_artefact=filter_150_stim)
                elif reg == 's2':
                    data_mat = np.mean(data_use_mat_norm_s2[:, outcome_arr == xx, :], 1)  # S2
                    plot_single_raster_plot(data_mat=data_mat, session=session, ax=ax, reg='S2', tt=xx, c_lim=c_lim,
                                imshow_interpolation=imshow_interpolation, plot_cbar=False, print_ylabel=(xx == 'hit'),
                                sort_tt_list=sort_tt_list, n_trials=np.sum(outcome_arr == xx), time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                                s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1,
                                ol_neurons_s2=ol_neurons_s2, time_axis=time_axis, filter_150_artefact=filter_150_stim)
            else:
                if reg == 's1':
                    data_mat = np.mean(data_spont_mat_norm[session.s1_bool, :, :], 1)  # Spont S1
                    plot_single_raster_plot(data_mat=data_mat[ol_neurons_s1, :], session=session, ax=ax, reg='S1', tt='spont', c_lim=c_lim,
                                    imshow_interpolation=imshow_interpolation, plot_cbar=bool_cb, cbar_pad=cbar_pad, cax=cax, print_ylabel=False,
                                    sort_tt_list=sort_tt_list, n_trials=data_spont_mat_norm.shape[1], time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                                    s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1,
                                    ol_neurons_s2=ol_neurons_s2, time_axis=time_axis, filter_150_artefact=filter_150_stim)
                elif reg == 's2':
                    data_mat = np.mean(data_spont_mat_norm[session.s2_bool, :, :], 1)  # Spont S2
                    plot_single_raster_plot(data_mat=data_mat[ol_neurons_s2, :], session=session, ax=ax, reg='S2', tt='spont', c_lim=c_lim,
                                    imshow_interpolation=imshow_interpolation, plot_cbar=False,print_ylabel=False,
                                    sort_tt_list=sort_tt_list, n_trials=data_spont_mat_norm.shape[1], time_ticks=time_ticks, time_tick_labels=time_tick_labels,
                                    s1_lim=s1_lim, s2_lim=s2_lim, plot_targets=True, ol_neurons_s1=ol_neurons_s1,
                                    ol_neurons_s2=ol_neurons_s2, time_axis=time_axis, filter_150_artefact=filter_150_stim)
            ax.set_ylabel(f'Sorted {reg.upper()} neurons', fontdict={'weight': 'normal'})
            ax.set_title(f'{label_tt[xx]} {reg.upper()}', fontdict={'color': color_tt[xx]})

            tmp_n_neurons = np.shape(data_mat)[0]
            if tmp_n_neurons > 400:
                multiplier = 100
            elif tmp_n_neurons > 50:
                multiplier = 50
            elif tmp_n_neurons > 25:
                multiplier = 25
            else:
                multiplier = 10
                
            if reg == 's1':
                ax.set_yticks(np.arange(int(np.ceil(np.sum(session.s1_bool) / multiplier))) * multiplier)
            elif reg == 's2':
                ax.set_yticks(np.arange(int(np.ceil(np.sum(session.s2_bool) / multiplier))) * multiplier)

            ax.tick_params(which='minor', left=False)
            if xx != 'hit':  ## assuming hit is the first one 
                assert list(tt_dict.keys())[0] == 'hit'
                ax.tick_params(labelleft=False)
            if reg != 's2': ## assuming s2 is the last 
                assert list(ax_dict.keys())[-1] == 's2'
                ax.tick_params(labelbottom=False)


        # ax[0][2].annotate(s=f'{str(session)}, sorted by {sorting_method} using {imshow_interpolation} interpolation',
        #                 xy=(0.8, 1.1), xycoords='axes fraction', weight= 'bold', fontsize=14)

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
                                             tuple_list_tt=[('hit', 'miss'), ('fp', 'cr'), ('urh', 'arm')]):
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
    df_prediction_train, df_prediction_test, _, __ = pof.train_test_all_sessions(sessions=ss_dict, verbose=1, n_split=n_splits,
                                                trial_times_use=np.array([time_frame]), return_decoder_weights=True,
                                            hitmiss_only=False,# list_test=['dec', 'stim'],
                                            include_autoreward=False, neurons_selection='s1',
                                            C_value=50, train_projected=False, proj_dir='same')
    fig, ax = plt.subplots(2,2, figsize=(7, 6), gridspec_kw={'wspace': 0.3, 'hspace': 0.4})

    ## NB: To not plot with hue, set hh=None
    # for mouse in df_prediction_train.keys():
    for mouse in [ss_dict[0].mouse]:
        if 'pred_stim_train' in df_prediction_train[mouse].columns:
            plot_df_stats(df=df_prediction_train[mouse], xx='true_stim_train', yy='pred_stim_train', # _proj
                              hh='true_dec_train', ax=ax[0][0])  # set hh=None or hh='dec_train'
            ax[0][0].set_xlabel('Number of cells PS'); ax[0][0].set_ylabel('Predicted probability of PS');
            ax[0][0].set_title('TRAIN - PS', weight='bold')
            ax[0][0].legend([], frameon=False)

            plot_df_stats(df=df_prediction_test[mouse], xx='true_stim_test', yy='pred_stim_test',
                              hh='true_dec_test', ax=ax[0][1])  # set hh=None or hh='dec_test'
            ax[0][1].set_xlabel('Number of cells PS'); ax[0][1].set_ylabel('Predicted probability of PS');
            ax[0][1].set_title('TEST - PS', weight='bold')
            ax[0][1].legend([], frameon=False)
    #         print(f'Accuracy PS: {pof.class_av_mean_accuracy(binary_truth=(df_prediction_test[mouse]["true_stim_test"] > 0).astype("int"), estimate=df_prediction_test[mouse]["pred_stim_test_proj"])[0]}')

        if 'pred_dec_train' in df_prediction_train[mouse].columns:
            plot_df_stats(df=df_prediction_train[mouse], xx='true_dec_train', yy='pred_dec_train',
                              hh=None, xticklabels=['no lick', 'lick'], ax=ax[1][0])
            ax[1][0].set_xlabel('Decision'); ax[1][0].set_ylabel('Predicted probability of lick');
            ax[1][0].set_title('TRAIN - LICK', weight='bold')
            ax[1][0].legend('No lick', 'lick', frameon=False)

            plot_df_stats(df=df_prediction_test[mouse], xx='true_dec_test', yy='pred_dec_test',
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
                                smooth_traces=False, one_sided_window_size=1,
                                plot_indiv=False, plot_std_area=True, plot_mean=True):
    if ax is None:
        ax = plt.subplot(111)

    if np.sum(np.isnan(time_array)) > 0:
        arr_nan = np.where(np.isnan(time_array))[0]
        assert len(np.unique(np.diff(arr_nan))) == 1 and np.diff(arr_nan)[0] == 1
        time_breakpoint = arr_nan[0]
        time_breakpoint_postnan = arr_nan[-1]
    else:
        time_breakpoint = 1
        time_breakpoint_postnan = None

    plot_interrupted_trace_simple(ax=ax, time_array=time_array,
                                    plot_array=np.zeros_like(time_array) + 0.5,
                                    ccolor='k', aalpha=0.6, llinewidth=3, linest=':')
    for i_lick, dict_part in ps_acc_split.items():  # PS accuracy split per lick /no lick trials
        # print(i_lick)
        plot_interrupted_trace_average_per_mouse(ax=ax, time_array=time_array, plot_array=dict_part, llabel=label_split[i_lick],
                            ccolor=colors_plot[reg][i_lick], plot_indiv=plot_indiv, plot_groupav=plot_mean,
                             plot_laser=False, #i_lick,
                            plot_errorbar=False, plot_std_area=plot_std_area, region_list=[reg],
                            running_average_smooth=smooth_traces, one_sided_window_size=one_sided_window_size,
                            time_breakpoint=time_breakpoint, time_breakpoint_postnan=time_breakpoint_postnan)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Accuracy')
    if plot_indiv is False:
        ax.legend(loc='upper left', frameon=False);
    ax.set_title(f'Dynamic PS decoding in {reg.upper()}', weight='bold')
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
                                      smooth_traces=True, one_sided_window_size=1, ax_acc_ps=None,
                                      plot_std_area=True, plot_indiv=False, title_lick_dec=False, plot_legend=True,
                                      fn_suffix='',bottom_yax_tt='CR', top_yax_tt='Hit', xlims=[-3, 4], plot_mean=True):

    if ax_acc_ps is None:
        fig = plt.figure(constrained_layout=False, figsize=(12, 4))
        gs_top = fig.add_gridspec(ncols=2, nrows=1, wspace=0.3,
                                bottom=0.15, top=0.9, left=0.10, right=0.9, hspace=0.4)
        ax_acc_ps = {reg: fig.add_subplot(gs_top[i_reg]) for i_reg, reg in enumerate(['s1', 's2'])}
    for i_reg, reg in enumerate(['s1', 's2']):
        _ = plot_dynamic_decoding_panel(time_array=time_array, ps_acc_split=ps_acc_split,
                                    reg=reg, ax=ax_acc_ps[reg], smooth_traces=smooth_traces,
                                    one_sided_window_size=one_sided_window_size, plot_indiv=plot_indiv,
                                    plot_std_area=plot_std_area, plot_mean=plot_mean)

        if yaxis_type == 'accuracy':
            ax_acc_ps[reg].set_ylabel('Prediction accuracy')
        elif yaxis_type == 'prediction':
            ax_acc_ps[reg].set_ylabel('Network prediction P(PS)')
        else:
            print('WARNING: yaxis_type not recognised')

        ax_acc_ps[reg].set_ylim([0, 1])
        ax_acc_ps[reg].tick_params(reset=True, top=False, right=False)
        despine(ax_acc_ps[reg])
        ax_acc_ps[reg].set_xlim([-4, 6])
        ax_acc_ps[reg].set_xticks(np.arange(10) - 3)
        if xlims == [-3, 4]:
            ax_acc_ps[reg].arrow(-2.15, 0.52, 0, 0.4, head_width=0.3, head_length=0.05, linewidth=3,
                            color=color_tt[top_yax_tt.lower()], length_includes_head=True, clip_on=False)
            ax_acc_ps[reg].arrow(-2.15, 0.48, 0, -0.4, head_width=0.3, head_length=0.05, linewidth=3,
                            color=color_tt[bottom_yax_tt.lower()], length_includes_head=True, clip_on=False)
            ax_acc_ps[reg].text(s=top_yax_tt, x=-2.75, y=0.73, rotation=90,
                            fontdict={'weight': 'bold', 'va': 'center', 'color': color_tt[top_yax_tt.lower()]})
            ax_acc_ps[reg].text(s=bottom_yax_tt, x=-2.75, y=0.33, rotation=90,
                            fontdict={'weight': 'bold', 'va': 'center', 'color': color_tt[bottom_yax_tt.lower()]})
        elif xlims == [-2, 2]:
            ax_acc_ps[reg].arrow(-1.65, 0.52, 0, 0.4, head_width=0.15, head_length=0.05, linewidth=3,
                            color=color_tt[top_yax_tt.lower()], length_includes_head=True, clip_on=False)
            ax_acc_ps[reg].arrow(-1.65, 0.48, 0, -0.4, head_width=0.15, head_length=0.05, linewidth=3,
                            color=color_tt[bottom_yax_tt.lower()], length_includes_head=True, clip_on=False)
            ax_acc_ps[reg].text(s=top_yax_tt, x=-1.75, y=0.73, rotation=90,
                            fontdict={'weight': 'bold', 'va': 'center', 'color': color_tt[top_yax_tt.lower()]})
            ax_acc_ps[reg].text(s=bottom_yax_tt, x=-1.75, y=0.33, rotation=90,
                            fontdict={'weight': 'bold', 'va': 'center', 'color': color_tt[bottom_yax_tt.lower()]})
        if title_lick_dec:
            ax_acc_ps[reg].set_title(f'Dynamic lick decoding in {reg.upper()}', fontdict={'weight': 'bold'})
        if top_yax_tt == 'Rew. only':  # too long for 1 line
            ax_acc_ps[reg].set_ylabel(f'{top_yax_tt} vs {bottom_yax_tt}\nclassification')
        else:
            ax_acc_ps[reg].set_ylabel(f'{top_yax_tt} vs {bottom_yax_tt} classification')


    if plot_legend:
        ax_acc_ps['s1'].legend(loc='upper left', bbox_to_anchor=(1.1, 0.98), 
                                frameon=False)
    else:
        if ax_acc_ps['s1'].get_legend() is not None:
            ax_acc_ps['s1'].get_legend().remove()
    if ax_acc_ps['s2'].get_legend() is not None:
        ax_acc_ps['s2'].get_legend().remove()
    if save_fig:
        fn = f'dyn-dec_{int(len(ps_acc_split["hit"]) / 2)}-mice_{fn_suffix}'
        plt.savefig(f'/home/tplas/repos/popping-off/figures/dyn_decoding/{fn}.pdf', bbox_to_inches='tight')

    return ax_acc_ps

def plot_dynamic_decoding_two_regions_wrapper(ps_pred_split, lick_pred_split, decoder_key='hit/cr',
                                              plot_tt=['hit', 'spont', 'miss', 'fp', 'cr'],
                                              ax_acc_ps=None, time_array=None, smooth_traces=False,
                                              one_sided_window_size=2, plot_indiv=False, plot_legend=True,
                                              indicate_spont=False, indicate_fp=False, xlims=[-3, 4],
                                              plot_ci=True, plot_mean=True,
                                              plot_artefact=True, plot_significance=True, bottom_sign_bar=1):
    ## Plot:
    if decoder_key == 'spont/cr':
        plot_dict_split = {x: lick_pred_split[decoder_key][x] for x in plot_tt} # separated by lick condition
        top_yax_tt = 'Rew. only'
        bottom_yax_tt = 'CR'
    elif decoder_key == 'hit/cr':
        plot_dict_split = {x: ps_pred_split[decoder_key][x] for x in plot_tt}   # separated by ps condition
        top_yax_tt = 'Hit'
        bottom_yax_tt = 'CR'
    elif decoder_key == 'hit/cr 10 trials':
        plot_dict_split = {x: ps_pred_split[decoder_key][x] for x in plot_tt}   # separated by ps condition
        top_yax_tt = 'Hit'
        bottom_yax_tt = 'CR'
    elif decoder_key == 'miss/cr':
        plot_dict_split = {x: ps_pred_split[decoder_key][x] for x in plot_tt}
        top_yax_tt = 'Miss'
        bottom_yax_tt = 'CR'
    elif decoder_key == 'hit/miss':
        plot_dict_split = {x: lick_pred_split[decoder_key][x] for x in plot_tt}   # separated by ps condition
        top_yax_tt = 'Hit'
        bottom_yax_tt = 'Miss'

    plot_dynamic_decoding_two_regions(ps_acc_split=plot_dict_split,
                                        time_array=time_array,
                                        yaxis_type='prediction',
                                        smooth_traces=smooth_traces,
                                        ax_acc_ps=ax_acc_ps,
                                        one_sided_window_size=one_sided_window_size,
                                        save_fig=False,
                                        fn_suffix='subsampled_SpontCr_10-sessions_ws2',
                                        top_yax_tt=top_yax_tt,
                                        bottom_yax_tt=bottom_yax_tt,
                                        plot_indiv=plot_indiv,
                                        plot_legend=plot_legend,
                                        plot_std_area=plot_ci,
                                        xlims=xlims,
                                        plot_mean=plot_mean)

    if ax_acc_ps is not None:
        for reg in ['s1', 's2']:
            if plot_artefact:
                add_ps_artefact(ax_acc_ps[reg], time_axis=time_array)
            ax_acc_ps[reg].set_xlim(xlims)
            names_tt = [label_tt[x] for x in decoder_key.split("/")]
            ax_acc_ps[reg].set_title(f'Dynamic {names_tt[0]}/{names_tt[1]} encoding in {reg.upper()}', 
                                     fontdict={'weight': 'bold'}, y=1.15)
            ax_acc_ps[reg].set_xticks([-2, -1, 0, 1, 2, 3, 4])
            ax_acc_ps[reg].set_yticks([0, 0.5, 1])
            
            if plot_significance:
                for i_tt, tt in enumerate(plot_tt):
                    _, signif_arr = pof.stat_test_dyn_dec(pred_dict=plot_dict_split, decoder_name='NA',
                                                        time_array=time_array, tt=tt, region=reg)
                    ax_acc_ps[reg].plot(time_array, [bottom_sign_bar + (i_tt  *0.03) if x == 1 else np.nan for x in signif_arr],
                                    linewidth=2, c=color_tt[tt], clip_on=False) 
            if decoder_key == 'hit/cr 10 trials':
                ax_acc_ps[reg].set_ylabel('Hit vs CR classification\nusing 10 trials only')

        if indicate_spont:
            ax_acc_ps['s1'].text(s='Reward only', x=4, y=0.3,
                                fontdict={'weight': 'bold', 'color': color_tt['spont'], 'ha': 'right'})
        if indicate_fp:
            ax_acc_ps['s1'].text(s='FP', x=1.4, y=0.62,
                                fontdict={'weight': 'bold', 'color': color_tt['fp']})
       

def plot_dynamic_decoding_two_regions_wrapper_split(ps_pred_split, lick_pred_split, decoder_key='hit/cr',
                                            #   plot_tt=['hit_n1', 'hit_n2', 'hit_n3', 
                                            #            'miss_n1', 'miss_n2', 'miss_n3'],
                                              plot_tt=['hit_c1', 'hit_c2', 'hit_c3', 
                                                       'miss_c1', 'miss_c2', 'miss_c3'],
                                              name_cov='variance_cell_rates_s1',
                                              ax_acc_ps=None, time_array=None, smooth_traces=False,
                                              one_sided_window_size=2, plot_indiv=False, plot_legend=True,
                                              indicate_spont=False, indicate_fp=False, xlims=[-3, 4],
                                              plot_ci=True, plot_mean=True,
                                              plot_artefact=True, plot_significance=True, bottom_sign_bar=1):
    ## Plot:
    if decoder_key == 'spont/cr':
        tmp_dict = lick_pred_split[decoder_key]
        top_yax_tt = 'Rew. only'
        bottom_yax_tt = 'CR'
    elif decoder_key == 'hit/cr':
        tmp_dict = ps_pred_split[decoder_key]   # separated by ps condition
        top_yax_tt = 'Hit'
        bottom_yax_tt = 'CR'
    elif decoder_key == 'hit/cr 10 trials':
        tmp_dict = ps_pred_split[decoder_key]   # separated by ps condition
        top_yax_tt = 'Hit'
        bottom_yax_tt = 'CR'
    elif decoder_key == 'miss/cr':
        tmp_dict = ps_pred_split[decoder_key]
        top_yax_tt = 'Miss'
        bottom_yax_tt = 'CR'
    elif decoder_key == 'hit/miss':
        tmp_dict = lick_pred_split[decoder_key]   # separated by ps condition
        top_yax_tt = 'Hit'
        bottom_yax_tt = 'Miss'
    if name_cov is not None:
        tmp_dict = tmp_dict[name_cov]
    plot_dict_split = {x: tmp_dict[x] for x in plot_tt} # separated by lick condition
       
    plot_dynamic_decoding_two_regions(ps_acc_split=plot_dict_split,
                                        time_array=time_array,
                                        yaxis_type='prediction',
                                        smooth_traces=smooth_traces,
                                        ax_acc_ps=ax_acc_ps,
                                        one_sided_window_size=one_sided_window_size,
                                        save_fig=False,
                                        fn_suffix='subsampled_SpontCr_10-sessions_ws2',
                                        top_yax_tt=top_yax_tt,
                                        bottom_yax_tt=bottom_yax_tt,
                                        plot_indiv=plot_indiv,
                                        plot_legend=plot_legend,
                                        plot_std_area=plot_ci,
                                        xlims=xlims,
                                        plot_mean=plot_mean)

    if ax_acc_ps is not None:
        for reg in ['s1', 's2']:
            if plot_artefact:
                add_ps_artefact(ax_acc_ps[reg], time_axis=time_array)
            ax_acc_ps[reg].set_xlim(xlims)
            ax_acc_ps[reg].set_title(f'Dynamic {decoder_key} encoding in {reg.upper()}', 
                                     fontdict={'weight': 'bold'}, y=1.05)
            ax_acc_ps[reg].set_xticks([-2, -1, 0, 1, 2, 3, 4])
            ax_acc_ps[reg].set_yticks([0, 0.5, 1])
            if plot_significance:
                for i_tt, tt in enumerate(plot_tt):
                    _, signif_arr = pof.stat_test_dyn_dec(pred_dict=plot_dict_split, decoder_name='NA',
                                                        time_array=time_array, tt=tt, region=reg)
                    ax_acc_ps[reg].plot(time_array, [bottom_sign_bar + (i_tt  *0.03) if x == 1 else np.nan for x in signif_arr],
                                    linewidth=2, c=color_tt[tt], clip_on=False)

        if indicate_spont:
            ax_acc_ps['s1'].text(s='Reward only', x=4, y=0.33,
                                fontdict={'weight': 'bold', 'color': color_tt['spont'], 'ha': 'right'})
        if indicate_fp:
            ax_acc_ps['s1'].text(s='FP', x=1.4, y=0.62,
                                fontdict={'weight': 'bold', 'color': color_tt['fp']})

def plot_regularisation_optimisation(all_data_dict, time_array_plot=None, decoder_key='hit/cr', 
                                     tt_pos='hit', tt_neg='cr', reg='s1', ax=None, c='k'):
    reg_arr = np.array(list(all_data_dict.keys()))
    plot_reg_arr = 1 / reg_arr
    if time_array_plot is not None:
        tp_min = np.where(np.isnan(time_array_plot))[0][-1] + 1
    else:
        tp_min = 0

    mean_diff_arr = np.zeros(len(reg_arr))
    ci_diff_arr = np.zeros(len(reg_arr))
    for i_reg, reg_strength in enumerate(reg_arr):
        dict_pos = all_data_dict[reg_strength][decoder_key][tt_pos]
        dict_neg = all_data_dict[reg_strength][decoder_key][tt_neg]
        mouse_list = [x for x in dict_pos.keys() if x[-2:] == reg]
        mat_diff = np.zeros((len(mouse_list), dict_pos[mouse_list[0]].shape[0] - tp_min))
        for i_m, mouse in enumerate(mouse_list):
            mean_pos = dict_pos[mouse][tp_min:, 0]
            mean_neg = dict_neg[mouse][tp_min:, 0]
            mat_diff[i_m, :] = mean_pos - mean_neg
        time_av_diff = np.mean(mat_diff, 1)
        mean_diff_arr[i_reg] = np.mean(time_av_diff)
        ci_diff_arr[i_reg] = np.std(time_av_diff) / np.sqrt(len(time_av_diff)) * 1.96  # 95% ci

    if ax is None:
        ax = plt.subplot(111)
    ax.plot(plot_reg_arr, mean_diff_arr, linewidth=3, c=c, label=decoder_key + ' ' + reg.upper())
    ax.fill_between(plot_reg_arr, mean_diff_arr - ci_diff_arr, mean_diff_arr + ci_diff_arr, facecolor=c, alpha=0.2)
    ax.set_xscale('log')
    ax.set_xlabel('L2 regularisation strength')
    ax.set_ylabel('Classification performance\n(= difference in predictions)')
    ax.legend(loc='best', frameon=False)
    despine(ax)

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

def single_cell_plot(session, cell_id, tt=['hit'], smooth_traces=False, smooth_window=2,
                     filter_150_stim=False, ylims=(-1, 2), plot_artefact=False,
                     ax=None, plot_ylabel=True, plot_title=False, plot_indiv=True,
                     plot_total_mean=True, plot_n_cell_split=False,
                     plot_legend_n_cells=False, plot_tt_descr=False):
    assert len(tt) == 1, 'trial types are averaged'
    if tt[0] == 'prereward':
        if plot_n_cell_split and not plot_total_mean and not plot_indiv:
            plot_total_mean = True
            plot_n_cell_split = False
        elif plot_n_cell_split:
            plot_n_cell_split = False

    (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2, data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
        time_ticks, time_tick_labels, time_axis) = normalise_raster_data(session=session,
                            sort_neurons=False, filter_150_stim=filter_150_stim)

    if ax is None:
        ax = plt.subplot(111)
    assert type(tt) == list and len(tt) == 1

    ## Get cell data
    if tt[0] == 'prereward':
        arr = data_spont_mat_norm[cell_id, :, :]  # session.pre_rew_trials[cell_id, :, :]
    else:
        arr = data_use_mat_norm[cell_id, :, :]  #session.behaviour_trials[cell_id, :, :]
        trial_idx = np.isin(outcome_arr, tt)
        arr = arr[trial_idx, :]

        if filter_150_stim:
            n_cells_stim_per_trial = session.trial_subsets[session.photostim < 2]
        else:
            n_cells_stim_per_trial = session.trial_subsets
        n_cells_stim_arr = n_cells_stim_per_trial[trial_idx]
    x_axis = copy.deepcopy(time_axis)

    ## Remove photostim artefact:
    if filter_150_stim:
        remove_photostim = np.logical_or(x_axis <= -0.07,
                                     x_axis > 0.35)
    else:
        remove_photostim = np.logical_or(x_axis <= -0.07,
                                     x_axis > 0.83)
    if 'prereward' not in tt:
        x_axis[~remove_photostim] = np.nan

    ## Plot:
    ax.plot([-2.1, 4], [0, 0], linestyle=':', c='grey')
    if plot_indiv:
        for i_trial in range(arr.shape[0]): ## individual trials
            if smooth_traces:
                trial = smooth_trace(arr[i_trial, :], smooth_window)
            else:
                trial = arr[i_trial, :]
            ax.plot(x_axis, trial, alpha=0.3, color='grey')
    if plot_total_mean:
        if smooth_traces:
            meaned = smooth_trace(np.mean(arr, 0), smooth_window)
        else:
            meaned = np.mean(arr, 0)
        ax.plot(x_axis, meaned, color=color_tt[tt[0]], linewidth=2)#, label=label_tt[tt[0]])  # trial-average

    if plot_n_cell_split:
        alpha_arr = np.linspace(0.3, 1, 7)
        for i_n_cells, n_cells in enumerate(np.unique(n_cells_stim_arr)):
            trial_inds = np.where(n_cells_stim_arr == n_cells)[0]
            plot_trace = np.mean(arr[trial_inds, :], 0)
            if smooth_traces:
                plot_trace = smooth_trace(plot_trace, smooth_window)
            ax.plot(x_axis, plot_trace, alpha=alpha_arr[i_n_cells],
                    color=color_tt[tt[0]], linewidth=2, label=(int(n_cells) if tt[0] == 'hit' else None))
        if plot_legend_n_cells:
            ax.annotate(s='Cells\ntargeted:', xy=(1.2, 0.9), xycoords='axes fraction')
            ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.15, 0.88))


    if 'prereward' not in tt and plot_artefact:
        add_ps_artefact(ax=ax, time_axis=x_axis)
    ax.set_xlim(-2, 4)
    ax.set_ylim(ylims)
    if plot_title:
        ax.set_title(f'{tt[0]} Trials', fontdict={'color': color_tt[tt[0]]})
    if plot_ylabel:
        ax.set_ylabel(r'$\Delta$F/F')
    # ax.set_xlabel('Time (s)')
    if plot_tt_descr:
        x_coords_hm = ax.get_xlim()[0]
        ax.text(s='Hit', x=x_coords_hm, y=ylims[1] * 0.8, fontdict={'color': color_tt['hit']})
        ax.text(s='Miss', x=x_coords_hm, y=ylims[1] * 0.6, fontdict={'color': color_tt['miss']})
        ax.text(s='Reward\nonly', x=x_coords_hm, y=ylims[1] * 0.2, fontdict={'color': color_tt['spont']})
        ax.text(s='Photostimulation', x=0.5, y=ylims[0], fontdict={'color': color_tt['photostim']})

    naked(ax)

    return arr


def plot_transfer_function(dict_activ, label=None, ax=None, verbose=0, plot_logscale=False,
                            plot_indiv_data=True, plot_mean_ci=True, plot_lin_fit=True,
                            weighted_regression=False, sqrt_weights=False, clip_weights=True,
                            dict_var=None, indicate_spont_ci=False):
    if ax is None:
        ax = plt.subplot(111)
    fit_x = []
    fit_y = []
    if weighted_regression:
        assert dict_var is not None
        wls_weights = []
    color = color_tt[label]
    for key, val in dict_activ.items():  # key = n_ps, val = data point per session
        x = np.repeat(key, len(val))  # number of cells targeted
        y = np.array(val)
        if weighted_regression:
            median = np.nanmedian(dict_var[key][dict_var[key] > 0])  # filter 0s for median 
            var_y = np.array([1 / tmp_var if tmp_var > 0.001 else 1 / median for tmp_var in dict_var[key]])  # inverse variance
            if sqrt_weights:
                var_y = np.sqrt(var_y)
        # Need to take a nan out where there's no misses of some cell number
        nn = np.where(~np.isnan(y))
        x = x[nn]
        y = y[nn]
        if weighted_regression:
            var_y = var_y[nn]

        fit_x.extend(x)
        fit_y.extend(y)
        if weighted_regression:
            wls_weights.extend(var_y)

        if plot_indiv_data:
            if weighted_regression:
                ax.scatter(x, y, c=color, s=8)  # plot individual sessions
            else:
                ax.scatter(x, y, c=color, s=8)  # plot individual sessions
        if plot_mean_ci:
            ci = np.std(y) / np.sqrt(len(y)) * 1.96  # 95% ci
            ax.errorbar(key, np.mean(y), yerr=ci, color=color,
                        markersize=12, label=label, capsize=6, linewidth=1.5, fmt='.')
        # label = None
    if weighted_regression and clip_weights:
        wls_weights = np.clip(wls_weights, np.percentile(wls_weights, 25), np.percentile(wls_weights, 75))
    if plot_lin_fit:
        ## Linear regression on session data points:
        fit_x = np.array(fit_x)
        fit_y = np.array(fit_y)
        if weighted_regression:
            # sklearn_wls_model = sklearn.linear_model.LinearRegression()
            # sklearn_wls_model.fit(fit_x.reshape(-1, 1), fit_y, sample_weight=wls_weights)
            # print(sklearn_wls_model.intercept_, sklearn_wls_model.coef_)
            fit_x_with_intercept = statsmodels.api.add_constant(fit_x)
            wls_model = statsmodels.regression.linear_model.WLS(fit_y, fit_x_with_intercept,
                                                    weights=wls_weights)
            results = wls_model.fit()
            intercept, slope = results.params
            if verbose:
                print('weighted regression')
                if verbose > 1:
                    print(results.summary())
                print('statsmodels intercept + slope:', intercept, slope)
                print('p values', results.pvalues, 'r squard', results.tvalues)
                print('weighted corr', weighted_pearson_corr(x=fit_x, y=fit_y, w=wls_weights))
                print('linear corr', np.corrcoef(fit_x, fit_y)[1, 0])
                print("\n")
        else:
            slope, intercept, r, p, se = scipy.stats.linregress(fit_x, fit_y)
            if verbose:
                # print(label)
                print('linear regression')
                print(f'r={r}')
                print(f'p={p}')
                print('\n')
        ax.plot(fit_x, fit_x * slope + intercept, color=color, linewidth=2)
       
    if indicate_spont_ci:
        ax.text(s='R.O. 95% CI', x=72, y=3)
    despine(ax)
    if plot_logscale:
        ax.set_xscale('log')
    ax.set_xlabel('Number of cells targeted')

def plot_scatter_balance_stim(dict_activ_full, ax_s1=None, ax_s2=None, tt='hit', plot_legend=True, verbose=0):
    if ax_s1 is None or ax_s2 is None:
        fig, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={'wspace': 0.4})

    ax_dict = {'s1': ax_s1, 's2': ax_s2}
    full_arr_exc = {x: np.array([]) for x in ['s1', 's2']}
    full_arr_inh = {x: np.array([]) for x in ['s1', 's2']}
    for i_reg, reg in enumerate(['s1', 's2']):
        dict_exc = dict_activ_full[reg]['positive']
        dict_inh = dict_activ_full[reg]['negative']
        assert (dict_exc.keys() == dict_inh.keys())
        for n_stim in list(dict_exc.keys()):
            arr_exc = dict_exc[n_stim].copy()
            arr_inh = dict_inh[n_stim].copy()
            nn = np.where(~np.isnan(arr_inh))  # filter nans (when no trial present)
            arr_exc = arr_exc[nn]
            arr_inh = arr_inh[nn]
            full_arr_exc[reg] = np.concatenate((full_arr_exc[reg], arr_exc.copy()))
            full_arr_inh[reg] = np.concatenate((full_arr_inh[reg], arr_inh.copy()))
            ax_dict[reg].scatter(arr_exc, arr_inh, color=color_tt[tt], s=np.power(n_stim, 0.7), label=int(n_stim))
        ax_dict[reg].set_xlabel('Fraction excited (%)')
        ax_dict[reg].set_ylabel('Fraction inhibited (%)')
        # ax_dict[reg].set_title(f'E/I balance in {reg.upper()} on {tt} trials')
        despine(ax_dict[reg])
        equal_xy_lims(ax=ax_dict[reg], start_zero=True)
        pearson_r, pearson_p = scipy.stats.pearsonr(full_arr_exc[reg], full_arr_inh[reg])
        if verbose > 0:
            print(reg, pearson_r, pearson_p)
        # ax_dict[reg].annotate(s=f'Pearson r = {np.round(pearson_r, 2)}, p < {readable_p(pearson_p)}',
        #                   xy=(0.05, 0.91), xycoords='axes fraction')  # top
        # xy = (0.51, 0.045) if tt == 'miss' and reg == 's2' else (0.585, 0.045)
        xy = (1.1, 0.045)
        ax_dict[reg].annotate(s='r={:.2f}'.format(np.round(pearson_r, 2)) + f'\np < {readable_p(pearson_p)}',
                          xy=xy, xycoords='axes fraction', ha='right')
    if plot_legend:
        ax_dict['s2'].annotate(s='Cells\ntargeted:', xy=(1.25, 0.77), xycoords='axes fraction')
        ax_dict['s2'].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.15, 0.75))

def plot_spont(msm, region='s1', direction='positive', ax=None):
    if ax is None:
        ax = plt.subplot(111)

    n_responders = []
    for session_idx in range(len(msm.linear_models)):
        session = msm.linear_models[session_idx].session
        n_responders.append(np.mean(pof.get_percent_cells_responding(session, region, direction,
                                         prereward=True)))
    meaned = np.mean(n_responders)
    ci = np.std(n_responders) / np.sqrt(len(n_responders)) * 1.96
    ax.fill_between([5, 150], [meaned - ci, meaned - ci], [meaned + ci, meaned + ci],
                     color=color_tt['spont'], alpha=0.2)


def plot_multisesssion_flu(msm, region, outcome, frames, n_cells, stack='all-trials', ax=None,
                           plot_ps_artefact=False, art_150_included=True, verbose=1):

    if ax is None:
        ax = plt.subplot(111)

    flu = []
    for lm in msm.linear_models:
        sf, time_axis = pof.session_flu(lm, region=region, outcome=outcome, frames=frames,
                         n_cells=n_cells)
        if stack == 'all-trials':
            flu.append(sf)  # stack every trial from every session in a big array
        else:
            flu.append(np.mean(sf, 0))  # stack the session mean into a big array
    x_axis = copy.deepcopy(time_axis)  # use last session, they are all the same
    flu = np.vstack(flu)  # Go from list of 2D arrays (trials x time points) to stacked 2D (along trial axis)
    mean_flu = np.mean(flu, 0)  # average across trials
    z = 1.96  # 95% confidence interval value
    ci = z * (np.std(flu, 0) / np.sqrt(flu.shape[0]))

    # Remove the artifact
    if art_150_included:
        artifact_frames = np.where((x_axis >= -0.07) & (x_axis < 0.9))
    else:
        artifact_frames = np.where((x_axis >= -0.07) & (x_axis < 0.35))
    mean_flu[artifact_frames] = np.nan
    x_axis[artifact_frames] = np.nan
    label = outcome.capitalize()

    if outcome == 'pre_reward':
        label = 'Reward\nonly'

    ax.plot(x_axis, mean_flu, color=color_tt[outcome], label=label)
    ax.fill_between(x=x_axis, y1=mean_flu + ci, y2=mean_flu - ci, color=color_tt[outcome], alpha=0.2)
    ax.set_xlabel('Time (seconds)')
    ax.axhline(0, ls=':', color='grey')
    ax.set_xlim(-2, 4)
    if plot_ps_artefact:
        add_ps_artefact(ax, time_axis=x_axis)

    if region != 's2':
        ax.set_ylabel(r'$\Delta$F/F')

    if verbose > 0:
        ## find max:
        ind_max = np.nanargmax(mean_flu)
        print(outcome, region)
        print('Grand average max value: ', mean_flu[ind_max], 'pm ', ci[ind_max], 'at time ', x_axis[ind_max])
        print('---')

def return_fraction_interval(min_lim=0, max_lim=1, frac=0.5):
    len_int = max_lim - min_lim 
    frac_relative = frac  * len_int 
    frac_abs = frac_relative + min_lim 
    return frac_abs

def plot_average_tt_s1_s2(msm, n_cells, ax_s1=None, ax_s2=None, save_fig=False, plot_legend=False,
                          tts_plot=['hit', 'miss'], frames='all', stack='all-trials',
                          main_ylims=(-0.2, 0.2), zoom_ylims=(-0.025, 0.055), zoom_inset=True):
    if ax_s1 is None or ax_s2 is None:
        fig, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(10,3))

    ## S1 plot
    ax_list = [ax_s1, ax_s2]
    for i_plot, reg in enumerate(['s1', 's2']):
        for tt in tts_plot:
            plot_multisesssion_flu(msm, region=reg, outcome=tt, frames=frames, n_cells=n_cells,
                            stack=stack, ax=ax_list[i_plot], plot_ps_artefact=(tt == 'hit'),
                            art_150_included=(150 in n_cells))
        ax_list[i_plot].set_title(f'Average activity {reg.upper()}', y=1.1)
        ax_list[i_plot].set_ylim(main_ylims)
        ax_list[i_plot].set_xlabel('Time (s)')
        despine(ax_list[i_plot])
    ax_list[0].set_ylabel('Average ' + r'$\Delta$F/F')
    ax_list[1].set_yticklabels(['' for x in ax_list[1].get_yticks()])

    if plot_legend:
        # leg = ax_s2.legend(frameon=False, loc='upper right')
        # lines = leg.get_lines()
        # _ = [line.set_linewidth(4) for line in lines]
        start_y = 0.2
        for idx, tt in enumerate(tts_plot):
            # tt_txt = tt if tt!= 'pre_reward' else 'reward only'

            ax_s1.text(s=label_tt[tt_txt], x=-1.9,
                       y=start_y-idx*0.045, fontdict={'color': color_tt[tt]})

        # ax_s1.text(s='Miss', x=-2.5, y=-0.36, fontdict={'color': color_tt['miss']}, fontsize=25)

    if zoom_inset:
        ## set box size, coords relative to fraction of axes
        if 150 not in n_cells:
            x_box_min, y_box_min = 0.6, 0.665
            x_box_len, y_box_len = 0.5, 0.47
            x_lims_box = [0, 2]  # In data units
        else:
            x_box_min, y_box_min = 0.7, 0.665
            x_box_len, y_box_len = 0.5, 0.47
            x_lims_box = [0.5, 2.5]  # In data units

        ax_zoom = {}
        for i_plot, reg in enumerate(['s1', 's2']):

            x_box_max, y_box_max = x_box_min + x_box_len, y_box_min + y_box_len
            ax_zoom[i_plot] = ax_list[i_plot].inset_axes([x_box_min, y_box_min, x_box_len, y_box_len])
            for tt in tts_plot:
                plot_multisesssion_flu(msm, region=reg, outcome=tt, frames=frames, n_cells=n_cells,
                                stack=stack, ax=ax_zoom[i_plot], plot_ps_artefact=(tt == 'hit'),
                                art_150_included=(150 in n_cells), verbose=0)
            # despine(ax_zoom[i_plot])
            ax_zoom[i_plot].set_ylim(zoom_ylims)

            ax_zoom[i_plot].set_xlim(x_lims_box)
            ax_zoom[i_plot].set_xlabel('')
            ax_zoom[i_plot].set_ylabel('')
            ax_zoom[i_plot].set_xticks([])
            ax_zoom[i_plot].set_yticks([])

            ## get coords of inset box in real coords of main panel:
            inset_x_min_coord = return_fraction_interval(min_lim=ax_list[i_plot].get_xlim()[0],
                                                         max_lim=ax_list[i_plot].get_xlim()[1],
                                                         frac=x_box_min)
            inset_x_max_coord = return_fraction_interval(min_lim=ax_list[i_plot].get_xlim()[0],
                                                         max_lim=ax_list[i_plot].get_xlim()[1],
                                                         frac=x_box_max)
            inset_y_min_coord = return_fraction_interval(min_lim=ax_list[i_plot].get_ylim()[0],
                                                         max_lim=ax_list[i_plot].get_ylim()[1],
                                                         frac=y_box_min)
            inset_y_max_coord = return_fraction_interval(min_lim=ax_list[i_plot].get_ylim()[0],
                                                         max_lim=ax_list[i_plot].get_ylim()[1],
                                                         frac=y_box_max)
            ## Plot box in main panel
            lw_box = plt.rcParams['axes.linewidth']
            ax_list[i_plot].plot(x_lims_box, [zoom_ylims[0], zoom_ylims[0]], c='k', linewidth=lw_box, clip_on=False)
            ax_list[i_plot].plot(x_lims_box, [zoom_ylims[1], zoom_ylims[1]], c='k', linewidth=lw_box, clip_on=False)
            ax_list[i_plot].plot([x_lims_box[0], x_lims_box[0]], [zoom_ylims[0], zoom_ylims[1]], c='k', linewidth=lw_box, clip_on=False)
            ax_list[i_plot].plot([x_lims_box[1], x_lims_box[1]], [zoom_ylims[0], zoom_ylims[1]], c='k', linewidth=lw_box, clip_on=False)
            ## Plot lines to box:
            ax_list[i_plot].plot([x_lims_box[0], inset_x_min_coord], [zoom_ylims[1], inset_y_max_coord],
                                 c='k', linewidth=lw_box, clip_on=False)
            ax_list[i_plot].plot([x_lims_box[1], inset_x_max_coord], [zoom_ylims[0], inset_y_min_coord],
                                 c='k', linewidth=lw_box, clip_on=False) 

    if save_fig:
        name_plot  = '-'.join(tts_plot)
    #     save_figure('Figure2_grandAverageTraces', figure_path)

def firing_rate_dist(lm, region, match_tnums=False, sort=False,
                     ax_hit=None, ax_miss=None):
    if ax_hit is None or ax_miss is None:
        fig, ax = plt.subplots(2, 1, figsize=(15,4))
        ax_hit, ax_miss = ax[0], ax[1]

    flu = pof.select_cells_and_frames(lm, region=region, frames='pre')
    flu = np.mean(flu, 2)  # Mean across time

    miss = flu[:, lm.session.outcome == 'miss']
    n_misses = miss.shape[1]

    hit = flu[:, lm.session.outcome == 'hit']
    n_hits = hit.shape[1]

    if match_tnums:
        keep_idx = np.random.choice(np.arange(n_hits), size=n_misses, replace=False)
        hit = hit[:, keep_idx]
        assert hit.shape == miss.shape

    # Mean across trials
    hit = np.mean(hit, 1)
    miss = np.mean(miss, 1)

    if sort:
        hit = np.sort(hit)
        miss = np.sort(miss)


    ax_miss.bar(np.arange(len(miss)), miss, width=2, color=color_tt['miss'], label='miss')
    ylims = ax_miss.get_ylim()
    naked(ax_miss)
    ax_miss.spines['bottom'].set_visible(True)
    ax_miss.set_xticks([0, 100, 200, 300])
    ax_miss.set_xlabel('Neuron ID')

    ax_hit.bar(np.arange(len(hit)), hit, width=2, color=color_tt['hit'], label='hit')
    ax_hit.set_ylim(ylims)
    naked(ax_hit)

    ax_hit.text(s='Hit', x=0, y=0.4, fontdict={'color': color_tt['hit']})
    ax_miss.text(s='Miss', x=0, y=0.6, fontdict={'color': color_tt['miss']})
    ax_hit.set_title('Pre-stimulus average activity')
    # ax_miss.text(x=-50, y=0.2, s='DF/F meaned across\nframes pre-stim and\nacross trials',
    #          rotation=90)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def covar_sketch(ax=None, plot_pc_var=True, plot_corr=True, mid_x = 0.05, translate_y=0.5):
    if ax is None:
        ax = plt.subplot(111)

    ## Pop mean and var:
    x_arr_gauss = np.linspace(mid_x -0.25, mid_x + 0.25, 100)
    
    y_arr_gauss = gaussian(x=x_arr_gauss, mu=mid_x, sig=0.1) * 0.4 + translate_y

    ax.plot(x_arr_gauss, y_arr_gauss, c='k', linewidth=1.5, clip_on=False)

    ax.plot([mid_x - 0.3, mid_x + 0.3], [translate_y - 0.03, translate_y - 0.03], c='k', linewidth=1.0, clip_on=False)
    ax.text(s=r"$\Delta F/F$" + ' distr.', x=mid_x, y=translate_y - 0.13, ha='center')
    ax.arrow(mid_x, translate_y + 0.2, 0.1, 0, head_width=0.04, head_length=0.02, linewidth=1.5,
                            color='k', length_includes_head=True, clip_on=False)
    ax.arrow(mid_x, translate_y + 0.2, -0.1, 0, head_width=0.04, head_length=0.02, linewidth=1.5,
                            color='k', length_includes_head=True, clip_on=False)
    ax.text(s='Pop var.', x=mid_x, y=translate_y + 0.05, ha='center')
    
    ax.arrow(mid_x + 0.1, translate_y + 0.5, -0.08, -0.08, head_width=0.04, head_length=0.02, linewidth=1.5,
                            color='k', length_includes_head=True, clip_on=False)
    ax.text(s='Pop mean', x=mid_x + 0.3, y=translate_y + 0.5, ha='center')
    
    # ## Correlation:
    if plot_corr:
        n_scatter = 40
        # mat_scatter = np.random.multivariate_normal(np.array([0, 0]), np.array([[1, 0.5], [0.5, 1]]), size=n_scatter) * 0.05
        mat_scatter = np.array([[-0.10894625, -0.03707173],
                                [ 0.04927918,  0.0558435 ],
                                [ 0.07458346,  0.03630893],
                                [ 0.0737646 ,  0.09700735],
                                [ 0.02697377, -0.00113548],
                                [-0.04998247,  0.04741769],
                                [ 0.0752935 ,  0.0101812 ],
                                [-0.07316041,  0.00448425],
                                [ 0.02393236,  0.02762098],
                                [-0.0541065 ,  0.01168037],
                                [-0.0008373 , -0.02861629],
                                [ 0.02036253, -0.00301528],
                                [ 0.02301268, -0.00663146],
                                [ 0.00729329,  0.08739137],
                                [-0.06424448,  0.02205373],
                                [-0.10345242, -0.04532926],
                                [-0.04049523, -0.08862355],
                                [-0.08370238, -0.08036568],
                                [ 0.0944889 ,  0.11319315],
                                [-0.01612632, -0.033244  ],
                                [-0.03482069, -0.09758172],
                                [-0.01408344, -0.01694928],
                                [ 0.01295038,  0.05611397],
                                [ 0.08962012,  0.03492652],
                                [ 0.02143228,  0.06839815],
                                [-0.00308122,  0.05854908],
                                [-0.01016929, -0.0544294 ],
                                [-0.07146149, -0.01729002],
                                [ 0.01721404, -0.03629429],
                                [ 0.0340228 ,  0.04326364],
                                [-0.06413212, -0.0580089 ],
                                [ 0.01786435,  0.0595894 ],
                                [-0.00896541,  0.02284251],
                                [-0.00737659,  0.00476788],
                                [ 0.00955506, -0.0185146 ],
                                [-0.06771121, -0.04626916],
                                [-0.0375874 , -0.05779237],
                                [-0.0850151 , -0.01057411],
                                [-0.03711363, -0.02676278],
                                [ 0.03556763,  0.08671738]])  # for reproduc. 
        mat_scatter *= 1.3
        # axes:
        left_scat = 0.5
        ax.plot([left_scat, left_scat + 0.34], [0.1, 0.1], c='k', clip_on=False)
        ax.plot([left_scat, left_scat], [0.1, 0.5], c='k', clip_on=False)
        ax.text(s='Neuron 1', x=left_scat + 0.0, y=0.02, c='k')
        ax.text(s='Neuron 2', x=left_scat - 0.08, y=0.15, c='k', rotation=90)
        
        ax.scatter(mat_scatter[:, 0] + left_scat + 0.2, mat_scatter[:, 1] + 0.3, c='k', s=5, clip_on=False)


        ax.arrow(0.95, 0.43, -0.06, 0.06, head_width=0.04, head_length=0.02, linewidth=1.5,
                                color='k', length_includes_head=True, clip_on=False)
        ax.arrow(0.81, 0.57, 0.06, -0.06, head_width=0.04, head_length=0.02, linewidth=1.5,
                                color='k', length_includes_head=True, clip_on=False)
        ax.text(s='Corr.', x=0.925, y=0.47, ha='center', rotation=-45)
        
        if plot_pc_var:
            ax.arrow(0.76, 0.21, 0.15, 0.15, head_width=0.04, head_length=0.02, linewidth=1.5,
                                    color='k', length_includes_head=True, clip_on=False)
            ax.text(s='Var.\n1st PC', x=0.8, y=0.1, c='k', rotation=45)
 
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    naked(ax)

def pre_stim_sketch(session, ax=None, x_min=-1, x_max=2, pre_stim_start=-0.5):
    if ax is None:
        ax = plt.subplot(111)
    time_axis = np.arange(x_min, x_max, 1/30)
    remove_stim_art_inds = np.logical_and(time_axis >= -0.07, time_axis < 0.35)
    time_axis[remove_stim_art_inds] = np.nan
    add_ps_artefact(ax=ax, time_axis=time_axis)
    
    (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2, data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
        time_ticks, time_tick_labels, time_axis_norm) = normalise_raster_data(session=session, start_time=x_min, filter_150_stim=True,
                                        sort_neurons=True, end_time=x_max)

    inds_cells = np.array([123, 34, 54, 65, 76, 86, 102])
    ind_trial = 62  # used to be 64 (with 250 ms)
    # print({k: v for k, v in enumerate(outcome_arr)})
    # print(outcome_arr[ind_trial])
    assert len(time_axis) == len(time_axis_norm)# and time_axis[0] == time_axis_norm[0] and time_axis[-1] == time_axis_norm[-1]  # one contains nans at artefact and other one doesnt
    data_plot = data_use_mat_norm_s1[inds_cells, :, :][:, ind_trial, :]
    assert data_plot.ndim == 2
    n_cells = len(inds_cells)
    for i_cell, ind_cell in enumerate(inds_cells):
        ax.plot(time_axis, data_plot[i_cell, :] + 1.2 * i_cell - 5, c='k')

    ## cosmetics:
    ax.set_ylim([-6, ax.get_ylim()[1]])
    naked(ax)
    right_edge = -0.07 # -0.0826
    color_patch = (211 /256, 211 / 256, 211 / 256, 0.5)
    color_patch_edge = (211 /256, 211 / 256, 211 / 256, 0.0)
    ax.axvspan(pre_stim_start, right_edge, fc=color_patch, ec=color_patch_edge, alpha=0.3)
    trap_x = [x_min, x_max, right_edge, pre_stim_start]
    trap_y = [-9, -9, -6, -6] #-6.037, -6.039]
    color_patch = (211 /256, 211 / 256, 211 / 256, 0.3)
    color_patch_edge = (211 /256, 211 / 256, 211 / 256, 0.0)
    ax.add_patch(matplotlib.patches.Polygon(xy=list(zip(trap_x, trap_y)), fill=True, 
                                            fc=color_patch, ec=color_patch_edge, clip_on=False))
    ax.add_patch(matplotlib.patches.Rectangle(xy=(-1, -11.50), width=3, height=2.5, fill=True,   #11.04
                                            fc=color_patch, ec=color_patch_edge, clip_on=False))
    
    ax.text(s='Pre-stimulus\nactivity metrics:', x=-0.7, y=-10.9, fontdict={'weight':'bold'})
    ax.set_ylabel('')
    ax.set_title(r"$\Delta F/F$" + ' activity', y=1.17)
 
def dyn_dec_sketch(session, 

                  fig=None, x_min=-1, x_max=2, pre_stim_start=-0.5,
                  hit_dyn_dec_trace=None, cr_dyn_dec_trace=None):

    ##
    if fig is None:
        fig = plt.figure(constrained_layout=False, figsize=(8, 3.25))
    else:
        figsize = fig.get_size_inches()
        if figsize[0] > figsize[1]:  # assume only the sketch should be shown 
            gs_sketch_activity = fig.add_gridspec(ncols=1, nrows=1, bottom=0.05, top=0.95, right=0.5, left=0.01, 
                                hspace=0.1, wspace=0.1)
            gs_scatter = fig.add_gridspec(ncols=3, nrows=1, bottom=0.65, top=0.9, right=0.95, left=0.6, 
                                hspace=0.1, wspace=0.25)
            gs_trace = fig.add_gridspec(ncols=1, nrows=1, bottom=0.05, top=0.5, right=0.95, left=0.6, 
                                hspace=0.1, wspace=0.1)
        else: # assume it is part of Fig 3
            gs_sketch_activity = fig.add_gridspec(ncols=1, nrows=1, bottom=0.68, top=0.94, right=0.5, left=0.03, 
                                hspace=0.1, wspace=0.1)
            gs_scatter = fig.add_gridspec(ncols=3, nrows=1, bottom=0.85, top=0.91, right=0.95, left=0.61, 
                                hspace=0.1, wspace=0.25)
            gs_trace = fig.add_gridspec(ncols=1, nrows=1, bottom=0.69, top=0.81, right=0.95, left=0.61, 
                                hspace=0.1, wspace=0.1)

    ## Activity sketch
    ## -----------------------
    ax_sketch = fig.add_subplot(gs_sketch_activity[0])
    time_axis = np.arange(x_min, x_max, 1/30)
    remove_stim_art_inds = np.logical_and(time_axis >= -0.07, time_axis < 0.35)
    time_axis[remove_stim_art_inds] = np.nan
    c_dict = {**{x: color_tt['hit'] for x in [0, 2, 4]},
              **{x: color_tt['cr'] for x in [1, 3]}}
    
    (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2, data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
        time_ticks, time_tick_labels, time_axis_norm) = normalise_raster_data(session=session, start_time=x_min, filter_150_stim=True,
                                        sort_neurons=True, end_time=x_max)

    inds_cells = np.array([123, 34, 54, 65, 76, 86, 102])
    ind_trial = 62  # used to be 64 (with 250 ms)
    # print({k: v for k, v in enumerate(outcome_arr)})
    # print(outcome_arr[ind_trial])
    assert len(time_axis) == len(time_axis_norm)# and time_axis[0] == time_axis_norm[0] and time_axis[-1] == time_axis_norm[-1]  # one contains nans at artefact and other one doesnt
    data_plot = data_use_mat_norm_s1[inds_cells, :, :][:, ind_trial, :]
    assert data_plot.ndim == 2
    n_cells = len(inds_cells)
    ## First one:
    y_incr = 0.9
    translate_y = -5
    for i_cell, ind_cell in enumerate(inds_cells):
        ax_sketch.plot(time_axis, data_plot[i_cell, :] + y_incr * i_cell + translate_y, 
        c=c_dict[0])
    
    add_ps_artefact(ax=ax_sketch, time_axis=time_axis, y_min=0.18, y_max=0.59)
    time_incr = 0.4
    dist_y = 0.3
    for increment in range(5):
        ## top:
        ax_sketch.plot(time_axis + time_incr, 
                data_plot[i_cell, :] + y_incr * (n_cells + increment + dist_y) + translate_y, 
                c=c_dict[increment])
        # others:
        for i_cell, ind_cell in enumerate(inds_cells):
            if i_cell != n_cells - 1:  # last one is not necessary because already done baove
                ax_sketch.plot(time_axis[-5:] + time_incr, 
                data_plot[i_cell, :][-5:] + y_incr * (i_cell + 1 + + increment + dist_y) + translate_y,
                c=c_dict[increment])
        time_incr += 0.3
        
    ylims = ax_sketch.get_ylim()
    ylims = (-7.54, 6.83)  # these limits are meticiously set to have the ps artefacts line up nicely .. 
    func_map_ax_to_data_coords = lambda x: (ylims[1] - ylims[0]) * x + ylims[0]
    func_map_data_to_ax_coords = lambda x: (x - ylims[0]) / (ylims[1] - ylims[0])
    time_incr = 0.4

    for increment in range(5):
        ymin_data_coords = y_incr * (n_cells + increment + dist_y - 0.3) + translate_y
        ymax_data_coords = y_incr * (n_cells + increment + dist_y + 0.3) + translate_y

        add_ps_artefact(ax=ax_sketch, time_axis=time_axis + time_incr, 
                        y_min=func_map_data_to_ax_coords(ymin_data_coords),
                        y_max=func_map_data_to_ax_coords(ymax_data_coords))
        time_incr += 0.3

    ## axes 
    ax_sketch.plot([-1.15, 2.1], [-6, -6], c='k')  # time 
    ax_sketch.plot([-1.15, -1.15], [-6, 1], c='k')  # neurons 
    ax_sketch.plot([-1.15, 0.7], [1, 6.2], c='k')  # trials 
    
    ## cosmetics:
    naked(ax_sketch)
    
    for i_arrow, time_arrow in enumerate([-0.67, 0.5, 1.67]):
        ax_sketch.arrow(time_arrow, -7.5, 0.0, 1.2, alpha=(i_arrow + 2) / 4,
                        head_width=0.15, head_length=0.5, linewidth=2, zorder=10,
                        color='k', length_includes_head=True, clip_on=False)

    ax_sketch.text(s='Time points', x=0.5, y=-9.4, ha='center')

    ax_sketch.text(s='Time points', x=6.65, y=-9.4, ha='center')
    ax_sketch.text(s='Neurons', x=-1.35, y=-2.5, ha='center', va='center', rotation=90)
    # ax_sketch.text(s='Hit & CR trials', x=-0.4, y=4.1, ha='center', va='center', rotation=33.5)

    ax_sketch.text(s='Hit', x=-0.95, y=2.65, ha='center', va='center', rotation=33.5, C=color_tt['hit'])
    ax_sketch.text(s='&', x=-0.69, y=3.34, ha='center', va='center', rotation=33.5, C='k')
    ax_sketch.text(s='CR', x=-0.4, y=4.1, ha='center', va='center', rotation=33.5, C=color_tt['cr'])
    ax_sketch.text(s='trials', x=0.05, y=5.35, ha='center', va='center', rotation=33.5, C='k')


    ## Scatter
    ## -------------
    ax_scatter = {x: fig.add_subplot(gs_scatter[x]) for x in range(3)}
    scatter_data_dict = {0: {'hit': np.array([[0.3, 0.4], [0.22, 0.76], [0.7, 0.63], [0.55, 0.25]]),
                             'cr': np.array([[0.17, 0.25], [0.76, 0.35], [0.45, 0.85], [0.6, 0.5]])},
                         1: {'hit': np.array([[0.73, 0.4], [0.42, 0.26], [0.75, 0.43], [0.55, 0.15]]),
                             'cr': np.array([[0.17, 0.35], [0.26, 0.65], [0.45, 0.85], [0.6, 0.75]])},                            
                         2: {'hit': np.array([[0.63, 0.14], [0.82, 0.16], [0.77, 0.18], [0.95, 0.25]]),
                             'cr': np.array([[0.37, 0.85], [0.06, 0.65], [0.14, 0.75], [0.16, 0.85]])}}
    for i_plot in range(3):
        for tt in ['hit', 'cr']:
            ax_scatter[i_plot].plot(scatter_data_dict[i_plot][tt][:, 0], 
                                    scatter_data_dict[i_plot][tt][:, 1],
                                    '.', c=color_tt[tt], markersize=12)
        despine(ax_scatter[i_plot])
        ax_scatter[i_plot].set_xlim(0, 1)
        ax_scatter[i_plot].set_ylim(0, 1)
        ax_scatter[i_plot].set_xticks([])
        ax_scatter[i_plot].set_yticks([])

    ax_scatter[0].set_xlabel('Neural dim. 1')
    ax_scatter[0].set_ylabel('Neural\ndim. 2')
    ax_scatter[1].set_title('Classify trials per time point')
    ## trace:
    ax_trace = fig.add_subplot(gs_trace[0])

    if hit_dyn_dec_trace is None:
        tmp_hit = np.array([0.491, 0.492, 0.489, 0.489, 0.489, 0.498, 0.503, 0.506, 0.512,
                            0.512, 0.508, 0.512, 0.512, 0.512, 0.512, 0.504, 0.499, 0.497,
                            0.499, 0.498, 0.496, 0.491, 0.493, 0.494, 0.487, 0.492, 0.496,
                            0.498, 0.493, 0.495, 0.498, 0.504, 0.502, 0.5  , 0.504, 0.505,
                            0.506, 0.501, 0.499, 0.498, 0.497, 0.499, 0.501, 0.501, 0.504,
                            0.501, 0.495, 0.496, 0.505, 0.506, 0.5, 0.484, 0.52, np.nan,
                            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                            np.nan, np.nan, np.nan, 0.64 , 0.633, 0.642, 0.65 , 0.659, 0.673,
                            0.686, 0.69 , 0.698, 0.707, 0.712, 0.717, 0.719, 0.718, 0.717,
                            0.722, 0.729, 0.738, 0.743, 0.744, 0.749, 0.754, 0.757, 0.763,
                            0.769, 0.77 , 0.771, 0.773, 0.773, 0.777, 0.777, 0.778, 0.782,
                            0.784, 0.786, 0.79 , 0.788, 0.787, 0.793, 0.794, 0.796, 0.794,
                            0.795, 0.795, 0.796, 0.792, 0.793, 0.792, 0.789, 0.786, 0.786,
                            0.786, 0.783, 0.783, 0.782, 0.783, 0.781, 0.782, 0.779, 0.781,
                            0.78 , 0.782, 0.784, 0.78 , 0.777, 0.776, 0.777, 0.775, 0.775,
                            0.772, 0.774, 0.771, 0.768, 0.767, 0.77 , 0.77 , 0.767, 0.764,
                            0.761, 0.759, 0.752, 0.749, 0.747, 0.749, 0.744, 0.741, 0.74 ,
                            0.741, 0.742, 0.74 , 0.742, 0.739, 0.734, 0.729, 0.725, 0.724,
                            0.722, 0.72 , 0.712, 0.714, 0.71 , 0.709, 0.706, 0.704, 0.703])
        tmp_hit = tmp_hit[25:115]
        tmp_hit = 0.5 + (tmp_hit - 0.5) * 1.3
        tmp_cr = 1- tmp_hit
    ax_trace.plot([-1, 2], [0.5, 0.5], linestyle=':', linewidth=2, color='grey', zorder=-1)
    ax_trace.plot(time_axis, tmp_hit, linewidth=3, c=color_tt['hit'], zorder=0)
    ax_trace.plot(time_axis, tmp_cr, linewidth=3, c=color_tt['cr'], zorder=0)
    add_ps_artefact(ax=ax_trace, time_axis=time_axis)
    despine(ax_trace)
    ax_trace.set_ylim(0, 1)
    ax_trace.set_xticks([])
    # ax_trace.set_xlabel('Time')# + r"$\to$")
    ax_trace.set_yticks([0, 0.5, 1])
    ax_trace.set_ylabel('Hit vs CR\nclassification')
    ax_trace.arrow(-0.67, 1.1, 0.0, -0.5, alpha=0.5,
                    head_width=0.1, head_length=0.1, linewidth=2, zorder=10,
                    color='k', length_includes_head=True, clip_on=False)
    ax_trace.arrow(0.5, 1.2, 0.0, -0.4, alpha=0.75,
                    head_width=0.1, head_length=0.1, linewidth=2, zorder=10,
                    color='k', length_includes_head=True, clip_on=False)
    ax_trace.arrow(1.67, 1.2, 0.0, -0.25, alpha=1,
                    head_width=0.1, head_length=0.1, linewidth=2, zorder=10,
                    color='k', length_includes_head=True, clip_on=False)
    ax_trace.text(s='Chance level', x=1, y=0.55, color='grey')


def scatter_plots_covariates(cov_dicts, ax_dict=None, lims=(-0.6, 0.6),
                            plot_type='scatter', bonf_n_tests=None, verbose=0,
                    cov_names=['mean_pre', 'variance_cell_rates', 'corr_pre', 'largest_PC_var']):
    if ax_dict is None:
        fig, ax = plt.subplots(1, len(cov_names), figsize=(3 * len(cov_names), 2),
                                gridspec_kw={'wspace': 0.5, 'hspace': 0.3})
        ax_dict = {}
        for i_ax, cov_name in enumerate(cov_names):
            ax_dict[cov_name] = ax[i_ax]

    n_sessions = len(cov_dicts)

    for i_plot, cov_name in enumerate(cov_names):
        tmp_ax = ax_dict[cov_name]
        all_hit = np.zeros(n_sessions)
        all_miss = np.zeros(n_sessions)
        for i_lm, cov_dict in cov_dicts.items():
            if len(np.unique(cov_dict[cov_name])) == 1:
                data = np.zeros(len(cov_dict[cov_name]))
            else:
                data = scipy.stats.zscore(cov_dict[cov_name])
            metric_hits = data[cov_dict['y'] == 1]
            metrics_misses = data[cov_dict['y'] == 0]
            all_hit[i_lm] = np.mean(metric_hits)
            all_miss[i_lm] = np.mean(metrics_misses)
        if len(np.unique(cov_dict[cov_name])) == 1:
            p_val = 1
        else:
            _, p_val = scipy.stats.wilcoxon(all_hit, all_miss)
        if bonf_n_tests is None:
            bonf_n_tests = len(cov_names)
        bool_sign = p_val < (5e-2 / bonf_n_tests) # bonferoni correction

        if plot_type == 'scatter':
            tmp_ax.scatter(all_hit, all_miss,
                        s=80, facecolors=('grey' if bool_sign else 'none'),
                        edgecolors=('k' if bool_sign else 'grey'),
                        linewidth=3)
            if lims is not None:
                tmp_ax.set_xlim(lims)
                tmp_ax.set_ylim(lims)
            else:
                equal_xy_lims(tmp_ax)
                lims = tmp_ax.get_xlim()
            tmp_ax.plot(lims, lims, linestyle=(0, (5, 10)), color='grey')
            tmp_ax.set_xlabel('Hit trials (z-score)')
            tmp_ax.set_ylabel('Miss trials (z-score)')
        elif plot_type == 'pointplot':
            tmp_df = pd.DataFrame({'zscore': np.concatenate((all_miss, all_hit)),
                                   'trial_type': ['miss'] * len(all_miss) + ['hit'] * len(all_hit)})
            sns.stripplot(data=tmp_df, x='trial_type', y='zscore', ax=tmp_ax,
                            s=8, color=('grey' if bool_sign else 'white'),
                            edgecolor=('k' if bool_sign else 'grey'), linewidth=3)
            sns.pointplot(data=tmp_df, x='trial_type', y='zscore', ax=tmp_ax,
                            s=80, color=('k' if bool_sign else 'grey'),
                            linewidth=3, ci=95)
            if i_plot == 0:
                tmp_ax.set_ylabel('Mean z-score')
            else:
                tmp_ax.set_ylabel('')
            # tmp_ax.set_xlabel('Trial type')
            tmp_ax.set_xlabel('')
            tmp_ax.set_ylim([-0.63, 0.9])
            tmp_ax.tick_params(bottom=False)
            despine(tmp_ax)
        elif plot_type == 'connecting_lines':
            tmp_df = pd.DataFrame({'hit_zscore': all_hit,
                                   'miss_zscore': all_miss})
            n_sessions = len(tmp_df)
            tmp_df['hit_xcoord'] = np.random.randn(n_sessions) * 0.1 + 1
            tmp_df['miss_xcoord'] = np.random.randn(n_sessions) * 0.1
            for tt in ['hit', 'miss']:
                tmp_ax.plot(tmp_df[f'{tt}_xcoord'], tmp_df[f'{tt}_zscore'], '.', 
                            color='k',#('k' if bool_sign else 'grey'), 
                            markersize=15)
            for i_s in range(n_sessions):
                tmp_ax.plot([tmp_df['miss_xcoord'][i_s], tmp_df['hit_xcoord'][i_s]],
                            [tmp_df['miss_zscore'][i_s], tmp_df['hit_zscore'][i_s]],
                            c='k', alpha=0.7) #alpha=(0.7 if bool_sign else 0.3))
            if i_plot == 0:
                tmp_ax.set_ylabel('Mean z-score')
            else:
                tmp_ax.set_ylabel('')
            # tmp_ax.set_xlabel('Trial type')
            tmp_ax.set_xlabel('')
            tmp_ax.set_xlim([-0.5, 1.5])
            tmp_ax.set_ylim([-0.63, 0.9])
            tmp_ax.set_xticks([0, 1])
            tmp_ax.set_xticklabels(['miss', 'hit'])
            tmp_ax.tick_params(bottom=False)
            despine(tmp_ax)

        if verbose > 0:
            print(cov_name, p_val, readable_p(p_val), np.sum(all_hit > all_miss))
        tmp_ax.set_title(f'{covar_labels[cov_name]}\np < {readable_p(p_val)}')

def plot_scatter_all_trials_two_covars(super_covar_df_dict, ax=None, covar_1='mean_pre', 
                                        n_bonferoni=3,
                                        covar_2='corr_pre', region='s1', verbose=0):
    ## if cov_dicts given:
    # arr_1, arr_2 = np.array([]), np.array([])
    # n_sessions = len(cov_dicts[region])
    
    # for i_sess in range(n_sessions):
    #     arr_1 = np.concatenate((arr_1, cov_dicts[region][i_sess][covar_1]))
    #     arr_2 = np.concatenate((arr_2, cov_dicts[region][i_sess][covar_2]))
    
    arr_1 = copy.deepcopy(super_covar_df_dict[region][covar_1])  # use copy because of jitter with reward history
    arr_2 = copy.deepcopy(super_covar_df_dict[region][covar_2])

    result_lr = scipy.stats.linregress(x=arr_1, y=arr_2)
    slope, _, corr_coef, p_val, __ = result_lr
    if verbose > 0:
        print(slope, corr_coef, p_val)

    if covar_1 == 'reward_history':
        assert len(np.unique(arr_1)) == 6
        arr_1 += np.random.uniform(low=-0.2, high=0.2, size=len(arr_1))
    elif covar_2 == 'reward_history':
        assert len(np.unique(arr_2)) == 6, np.unique(arr_2)
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
        arr_2 += np.random.uniform(low=-0.2, high=0.2, size=len(arr_2))
    if ax is None:
        ax = plt.subplot(111)
    c_dots = ('k' if p_val < (0.05 / n_bonferoni) else 'grey')
    ax.plot(arr_1, arr_2, '.', c=c_dots, markersize=5)
    ax.set_xlabel(covar_labels[covar_1])
    ax.set_ylabel(covar_labels[covar_2])
    ax.set_title(f'r={np.round(corr_coef, 2)}, p < {readable_p(p_val)}')
    despine(ax)
    
def plot_accuracy_covar(cov_dicts, cov_name='variance_cell_rates', zscore_covar=False,
                        one_sided_ws=20, ax=None, sessions=None, metric='fraction_hit',
                        verbose=0):
    if ax is None:
        ax = plt.subplot(111)
    n_sessions = len(cov_dicts)
    for i_ss in range(n_sessions):

        tmp_vcr = copy.deepcopy(cov_dicts[i_ss][cov_name])
        if zscore_covar:
            tmp_vcr = scipy.stats.zscore(tmp_vcr)
        tmp_y = cov_dicts[i_ss]['y']  #0 = miss, 1= hit
        assert len(tmp_vcr) == len(tmp_y)

        sorted_inds_vcr = np.argsort(tmp_vcr)  # sort by covar value
        sorted_vcr = tmp_vcr[sorted_inds_vcr]
        sorted_y = tmp_y[sorted_inds_vcr]

        n_dp = len(sorted_y)
        av_vcr_arr = np.zeros(n_dp - 2 * one_sided_ws)  # take centered running mean with size stated
        av_y_arr = np.zeros(n_dp - 2 * one_sided_ws)
        i_dp = 0
        for i_cdp in range(one_sided_ws, n_dp - one_sided_ws):  # running mean
            wind_inds = np.arange(i_cdp - one_sided_ws, i_cdp + one_sided_ws)  # window
            av_vcr = np.mean(sorted_vcr[wind_inds])
            if metric == 'fraction_hit':
                av_y = np.mean(sorted_y[wind_inds])
            elif metric == 'dprime':
                trial_inds = sorted_inds_vcr[wind_inds]
                av_y = pof.get_alltrials_dprime(session=sessions[i_ss], trial_inds=trial_inds)
            av_vcr_arr[i_dp] = av_vcr
            av_y_arr[i_dp] = av_y
            i_dp += 1
        result_lr = scipy.stats.linregress(x=av_vcr_arr, y=av_y_arr)
        slope, _, corr_coef, p_val, __ = result_lr
        label = ' '
        if slope < 0 and corr_coef < 0:
            bonf_correction = 2 * n_sessions  # multiply by 2 for one-sided p val & bonferroni
            label = asterisk_p(p_val=p_val, bonf_correction=bonf_correction)
        if verbose > 0:
            print(f'Session {i_ss}, {result_lr}')
        ax.plot(av_vcr_arr, av_y_arr, label=label, linewidth=3, alpha=0.7, c=color_dict_stand[i_ss])
    if zscore_covar:
        assert cov_name == 'variance_cell_rates'
        ax.set_xlabel('Z-scored population variance')
    else:
        assert cov_name == 'variance_cell_rates'
        ax.set_xlabel('Population variance')
    ax.set_ylabel('Probability hit')
    # ax.set_ylim([0, 1])
    ax.set_title(f'P(hit) as function of VCR per session\n({int(2 * one_sided_ws)}-trial running mean used.)', y=1.1)
    despine(ax)
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.text(s=f'{n_sessions} sessions:\n***: p < 0.001', x=2.1, y=0.81)
    ax.legend(bbox_to_anchor=(1.1, -0.25), loc='lower left', frameon=False, ncol=2)

def plot_density_hit_miss_covar(super_covar_df, n_bins_covar=7, ax=None,
                                covar_name='variance_cell_rates', zscored_covar=True,
                                metric='fraction_hit', plot_arrow=False, include_150=True):
    (mat_fraction, median_cov_perc_arr, cov_perc_arr,
        n_stim_arr), _ = pof.compute_density_hit_miss_covar(super_covar_df=super_covar_df,
                                             n_bins_covar=n_bins_covar, metric=metric,
                                             include_150=include_150)

    if ax is None:
        ax = plt.subplot(111)

    if metric == 'fraction_hit':
        sns.heatmap(mat_fraction, ax=ax, vmin=0, vmax=1,
                    cbar_kws={'label': 'Probability hit'}, rasterized=False,
                    cmap=sns.diverging_palette(h_neg=350, h_pos=140, s=85, l=23, sep=10, n=10, center='light'))
        ax.set_title('P(hit) depends on pop. variance\n and number of cells targeted', y=1.04)
    elif metric == 'occupancy':
        sns.heatmap(mat_fraction, ax=ax, vmin=0, vmax=30,
                    cbar_kws={'label': 'Number of trials per bin'},
                    cmap='magma', rasterized=False)
        ax.set_title('Occupancy as a function of VCR and N_stim')

    ax.invert_yaxis()
    ax.set_yticklabels(n_stim_arr, rotation=0)
    xticklabels = [str(np.round(x, 1)) for x in median_cov_perc_arr]
    xticklabels = [x.replace('-0.0', '0').replace('0.0', '0').replace("-", u"\u2212") for x in xticklabels]
    ax.set_xticklabels(xticklabels, rotation=45)
    if zscored_covar:
        ax.set_xlabel(f'Binned z-scored {covar_labels[covar_name].lower()}\n(median of bin)')
    else:
        ax.set_xlabel(f'Binned {covar_labels[covar_name]}')
    ax.set_ylabel('Number of cells targeted')
    if plot_arrow:
        ax.arrow(n_bins_covar - 0.5, 0.5, -n_bins_covar + 1, n_bins_covar - 1,
                            head_width=0.35, head_length=0.45, linewidth=3,
                            color='k', length_includes_head=True, clip_on=False)
        assert n_bins_covar == 7
        ax.text(s='SNR axis', x=3, y=2.5, rotation=-45, fontdict={'weight': 'bold'})
        ax.text(s='0%', x=5.75, y=1.00, rotation=-45, fontdict={'weight': 'bold'})
        ax.text(s='100%', x=1.0, y=5.2, rotation=-45, fontdict={'weight': 'bold'})

def plot_collapsed_hit_miss_covar(super_covar_df, n_bins_covar=7, ax=None,
                            covar_name='variance_cell_rates', #zscored_covar=True,
                            metric='fraction_hit', pool_trials=True, verbose=0,
                            include_150=True):
    if ax is None:
        ax = plt.subplot(111)
    (mat_fraction, median_cov_perc_arr, cov_perc_arr,
        n_stim_arr), (mean_mat_arr, ci_mat_arr) = pof.compute_density_hit_miss_covar(super_covar_df=super_covar_df,
                                             n_bins_covar=n_bins_covar, metric=metric, verbose=verbose,
                                             include_150=include_150)
                                            #  zscore_covar=zscore_covar)

    diag_arr = np.linspace(0, 100, len(mean_mat_arr))
    if pool_trials:
        mean_mat_arr = mean_mat_arr[::-1]  # reverse for plotting
        ci_mat_arr = ci_mat_arr[::-1]
        ax.plot(diag_arr, mean_mat_arr, linewidth=3,
                color='k')
        ax.fill_between(diag_arr, mean_mat_arr - ci_mat_arr, mean_mat_arr + ci_mat_arr,
                        facecolor='grey', alpha=0.2)
        ax.set_xlabel('SNR axis (%)')
    else:
        arr_diag_index = np.arange(-n_bins_covar + 1, n_bins_covar - 1)
        mean_mat_arr = np.zeros(len(arr_diag_index))
        ## The most negative number corresponds to the top left (ie 150, low VCR)
        for i_diag_index, diag_index in enumerate(arr_diag_index):
            mat_inds = np.where(np.eye(n_bins_covar, k=diag_index) == 1)
            mat_vals = mat_fraction[mat_inds]
            mean_val = np.mean(mat_vals)
            # return mean_val
            mean_mat_arr[i_diag_index] = mean_val

        ax.plot(diag_arr, mean_mat_arr, color='k', linewidth=3, label='mean of means')
        ax.set_xlabel('Diagonal element (top left to bottom right)')
    ax.set_ylabel('Probability hit')
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    despine(ax)
    ax.set_title('Diagonal of matrix', y=1.14)

def hist_covar(super_covar_df, ax=None, covar_name='variance_cell_rates'):
    if ax is None:
        ax = plt.subplot(111)
    ax.hist(super_covar_df[covar_name], bins=20)
    if covar_name == 'variance_cell_rates':
        ax.set_xlabel('VCR')
        ax.set_title(f'z scored logged VCR across \n{len(super_covar_df)} trials')
    else:
        ax.set_xlabel(covar_name)
        ax.set_title(f'{covar_name} across\n{len(super_covar_df)} trials')
    ax.set_ylabel('Frequency')
    despine(ax)

def scatter_covar_s1s2(super_covar_df_dict, cov_name='variance_cell_rates', ax=None):
    if ax is None:
        ax = plt.subplot(111)
    assert (super_covar_df_dict['s1']['n_cells_stimmed'] == super_covar_df_dict['s2']['n_cells_stimmed']).all()

    ax.plot(super_covar_df_dict['s1'][cov_name], super_covar_df_dict['s2'][cov_name],
            '.', markersize=4, color='k')
    pearson_r, pearson_p = scipy.stats.pearsonr(super_covar_df_dict['s1'][cov_name],
                                                super_covar_df_dict['s2'][cov_name])
    # ax.annotate(s=f'r={np.round(pearson_r, 2)}, p < {readable_p(pearson_p)}',
    #                       xy=(0, 1.15), xycoords='axes fraction')

    despine(ax)
    ax.set_xlabel(f'{covar_labels[cov_name]} S1')
    ax.set_ylabel(f'{covar_labels[cov_name]} S2')
    ax.set_title(f'r={np.round(pearson_r, 2)}, p < {readable_p(pearson_p)}')

def get_plot_trace(lm, ax=None, targets=False, region='s1', 
                    n_stim_list=[5, 10, 20, 30, 40, 50],
                    coords_photostim_text=(0.05, 1.06),
                    i_col=0, color_dict=None, plot_ci=True, lw_mean=1,
                    tt_list=['hit', 'miss', 'too_', 'urh', 'arm'],
                    verbose=0, baseline_by_prestim=True,
                    absolute_vals=False, plot_artefact=False, plot_legend=False,
                    text_photostim=False, text_xlabel=True, text_ylabel=True,
                    type_plot='trace'):

    
    if color_dict is None:
        color_dict = color_dict_stand
    if targets:
        mask = ~lm.session.is_target  # flip bool (with ~) because np.ma.array later only saves False, and masks True
    else:
        mask = lm.session.is_target

    (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2, data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
        time_ticks, time_tick_labels, time_axis) = normalise_raster_data(session=lm.session, #start_time=-1,
                                    filter_150_stim=False, # start_baseline_time=-2.1,
                                    sort_neurons=False, end_time=6, baseline_by_prestim=baseline_by_prestim)

    mask = mask[lm.region_map[region], :, :]
    if region == 's1':
        flu = data_use_mat_norm_s1
    elif region == 's2':
        flu = data_use_mat_norm_s2

    ## Take out catch trials & filter trial outcome
    n_stim_bool = np.isin(lm.session.trial_subsets, n_stim_list)
    stim_idx = np.logical_and(n_stim_bool, np.isin(lm.session.outcome, tt_list))
    if verbose > 0:
        print(lm.session.outcome[stim_idx])
    flu = flu[:, stim_idx, :]
    mask = mask[:, stim_idx, :]

    mask = np.mean(mask, 2)
    mask = np.repeat(mask[:, :, None], flu.shape[2], axis=2)

    # Fluoresence averaged across cells with (non)targets filtered
    flu = np.ma.array(flu, mask=mask)
    
    x_axis = time_axis 
    label = 'Targets S1' if targets else f'Non Targets {region.upper()}'

    if absolute_vals:
        flu = np.abs(flu)
    mean_arr = np.mean(flu, (0, 1))
    if plot_ci:
        std_arr = np.std(flu, (0, 1))
        ci_arr = std_arr * 1.96 / np.sqrt(flu.shape[0] * flu.shape[1])

    if 150 in n_stim_list:
        remove_stim_art_inds = np.logical_and(time_axis >= -0.07, time_axis < 0.83)
    else:
        remove_stim_art_inds = np.logical_and(time_axis >= -0.07, time_axis < 0.35)
    time_axis = copy.deepcopy(time_axis)
    time_axis[remove_stim_art_inds] = np.nan
    pre_stim_frame = np.where(np.isnan(time_axis))[0][0]

    if absolute_vals:
        mean_arr -= np.mean(mean_arr[:pre_stim_frame])

    if type_plot == 'trace':
        if ax is None:
            ax = plt.subplot(111)
        ax.plot(time_axis, np.zeros(len(time_axis)), linestyle=':', c='grey', zorder=-10)
        ax.plot(time_axis, mean_arr, color=color_dict[i_col], label=label, linewidth=lw_mean)
        if plot_ci:
            ax.fill_between(time_axis, mean_arr - ci_arr, mean_arr + ci_arr,
                            color=color_dict[i_col], alpha=0.2)

        if plot_artefact:
            add_ps_artefact(ax=ax, time_axis=time_axis)
            if text_photostim:
                ax.annotate(s='Photostimulation', xy=coords_photostim_text, xycoords='axes fraction',
                            color=color_tt['photostim'], alpha=1)
        if text_ylabel:
            if absolute_vals:
                ax.set_ylabel('Absolute ' + r"$\Delta$F/F")
            else:
                ax.set_ylabel('Average ' + r"$\Delta$F/F")
        else:
            remove_yticklabels(ax)
        ax.set_xticks([-2, 0, 2, 4, 6])
        if text_xlabel:
            ax.set_xlabel('Time (s)')
        else:
            remove_xticklabels(ax)
        if absolute_vals:
            ax.set_ylim([ -0.06, 0.15])
        else:
            ax.set_ylim(-0.06, 0.23)
            ax.set_yticks([0, 0.1, 0.2])
        if plot_legend:
            if absolute_vals:
                legend = ax.legend(bbox_to_anchor=(1, 0), frameon=False, loc='lower right')
            else:
                start_y = 0.18
                for idx, tt in enumerate(['Targets', 'Non-targets S1', 'Non-targets S2']):
                    ax.text(s=tt, x=2.15, va='top',
                            y=start_y - idx * 0.025, fontdict={'color': color_dict[idx]})
                # legend = ax.legend(bbox_to_anchor=(1.4, 1.3), frameon=False, loc='upper left')
        despine(ax)
    elif type_plot == 'return_mean':
        return (time_axis, mean_arr)  
    elif type_plot == 'return_bar':
        post_stim_sum = np.sum(mean_arr[np.max(np.where(np.isnan(time_axis))[0]):])  # sum from last nan value of time axis onwards
        return post_stim_sum
    else:
        assert False, 'plot type not recognised'
    
def plot_bar_plot_targets(lm_list, dict_auc=None, baseline_by_prestim=True, 
                          ax=None, color_dict=None, plot_legend=True):
    n_sessions = len(lm_list)
    bar_names = ['targets_s1', 'nontargets_s1', 'nontargets_s2']
    if dict_auc == None:
        dict_auc = {x: np.zeros(n_sessions) for x in bar_names}
        for i_s in tqdm(range(n_sessions)):
            lm = lm_list[i_s]
            dict_auc['targets_s1'][i_s] = get_plot_trace(lm, targets=True, region='s1',
                                        absolute_vals=False, type_plot='return_bar',
                                        baseline_by_prestim=baseline_by_prestim)
            dict_auc['nontargets_s1'][i_s] = get_plot_trace(lm, targets=False, region='s1', 
                                        absolute_vals=False, type_plot='return_bar',
                                        baseline_by_prestim=baseline_by_prestim)
            dict_auc['nontargets_s2'][i_s] = get_plot_trace(lm, targets=False, region='s2',
                                        absolute_vals=False, type_plot='return_bar',
                                        baseline_by_prestim=baseline_by_prestim)
    
    if color_dict == None:
        color_dict = color_dict_stand

    mean_arr = np.array([np.mean(dict_auc[bar_name]) for bar_name in bar_names])
    std_arr = np.array([np.std(dict_auc[bar_name]) for bar_name in bar_names])
    ci_arr = std_arr * 1.96 / np.sqrt(n_sessions)

    if ax is None:
        ax = plt.subplot(111)

    ax.bar(np.arange(len(bar_names)), mean_arr, width=0.8, 
           color=[color_dict[ii] for ii in range(len(bar_names))],
           yerr=ci_arr)

    # ax.set_ylabel('AUC')
    ax.set_ylabel('Total ' + r"$\Delta F/F$" + ' response\npost-stimulus (AUC)')
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    # ax.set_xticklabels(['Targets S1', 'Non-targets S1', 'Non-targets S2'])
    despine(ax)

    if plot_legend:
        start_y = mean_arr[0]
        for idx, tt in enumerate(['Targets', 'Non-targets S1', 'Non-targets S2']):
            ax.text(s=tt, x=0.6, va='top',
                    y=start_y - idx * 1.75, fontdict={'color': color_dict[idx]})

    return dict_auc

def plot_accuracy_n_cells_stim(ax=None, subset_dprimes=None, verbose=0, fit_in_logspace=True,
                              midpoint_fit=True, plot_labels=True, min_x=None, max_x=None,
                              translate_to_min=False):
    np.seterr(divide='ignore')  # Ugly division by 0 error
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x_axis = np.array([5, 10, 20, 30, 40, 50, 150])
    if fit_in_logspace:  # if fit is to be computed in logspace, transform x:
        x_axis = np.log10(x_axis)
    all_dp = []
    n = 0

    if min_x is None:
        min_x = np.min(x_axis)
    if max_x is None:
        max_x = np.max(x_axis)
    
    for idx, dp in enumerate(subset_dprimes):

        all_dp.append(dp)

        if translate_to_min:
            # Subtract the min val from the data and then add it
            # back on later, to get the fit below 0 if required
            min_val = np.min(dp)
            dp = dp - min_val
        if fit_in_logspace: # set initial parameter values based on logspace bool
            init_param = [np.max(dp), 1.2, 0.01]
        else:
            init_param = [np.max(dp), 50, 200]
        try:
            popt, pcov = scipy.optimize.curve_fit(pof.pf, x_axis, dp, method='dogbox', p0=init_param)
        except RuntimeError:
            print('WARNING: an individual fit has not converged')
            popt = init_param
        x_arr = np.linspace(min_x, max_x, 1000)
        if translate_to_min:
            dp = dp + min_val
            indiv_fit = pof.pf(x_arr, *popt) + min_val
        else:
            indiv_fit = pof.pf(x_arr, *popt)

        ax.plot(x_axis, dp, '.', color='grey', alpha=0.2)
        ax.plot(x_arr, indiv_fit,
                color='grey', alpha=0.3, label='Individual sessions')

    y = np.concatenate(all_dp)
    x = np.tile(x_axis, subset_dprimes.shape[0])
    if translate_to_min:
        min_val = np.min(y)
        y = y - min_val
    if fit_in_logspace:
        init_param = [np.max(y), 1.2, 0.005]
    else:
        init_param = [np.max(y), 50, 200]
    popt, pcov = scipy.optimize.curve_fit(pof.pf, x, y, method='dogbox', p0=init_param)
    if translate_to_min:
        y = y + min_val
        if verbose:
            print('; min value: ', min_val)
    ## Compute error on midpoint:
    p_std = np.sqrt(np.diag(pcov))  # standard deviations of each parameter fit
    p_95ci = p_std / np.sqrt(len(x)) * 1.96  # convert to 95 ci
    if verbose:
        print('max_val, midpoint, growth:', popt, '95 ci of these', p_95ci)

    x_range = np.linspace(min_x, max_x, 10001)
    fit = pof.pf(x_range, *popt)
    if translate_to_min:
        fit += min_val
    ax.plot(x_range, fit, color='k', label='Average across sessions', linewidth=3)
    ax.set_ylabel('Behavioral accuracy (d\')')
    ax.set_xlabel('Number of cells targeted')
   
    color_mean = 'k'
    if midpoint_fit:  # find midpoint of full fit (that can go beyond data range)
        n_cells_mid = popt[1]
        if fit_in_logspace:
            n_cells_mid_95ci = int(np.round(10 ** p_95ci[1]))
        else:
            n_cells_mid_95ci = int(np.round(p_95ci[1]))
        dprime_mid = pof.pf(n_cells_mid, *popt)
        if translate_to_min:
            dprime_mid += min_val
    else:  # find midpoint of data range
        n_cells_mid, dprime_mid = get_percentile_value(x_range, fit)
    
    if fit_in_logspace:
        ax.set_xticks(ticks=np.log10([5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150]),
                    minor=True)
        ax.set_xticks(ticks=np.log10([10, 100]), minor=False)
        ax.set_xticklabels([''] * 18, minor=True)
        ax.set_xticklabels(['10', '100'])
        ax.vlines(x=n_cells_mid, ymin=ax.get_ylim()[0], ymax=dprime_mid, color=color_mean, ls=':', lw=2)
        ax.hlines(y=dprime_mid, xmin=ax.get_xlim()[0], xmax=n_cells_mid, color=color_mean, ls=':', lw=2)
        if midpoint_fit:
            ax.text(x=n_cells_mid + 0.1, y=-1, s=f'{round(10 ** n_cells_mid)} ' + r"$\pm$" + f' {n_cells_mid_95ci} cells', color=color_mean) 
        else:
            ax.text(x=n_cells_mid + 0.1, y=-1, s=f'{round(10 ** n_cells_mid)} cells', color=color_mean) 
    else:
        ax.vlines(x=n_cells_mid, ymin=ax.get_ylim()[0], ymax=dprime_mid, color=color_mean, ls=':', lw=2)
        ax.hlines(y=dprime_mid, xmin=5, xmax=n_cells_mid, color=color_mean, ls=':', lw=2)
        # ax.set_xscale('log')
        # ax.set_xticks(ticks=[5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150],
        #             minor=True)
        # ax.set_xticks(ticks=[10, 100], minor=False)
        # ax.set_xticklabels([''] * 18, minor=True)
        # ax.set_xticklabels(['10', '100'])
        if midpoint_fit:
            ax.text(x=n_cells_mid + 1, y=-1, s=f'{round(n_cells_mid)}' + r"$\pm$" + f'{n_cells_mid_95ci} cells', color=color_mean)
        else:
            ax.text(x=n_cells_mid + 1, y=-1, s=f'{round(n_cells_mid)} cells', color=color_mean)
  
        if plot_labels:
            xcoord_labels = ax.get_xlim()[0]
            ax.text(x=xcoord_labels, y=3.5, s='Individual sessions', color='grey', alpha=1)
            ax.text(x=xcoord_labels, y=3.2, s='Average across sessions', color=color_mean)
    ax.set_yticks([-1, 0, 1, 2])
    despine(ax)
    
def get_percentile_value(x_range, curve, p=0.5):

    y_point = np.min(curve) + ((np.max(curve) - np.min(curve)) * p)
    x_point = x_range[np.argmin(np.abs(curve - y_point))]
    return x_point, y_point

def plot_accuracy_n_cells_stim_CI(ax=None, subset_dprimes=None):

    if ax is None:
        ax = plt.subplot(111)
    x_axis = [5, 10, 20, 30, 40, 50, 150]
    y = np.concatenate(subset_dprimes)
    x = np.tile(x_axis, subset_dprimes.shape[0])

    min_val = np.min(y)

    y = y - min_val
    popt, pcov = scipy.optimize.curve_fit(pof.pf, x, y, method='dogbox', p0=[np.max(y), 50, 200])

    y = y + min_val
    x_range = np.linspace(np.min(x_axis), np.max(x_axis) + 1, 10001)

    perr = np.sqrt(np.diag(pcov))
    ci_95 = perr * 1.96 / np.sqrt(subset_dprimes.size)
    # The midpoint should be lower for the upper bound and higher for the lower bound
    ci_95[1] = ci_95[1] * -1

    fit = pof.pf(x_range, *popt) + min_val
    bound_upper = pof.pf(x_range, *(popt + ci_95)) + min_val
    bound_lower = pof.pf(x_range, *(popt - ci_95)) + min_val

    ax.plot(x_range, fit, 'black')
    ax.fill_between(x_range, bound_lower, bound_upper,
                     color = 'grey', alpha = 0.5)

    for curve, color in zip([bound_lower, fit, bound_upper], ['grey', 'red', 'grey']):

        n_cells_mid, dprime_mid = get_percentile_value(x_range, curve)
        print(n_cells_mid)
        ax.vlines(x=n_cells_mid, ymin=ax.get_ylim()[0], ymax=dprime_mid, color=color, ls='-')
        ax.hlines(y=dprime_mid, xmin=5, xmax=n_cells_mid, color=color, ls='-')

    plt.xscale('log')

    ax.set_xscale('log')

def lick_hist_all_sessions(lms, ax=None, ax_extra=None, cutoff_time=2000,
                          tt_plot_list=['hit', 'miss', 'fp', 'cr', 'too_', 'urh', 'arm']):
    arr_first_lick_total = np.array([])
    arr_outcome_total = np.array([])
    arr_session_total = np.array([])
    dict_count_no_lick = {x: 0 for x in tt_plot_list}
    dict_count_greater_cutoff = {x: 0 for x in tt_plot_list}
    for i_lm, lm in enumerate(lms):
        session = lm.session
        binned_licks = np.array(session.spiral_lick)[lm.session.nonnan_trials]
        trial_subsets = session.trial_subsets
        outcome = session.outcome
        assert len(binned_licks) == len(outcome)
        assert len(binned_licks) == len(trial_subsets)
        for tt in np.unique(outcome):
            inds_tt = np.where(outcome == tt)[0]
            arr_tt = binned_licks[inds_tt]  # array of trials; for each trial an array of lick times
            arr_first_lick = np.array([(x[0] if len(x) > 0 else np.nan) for x in arr_tt])
            n_no_lick = np.sum(np.array([len(x) for x in arr_tt]) == 0)
            n_greater_cutoff = np.sum(arr_first_lick > cutoff_time)
            if tt in tt_plot_list:  # add no licks trials and greater than cutoff
                dict_count_no_lick[tt] += n_no_lick
                dict_count_greater_cutoff[tt] += n_greater_cutoff
                if tt == 'hit' and n_no_lick > 0:
                    print(lm.session, tt, n_no_lick, np.sum(lm.session.first_lick[inds_tt] == None))
                    # return arr_tt
            arr_outcome = outcome[inds_tt]
            arr_session = [session.name] * len(arr_outcome)
            arr_first_lick_total = np.concatenate((arr_first_lick_total, arr_first_lick))
            arr_outcome_total = np.concatenate((arr_outcome_total, arr_outcome))
            arr_session_total = np.concatenate((arr_session_total, arr_session))
    df = pd.DataFrame({'session': arr_session_total,
                       'first_lick': arr_first_lick_total,
                       'outcome': arr_outcome_total})    
        
    if ax is None:
        ax = plt.subplot(111)
    
    ## Main hist:
    plot_bins = np.arange(0, 2000, 50)
    n, bins, patches= ax.hist([df[df['outcome'] == tt]['first_lick'] for tt in tt_plot_list], 
            bins=plot_bins, stacked=True)
    for i_tt, tt in enumerate(tt_plot_list):
        for patch in patches[i_tt]:
            patch.set_facecolor(color_tt[tt])
        ax.annotate(s=label_tt[tt], xy=(0.8, 0.9 - 0.08 * i_tt),
                    xycoords='axes fraction', color=color_tt[tt],
                    weight='bold')

    despine(ax)
    ax.set_xlabel('Response time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Lick response times of all sessions', weight='bold')

    ## Extra hist:
    if ax_extra is not None:
        print('no lick:', dict_count_no_lick)      
        print('cutoff:', dict_count_greater_cutoff) 
        prev_bottoms = np.array([0, 0])     
        for i_tt, tt in enumerate(tt_plot_list):
            ax_extra.bar(x=[0.6, 1.3], height=[dict_count_greater_cutoff[tt], dict_count_no_lick[tt]],
                    width=0.3, bottom=prev_bottoms, color=[color_tt[tt]] * 2)
            prev_bottoms[0] += dict_count_greater_cutoff[tt]
            prev_bottoms[1] += dict_count_no_lick[tt]
        despine(ax_extra)
        ax_extra.set_ylabel('Frequency')
        ax_extra.set_xticks([0.6, 1.3])
        ax_extra.set_xlim([0.2, 1.8])
        ax_extra.set_xticklabels(['>2000 ms', 'No lick'], rotation=30, ha='right')
    return df

def lick_raster(lm, fig=None, trial_schematic=False):
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    session = lm.session
    binned_licks = np.array(session.spiral_lick)[lm.session.nonnan_trials]
    trial_subsets = session.trial_subsets
    outcome = session.outcome
    assert len(binned_licks) == len(trial_subsets) == len(outcome)

    # # Current putting too_soons back in as hits. You can also
    # # remove them completely
    # ## Its currently called 'too_' for some reason
    # outcome[outcome=='too_'] = 'hit'

    # # Switch out thijs trial definitions for simlicity 
    # outcome[outcome=='urh'] = 'hit'
    # outcome[outcome=='arm'] = 'miss'

    # Sort variables by whether the number of cells stimmed
    tt_plot_list = ['hit', 'miss', 'fp', 'cr']
    trial_idxs = np.isin(outcome, tt_plot_list)
    # trial_idxs = np.argsort(trial_subsets, kind='mergesort') # old, when all tts were used
    inds_sorted_trials = np.argsort(trial_subsets[trial_idxs], kind='mergesort')
    trial_idxs = np.where(trial_idxs)[0][inds_sorted_trials]

    sorted_licks = binned_licks[trial_idxs]
    sorted_outcome = outcome[trial_idxs]
    sorted_subsets = trial_subsets[trial_idxs]
    sorted_subsets = pof.trial_binner(sorted_subsets)
    subsets = ['150', '40-50', '20-30', '5-10', '0']

    # Map a plot color to each subset size
    # colors = ['pink'] * 4
    # colors.append('c')
    # colors = ['black'] * 5
    # color_map = {}
    # for i, sub in enumerate(subsets):
    #     color_map[sub] = colors[i]
    # subset_colors = [color_map[i] for i in sorted_subsets]

    if fig is None:
        fig = plt.figure(figsize=(5,10))
  
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, fig, width_ratios=[4, 1], wspace=0.0) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    # Plot lick rasters
    line_pos = []
    reaction_times = []

    for i, rast in enumerate(sorted_licks):
        # color = subset_colors[i]
        y_axis = np.ones(len(rast)) + i
        if sorted_subsets[i] != sorted_subsets[i - 1] and i != 0:
            ax0.axhline(y=i, ls='--', color='black')
            line_pos.append(i)
        try:
            point = rast[0] / 1000
            if point <= 1 and point >= 0.15:
                marker = '.'
                fillstyle = 'full'
            else:
                marker = '.'
                fillstyle = 'none'
            ax0.plot(point, y_axis[0], marker=marker,
                     fillstyle=fillstyle, c='k', markersize=8.5)  # set c = color for subset dependent colors
            reaction_times.append(rast[0])
        except IndexError:
            reaction_times.append(np.nan)
            pass

    # Get positions of dividing lines so know where to put y tick labels
    line_pos.insert(0, 0)
    line_pos.append(len(sorted_subsets))
    subset_centre = [(line_pos[i] + line_pos[i + 1]) / 2 for i in range(len(line_pos) - 1)]

    # Setup raster axis
    if trial_schematic:
        # added_height_rw = 80
        added_height_rw = 100
    else:
        added_height_rw = 3
    ax0.fill_between([0.15, 1], 0, len(sorted_licks) + added_height_rw, color='gainsboro', alpha=0.5,
                     clip_on=False)
    ax0.set_xlim((0, 1.95))
    ax0.set_xlabel('Time from photostimulation onset (s)')
    ax0.set_ylim((0, len(sorted_licks) + 3))
    ax0.set_yticks(subset_centre)
    ax0.set_yticklabels(np.flip(subsets))
    ax0.tick_params(left=False)
    ax1.tick_params(left=False)# cbar y
    ax0.set_ylabel('Number of cells targeted')
    despine(ax0)
    despine(ax1)

    # Map trial types to ints so can express as colormap
    int_map = {'hit': 3,
               'miss': 2,
               'fp': 1,
               'cr': 0}

    int_mapped = [int_map[out] for out in np.flip(sorted_outcome)]
    bar_width = 10
    bar_data = np.stack([int_mapped for _ in range(bar_width)], axis=1)

    to_rbga = lambda col: hex2color(col)
    cols =  [color_dict_stand[0], color_dict_stand[1],
             CB_color_cycle[7], color_dict_stand[2]]

    cols = [color_tt['cr'], color_tt['fp'], color_tt['miss'], color_tt['hit']]

    cm = matplotlib.colors.LinearSegmentedColormap.from_list('hit_miss', cols, N=4)

    # Build the colorbar
    im = ax1.imshow(bar_data, cmap=cm)
    ax1.set_xticks([])
    trial_scale = [0, len(sorted_outcome)]
    ax1.set_yticks(trial_scale)
    # This gives you the trial number on the axis
#     ax1.set_yticklabels([('#'+str(t)) for t in np.flip(trial_scale)])
    ax1.set_yticklabels([])
    for s in ['top', 'bottom', 'right', 'left']:
        ax1.spines[s].set_visible(False)

    ## Matrix with outcomes:
    mat_x = -4.7
    mat_y = 105 #-15
    ax0.plot([mat_x + 2.4, mat_x + 3.6], [mat_y + 50, mat_y + 50], c='grey', linestyle=':', clip_on=False, zorder=1)
    ax0.plot([mat_x + 2.76, mat_x + 2.76], [mat_y, mat_y + 70], c='grey', linestyle=':', clip_on=False, zorder=1)
    ax0.text(s='Go', x=mat_x + 2.73, y=mat_y + 30, ha='right')
    ax0.text(s='Catch', x=mat_x + 2.73, y=mat_y + 10, ha='right')
    ax0.text(s='Hit', x=mat_x + 2.8, y=mat_y + 30, c=color_tt['hit'])
    ax0.text(s='Miss', x=mat_x + 3.15, y=mat_y + 30, c=color_tt['miss'])
    ax0.text(s='FP', x=mat_x + 2.8, y=mat_y + 10, c=color_tt['fp'])
    ax0.text(s='CR', x=mat_x + 3.15, y=mat_y + 10, c=color_tt['cr'])
    ax0.text(s='Lick', x=mat_x + 2.8, y=mat_y + 55)
    ax0.text(s='No lick', x=mat_x + 3.15, y=mat_y + 55)
    box_behav_mat = matplotlib.patches.Rectangle(xy=(mat_x + 2.24, mat_y - 5),
                                                      width=1.5, height=80, clip_on=False,
                                                      facecolor='none', edgecolor='grey', lw=1)#,
                                        # boxstyle=matplotlib.patches.BoxStyle("Round", pad=0.05))
    ax0.add_patch(box_behav_mat)
    ## Legend dots:
    
    leg_x = -1.3 #-4.5
    leg_y = 35 #-30
    # ax0.plot(leg_x + 2.42, leg_y + 179, marker='.', clip_on=False, zorder=1,
    #         fillstyle='full', c='k', markersize=8.5)
    # ax0.plot(leg_x + 2.42, leg_y + 134, marker='.', clip_on=False, zorder=1,
    #         fillstyle='none', c='k', markersize=8.5)
    
    # ax0.text(s='Lick in\nwindow', x=leg_x + 2.52, y=leg_y + 185, va='top')
    # ax0.text(s='Lick outside\nwindow', x=leg_x + 2.52, y=leg_y + 140, va='top')
    # leg_x = -1.2
    # leg_y = 5
    ax0.plot(leg_x + 2.42, leg_y + 179, marker='.', clip_on=False, zorder=1,
            fillstyle='full', c='k', markersize=8.5)
    ax0.plot(leg_x + 2.42, leg_y + 157, marker='.', clip_on=False, zorder=1,
            fillstyle='none', c='k', markersize=8.5)
    ax0.text(s='Lick responses:', x=leg_x + 2.375, y=leg_y + 205, va='top')
    ax0.text(s='In window', x=leg_x + 2.52, y=leg_y + 185, va='top')
    ax0.text(s='Outside\nwindow', x=leg_x + 2.52, y=leg_y + 165, va='top')

    ## Trial schematic:
    if trial_schematic:
        main_ylims = ax0.get_ylim()
        print(main_ylims)
        bottom_vbars = added_height_rw - 45
        height_vbars = added_height_rw - 3  # because ylim is + 3
        lift_text = (height_vbars  + bottom_vbars) / 2
        left_end = -1.35
        right_end = 2.5
        for xcoord in [left_end, 0, 0.15, 1, right_end]:
            ax0.plot([xcoord, xcoord], [main_ylims[1] + bottom_vbars, 
                                        main_ylims[1] + height_vbars], 
                        c='k', clip_on=False)
            
        for ycoord in [bottom_vbars, height_vbars]:
            ax0.plot([left_end, right_end], [main_ylims[1] + ycoord, main_ylims[1] + ycoord], 
                            c='k', clip_on=False)
        ## Photostim bar in trial structure:
        # ax0.fill_between([0.0, 0.15], main_ylims[1] + bottom_vbars, len(sorted_licks) + added_height_rw, 
        #                 color=color_tt['photostim'], alpha=0.3, clip_on=False)
        ax0.fill_between([0.0, 0.25], main_ylims[1] + 10, main_ylims[1] + 25, 
                        color=color_tt['photostim'], alpha=0.3, clip_on=False)
        ax0.fill_between([0.0, 0.76], main_ylims[1] + 35, main_ylims[1] + 50, 
                        color=color_tt['photostim'], alpha=0.3, clip_on=False)
        ax0.annotate('5-50 cells stim.', xy=(-0.03, main_ylims[1] + 16), c=color_tt['photostim'],
                     xycoords='data', ha='right', va='center', annotation_clip=False)
        ax0.annotate('150 cells stim.', xy=(-0.03, main_ylims[1] + 41),c=color_tt['photostim'],
                     xycoords='data', ha='right', va='center', annotation_clip=False)
        ax0.annotate('Response\nwindow', xy=(0.575, main_ylims[1] + lift_text), 
                     xycoords='data', ha='center', va='center', annotation_clip=False)
        ax0.annotate('Lick withhold\n(4 - 6 seconds)', xy=(left_end / 2, main_ylims[1] + lift_text), 
                     xycoords='data', ha='center', va='center', annotation_clip=False)
        ax0.annotate('Inter-trial interval\n(5 seconds)', xy=((right_end - 1) / 2 + 1, main_ylims[1] + lift_text), 
                     xycoords='data', ha='center', va='center', annotation_clip=False)
        # ax0.annotate('Trial structure', xy=(left_end, main_ylims[1] + height_vbars + 9), weight='bold',
        #              xycoords='data', ha='left', va='center', annotation_clip=False)
        # ax0.annotate('Response times', xy=(left_end, main_ylims[1] + 3), weight='bold', 
        #              xycoords='data', ha='left', va='bottom', annotation_clip=False)



def percent_responding_tts(lm_list, axes=None, verbose=1, 
                            p_val_significant=0.05, bonf_correction=4):
    
    if axes is None:
        _, axes = plt.subplots(1,2)
    
    for idx, (region, ax) in enumerate(zip(['s1', 's2'], axes)):

        data_dict = pof.get_data_dict(region=region, lm_list=lm_list)
        data_df = pd.DataFrame.from_dict(data_dict)

        boxprops = {'facecolor':'none', "zorder":10}
        sns.boxplot(data=data_df, color='black', width=0.35, boxprops=boxprops, ax=ax)
        sns.stripplot(data=data_df, palette=color_tt, ax=ax)

        if region == 's1':
            ax.set_ylabel('Fraction excited\nor inhibited (%)')
            ax.set_ylim(0, 27)        
            y_coord_tt = 26 
        else:
            ax.set_ylim(0, 12)
            ax.set_yticks([0,10], [0,10])
            y_coord_tt = 11.3
        
        alpha = 0.05 / (len(data_dict) - 1)
        if verbose > 0:
            print('\n')
            print(region)
        x_coord_tt = 1

        for tt_against in data_dict.keys():

            if tt_against == 'Hit':
                continue

            _, p = scipy.stats.wilcoxon(data_dict['Hit'], data_dict[tt_against])

            s = f'Hit vs {tt_against}, p = {p}, significant = {p < alpha}'
            if verbose > 0:
                print(s)
            label_sign = asterisk_p(p_val=p, bonf_correction=bonf_correction)
            ax.text(s=label_sign, x=x_coord_tt, y=y_coord_tt, ha='center', va='center')
            x_coord_tt += 1
        ax.tick_params(bottom=False)