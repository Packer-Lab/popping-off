## general imports (also for subsequent analysis notebooks)
import sys
import os
path_to_vape = os.path.expanduser('~/repos/Vape')
sys.path.append(path_to_vape)
sys.path.append(os.path.join(path_to_vape, 'jupyter'))
sys.path.append(os.path.join(path_to_vape, 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import utils_funcs as utils
import run_functions as rf
from subsets_analysis import Subsets
import pickle
import sklearn.decomposition
from cycler import cycler
import pandas as pd
import math, cmath
from tqdm import tqdm
import scipy.stats
from Session import Session  # class that holds all data per session
plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))

def beh_metric(sessions, metric='accuracy',
               stim_array=[0, 5, 10, 20, 30, 40, 50]):
    acc = np.zeros((len(sessions), len(stim_array)))
    for i_session, session in sessions.items():
        for i_stim, stim in enumerate(stim_array):
            trial_inds = np.where(session.trial_subsets == stim)[0]
            tp = np.sum(session.outcome[trial_inds] == 'hit')
            fp = np.sum(session.outcome[trial_inds] == 'fp')
            tn = np.sum(session.outcome[trial_inds] == 'cr')
            fn = np.sum(session.outcome[trial_inds] == 'miss')
            assert (tp + fp + tn + fn) == len(session.outcome[trial_inds])
            if metric == 'accuracy':
                acc[i_session, i_stim] = (tp + tn) / (tp + fp + tn + fn)
            elif metric == 'sensitivity':
                acc[i_session, i_stim] = tp / (tp + fp)
    return acc

def fun_return_2d(data):  # possibly add fancy stuff
    return np.mean(data, 2)

def angle_vecs(v1, v2):
    assert v1.shape == v2.shape
    v1, v2 = np.squeeze(v1), np.squeeze(v2)
    tmp = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    rad = np.arccos(tmp)
    deg = rad * 360 / (2 * np.pi)
    return deg

def mean_angle(deg):
    return math.degrees(cmath.phase(sum([cmath.rect(1, math.radians(d)) for d in deg])/len(deg)))

def create_dict_pred(nl, train_proj, lt):
        dict_predictions_test = {x + '_test': np.array([]) for x in nl}  # make dicts to save
        dict_predictions_train = {x + '_train': np.array([]) for x in nl}
        if train_proj:
            for x in lt:
                dict_predictions_train[f'pred_{x}_train_proj'] = np.array([])
                dict_predictions_test[f'pred_{x}_test_proj'] = np.array([])
        if len(lt) == 2:
            dict_predictions_train['angle_decoders'] = np.array([])
        return dict_predictions_train, dict_predictions_test

def train_test_all_sessions(sessions, trial_times_use=None, verbose=2, list_test = ['dec', 'stim'],
                            hitmiss_only=False, include_150 = False, return_decoder_weights=False,
                            n_split = 4, include_autoreward=True, neurons_selection='all',
                            C_value=0.2, reg_type='l2', train_projected=False, proj_dir='different'):
    if hitmiss_only:
        if verbose >= 1:
            print('Using hit/miss trials only.')
        if 'stim' in list_test:
            list_test.remove('stim')  # no point in estimating stim, because only PS
            
    name_list = ['autorewarded']  # names of details to save - whether autorewrd trial or not
    for nn in list_test:
        name_list.append('pred_' + nn)  # prediction
    for nn in ['dec', 'stim']:
        name_list.append('true_' + nn)  # ground truth
        
    mouse_list = np.unique([ss.mouse for _, ss in sessions.items()])
    df_prediction_train, df_prediction_test = dict(), dict()
    if verbose >= 2:
        print(mouse_list)
    if return_decoder_weights:
        dec_weights = {xx: {} for xx in list_test}
    for mouse in mouse_list:
        angle_decoders = np.zeros((len(sessions), n_split))
        dict_predictions_train, dict_predictions_test = create_dict_pred(nl=name_list, train_proj=train_projected, lt=list_test)
        for i_session, session in sessions.items():  # loop through sessions/runs and concatenate results (in dicts)
            if session.mouse == mouse:  # only evaluate current mouse
                if verbose >= 1:
                    print(f'Mouse {mouse}, Starting loop {i_session + 1}/{len(sessions)}')
                if trial_times_use is None:
                    trial_frames_use = session.filter_ps_array[(session.final_pre_gap_tp + 1):(session.final_pre_gap_tp + 6)]
                    print('WARNING: trial_times undefined so hard-coding them (to 5 post-stim frames)')
                else:
                    trial_frames_use = []
                    for tt in trial_times_use:
                        trial_frames_use.append(session.filter_ps_array[np.where(session.filter_ps_time == tt)[0][0]])  # this will throw an error if tt not in filter_ps_time
                    trial_frames_use = np.array(trial_frames_use)
                    assert len(trial_times_use) == len(trial_frames_use)
                    if verbose >= 2:
                        print(trial_times_use, trial_frames_use)

                ## Set neuron inds
                if neurons_selection == 'all':
                    neurons_include = np.arange(session.behaviour_trials.shape[0])
                elif neurons_selection == 's1':
                    neurons_include = np.where(session.s1_bool)[0]
                elif neurons_selection == 's2':
                    neurons_include = np.where(session.s2_bool)[0]
                if verbose >= 2:
                    print(f'n neurons: {len(neurons_include)}/{session.n_cells}, {neurons_selection}')
                ## Set trial inds
                if include_150 is False:
                    trial_inds = np.where(session.photostim < 2)[0]
                else:
                    trial_inds = np.arange(len(session.photostim))

                if hitmiss_only:
                    hitmiss_trials = np.where(np.logical_or(session.outcome == 'hit', session.outcome == 'miss'))[0]
                    if verbose == 2:
                        print(f'Size hm {hitmiss_trials.size}, trial inds {trial_inds.size}')
                    trial_inds = np.intersect1d(trial_inds, hitmiss_trials)

                if include_autoreward is False:
                    ar_exclude = np.where(session.autorewarded == False)[0]
                    if verbose == 2:
                        print(f'{np.sum(session.autorewarded)} autorewarded trials found and excluded')
                    trial_inds = np.intersect1d(trial_inds, ar_exclude)

                if verbose == 2:
                    print(f'final size {trial_inds.size}')
                n_trials = len(trial_inds)
                if verbose == 2:
                    print(f'Total number of trials is {n_trials}. Number of splits is {n_split}')

                # Prepare data with selections
                data_use = session.behaviour_trials[neurons_include, :, :]
                data_use = data_use[:, :, trial_frames_use]
                data_use = data_use[:, trial_inds, :]
                data_use = fun_return_2d(data_use)
                stand_scale = sklearn.preprocessing.StandardScaler()
                data_use = stand_scale.fit_transform(data_use)
  
                sss = sklearn.model_selection.StratifiedKFold(n_splits=n_split)  # split into n_split data folds of trials
                if verbose == 2:
                    print(f'Number of licks: {np.sum(session.decision[trial_inds])}')
                    dict_outcomes = {x: np.sum(session.outcome[trial_inds] == x) for x in np.unique(session.outcome[trial_inds])}
                    print(f'Possible trial outcomes: {dict_outcomes}')
                    dict_n_ps = {x: np.sum(session.trial_subsets[trial_inds] == x) for x in np.unique(session.trial_subsets[trial_inds])}
                    print(f'Possible stimulations: {dict_n_ps}')
                               
                i_loop = 0
                if return_decoder_weights:
                    for x in list_test:
                        dec_weights[x][session.signature] = np.zeros((n_split, len(neurons_include)))
                for train_inds, test_inds in sss.split(X=np.zeros(n_trials), y=session.outcome[trial_inds]):  # loop through different train/test folds, concat results
                    train_data, test_data = data_use[:, train_inds], data_use[:, test_inds]
                    if i_loop == 0:
                        if verbose == 2:
                            print(f'Shape train data {train_data.shape}, test data {test_data.shape}')

                    ## Get labels and categories of trials
                    train_labels = {'stim': session.photostim[trial_inds[train_inds]],
                                    'dec': session.decision[trial_inds[train_inds]]}
                    test_labels = {'stim': session.photostim[trial_inds[test_inds]], 
                                   'dec': session.decision[trial_inds[test_inds]]}
                    if verbose == 2:
                        print(f' Number of test licks {np.sum(test_labels["dec"])}')
                    detailed_ps_labels = session.trial_subsets[trial_inds].astype('int')
                    autorewarded = session.autorewarded[trial_inds]
                    assert len(train_labels['dec']) == train_data.shape[1]
                    assert len(test_labels['stim']) == test_data.shape[1]

                    ## Train logistic regression model on train data
                    dec = {}
                    for x in list_test:
                        dec[x] = sklearn.linear_model.LogisticRegression(penalty=reg_type, C=C_value, class_weight='balanced').fit(
                                        X=train_data.transpose(), y=train_labels[x])
                        if return_decoder_weights:
                            dec_weights[x][session.signature][i_loop, :] = dec[x].coef_.copy()
                        
                    if len(list_test) == 2:
                        angle_decoders[i_session, i_loop] = angle_vecs(dec[list_test[0]].coef_, dec[list_test[1]].coef_)

                    if train_projected:  # project and re decode
                        dec_proj = {}
                        assert len(list_test) == 2  # hard coded that len==2 further on
                        for i_x, x in enumerate(list_test):
                            i_y = 1 - i_x
                            y = list_test[i_y]
                            assert x != y
                            if proj_dir == 'same':
                                enc_vector = dec[x].coef_ / np.linalg.norm(dec[x].coef_)
                            elif proj_dir == 'different':
                                enc_vector = dec[y].coef_ / np.linalg.norm(dec[y].coef_)
                            train_data_proj = enc_vector.copy() * train_data.transpose()
                            test_data_proj = enc_vector.copy() * test_data.transpose()
                            dec_proj[x] = sklearn.linear_model.LogisticRegression(penalty=reg_type, C=C_value, class_weight='balanced').fit(
                                            X=train_data_proj, y=train_labels[x])

                    ## Predict test data
                    pred_proba_train = {x: dec[x].predict_proba(X=train_data.transpose())[:, 1] for x in list_test}
                    pred_proba_test = {x: dec[x].predict_proba(X=test_data.transpose())[:, 1] for x in list_test}
                    if train_projected:
                        pred_proba_train_proj = {x: dec_proj[x].predict_proba(X=train_data_proj)[:, 1] for x in list_test}
                        pred_proba_test_proj = {x: dec_proj[x].predict_proba(X=test_data_proj)[:, 1] for x in list_test}

                    ## Save results
                    for x in list_test: 
                        dict_predictions_train[f'pred_{x}_train'] = np.concatenate((dict_predictions_train[f'pred_{x}_train'], pred_proba_train[x]))
                        dict_predictions_test[f'pred_{x}_test'] = np.concatenate((dict_predictions_test[f'pred_{x}_test'], pred_proba_test[x]))
                        if train_projected:
                            dict_predictions_train[f'pred_{x}_train_proj'] = np.concatenate((dict_predictions_train[f'pred_{x}_train_proj'], pred_proba_train_proj[x]))
                            dict_predictions_test[f'pred_{x}_test_proj'] = np.concatenate((dict_predictions_test[f'pred_{x}_test_proj'], pred_proba_test_proj[x]))
                    if len(list_test) == 2:
                        dict_predictions_train['angle_decoders'] = np.concatenate((dict_predictions_train['angle_decoders'], np.zeros_like(pred_proba_train[x]) + angle_decoders[i_session, i_loop]))
                    dict_predictions_train['true_stim_train'] = np.concatenate((dict_predictions_train['true_stim_train'], detailed_ps_labels[train_inds]))
                    dict_predictions_test['true_stim_test'] = np.concatenate((dict_predictions_test['true_stim_test'], detailed_ps_labels[test_inds]))
                    dict_predictions_train['autorewarded_train'] = np.concatenate((dict_predictions_train['autorewarded_train'], autorewarded[train_inds]))
                    dict_predictions_test['autorewarded_test'] = np.concatenate((dict_predictions_test['autorewarded_test'], autorewarded[test_inds]))
                    dict_predictions_train['true_dec_train'] = np.concatenate((dict_predictions_train['true_dec_train'], train_labels['dec']))
                    dict_predictions_test['true_dec_test'] = np.concatenate((dict_predictions_test['true_dec_test'], test_labels['dec']))
                    i_loop += 1
        if verbose == 2:
            print(f'length test: {len(dict_predictions_test["true_dec_test"])}')
        ## Put dictionary results into dataframes:
        df_prediction_train[mouse] = pd.DataFrame(dict_predictions_train)
        df_prediction_test[mouse] = pd.DataFrame(dict_predictions_test)
        
    if return_decoder_weights is False:
        return df_prediction_train, df_prediction_test
    elif return_decoder_weights:
        return df_prediction_train, df_prediction_test, dec_weights
    
def plot_df_stats(df, xx, yy, hh, plot_line=True, xticklabels=None,
                  type_scatter='strip', ccolor='grey', aalpha=1):
    """Plot individual trials
    
    Parameters:
    -------------
        type_scatter: str, default='strip
            'strip' for stripplot, 'violin' for violin plot
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
        

## Some functions that can be used as accuracy assessment 
def prob_correct(binary_truth, estimate):    
    """Return probability of correct estimate, where bt = {0, 1} and est = (0, 1)"""
    prob = (binary_truth * estimate + (1 - binary_truth) * (1 - estimate))
    return prob

def mean_accuracy(binary_truth, estimate):
    """Mean accuracy (average over all trials)"""
    assert len(binary_truth) == len(estimate)
    pp = prob_correct(binary_truth=binary_truth, estimate=estimate)
    return np.mean(pp), np.std(pp)

def mean_accuracy_pred(binary_truth, estimate):
    """Mean accuracy with hard >0.5 threshold (average of all trials)"""
    round_est = np.round(estimate)
    return sklearn.metrics.accuracy_score(binary_truth, round_est), 0

def llh(binary_truth, estimate):
    """Log likelihood of all trials"""
    assert len(binary_truth) == len(estimate)
    pp = prob_correct(binary_truth=binary_truth, estimate=estimate)
    llh = np.mean(np.log(np.clip(pp, a_min=1e-3, a_max=1)))
    return llh, 0

def r2_acc(binary_truth, estimate):
    """R2, plainly averaged over all trials (not variance-weighted)"""
    return sklearn.metrics.r2_score(y_true=binary_truth, y_pred=estimate), 0
                  
def separability(binary_truth, estimate):
    """Measure difference between averages P(1) and P(0)."""
    av_pred_0 = np.mean(estimate[binary_truth == 0])
    av_pred_1 = np.mean(estimate[binary_truth == 1])
    sep = av_pred_1 - av_pred_0
    return sep, 0

def min_mean_accuracy(binary_truth, estimate):
    """Minimum of averages P(1) and P(0)"""
    mean_acc_true = np.mean(estimate[binary_truth == 1])
    mean_acc_false = 1 - np.mean(estimate[binary_truth == 0])
    return np.minimum(mean_acc_true, mean_acc_false), 0

def class_av_mean_accuracy(binary_truth, estimate):
    """Mean of averages P(1) and P(0)"""
    mean_acc_true = np.mean(estimate[binary_truth == 1])
    mean_acc_false = 1 - np.mean(estimate[binary_truth == 0])
    return 0.5 * (mean_acc_true + mean_acc_false), 0

## Main function to compute accuracy of decoders per time point
def compute_accuracy_time_array(sessions, time_array, average_fun=class_av_mean_accuracy, reg_type='l2',
                                region_list=['s1', 's2'], regularizer=0.02, projected_data=False):
    """Compute accuracy of decoders for all time steps in time_array, for all sessions (concatenated per mouse)"""
    mouse_list = np.unique([ss.mouse for _, ss in sessions.items()])
    stim_list = [0, 5, 10, 20, 30, 40, 50]  # hard coded!
    dec_list = [0, 1]  # hard_coded!! 
    mouse_s_list = []
    for mouse in mouse_list:
        for reg in region_list:
            mouse_s_list.append(mouse + '_' + reg)
    n_timepoints = len(time_array)
    signature_list = [session.signature for _, session in sessions.items()]
    lick_acc = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} #mean, std
    lick_acc_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in stim_list}  # split per ps conditoin
    lick_half = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list}  # naive with P=0.5 for 2 options (lick={0, 1})
    ps_acc = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} 
    ps_acc_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in dec_list}  # split per lick conditoin
    angle_dec = {mouse: np.zeros(n_timepoints) for mouse in mouse_s_list}
    decoder_weights = {'s1_stim': {session.signature: np.zeros((np.sum(session.s1_bool), len(time_array))) for _, session in sessions.items()},
                       's2_stim': {session.signature: np.zeros((np.sum(session.s2_bool), len(time_array))) for _, session in sessions.items()},
                       's1_dec': {session.signature: np.zeros((np.sum(session.s1_bool), len(time_array))) for _, session in sessions.items()},
                       's2_dec': {session.signature: np.zeros((np.sum(session.s2_bool), len(time_array))) for _, session in sessions.items()}}
    for i_tp, tp in tqdm(enumerate(time_array)):  # time array IN SECONDS
        for reg in region_list:
            df_prediction_train, df_prediction_test, dec_w = train_test_all_sessions(sessions=sessions, trial_times_use=np.array([tp]), 
                                                          verbose=0, hitmiss_only=False, include_150=False,
                                                          include_autoreward=True, C_value=regularizer, reg_type=reg_type,
                                                          train_projected=projected_data, return_decoder_weights=True,
                                                          neurons_selection=reg)
            for xx in ['stim', 'dec']:
                for signat in signature_list:
                    decoder_weights[f'{reg}_{xx}'][signat][:, i_tp] = np.mean(dec_w[xx][signat], 0)
            for mouse in df_prediction_train.keys():
                lick = df_prediction_test[mouse]['true_dec_test'].copy()
                ps = (df_prediction_test[mouse]['true_stim_test'] > 0).astype('int').copy()
                if projected_data is False:
                    pred_lick = df_prediction_test[mouse]['pred_dec_test'].copy()
                else:
                    pred_lick = df_prediction_test[mouse]['pred_dec_test_proj']  
                lick_half[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=lick, estimate=(np.zeros_like(lick) + 0.5))  # control for P=0.5
                lick_acc[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=lick, estimate=pred_lick)
#                 lick_acc[mouse + '_' + reg][i_tp, :] = 0
#                 for i_lick in np.unique(lick):
#                     lick_acc[mouse + '_' + reg][i_tp, :] += np.array(average_fun(binary_truth=lick[lick == i_lick], estimate=pred_lick[lick == i_lick])) / len(np.unique(lick))
                
                for x, arr in lick_acc_split.items():
                    arr[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=lick[np.where(df_prediction_test[mouse]['true_stim_test'] == x)[0]], 
                                              estimate=pred_lick[np.where(df_prediction_test[mouse]['true_stim_test'] == x)[0]])

                if 'pred_stim_test' in df_prediction_test[mouse].columns:
                    if projected_data is False:
                        pred_ps = df_prediction_test[mouse]['pred_stim_test'] 
                    else:
                        pred_ps = df_prediction_test[mouse]['pred_stim_test_proj'] 
                    ps_acc[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=ps, estimate=pred_ps)
#                     ps_acc[mouse + '_' + reg][i_tp, :] = 0
#                     for i_ps in np.unique(lick):
#                         ps_acc[mouse + '_' + reg][i_tp, :] += np.array(average_fun(binary_truth=ps[lick == i_ps], estimate=pred_ps[lick == i_ps])) / len(np.unique(lick))

                    for x, arr in ps_acc_split.items():
                        arr[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=ps[lick == x], 
                                                  estimate=pred_ps[lick == x])
                angle_dec[mouse + '_' + reg][i_tp] = np.mean(df_prediction_train[mouse]['angle_decoders'])
                
    return (lick_acc, lick_acc_split, ps_acc, ps_acc_split, lick_half, angle_dec, decoder_weights)

## Create list with standard colors:
color_dict_stand = {}
for ii, x in enumerate(plt.rcParams['axes.prop_cycle']()):
    color_dict_stand[ii] = x['color']
    if ii > 8:
        break  # after 8 it repeats (for ever)
    
def plot_interrupted_trace_simple(ax, time_array, plot_array, llabel='', ccolor='grey',
                                 linest='-', aalpha=1, zero_mean=False, llinewidth=3):
    """Plot plot_array vs time_array, where time_array has 1 gap, which is not plotted."""
    assert len(time_array) == len(plot_array)
    breakpoint = np.argmax(np.diff(time_array)) + 1# finds the 0, equivalent to art_gap_start
    if zero_mean:
        plot_array = plot_array - np.mean(plot_array)
    ax.plot(time_array[:breakpoint], plot_array[:breakpoint],  linewidth=llinewidth, linestyle=linest, 
                markersize=12, color=ccolor, label=llabel, alpha=aalpha)
    ax.plot(time_array[breakpoint:], plot_array[breakpoint:],  linewidth=llinewidth, linestyle=linest, 
                markersize=12, color=ccolor, alpha=aalpha, label=None)
    return ax
    
    
    
def plot_interrupted_trace(ax, time_array, plot_array, llabel='', bool_plot_std=False,
                           plot_laser=True, ccolor='grey', plot_indiv=True,
                           plot_groupav=True, individual_mouse_list=None,
                           plot_errorbar=False, plot_std_area=False, region_list=['s1', 's2'],
                           plot_diff_s1s2=False, freq=5):
    """Same as plot_interrupted_trace_simple(), but customised to plot_array being a dictionary
    of individual mouse traces. Can plot individual traces & group average."""
    breakpoint = np.argmax(np.diff(time_array)) + 1# finds the 0, equivalent to art_gap_start
    time_1 = time_array[:breakpoint]  # time before & after PS
    time_2 = time_array[breakpoint:]
    mouse_list = list(plot_array.keys())  # all data sets (including _s1 and _s2 )
    region_list = np.array(region_list)
    if plot_diff_s1s2:
        assert len(region_list) == 2
    if individual_mouse_list is None:
        individual_mouse_list = mouse_list
    linest = {'s1': '-', 's2': '--'}
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
                                        markersize=12, color=ccolor, label=llabel + f' {rr.upper()}', alpha=0.9)
                        ax.plot(time_2, av_mean[breakpoint:], linewidth=4, linestyle=linest[rr], 
                                    markersize=12, color=ccolor, alpha=0.9, label=None)
                    elif plot_errorbar is True:  # plot group means with error bars
                        ax.errorbar(time_1, av_mean[:breakpoint], yerr=std_means[:breakpoint], linewidth=4, linestyle=linest[rr], 
                                        markersize=12, color=ccolor, label=llabel + f' {rr.upper()}', alpha=0.9)
                        ax.errorbar(time_2, av_mean[breakpoint:], yerr=std_means[breakpoint:], linewidth=4, linestyle=linest[rr], 
                                    markersize=12, color=ccolor, alpha=0.9, label=None)
                    if plot_std_area:  # plot std area
                        ax.fill_between(x=time_1, y1=av_mean[:breakpoint] - std_means[:breakpoint],
                                        y2=av_mean[:breakpoint] + std_means[:breakpoint], color=ccolor, alpha=0.1, 
                                        label=f'Group std {llabel}')#, hatch=region_hatch[rr])
                        ax.fill_between(x=time_2, y1=av_mean[breakpoint:] - std_means[breakpoint:],
                                       y2=av_mean[breakpoint:] + std_means[breakpoint:], color=ccolor, alpha=0.1, 
                                        label=None)#, hatch=region_hatch[rr])
        elif plot_diff_s1s2:
            assert (region_list == np.array(['s1', 's2'])).all() and len(region_list) == len(average_mean)
            diff_mean = average_mean['s1'] - average_mean['s2']
            ax.plot(time_1, diff_mean[:breakpoint],  linewidth=4, linestyle='-', 
                        markersize=12, color=ccolor, label=f'S1 - S2 diff. {llabel}', alpha=0.9)
            ax.plot(time_2, diff_mean[breakpoint:], linewidth=4, linestyle='-', 
                        markersize=12, color=ccolor, alpha=0.9, label=None)
    if len(region_list) == 2:
        assert count_means[region_list[0]] == count_means[region_list[1]]
    if plot_laser:  # plot laser
        ax.axvspan(xmin=time_1[-1] + 1 / freq, xmax=time_2[0] - 1 / freq, ymin=0.1, ymax=0.9, alpha=0.2, label=None)
    return ax, average_mean

def wilcoxon_test(acc_dict):
    """Perform wilcoxon signed rank test for dictionoary of S1/S2 measurements. Each
    S1/S2 pair per mouse is a paired sample for the test. Perform test on each time point."""
    reg_mouse_list = list(acc_dict.keys())
    mouse_list = np.unique([xx[:-3] for xx in reg_mouse_list])
    reg_list = ['s1', 's2']
    mouse_s1_list = [mouse + '_s1' for mouse in mouse_list]
    mouse_s2_list = [mouse + '_s2' for mouse in mouse_list]
    
    n_tp = acc_dict[reg_mouse_list[0]].shape[0]
    p_vals = np.zeros(n_tp)
    for tp in range(n_tp):
        if acc_dict[reg_mouse_list[0]].ndim == 2:
            s1_array = [acc_dict[ms1][tp, 0] for ms1 in mouse_s1_list]
            s2_array = [acc_dict[ms2][tp, 0] for ms2 in mouse_s2_list]
        elif acc_dict[reg_mouse_list[0]].ndim == 1:
            s1_array = [acc_dict[ms1][tp] for ms1 in mouse_s1_list]
            s2_array = [acc_dict[ms2][tp] for ms2 in mouse_s2_list]

        stat, pval = scipy.stats.wilcoxon(x=s1_array, y=s2_array, alternative='two-sided')
        p_vals[tp] = pval.copy()
    return p_vals

def make_violin_df_custom(input_dict_df, flat_normalise_ntrials=False, verbose=0):
    """Function to turn my custom dictionary structure of DF to a single DF, suitable for a violin plot.
    
    Parameters:
    ---------------
        input_dict_df: dcit with structure [reg][tp][mouse]
            Assuming all mice/tp/reg combinations (regular grid)
        flat_normalise_ntrials: bool, default=False
            whether to normalise mice by number of trials. This can be required by the group averaging
            of decoders also ignores number of trials per mouse. If true, this is achieved by 
            creating multiple copies of each mouse such that in the end each mouse has approximately
            the same number of trials
        verbose: int, default=0
            verbosity index
    
    Returns:
    ---------------
    new_df: dict with structure [tp]
    """
    dict_df = input_dict_df.copy()
    region_list = list(dict_df.keys())
    assert (region_list == ['s1', 's2'])
    timepoints = list(dict_df[region_list[0]].keys())
    mouse_list = list(dict_df[region_list[0]][timepoints[0]].keys())
    n_multi = {}
    mouse_multi = 1
    ## add labels:
    for reg in region_list:
        for tp in timepoints:
            for mouse in mouse_list:
                dict_df[reg][tp][mouse]['region'] = reg.upper()
                dict_df[reg][tp][mouse]['mouse'] = mouse
                dict_df[reg][tp][mouse]['n_trials_mouse'] = len(dict_df[reg][tp][mouse])
                if flat_normalise_ntrials and mouse_multi:
                    n_multi[mouse] = np.round(10000 / len(dict_df[reg][tp][mouse])).astype('int')
                    assert n_multi[mouse] >= 10  # require at least 10 multiplications (so that relative error <10% for 1 mouse)
                    if verbose:
                        print(f'Number of trials for mouse {mouse}: {len(dict_df[reg][tp][mouse])}, multiplications: {np.round(10000 / len(dict_df[reg][tp][mouse]), 2)}')               
            mouse_multi = 0 #  after first iteration
    ## Concatenate:
    new_df = {}
    for tp in timepoints:
        if flat_normalise_ntrials is False:
            new_df[tp] = pd.concat([dict_df['s1'][tp][mouse] for mouse in mouse_list] + 
                                   [dict_df['s2'][tp][mouse] for mouse in mouse_list])
        elif flat_normalise_ntrials:
            new_df[tp] = pd.concat([pd.concat([dict_df['s1'][tp][mouse] for x in range(n_multi[mouse])]) for mouse in mouse_list] + 
                                   [pd.concat([dict_df['s2'][tp][mouse] for x in range(n_multi[mouse])]) for mouse in mouse_list])  
    if verbose:
        for mouse in mouse_list:
            print(f'Corrected number of trials for mouse {mouse}: {len(new_df[1.0][new_df[1.0]["mouse"] == mouse])}')
    return new_df









