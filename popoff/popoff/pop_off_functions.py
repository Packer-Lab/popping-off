## general imports (also for subsequent analysis notebooks)
import sys
import os
import loadpaths
user_paths_dict = loadpaths.loadpaths()
path_to_vape = user_paths_dict['vape_path']
sys.path.append(str(path_to_vape))
sys.path.append(str(os.path.join(path_to_vape, 'jupyter')))
sys.path.append(str(os.path.join(path_to_vape, 'utils')))
sys.path.append(user_paths_dict['base_path'])
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# import utils_funcs as utils
# import run_functions as rf
# from subsets_analysis import Subsets
import pickle, copy
import sklearn.decomposition, sklearn.discriminant_analysis
from cycler import cycler
import pandas as pd
import math, cmath
from tqdm import tqdm
import scipy.stats, scipy.spatial
from statsmodels.stats import multitest
import statsmodels.api as sm
from Session import Session  # class that holds all data per session
import pop_off_plotting as pop
import utils  # from Vape
plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))

def save_figure(name, base_path='/home/jrowland/mnt/qnap/Figures/bois'):
    plt.rcParams['pdf.fonttype'] = 42
    plt.savefig(os.path.join(base_path, f'{name}.pdf'),
                bbox_inches='tight', transparent=True)

def beh_metric(sessions, metric='accuracy',
               stim_array=[0, 5, 10, 20, 30, 40, 50]):
    """Compute metric to quantify behavioural performance for sessions.

    Parameters
    ----------
    sessions : dict of Sessions
        all sessions.
    metric : str, default='accuracy'
        what metric to compute. possibilities; 'accuracy' and 'sensitivity'.
    stim_array : list, default=[0, 5, 10, 20, 30, 40, 50]
        list of number of cells PS.

    Returns
    -------
    acc: np.array of size (n_sessions x n_stims)
        Array with metric per stim.

    """
    acc = np.zeros((len(sessions), len(stim_array)))
    for i_session, session in sessions.items():
        for i_stim, stim in enumerate(stim_array):
            trial_inds = np.where(session.trial_subsets == stim)[0]
#             if len(trial_inds) == 0:  # if no trials have this stimulus
#                 continue
            tp = np.sum(session.outcome[trial_inds] == 'hit')
            fp = np.sum(session.outcome[trial_inds] == 'fp')
            tn = np.sum(session.outcome[trial_inds] == 'cr')
            fn = np.sum(session.outcome[trial_inds] == 'miss')
            too_early = np.sum(session.outcome[trial_inds] == 'too_')
            arm = np.sum(session.outcome[trial_inds] == 'arm')
            urh = np.sum(session.outcome[trial_inds] == 'urh')
            assert (tp + fp + tn + fn + too_early + arm + urh) == len(session.outcome[trial_inds]), f'{np.unique(session.outcome[trial_inds])}'
            if metric == 'accuracy':
                acc[i_session, i_stim] = (tp + tn) / (tp + fp + tn + fn)
            elif metric == 'sensitivity':
                acc[i_session, i_stim] = tp.copy() / (tp.copy() + fp.copy())
    return acc

def fun_return_2d(data):  # possibly add fancy stuff
    """Function that determines how multiple time points are handled in train_test_all_sessions().

    Parameters
    ----------
    data : 3D np array, last dim = Time
        neural data.

    Returns
    -------
    2D np.array
        where Time is averaged out.

    """
    return np.mean(data, 2)

def angle_vecs(v1, v2):
    """Compute angle between two vectors with cosine similarity.

    Parameters
    ----------
    v1 : np.array
        vector 1.
    v2 : np.array
        vector 2.

    Returns
    -------
    deg: float
        angle in degrees .

    """
    assert v1.shape == v2.shape
    v1, v2 = np.squeeze(v1), np.squeeze(v2)
    tmp = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    rad = np.arccos(tmp)
    deg = rad * 360 / (2 * np.pi)
    return deg

def mean_angle(deg):
    """Average angles (take periodicity into account).

    Parameters
    ----------
    deg : np.array of floats
        angles

    Returns
    -------
    mean
        mean of angles .

    """
    return math.degrees(cmath.phase(sum([cmath.rect(1, math.radians(d)) for d in deg])/len(deg)))

def create_dict_pred(nl, train_proj, lt):
    """Create dictionaries to save decoders predictions (used in train_test_all_sessions()).

    Parameters
    ----------
    nl : list
        name list of items to save.
    train_proj : bool
        whether to create dictionary keys for projected data.
    lt : list
        list of decoder names (e.g. ['stim', 'dec']).

    Returns
    -------
    dict_predictions_train
        dictionary for training data
    dict_predictions_test
        dictionary for test data

    """
    dict_predictions_test = {x + '_test': np.array([]) for x in nl}  # make dicts to save
    dict_predictions_train = {x + '_train': np.array([]) for x in nl}
    if train_proj:
        for x in lt:
            dict_predictions_train[f'pred_{x}_train_proj'] = np.array([])
            dict_predictions_test[f'pred_{x}_test_proj'] = np.array([])
    if len(lt) == 2:
        dict_predictions_train['angle_decoders'] = np.array([])
    return dict_predictions_train, dict_predictions_test

def train_test_all_sessions(sessions, trial_times_use=None, verbose=2, list_test=['dec', 'stim'],
                            list_tt_training=['hit', 'miss', 'fp', 'cr', 'spont'], include_150=False,
                            return_decoder_weights=False, zscore_data=False,
                            n_split=4, include_autoreward=False, include_unrewardedhit=False,
                            neurons_selection='all', include_too_early=False,
                            C_value=0.2, reg_type='l2', train_projected=False, proj_dir='different',
                            concatenate_sessions_per_mouse=True, hard_set_10_trials=False,
                            list_save_covs=[], equalize_n_trials_per_tt=True):
    """Major function that trains the decoders. It trains the decoders specified in
    list_test, for the time trial_times_use, for all sessions in sessions. The trials
    are folded n_split times (stratified over session.outcome), new decoders
    are trained for each fold and the test results  for each fold are concatenated.
    It returns pd.Dataframes with predictions, and optionally the decoder weights (if return_decoder_weights is True)

    Parameters
    ----------
    sessions : dict
        dictionary of sessions
    trial_times_use : np.array
        np.array of time points to use. len(trial_times_use) > 1, its elements are averaged with fun_return_2d()
    verbose : int, default=2
        verbosiness
    list_test : list, default=['dec' ,'stim']
        list of decoder name
    include_150 : bool, default=False
        if true, include n_PS=150 trials
    return_decoder_weights : bool, default=False
        if True, also return decoder weights
    n_split : int, default=4
        number of data Folds
    include_autoreward : bool, default=True
        if True, include autoreward trials
    neurons_selection : str:, default='all'
        which neurons to include. possibilities: 'all', 's1', 's2'
    C_value : float, default=0.2
        regularisation value (if reg_type is an adeqaute regulariser type for sklearn.linear_model.LogisticRegression)
    reg_type : str, default='l2'
        regulariser type, possibilities: 'l2', 'l1', 'none' (and elastic net I think)
    train_projected : bool, default=False
        if True, also evaluate decoders on projected data
    proj_dir : str. default'different
        if train_projected is True, specifies on which axis to project. possibilities: 'different' or 'same'

    Returns
    -------
    df_prediction_train: pd.DataFrame
        train data predictions
    df_prediction_test: pd.DataFrame
        test data predictions
    if return_decoder_weights:
        dec_weights: dict
            weights of all decoders

    """
    if train_projected:
        print("TRAINING Projected")
    if 'spont' in list_tt_training:
        spont_used_for_training = True 
    else:
        spont_used_for_training = False

    if hard_set_10_trials and False:
        print('WARNING: only using 10 trials per trial type!!')

    ## If trial types used for training are all of the same response type for either stim or dec, do not train that decoder
    dict_response_matrix = {'hit': [1, 1], 'miss': [1, 0], 'fp': [0, 1], 'cr': [0, 0], 'spont': [0, 1]}
    stim_response_list, dec_response_list = [], []
    for tt in list_tt_training:
        stim_response_list.append(dict_response_matrix[tt][0])
        dec_response_list.append(dict_response_matrix[tt][1])
    list_test = ['dec', 'stim']
    if len(np.unique(stim_response_list)) < 2 and 'stim' in list_test:
        list_test.remove('stim')
    if len(np.unique(dec_response_list)) < 2 and 'dec' in list_test:
        list_test.remove('dec')
 
    name_list = ['autorewarded_miss', 'unrewarded_hit', 'outcome']  # names of details to save - whether autorewrd trial or not
    for nn in list_test:
        name_list.append('pred_' + nn)  # prediction
    for nn in ['dec', 'stim', 'reward']:
        name_list.append('true_' + nn)  # ground truth

    if len(list_save_covs) > 0:  # names of covariates to be saved
        for name_cov in list_save_covs:
            name_list.append(name_cov)

    df_prediction_train, df_prediction_test = dict(), dict()
    if return_decoder_weights:
        dec_weights = {xx: {} for xx in list_test}
    
    if concatenate_sessions_per_mouse:
        mouse_list = np.unique([ss.mouse for _, ss in sessions.items()])
    else:
        mouse_list = [ss.signature for ss in sessions.values()]    

    if equalize_n_trials_per_tt:
        ## Computing distr of lick times of reward only trials across all mice
        ## Can be used later when sub sampling if hard_set_10_trials == True
        all_spont_lick_times = np.array([])  # will be distr of all lickt times of reward only trials
        for ss in sessions.values():
            all_spont_lick_times = np.concatenate((all_spont_lick_times, ss.first_lick_spont))
        assert len(all_spont_lick_times) == (len(sessions) * 10)
    
    angle_decoders = np.zeros((len(sessions), n_split))
    for mouse in mouse_list:
        dict_predictions_train, dict_predictions_test = create_dict_pred(nl=name_list, train_proj=train_projected, lt=list_test)
        dict_predictions_test['used_for_training'] = np.array([])
        if concatenate_sessions_per_mouse:  # get sessions that correspond to this mouse identity 
            curr_sessions = {i_session: session for i_session, session in sessions.items() if session.mouse == mouse}
        else:  # get the one session that corresponds to this session signature (called mouse_list for historical reasons)
            curr_sessions = {i_session: session for i_session, session in sessions.items() if session.signature == mouse}
        for i_session, session in curr_sessions.items():  # loop through sessions/runs and concatenate results (in dicts)
            
            # if session.mouse == mouse:
            if concatenate_sessions_per_mouse:
                assert session.mouse == mouse
            else:
                assert session.signature == mouse 
                assert len(curr_sessions) == 1
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
            ## trial_inds: used for training & testing
            ## eval_only_inds: only used for testing (Auto Rew Miss; Un Rew Hit)
            trial_inds = np.array([], dtype='int')
            for tt in list_tt_training:
                curr_tt_trials = np.where(session.outcome == tt)[0]
                trial_inds = np.concatenate((trial_inds, curr_tt_trials))

            if include_150 is False:
                trial_inds = np.intersect1d(trial_inds, np.where(session.photostim < 2)[0])
            else:
                print('150 n_stim is Used!!')
            if include_autoreward is False:
                ar_exclude = np.where(session.autorewarded == False)[0]
                if verbose == 2:
                    print(f'{np.sum(session.autorewarded)} autorewarded trials found and excluded')
                trial_inds = np.intersect1d(trial_inds, ar_exclude)
            else:
                print('WARNING: ARM not excluded!')

            if include_unrewardedhit is False:
                uhr_excluded = np.where(session.unrewarded_hits == False)[0]
                if verbose == 2:
                    print(f'{np.sum(session.unrewarded_hits)} unrewarded_hits found and excluded')
                trial_inds = np.intersect1d(trial_inds, uhr_excluded)
            else:
                print('WARNING: URH not excluded!')

            if include_too_early is False:
                too_early_excl = np.where(session.outcome != 'too_')[0]
                trial_inds = np.intersect1d(trial_inds, too_early_excl)
            else:
                print('WARNING: too early not excluded!')

            if equalize_n_trials_per_tt:
                # print('start', len(trial_inds))#, session.outcome[trial_inds])
                dict_trials_per_tt = {x: np.where(session.outcome[trial_inds] == x)[0] for x in list_tt_training if x is not 'spont'}
                dict_firstlick_per_tt = {x: session.first_lick[trial_inds][dict_trials_per_tt[x]] for x in list_tt_training if x is not 'spont'}
                # dict_firstlick_per_tt['spont'] = session.first_lick_spont  # add manually

                if 'spont' not in list_tt_training:
                    if hard_set_10_trials is False:
                        min_n_trials = np.min([len(v) for v in dict_trials_per_tt.values()])
                    elif hard_set_10_trials:
                        min_n_trials = 10  # use to control for n_trials of spont
                        # print('only using 10 trials per trial type!!')  # always give warning
                elif 'spont' in list_tt_training and 'hit' in list_tt_training and len(list_tt_training) == 2:
                    min_n_trials = 10
                else:
                    # assert 'spont' not in list_tt_training, 'if spont is used for training, this results in a slight bias towards no-PS'
                    min_n_trials = np.minimum(np.min([len(v) for v in dict_trials_per_tt.values()]), 10)
                    if min_n_trials != 10:
                        print(f'{min_n_trials} trials')
                new_trial_inds = np.array([], dtype='int')
                for tt, v in dict_trials_per_tt.items():  # loop through trial type + all corresponding trial inds
                    # new_trial_inds = np.concatenate((new_trial_inds, v[-min_n_trials:]))  # late trials subsampe (or early with :min_n_trials)
                    if tt == 'hit':
                        sample_according_to_spont_lick_time_distr = True
                    else:  # no need to match for CR etc
                        sample_according_to_spont_lick_time_distr = False

                    if sample_according_to_spont_lick_time_distr:
                        sorted_v, new_density_v = pop.subsample_lick_times(truth_lick_times=all_spont_lick_times,  # use distr across all mice to get better distr (because of low number of reward-only trials per recording, which can lead to large zero-density gaps in distr)
                                                                            # truth_lick_times=dict_firstlick_per_tt['spont'],  # alternative; only use lick times of this recording
                                                                            sampled_lick_times=dict_firstlick_per_tt[tt],
                                                                            sampled_data=v, n_bins=5)
                        new_trial_inds = np.concatenate((new_trial_inds, np.random.choice(a=sorted_v, size=min_n_trials, replace=False, p=new_density_v)))  # random subsample of trials, using density of spont lick times
                    else:
                        new_trial_inds = np.concatenate((new_trial_inds, np.random.choice(a=v, size=min_n_trials, replace=False)))  # random subsample of trials
                trial_inds = trial_inds[new_trial_inds]
                # print('end', len(trial_inds))#, session.outcome[trial_inds])
            
            ## set evaluation only indices
            eval_only_inds = np.concatenate((np.where(session.autorewarded == True)[0],
                                                np.where(session.unrewarded_hits == True)[0]))
            eval_only_labels = ['arm'] * np.sum(session.autorewarded) + ['urh'] * np.sum(session.unrewarded_hits)
            assert len(eval_only_inds) == np.sum(session.autorewarded) + np.sum(session.unrewarded_hits)

            for tt in ['hit', 'miss', 'fp', 'cr']:
                if tt not in list_tt_training:
                    eval_only_inds = np.concatenate((eval_only_inds, np.where(session.outcome == tt)[0]))
                    eval_only_labels = eval_only_labels + [tt] * len(np.where(session.outcome == tt)[0])
            trial_outcomes = session.outcome[trial_inds]

            ## Prepare data with selections

            ## Retrieve normalized data:
            (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2, data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
                time_ticks, time_tick_labels, time_axis) = pop.normalise_raster_data(session, sort_neurons=False, start_time=session.filter_ps_time.min(), 
                                                                                     end_time=session.filter_ps_time.max(), filter_150_stim=False)
            assert data_use_mat_norm.shape[1] == session.behaviour_trials.shape[1], (data_use_mat_norm.shape, session.behaviour_trials.shape, len(trial_inds))
            ## Filter neurons
            data_use = data_use_mat_norm[neurons_include, :, :]
            data_eval = data_use_mat_norm[neurons_include, :, :]
            data_spont = data_spont_mat_norm[neurons_include, :, :]

            ## Select time frame(s)
            data_use = data_use[:, :, trial_frames_use]
            data_eval = data_eval[:, :, trial_frames_use]
            data_spont = data_spont[:, :, trial_frames_use]  # use all trials

            ## Select trials
            data_use = data_use[:, trial_inds, :]
            assert len(eval_only_inds) > 0, f'ERROR: {session} has no eval-only trials, which has not been really taken care of here'
            data_eval = data_eval[:, eval_only_inds, :]
            assert data_spont.ndim == 3
            n_spont_trials = data_spont.shape[1]
            assert n_spont_trials == 10 or n_spont_trials == 9
            if n_spont_trials == 0:
                print('NO SPONT TRIALS in ', session)

            cov_dict = {}
            if spont_used_for_training:
                data_use = np.hstack((data_use, data_spont))
                trial_outcomes = np.concatenate((trial_outcomes, ['spont'] * n_spont_trials))
                stim_trials = np.concatenate((session.photostim[trial_inds], np.zeros(n_spont_trials, dtype='int')))
                dec_trials = np.concatenate((session.decision[trial_inds], np.ones(n_spont_trials, dtype='int')))
                detailed_ps_labels = np.concatenate((session.trial_subsets[trial_inds].astype('int'), np.zeros(n_spont_trials, dtype='int')))
                rewarded_trials = np.concatenate((np.array([x in ['hit', 'too_', 'arm'] for x in session.outcome[trial_inds]]), np.ones(n_spont_trials, dtype='int')))
                autorewarded = np.concatenate((session.autorewarded[trial_inds], np.zeros(n_spont_trials, dtype='bool')))
                rewarded_trials[autorewarded] = True
                unrewarded_hits = np.concatenate((session.unrewarded_hits[trial_inds], np.zeros(n_spont_trials, dtype='bool')))
                rewarded_trials[unrewarded_hits] = False
                if len(list_save_covs) > 0:
                    for name_cov in list_save_covs:
                        cov_dict[name_cov] = np.concatenate((session.cov_dict[name_cov][trial_inds], session.cov_dict_reward_only[name_cov]))
            else:
                cov_dict_reward_only = {}
                stim_trials = session.photostim[trial_inds]
                dec_trials = session.decision[trial_inds]
                detailed_ps_labels = session.trial_subsets[trial_inds].astype('int')
                rewarded_trials = np.array([x in ['hit', 'too_', 'arm'] for x in session.outcome[trial_inds]])
                autorewarded = session.autorewarded[trial_inds]
                rewarded_trials[autorewarded] = True
                unrewarded_hits = session.unrewarded_hits[trial_inds]
                rewarded_trials[unrewarded_hits] = False
                if len(list_save_covs) > 0:
                    for name_cov in list_save_covs:
                        cov_dict[name_cov] = session.cov_dict[name_cov][trial_inds]
                        cov_dict_reward_only[name_cov] = session.cov_dict_reward_only[name_cov]

            assert len(trial_outcomes) == data_use.shape[1]
            ## Squeeze time frames
            data_use = fun_return_2d(data_use)
            data_eval = fun_return_2d(data_eval)
            data_spont = fun_return_2d(data_spont)

            ## Stack & fit scaler
            if zscore_data:
                assert False, 'z score is used'
                if spont_used_for_training:
                    data_stacked = np.hstack((data_use, data_eval))  # stack for scaling (spont already included)
                    assert (data_stacked[:, (len(trial_inds) + n_spont_trials):(len(trial_inds) + len(eval_only_inds) + n_spont_trials)] == data_eval).all()
                else:
                    data_stacked = np.hstack((data_use, data_eval, data_spont))  # stack for scaling
                    assert (data_stacked[:, len(trial_inds):(len(trial_inds) + len(eval_only_inds))] == data_eval).all()
                stand_scale = sklearn.preprocessing.StandardScaler().fit(data_stacked.T) # use all data to fit scaler, then scale indiviudally
                ## Scale
                data_use = stand_scale.transform(data_use.T)
                data_eval = stand_scale.transform(data_eval.T)
                data_spont = stand_scale.transform(data_spont.T)
                data_use = data_use.T
                data_eval = data_eval.T
                data_spont = data_spont.T

            sss = sklearn.model_selection.StratifiedKFold(n_splits=n_split)  # split into n_split data folds of trials (strat shuffle split is not appropriate generally because it changes the test set)
            if verbose == 2:
                if np.abs(trial_times_use[0] + 3) < 0.1:
                    print(f'Number of licks: {np.sum(session.decision[trial_inds])}')
                    dict_outcomes = {x: np.sum(trial_outcomes == x) for x in np.unique(trial_outcomes)}
                    print(f'Possible trial outcomes: {dict_outcomes}')
                    dict_n_ps = {x: np.sum(session.trial_subsets[trial_inds] == x) for x in np.unique(session.trial_subsets[trial_inds])}
                    print(f'Possible stimulations: {dict_n_ps}')

            i_loop = 0
            if return_decoder_weights:
                for x in list_test:
                    dec_weights[x][session.signature] = np.zeros((n_split, len(neurons_include)))

            n_trials = data_use.shape[1]
            if verbose == 2:
                print(f'Total number of trials is {n_trials}. Number of splits is {n_split}')

            pred_proba_eval = {x: {} for x in range(n_split)} # dict per cv loop, average later.
            pred_proba_spont = {x: {} for x in range(n_split)}
            for train_inds, test_inds in sss.split(X=np.zeros(n_trials), y=trial_outcomes):  # loop through different train/test folds, concat results
                train_data, test_data = data_use[:, train_inds], data_use[:, test_inds]
                if i_loop == 0:
                    if verbose == 2:
                        print(f'Shape train data {train_data.shape}, test data {test_data.shape}')

                ## Get labels and categories of trials
                train_labels = {'stim': stim_trials[train_inds],
                                'dec': dec_trials[train_inds]}
                test_labels = {'stim': stim_trials[test_inds],
                                'dec': dec_trials[test_inds]}
                if verbose == 2:
                    print(f' Number of test licks {np.sum(test_labels["dec"])}')
                assert len(train_labels['dec']) == train_data.shape[1]
                assert len(test_labels['stim']) == test_data.shape[1]

                ## Train logistic regression model on train data
                dec = {}
                for x in list_test:
                    assert len(np.unique(train_labels[x])) == 2 , f'{x} training will be perfect'
                    assert len(np.unique(test_labels[x])) == 2, 'not stricitly necessary, could be loosened'
                    # cw_dict = {ww: np.sum(train_labels[x] == ww) / len(train_labels[x]) for ww in [0, 1]}
                    dec[x] = sklearn.linear_model.LogisticRegression(penalty=reg_type, C=C_value, class_weight='balanced').fit(
                                    X=train_data.transpose(), y=train_labels[x])
                    # print(train_labels[x])
                    if return_decoder_weights:
                        dec_weights[x][session.signature][i_loop, :] = dec[x].coef_.copy()

                if len(list_test) == 2:
                    angle_decoders[i_session, i_loop] = None #angle_vecs(dec[list_test[0]].coef_, dec[list_test[1]].coef_)

                if train_projected:  # project and re decode
                    assert False, 'proj not implemented'
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
                pred_proba_eval[i_loop] = {x: dec[x].predict_proba(X=data_eval.transpose())[:, 1] for x in list_test}
                pred_proba_spont[i_loop] = {x: dec[x].predict_proba(X=data_spont.transpose())[:, 1] for x in list_test}

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
                dict_predictions_train['true_reward_train'] = np.concatenate((dict_predictions_train['true_reward_train'], rewarded_trials[train_inds]))
                dict_predictions_test['true_reward_test'] = np.concatenate((dict_predictions_test['true_reward_test'], rewarded_trials[test_inds]))
                dict_predictions_train['outcome_train'] = np.concatenate((dict_predictions_train['outcome_train'], trial_outcomes[train_inds]))
                dict_predictions_test['outcome_test'] = np.concatenate((dict_predictions_test['outcome_test'], trial_outcomes[test_inds]))
                dict_predictions_train['autorewarded_miss_train'] = np.concatenate((dict_predictions_train['autorewarded_miss_train'], autorewarded[train_inds]))
                dict_predictions_test['autorewarded_miss_test'] = np.concatenate((dict_predictions_test['autorewarded_miss_test'], autorewarded[test_inds]))
                dict_predictions_train['unrewarded_hit_train'] = np.concatenate((dict_predictions_train['unrewarded_hit_train'], unrewarded_hits[train_inds]))
                dict_predictions_test['unrewarded_hit_test'] = np.concatenate((dict_predictions_test['unrewarded_hit_test'], unrewarded_hits[test_inds]))
                dict_predictions_train['true_dec_train'] = np.concatenate((dict_predictions_train['true_dec_train'], train_labels['dec']))
                dict_predictions_test['true_dec_test'] = np.concatenate((dict_predictions_test['true_dec_test'], test_labels['dec']))
                dict_predictions_test['used_for_training'] = np.concatenate((dict_predictions_test['used_for_training'], np.ones(len(test_inds))))
                if len(list_save_covs) > 0:
                    for name_cov in list_save_covs:
                        dict_predictions_train[name_cov + '_train'] = np.concatenate((dict_predictions_train[name_cov + '_train'], cov_dict[name_cov][train_inds]))
                        dict_predictions_test[name_cov + '_test'] = np.concatenate((dict_predictions_test[name_cov + '_test'], cov_dict[name_cov][test_inds]))
                i_loop += 1

            ## Add results of eval_only trials (average of decoder CVs):

            ## eval onlY:
            assert (np.array(list(pred_proba_eval.keys())) == np.arange(n_split)).all()
            for x in list_test:
                mat_predictions = np.array([pred_proba_eval[nn][x] for nn in range(n_split)])
                assert mat_predictions.shape[0] == n_split
                dict_predictions_test[f'pred_{x}_test'] = np.concatenate((dict_predictions_test[f'pred_{x}_test'], np.mean(mat_predictions, 0)))
            dict_predictions_test['true_stim_test'] = np.concatenate((dict_predictions_test['true_stim_test'], session.trial_subsets[eval_only_inds].astype('int')))
            tmp_rewarded_all_trials = np.logical_or(session.outcome == 'hit', session.outcome == 'too_')
            tmp_rewarded_all_trials[session.autorewarded] = True
            tmp_rewarded_all_trials[session.unrewarded_hits] = False
            dict_predictions_test['true_reward_test'] = np.concatenate((dict_predictions_test['true_reward_test'], tmp_rewarded_all_trials[eval_only_inds]))
            dict_predictions_test['outcome_test'] = np.concatenate((dict_predictions_test['outcome_test'], eval_only_labels))
            dict_predictions_test['autorewarded_miss_test'] = np.concatenate((dict_predictions_test['autorewarded_miss_test'], session.autorewarded[eval_only_inds]))
            dict_predictions_test['unrewarded_hit_test'] = np.concatenate((dict_predictions_test['unrewarded_hit_test'], session.unrewarded_hits[eval_only_inds]))
            dict_predictions_test['true_dec_test'] = np.concatenate((dict_predictions_test['true_dec_test'], session.decision[eval_only_inds]))
            dict_predictions_test['used_for_training'] = np.concatenate((dict_predictions_test['used_for_training'], np.zeros(len(eval_only_inds))))
            if len(list_save_covs) > 0:
                for name_cov in list_save_covs:
                    dict_predictions_test[name_cov + '_test'] = np.concatenate((dict_predictions_test[name_cov + '_test'], session.cov_dict[name_cov][eval_only_inds])) 
                
            ## spontaneous:
            if n_spont_trials > 0 and (spont_used_for_training is False):
                assert (np.array(list(pred_proba_spont.keys())) == np.arange(n_split)).all()
                for x in list_test:
                    mat_predictions = np.array([pred_proba_spont[nn][x] for nn in range(n_split)])
                    assert mat_predictions.shape[0] == n_split, mat_predictions.shape[1] == n_spont_trials
                    dict_predictions_test[f'pred_{x}_test'] = np.concatenate((dict_predictions_test[f'pred_{x}_test'], np.mean(mat_predictions, 0)))
                dict_predictions_test['true_stim_test'] = np.concatenate((dict_predictions_test['true_stim_test'], np.zeros(n_spont_trials)))
                dict_predictions_test['true_reward_test'] = np.concatenate((dict_predictions_test['true_reward_test'], np.ones(n_spont_trials)))
                dict_predictions_test['outcome_test'] = np.concatenate((dict_predictions_test['outcome_test'], np.array(['spont'] * n_spont_trials)))
                dict_predictions_test['autorewarded_miss_test'] = np.concatenate((dict_predictions_test['autorewarded_miss_test'], np.zeros(n_spont_trials)))
                dict_predictions_test['unrewarded_hit_test'] = np.concatenate((dict_predictions_test['unrewarded_hit_test'], np.zeros(n_spont_trials)))
                dict_predictions_test['true_dec_test'] = np.concatenate((dict_predictions_test['true_dec_test'], np.ones(n_spont_trials)))
                dict_predictions_test['used_for_training'] = np.concatenate((dict_predictions_test['used_for_training'], np.zeros(n_spont_trials)))
                if len(list_save_covs) > 0:
                    for name_cov in list_save_covs:
                        dict_predictions_test[name_cov + '_test'] = np.concatenate((dict_predictions_test[name_cov + '_test'], cov_dict_reward_only[name_cov]))

        if verbose == 2:
            print(f'length test: {len(dict_predictions_test["true_dec_test"])}')
        ## Put dictionary results into dataframes:
        df_prediction_train[mouse] = pd.DataFrame(dict_predictions_train)
        df_prediction_test[mouse] = pd.DataFrame(dict_predictions_test)

    if return_decoder_weights is False:
        return df_prediction_train, df_prediction_test, None, (data_use, trial_outcomes)
    elif return_decoder_weights:
        return df_prediction_train, df_prediction_test, dec_weights, (data_use, trial_outcomes)


## Some functions that can be used as accuracy assessment
def prob_correct(binary_truth, estimate):
    """Return probability of correct estimate, where bt = {0, 1} and est = (0, 1).

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    prob, np.array of floats
        Accuracy (probability of being correct for each element)

    """
    prob = (binary_truth * estimate + (1 - binary_truth) * (1 - estimate))
    return prob

def mean_accuracy(binary_truth, estimate):
    """Mean accuracy (average over all trials)

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    mean, float
        Mean of accuracies
    std, float
        std of accuracies

    """
    assert len(binary_truth) == len(estimate)
    pp = prob_correct(binary_truth=binary_truth, estimate=estimate)
    return np.mean(pp), np.std(pp)

def mean_accuracy_pred(binary_truth, estimate):
    """Mean accuracy with hard >0.5 threshold (average of all trials)

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    mean_pred, float
        mean accuracy of thresholded predictions
    0

    """
    round_est = np.round(estimate)
    return sklearn.metrics.accuracy_score(binary_truth, round_est), 0

def llh(binary_truth, estimate):
    """Log likelihood of all trials.

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    llh, float
        Log likelihood of accuracies
    0

    """
    assert len(binary_truth) == len(estimate)
    pp = prob_correct(binary_truth=binary_truth, estimate=estimate)
    llh = np.mean(np.log(np.clip(pp, a_min=1e-3, a_max=1)))
    return llh, 0


def score_nonbinary(model, X, y):

    ''' JR helper for Thijs juice'''

    estimate = model.predict_proba(X)[:, 1]
    acc = mean_accuracy(binary_truth=y, estimate=estimate)
    return acc[0]


def r2_acc(binary_truth, estimate):
    """R2, plainly averaged over all trials (not variance-weighted).

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    r2, float
        R2 score of accuracies
    0

    """
    return sklearn.metrics.r2_score(y_true=binary_truth, y_pred=estimate), 0

def separability(binary_truth, estimate):
    """Measure difference between averages P(1) and P(0).

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    sep, float
        Separability between average class predictions
    0

    """
    av_pred_0 = np.mean(estimate[binary_truth == 0])
    av_pred_1 = np.mean(estimate[binary_truth == 1])
    sep = av_pred_1 - av_pred_0
    return sep, 0

def min_mean_accuracy(binary_truth, estimate):
    """Minimum of averages P(1) and P(0).

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    min_mean, float
        class-minimum of accuracies
    0

    """
    mean_acc_true = np.mean(estimate[binary_truth == 1])
    mean_acc_false = 1 - np.mean(estimate[binary_truth == 0])
    return np.minimum(mean_acc_true, mean_acc_false), 0

def class_av_mean_accuracy(binary_truth, estimate):
    """Mean of averages P(1) and P(0).
    #TODO: should we use sample correct (n-1)/n for std calculation?

    Parameters
    ----------
    binary_truth : np.array of 1s and 0s
        Binary ground truth array.
    estimate : np.array of floats 0 < f < 1
        Predictions of numbers in binary_truth.

    Returns
    -------
    class_av_mean, float
        Average accuracy where classes are weighted equally (indep of number of elements per class)
    0

    """
    if np.sum(binary_truth == 1) > 0:
        n_true = np.sum(binary_truth == 1)
        mean_acc_true = np.mean(estimate[binary_truth == 1])
        std_acc_true = np.std(estimate[binary_truth == 1])
        bin_truth_1 = True
    else:  # no BT == 1
        bin_truth_1 = False
    if np.sum(binary_truth == 0) > 0:
        n_false = np.sum(binary_truth == 0)
        mean_acc_false = 1 - np.mean(estimate[binary_truth == 0])
        std_acc_false = np.std(estimate[binary_truth == 0])
        bin_truth_0 = True
    else:  # no BT == 0
        bin_truth_0 = False
    if bin_truth_1 and bin_truth_0:
        comp_std = np.sqrt((n_true * (std_acc_true ** 2) + n_false * (std_acc_false ** 2)) / (n_true + n_false))
        return 0.5 * (mean_acc_true + mean_acc_false), comp_std
    elif bin_truth_1 and not bin_truth_0:  # if only 1 is present, return that accuracy only
        return mean_acc_true, std_acc_true
    elif not bin_truth_1 and bin_truth_0:
        return mean_acc_false, std_acc_false

## Main function to compute accuracy of decoders per time point
# def compute_accuracy_time_array(sessions, time_array, average_fun=class_av_mean_accuracy, reg_type='l2',
#                                 region_list=['s1', 's2'], regularizer=0.02, projected_data=False):
#     """
#     Deprecated now that compute_accuracy_time_array_average_per_mouse() also computes per sessions
#     
#     Compute accuracy of decoders for all time steps in time_array, for all sessions
#     Idea is that results here are concatenated overall, not saved per mouse only but
#     this needs checking #TODO

#     Parameters
#     ----------
#     sessions : dict of Session
#             data
#     time_array : np.array
#          array of time points to evaluate
#     average_fun : function
#         function that computes accuracy metric
#     reg_type : str, 'l2' or 'none'
#         type of regularisation
#     region_list : str, default=['s1', 's2']
#         list of regions to compute
#     regularizer : float
#         if reg_type == 'l2', this is the reg strength (C in scikit-learn)
#     projected_data : bool, default=False
#         if true, also compute test prediction on projected data (see train_test_all_sessions())

#     Returns
#     -------
#     tuple
#         (lick_acc,
#             lick accuracy of lick decoder per mouse/session
#         lick_acc_split,
#             lick accuracy split by trial type
#         ps_acc,
#             ps accuracy
#         ps_acc_split,
#             ps accuracy split by lick trial type
#         lick_half,
#             accuracy of naive fake data
#         angle_dec,
#             angle between decoders
#         decoder_weights)
#             weights of decoders

#     """
#     mouse_list = np.unique([ss.mouse for _, ss in sessions.items()])
#     stim_list = [0, 5, 10, 20, 30, 40, 50]  # hard coded!
#     tt_list = ['hit', 'fp', 'miss', 'cr']
#     dec_list = [0, 1]  # hard_coded!!
#     mouse_s_list = []
#     for mouse in mouse_list:
#         for reg in region_list:
#             mouse_s_list.append(mouse + '_' + reg)
#     n_timepoints = len(time_array)
#     signature_list = [session.signature for _, session in sessions.items()]

#     lick_acc = {reg: np.zeros((n_timepoints, 2)) for reg in region_list} #mean, std
# #     lick_acc_split = {x: {reg: np.zeros((n_timepoints, 2)) for reg in region_list} for x in stim_list}  # split per ps conditoin
#     lick_acc_split = {x: {reg: np.zeros((n_timepoints, 2)) for reg in region_list} for x in tt_list}  # split per tt
#     lick_half = {reg: np.zeros((n_timepoints, 2)) for reg in region_list}  # naive with P=0.5 for 2 options (lick={0, 1})
#     ps_acc = {reg: np.zeros((n_timepoints, 2)) for reg in region_list}
#     ps_acc_split = {x: {reg: np.zeros((n_timepoints, 2)) for reg in region_list} for x in dec_list}  # split per lick conditoin
#     angle_dec = {reg: np.zeros((n_timepoints, 2)) for reg in region_list}
#     decoder_weights = {'s1_stim': {session.signature: np.zeros((np.sum(session.s1_bool), len(time_array))) for _, session in sessions.items()},
#                        's2_stim': {session.signature: np.zeros((np.sum(session.s2_bool), len(time_array))) for _, session in sessions.items()},
#                        's1_dec': {session.signature: np.zeros((np.sum(session.s1_bool), len(time_array))) for _, session in sessions.items()},
#                        's2_dec': {session.signature: np.zeros((np.sum(session.s2_bool), len(time_array))) for _, session in sessions.items()}}

#     for i_tp, tp in tqdm(enumerate(time_array)):  # time array IN SECONDS

#         for reg in region_list:
#             df_prediction_test = {reg: {}}  # necessary for compability with violin plot df custom function
#             df_prediction_train, df_prediction_test[reg][tp], dec_w, _ = train_test_all_sessions(sessions=sessions, trial_times_use=np.array([tp]),
#                                                           verbose=0, include_150=False,
#                                                           include_autoreward=False, C_value=regularizer, reg_type=reg_type,
#                                                           train_projected=projected_data, return_decoder_weights=True,
#                                                           neurons_selection=reg)
#             for xx in ['stim', 'dec']:
#                 for signat in signature_list:
#                     decoder_weights[f'{reg}_{xx}'][signat][:, i_tp] = np.mean(dec_w[xx][signat], 0)

#             tmp_dict = make_violin_df_custom(input_dict_df=df_prediction_test,
#                                            flat_normalise_ntrials=False, verbose=0)
#             total_df_test = tmp_dict[tp]
#             lick = total_df_test['true_dec_test'].copy()
#             ps = (total_df_test['true_stim_test'] > 0).astype('int').copy()
#             if projected_data is False:
#                 pred_lick = total_df_test['pred_dec_test'].copy()
#             else:
#                 pred_lick = total_df_test['pred_dec_test_proj']
#             lick_half[reg][i_tp, :] = average_fun(binary_truth=lick, estimate=(np.zeros_like(lick) + 0.5))  # control for P=0.5
#             lick_acc[reg][i_tp, :] = average_fun(binary_truth=lick, estimate=pred_lick)

#             for x, arr in lick_acc_split.items():
#                 arr[reg][i_tp, :] = average_fun(binary_truth=lick[np.where(total_df_test['outcome_test'] == x)[0]],
#                                           estimate=pred_lick[np.where(total_df_test['outcome_test'] == x)[0]])

#             if 'pred_stim_test' in total_df_test.columns:
#                 if projected_data is False:
#                     pred_ps = total_df_test['pred_stim_test']
#                 else:
#                     pred_ps = total_df_test['pred_stim_test_proj']
#                 ps_acc[reg][i_tp, :] = average_fun(binary_truth=ps, estimate=pred_ps)

#                 for x, arr in ps_acc_split.items():
#                     arr[reg][i_tp, :] = average_fun(binary_truth=ps[lick == x],
#                                               estimate=pred_ps[lick == x])
#             tmp_all_angles = np.array([])
#             for mouse in df_prediction_train.keys():
#                 tmp_all_angles = np.concatenate((tmp_all_angles, df_prediction_train[mouse]['angle_decoders']))
#             angle_dec[reg][i_tp, 0] = mean_angle(tmp_all_angles)  # not sure about periodic std????

#     return (lick_acc, lick_acc_split, ps_acc, ps_acc_split, lick_half, angle_dec, decoder_weights)

## Main function to compute accuracy of decoders per time point
def compute_accuracy_time_array_average_per_mouse(sessions, time_array, average_fun=class_av_mean_accuracy, reg_type='l2',
                                                  region_list=['s1', 's2'], regularizer=0.02, projected_data=False, split_fourway=False,
                                                  list_tt_training=['hit', 'miss', 'fp', 'cr', 'spont'],
                                                  tt_list=['hit', 'fp', 'miss', 'cr', 'arm', 'urh', 'spont'],
                                                  concatenate_sessions_per_mouse=True, hard_set_10_trials=False):
    """Compute accuracy of decoders for all time steps in time_array, for all sessions (concatenated per mouse)

    Parameters
    ----------
    sessions : dict of Session
            data
    time_array : np.array
         array of time points to evaluate
    average_fun : function
        function that computes accuracy metric
    reg_type : str, 'l2' or 'none'
        type of regularisation
    region_list : str, default=['s1', 's2']
        list of regions to compute
    regularizer : float
        if reg_type == 'l2', this is the reg strength (C in scikit-learn)
    projected_data : bool, default=False
        if true, also compute test prediction on projected data (see train_test_all_sessions())

    Returns
    -------
    tuple
        (lick_acc,
            lick accuracy of lick decoder per mouse/session
        lick_acc_split,
            lick accuracy split by trial type
        ps_acc,
            ps accuracy
        ps_acc_split,
            ps accuracy split by lick trial type
        lick_half,
            accuracy of naive fake data
        angle_dec,
            angle between decoders
        decoder_weights)
            weights of decoders

    """
    if concatenate_sessions_per_mouse:
        mouse_list = np.unique([ss.mouse for _, ss in sessions.items()])
    else:
        mouse_list = [ss.signature for ss in sessions.values()]   
    stim_list = [0, 5, 10, 20, 30, 40, 50]  # hard coded!
    dec_list = [0, 1]  # hard_coded!!
    mouse_s_list = []
    for mouse in mouse_list:
        for reg in region_list:
            mouse_s_list.append(mouse + '_' + reg)
    n_timepoints = len(time_array)
    signature_list = [session.signature for _, session in sessions.items()]

    lick_acc = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} #mean, std
    lick_half = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list}  # naive with P=0.5 for 2 options (lick={0, 1})
    ps_acc = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list}
    if split_fourway is False:
        ps_acc_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in dec_list}  # split per trial type
        ps_pred_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in dec_list}
#     lick_acc_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in stim_list}  # split per ps conditoin
        assert False, 'implement lick_acc and lick_pred'
    elif split_fourway is True:
        ps_acc_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in tt_list}  # split per lick condition
        ps_pred_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in tt_list}
        lick_acc_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in tt_list}  # split per tt
        lick_pred_split = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in tt_list}  # split per tt
    angle_dec = {mouse: np.zeros(n_timepoints) for mouse in mouse_s_list}
    decoder_weights = {'s1_stim': {session.signature: np.zeros((np.sum(session.s1_bool), n_timepoints)) for _, session in sessions.items()},
                       's2_stim': {session.signature: np.zeros((np.sum(session.s2_bool), n_timepoints)) for _, session in sessions.items()},
                       's1_dec': {session.signature: np.zeros((np.sum(session.s1_bool), n_timepoints)) for _, session in sessions.items()},
                       's2_dec': {session.signature: np.zeros((np.sum(session.s2_bool), n_timepoints)) for _, session in sessions.items()}}
    for i_tp, tp in tqdm(enumerate(time_array)):  # time array IN SECONDS
        for reg in region_list:
            if type(tp) == np.ndarray:
                use_tp = tp 
            else:
                assert type(tp) == np.float64
                use_tp = np.array([tp])
            df_prediction_train, df_prediction_test, dec_w, _ = train_test_all_sessions(sessions=sessions, trial_times_use=use_tp,
                                                                                        verbose=0, include_150=False, list_tt_training=list_tt_training,
                                                                                        include_autoreward=False, C_value=regularizer, reg_type=reg_type,
                                                                                        train_projected=projected_data, return_decoder_weights=True,
                                                                                        neurons_selection=reg, concatenate_sessions_per_mouse=concatenate_sessions_per_mouse,
                                                                                        hard_set_10_trials=hard_set_10_trials)
            for xx in dec_w.keys():
                for signat in signature_list:
                    decoder_weights[f'{reg}_{xx}'][signat][:, i_tp] = np.mean(dec_w[xx][signat], 0)

            for mouse in df_prediction_train.keys():
                assert df_prediction_test[mouse][df_prediction_test[mouse]['used_for_training'] == 1]['unrewarded_hit_test'].sum() == 0
                assert df_prediction_test[mouse][df_prediction_test[mouse]['used_for_training'] == 1]['autorewarded_miss_test'].sum() == 0

                inds_training = np.where(df_prediction_test[mouse]['used_for_training'] == 1)[0]
                lick = df_prediction_test[mouse]['true_dec_test'].copy()
                ps = (df_prediction_test[mouse]['true_stim_test'] > 0).astype('int').copy()
                n_stim = df_prediction_test[mouse]['true_stim_test'].copy()

                if 'pred_dec_test' in df_prediction_test[mouse].columns:
                    if projected_data is False:
                        pred_lick = df_prediction_test[mouse]['pred_dec_test'].copy()
                    else:
                        pred_lick = df_prediction_test[mouse]['pred_dec_test_proj']
                    lick_half[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=lick[inds_training], estimate=(np.zeros_like(lick[inds_training]) + 0.5))  # control for P=0.5
                    lick_acc[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=lick[inds_training], estimate=pred_lick[inds_training])
    #                 lick_acc[mouse + '_' + reg][i_tp, :] = 0
    #                 for i_lick in np.unique(lick):
    #                     lick_acc[mouse + '_' + reg][i_tp, :] += np.array(average_fun(binary_truth=lick[lick == i_lick], estimate=pred_lick[lick == i_lick])) / len(np.unique(lick))
                    assert split_fourway is True
                    for x, arr in lick_acc_split.items():    
                        arr[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=lick[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]],
                                                                      estimate=pred_lick[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]])
                    for x, arr in lick_pred_split.items():
                        arr[mouse + '_' + reg][i_tp, :] = [np.mean(pred_lick[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]]), 
                                                           np.std(pred_lick[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]])]

                if 'pred_stim_test' in df_prediction_test[mouse].columns:
                    if projected_data is False:
                        pred_ps = df_prediction_test[mouse]['pred_stim_test']
                    else:
                        pred_ps = df_prediction_test[mouse]['pred_stim_test_proj']
                    ps_acc[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=ps[inds_training], estimate=pred_ps[inds_training])
#                     ps_acc[mouse + '_' + reg][i_tp, :] = 0
#                     for i_ps in np.unique(lick):
#                         ps_acc[mouse + '_' + reg][i_tp, :] += np.array(average_fun(binary_truth=ps[lick == i_ps], estimate=pred_ps[lick == i_ps])) / len(np.unique(lick))

                    for x, arr in ps_acc_split.items():
                        if split_fourway is False:  # split two ways by lick decision
                            arr[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=ps[np.intersect1d(np.where(lick == x)[0], inds_training)],
                                                                          estimate=pred_ps[np.intersect1d(np.where(lick == x)[0], inds_training)])
                        elif split_fourway is True:  
                            arr[mouse + '_' + reg][i_tp, :] = average_fun(binary_truth=ps[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]],
                                                                          estimate=pred_ps[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]])

                    for x, arr in ps_pred_split.items():
                        if split_fourway is False:  # split two ways by lick decision
                            arr[mouse + '_' + reg][i_tp, :] = [np.mean(pred_ps[lick == x]), np.std(pred_ps[lick == x])]
                        elif split_fourway is True:  
                            arr[mouse + '_' + reg][i_tp, :] = [np.mean(pred_ps[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]]), 
                                                                np.std(pred_ps[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]])]

                if 'angle_decoders' in df_prediction_train[mouse].columns:
                    angle_dec[mouse + '_' + reg][i_tp] = np.mean(df_prediction_train[mouse]['angle_decoders'])
    return (lick_acc, lick_acc_split, lick_pred_split, ps_acc, ps_acc_split, ps_pred_split, lick_half, angle_dec, decoder_weights)

## Main function to compute accuracy of decoders per time point
def compute_prediction_time_array_average_per_mouse_split(sessions, time_array, average_fun=class_av_mean_accuracy, reg_type='l2',
                                                        region_list=['s1', 's2'], regularizer=0.02, projected_data=False, 
                                                        list_tt_training=['hit', 'miss', 'fp', 'cr', 'spont'],
                                                        tt_list=['hit', 'fp', 'miss', 'cr', 'arm', 'urh', 'spont'],
                                                        concatenate_sessions_per_mouse=True, hard_set_10_trials=False,
                                                        list_save_covs=[]):
    """Compute accuracy of decoders for all time steps in time_array, for all sessions (concatenated per mouse)

    Parameters
    ----------
    sessions : dict of Session
            data
    time_array : np.array
         array of time points to evaluate
    average_fun : function
        function that computes accuracy metric
    reg_type : str, 'l2' or 'none'
        type of regularisation
    region_list : str, default=['s1', 's2']
        list of regions to compute
    regularizer : float
        if reg_type == 'l2', this is the reg strength (C in scikit-learn)
    projected_data : bool, default=False
        if true, also compute test prediction on projected data (see train_test_all_sessions())

    Returns
    -------
    tuple
        (lick_acc,
            lick accuracy of lick decoder per mouse/session
        lick_acc_split,
            lick accuracy split by trial type
        ps_acc,
            ps accuracy
        ps_acc_split,
            ps accuracy split by lick trial type
        lick_half,
            accuracy of naive fake data
        angle_dec,
            angle between decoders
        decoder_weights)
            weights of decoders

    """
    ## Prepare variables:
    assert projected_data is False, 'see old function for template of how to implement if True'
    if concatenate_sessions_per_mouse:
        mouse_list = np.unique([ss.mouse for _, ss in sessions.items()])
    else:
        mouse_list = [ss.signature for ss in sessions.values()]   
    stim_list = [0, 5, 10, 20, 30, 40, 50]  # hard coded!
    dec_list = [0, 1]  # hard_coded!!
    mouse_s_list = []
    for mouse in mouse_list:
        for reg in region_list:
            mouse_s_list.append(mouse + '_' + reg)
    n_timepoints = len(time_array)
    signature_list = [session.signature for _, session in sessions.items()]
    if len(list_save_covs) > 0:
        for ss in sessions.values():
            assert hasattr(ss, 'cov_dict'), f'{ss} does not have a cov_dict'
            for name_cov in list_save_covs:
                assert name_cov in ss.cov_dict.keys(), f'{ss} cov_dict does not contain {name_cov}'

    ## Accuracy & prediction split by trial type
    ps_pred_split_tt = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in tt_list}
    lick_pred_split_tt = {x: {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list} for x in tt_list}  # split per tt

    nstim_name_dict = {'n0': np.array([0]), 'n1': np.array([5, 10]), 
                       'n2': np.array([20, 30]), 'n3': np.array([40, 50])}
    ps_pred_split_tt_nstim, lick_pred_split_tt_nstim = {}, {}
    
    nstim_selection_dict, tt_selection_dict = {}, {}
    for nstim_name, nstim_list in nstim_name_dict.items():
        for tt in tt_list:
            key_name = tt + '_' + nstim_name
            nstim_selection_dict[key_name] = nstim_list.copy()
            tt_selection_dict[key_name] = tt 
            ps_pred_split_tt_nstim[key_name] = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list}
            lick_pred_split_tt_nstim[key_name] = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list}

    if len(list_save_covs) > 0:
        covar_perc_dict = {'c1': [0, 33.3], 'c2': [33.3001, 66.7], 'c3': [66.7001, 100]}
        ps_pred_split_tt_covar, lick_pred_split_tt_covar = {x: {} for x in list_save_covs}, {x: {} for x in list_save_covs}
        for cov_name in list_save_covs:
            for covar_perc_key, perc_value in covar_perc_dict.items():
                for tt in tt_list:        
                    key_name = tt + '_' + covar_perc_key
                    ps_pred_split_tt_covar[cov_name][key_name] = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list}
                    lick_pred_split_tt_covar[cov_name][key_name] = {mouse: np.zeros((n_timepoints, 2)) for mouse in mouse_s_list}
    else:
        ps_pred_split_tt_covar, lick_pred_split_tt_covar = None, None

    # angle_dec = {mouse: np.zeros(n_timepoints) for mouse in mouse_s_list}
    # decoder_weights = {'s1_stim': {session.signature: np.zeros((np.sum(session.s1_bool), len(time_array))) for _, session in sessions.items()},
    #                    's2_stim': {session.signature: np.zeros((np.sum(session.s2_bool), len(time_array))) for _, session in sessions.items()},
    #                    's1_dec': {session.signature: np.zeros((np.sum(session.s1_bool), len(time_array))) for _, session in sessions.items()},
    #                    's2_dec': {session.signature: np.zeros((np.sum(session.s2_bool), len(time_array))) for _, session in sessions.items()}}

    ## Train decoders & extract relevant results:
    for i_tp, tp in tqdm(enumerate(time_array)):  # time array IN SECONDS
        for reg in region_list:
            df_prediction_train, df_prediction_test, dec_w, _ = train_test_all_sessions(sessions=sessions, trial_times_use=np.array([tp]),
                                                                                        verbose=0, include_150=False, list_tt_training=list_tt_training,
                                                                                        include_autoreward=False, C_value=regularizer, reg_type=reg_type,
                                                                                        train_projected=projected_data, return_decoder_weights=True,
                                                                                        neurons_selection=reg, concatenate_sessions_per_mouse=concatenate_sessions_per_mouse,
                                                                                        hard_set_10_trials=hard_set_10_trials,
                                                                                        list_save_covs=list_save_covs)  # train decoders
            # for xx in dec_w.keys():  # extract decoder weights
            #     for signat in signature_list:
            #         decoder_weights[f'{reg}_{xx}'][signat][:, i_tp] = np.mean(dec_w[xx][signat], 0)
            # return df_prediction_train, df_prediction_test
            for mouse in df_prediction_train.keys():  # extract decoder predictions per mouse
                assert df_prediction_test[mouse][df_prediction_test[mouse]['used_for_training'] == 1]['unrewarded_hit_test'].sum() == 0
                assert df_prediction_test[mouse][df_prediction_test[mouse]['used_for_training'] == 1]['autorewarded_miss_test'].sum() == 0

                inds_training = np.where(df_prediction_test[mouse]['used_for_training'] == 1)[0]  # this excludes stuff like arm and urh
                lick = df_prediction_test[mouse]['true_dec_test'].copy()
                ps = (df_prediction_test[mouse]['true_stim_test'] > 0).astype('int').copy()
                n_stim = df_prediction_test[mouse]['true_stim_test'].copy()
                
                ## 1 or 2 classifiers could be have trained (decision & stimulus):
                if 'pred_dec_test' in df_prediction_test[mouse].columns:
                    pred_lick = df_prediction_test[mouse]['pred_dec_test'].copy()
                    ## Prediction split by trial type:
                    for x, arr in lick_pred_split_tt.items():
                        arr[mouse + '_' + reg][i_tp, :] = [np.mean(pred_lick[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]]), 
                                                           np.std(pred_lick[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]])]

                    ## Prediction split by trial type AND n cells stimulated:
                    for x, arr in lick_pred_split_tt_nstim.items():
                        trial_selection = np.where(np.logical_and(df_prediction_test[mouse]['outcome_test'] == tt_selection_dict[x],
                                                                  np.isin(df_prediction_test[mouse]['true_stim_test'], nstim_selection_dict[x])))[0]
                        arr[mouse + '_' + reg][i_tp, :] = [np.mean(pred_lick[trial_selection]), 
                                                           np.std(pred_lick[trial_selection])]

                    ## Prediction split by trial type AND covariate
                    if len(list_save_covs) > 0:
                        for cov_name in list_save_covs:
                            for x, arr in lick_pred_split_tt_covar[cov_name].items():
                                arr_covar = df_prediction_test[mouse][cov_name + '_test']
                                # assert np.sum(np.isnan(arr_covar)) == 0, f'NaNs in {mouse} and {cov_name}'
                                covar_perc_key = x.split('_')[1]  # second part of name 
                                perc_min = np.percentile(arr_covar, covar_perc_dict[covar_perc_key][0])
                                perc_max = np.percentile(arr_covar, covar_perc_dict[covar_perc_key][1])
                                tt_name = x.split('_')[0]
                                trial_selection = np.where(np.logical_and(np.logical_and(arr_covar <= perc_max,
                                                                          arr_covar >= perc_min),
                                                                          df_prediction_test[mouse]['outcome_test'] == tt_name))[0]
                                arr[mouse + '_' + reg][i_tp, :] = [np.mean(pred_lick[trial_selection]), 
                                                                   np.std(pred_lick[trial_selection])]

                if 'pred_stim_test' in df_prediction_test[mouse].columns:
                    pred_ps = df_prediction_test[mouse]['pred_stim_test'].copy()
                    ## Prediction split by trial types: 
                    for x, arr in ps_pred_split_tt.items():
                        arr[mouse + '_' + reg][i_tp, :] = [np.mean(pred_ps[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]]), 
                                                           np.std(pred_ps[np.where(df_prediction_test[mouse]['outcome_test'] == x)[0]])]

                    ## Prediction split by trial type AND n cells stimulated:
                    for x, arr in ps_pred_split_tt_nstim.items():
                        trial_selection = np.where(np.logical_and(df_prediction_test[mouse]['outcome_test'] == tt_selection_dict[x],
                                                                  np.isin(df_prediction_test[mouse]['true_stim_test'], nstim_selection_dict[x])))[0]
                        arr[mouse + '_' + reg][i_tp, :] = [np.mean(pred_ps[trial_selection]), 
                                                           np.std(pred_ps[trial_selection])]

                    ## Prediction split by trial type AND covariate
                    if len(list_save_covs) > 0:
                        for cov_name in list_save_covs:
                            for x, arr in ps_pred_split_tt_covar[cov_name].items():
                                arr_covar = df_prediction_test[mouse][cov_name + '_test']
                                # assert np.sum(np.isnan(arr_covar)) == 0, f'NaNs in {mouse} and {cov_name}'
                                covar_perc_key = x.split('_')[1]  # second part of name 
                                tt_name = x.split('_')[0]
                                perc_min = np.percentile(arr_covar, covar_perc_dict[covar_perc_key][0])
                                perc_max = np.percentile(arr_covar, covar_perc_dict[covar_perc_key][1])
                                trial_selection = np.where(np.logical_and(np.logical_and(arr_covar <= perc_max,
                                                                                         arr_covar >= perc_min),
                                                                          df_prediction_test[mouse]['outcome_test'] == tt_name))[0]
                                # print(arr_covar, np.sum(np.isnan(arr_covar)), trial_selection, perc_min, perc_max, x)
                                # break
                                arr[mouse + '_' + reg][i_tp, :] = [np.mean(pred_ps[trial_selection]), 
                                                                   np.std(pred_ps[trial_selection])]

                # if 'angle_decoders' in df_prediction_train[mouse].columns:
                #     angle_dec[mouse + '_' + reg][i_tp] = np.mean(df_prediction_train[mouse]['angle_decoders'])
    angle_dec, decoder_weights = None, None
    return (lick_pred_split_tt, lick_pred_split_tt_nstim, lick_pred_split_tt_covar,
            ps_pred_split_tt, ps_pred_split_tt_nstim, ps_pred_split_tt_covar)


def get_acc_array(pred_dict, decoder_name='hit/cr', covar_name=None, tt='hit', region='s1',
                  time_array=np.array([])):
    '''extract dataframe with accuracy array of dictionary (that is returned by dyn dec analysis)'''
    if decoder_name == 'NA':
        sub_dict = {k[:-3]: v for k, v in pred_dict[tt].items() if k[-2:] == region}
    elif covar_name is not None:
        sub_dict = {k[:-3]: v for k, v in pred_dict[decoder_name][covar_name][tt].items() if k[-2:] == region}
    else:
        sub_dict = {k[:-3]: v for k, v in pred_dict[decoder_name][tt].items() if k[-2:] == region}

    df_pred = {}
    df_pred['time_array'] = time_array  ## add time array. should be real time (in seconds)
    for name_session, mat in sub_dict.items():
        df_pred[name_session] = mat[:, 0]  # extract mean accuracy trace of a session 
        assert len(time_array) == mat.shape[0]
    df_pred = pd.DataFrame(df_pred)

    df_pred_collapsed = pd.melt(df_pred, id_vars='time_array',
                                value_vars=list(sub_dict.keys()), var_name='session',
                                value_name='accuracy')  # collapse into 3 columns (acc, time, session)
    return df_pred, df_pred_collapsed

def stat_test_dyn_dec(pred_dict, decoder_name='hit/cr', tt='hit', region='s1',
                      time_array=np.array([]), frames_bin=2, th=0.05):
    '''
    time array with time in seconds, should be same size as accuracy arrays 
    use nans to exclude (artefact) periods

    '''
    _, df_pred_collapsed = get_acc_array(pred_dict=pred_dict, time_array=time_array, 
                                        decoder_name=decoder_name, tt=tt, region=region)
    df_pred_collapsed['chance_level'] = 0.5  ## add column with chance level performance 
    signif_array = np.zeros(len(time_array))
    n_bins = int(np.floor(np.sum(~np.isnan(time_array)) / frames_bin))  # exclude artefact in test
    th_bonf = th / n_bins  # perform bonferroni correction for number of tests

    for i_bin in range(n_bins):  # loop through bins
        start_frame = int(i_bin * frames_bin)
        end_frame = int((i_bin + 1) * frames_bin)
        time_min = time_array[start_frame]
        if end_frame >= len(time_array):
            time_max = time_array[-1] + 0.1
            end_frame = len(time_array) 
        else:
            time_max = time_array[end_frame]
        
        if np.sum(np.isnan(time_array[start_frame:end_frame + 1])) > 0:
            continue  # skip bins that contains nans [during artefact]
        else:
            inds_rows = np.logical_and(df_pred_collapsed['time_array'] >= time_min, 
                                    df_pred_collapsed['time_array'] < time_max)
            sub_df = df_pred_collapsed[inds_rows]  # select df during this time bin

            stat, pval = scipy.stats.wilcoxon(x=sub_df['accuracy'], y=sub_df['chance_level'], 
                                            alternative='two-sided')
            
            # print(th_bonf, pval, i_bin, tt, region)
            # if i_bin == 1:
                # print(sub_df)
            if pval < th_bonf:
                signif_array[start_frame:end_frame] = 1  # indicate significance

    return df_pred_collapsed, signif_array

def stat_test_dyn_dec_two_arrays(pred_dict_1={}, decoder_name_1='hit/cr', tt_1='hit', region_1='s1',
                                 pred_dict_2={}, decoder_name_2='hit/cr', tt_2='hit', region_2='s1',
                                 time_array=np.array([]), frames_bin=2, th=0.05, covar_name=None,
                                 alternative='two-sided'):
    '''
    time array with time in seconds, should be same size as accuracy arrays 
    use nans to exclude (artefact) periods

    '''
    _, df_pred_collapsed_1 = get_acc_array(pred_dict=pred_dict_1, time_array=time_array, covar_name=covar_name,
                                        decoder_name=decoder_name_1, tt=tt_1, region=region_1)
    _, df_pred_collapsed_2 = get_acc_array(pred_dict=pred_dict_2, time_array=time_array, covar_name=covar_name,
                                        decoder_name=decoder_name_2, tt=tt_2, region=region_2)

    inds_non_nan = ~np.isnan(df_pred_collapsed_1['time_array'])
    df_pred_collapsed_1 = df_pred_collapsed_1[inds_non_nan]
    df_pred_collapsed_2 = df_pred_collapsed_2[inds_non_nan]

    signif_array = np.zeros(len(time_array))
    n_bins = int(np.floor(len(time_array) / frames_bin))
    th_bonf = th / n_bins  # perform bonferroni correction for number of tests

    for i_bin in range(n_bins):  # loop through bins
        start_frame = int(i_bin * frames_bin)
        end_frame = int((i_bin + 1) * frames_bin)
        time_min = time_array[start_frame]
        if end_frame >= len(time_array):
            time_max = time_array[-1] + 0.1
            end_frame = len(time_array) 
        else:
            time_max = time_array[end_frame]
        
        if np.sum(np.isnan(time_array[start_frame:end_frame + 1])) > 0:
            continue  # skip bins that contains nans [during artefact]
        else:
            inds_rows = np.logical_and(df_pred_collapsed_1['time_array'] >= time_min, 
                                       df_pred_collapsed_1['time_array'] < time_max)
            sub_df_1 = df_pred_collapsed_1[inds_rows]  # select df during this time bin
            sub_df_2 = df_pred_collapsed_2[inds_rows]  # select df during this time bin

            new_df = pd.merge(sub_df_1, sub_df_2,  how='left', left_on=['session','time_array'], 
                              right_on = ['session','time_array'], suffixes=('_1', '_2'))
            new_df = new_df.dropna()# new_df.fillna(0.5)  # Nans mess up the wilcoxon test. occurs very rarely, but ruins the whole test. 
            # stat, pval = scipy.stats.wilcoxon(x=sub_df_1['accuracy'], y=sub_df_2['accuracy'], 
            #                                 alternative=alternative)
            stat, pval = scipy.stats.wilcoxon(x=new_df['accuracy_1'], y=new_df['accuracy_2'], 
                                            alternative=alternative)
            if pval < th_bonf:
                signif_array[start_frame:end_frame] = 1  # indicate significance


        # if time_min > 0.5:
        #     print(new_df)
        #     assert False
    return df_pred_collapsed_1, df_pred_collapsed_2, signif_array


def stat_test_dyn_dec_two_difference_arrays(pred_dict_1={}, decoder_name_1='hit/cr', tt_1_pos='hit', tt_1_neg='cr', region_1='s1',
                                            pred_dict_2={}, decoder_name_2='hit/cr', tt_2_pos='hit', tt_2_neg='cr', region_2='s2',
                                            time_array=np.array([]), frames_bin=2, th=0.05, 
                                            alternative='two-sided'):
    '''
    time array with time in seconds, should be same size as accuracy arrays 
    use nans to exclude (artefact) periods

    '''
    _, df_pred_collapsed_1_pos = get_acc_array(pred_dict=pred_dict_1, time_array=time_array, 
                                        decoder_name=decoder_name_1, tt=tt_1_pos, region=region_1)
    _, df_pred_collapsed_2_pos = get_acc_array(pred_dict=pred_dict_2, time_array=time_array, 
                                        decoder_name=decoder_name_2, tt=tt_2_pos, region=region_2)

    _, df_pred_collapsed_1_neg = get_acc_array(pred_dict=pred_dict_1, time_array=time_array, 
                                        decoder_name=decoder_name_1, tt=tt_1_neg, region=region_1)
    _, df_pred_collapsed_2_neg = get_acc_array(pred_dict=pred_dict_2, time_array=time_array, 
                                        decoder_name=decoder_name_2, tt=tt_2_neg, region=region_2)
    inds_non_nan = ~np.isnan(df_pred_collapsed_1_pos['time_array'])
    df_pred_collapsed_1_pos = df_pred_collapsed_1_pos[inds_non_nan]
    df_pred_collapsed_2_pos = df_pred_collapsed_2_pos[inds_non_nan]
    df_pred_collapsed_1_neg = df_pred_collapsed_1_neg[inds_non_nan]
    df_pred_collapsed_2_neg = df_pred_collapsed_2_neg[inds_non_nan]
    
    
    signif_array = np.zeros(len(time_array))
    n_bins = int(np.floor(len(time_array) / frames_bin))
    th_bonf = th / n_bins  # perform bonferroni correction for number of tests

    for i_bin in range(n_bins):  # loop through bins
        start_frame = int(i_bin * frames_bin)
        end_frame = int((i_bin + 1) * frames_bin)
        time_min = time_array[start_frame]
        if end_frame >= len(time_array):
            time_max = time_array[-1] + 0.1
            end_frame = len(time_array) 
        else:
            time_max = time_array[end_frame]
        
        if np.sum(np.isnan(time_array[start_frame:end_frame + 1])) > 0:
            continue  # skip bins that contains nans [during artefact]
        else:
            inds_rows = np.logical_and(df_pred_collapsed_1_pos['time_array'] >= time_min, 
                                       df_pred_collapsed_1_pos['time_array'] < time_max)
            sub_df_1_pos = df_pred_collapsed_1_pos[inds_rows]  # select df during this time bin
            sub_df_2_pos = df_pred_collapsed_2_pos[inds_rows]  # select df during this time bin
            sub_df_1_neg = df_pred_collapsed_1_neg[inds_rows]  # select df during this time bin
            sub_df_2_neg = df_pred_collapsed_2_neg[inds_rows]  # select df during this time bin

            new_df_1 = pd.merge(sub_df_1_pos, sub_df_1_neg,  how='left', left_on=['session','time_array'], 
                              right_on = ['session','time_array'], suffixes=('_pos', '_neg'))
            new_df_1['accuracy_diff'] = new_df_1['accuracy_pos'] - new_df_1['accuracy_neg']

            new_df_2 = pd.merge(sub_df_2_pos, sub_df_2_neg,  how='left', left_on=['session','time_array'], 
                              right_on = ['session','time_array'], suffixes=('_pos', '_neg'))
            new_df_2['accuracy_diff'] = new_df_2['accuracy_pos'] - new_df_1['accuracy_neg']

            new_df_total = pd.merge(new_df_1, new_df_2,  how='left', left_on=['session','time_array'], 
                              right_on = ['session','time_array'], suffixes=('_1', '_2'))

            # stat, pval = scipy.stats.wilcoxon(x=sub_df_1['accuracy'], y=sub_df_2['accuracy'], 
            #                                 alternative=alternative)
            stat, pval = scipy.stats.wilcoxon(x=new_df_total['accuracy_diff_1'], y=new_df_total['accuracy_diff_2'], 
                                            alternative=alternative)
            if pval < th_bonf:
                signif_array[start_frame:end_frame] = 1  # indicate significance

            # print(i_bin, time_min, time_max, np.mean(new_df_1['accuracy_diff']), np.mean(new_df_2['accuracy_diff']),
            #  pval, th_bonf)
            # print(new_df_total)

    return new_df_total, signif_array

def wilcoxon_test(acc_dict):
    """Perform wilcoxon signed rank test for dictionoary of S1/S2 measurements. Each
    S1/S2 pair per mouse is a paired sample for the test. Perform test on each time point.

    Parameters:
    ----------------------
        acc_dict: dict
            dictionary of np.arrays of (n_timepoints x 2) or (n_timepoints)
            Wilcoxon test is performed across all mice, comparing regions, for each time point.

    Returns:
    ---------------------
        p_vals: np.array of size n_timepoints
            P values of W test.
    """
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
        p_vals[tp] = pval#.copy()
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
    bool_two_regions = (region_list == ['s1', 's2'])
    if not bool_two_regions:
        assert (region_list == ['s1']) or (region_list == ['s2'])
    timepoints = list(dict_df[region_list[0]].keys())
    assert False, 'check if mouse_list definition is still valid after change with concatenate_sessions per mice'
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
            if bool_two_regions:
                new_df[tp] = pd.concat([dict_df['s1'][tp][mouse] for mouse in mouse_list] +
                                       [dict_df['s2'][tp][mouse] for mouse in mouse_list])
            else:
                new_df[tp] = pd.concat([dict_df[region_list[0]][tp][mouse] for mouse in mouse_list])
        elif flat_normalise_ntrials:
            if bool_two_regions:
                new_df[tp] = pd.concat([pd.concat([dict_df['s1'][tp][mouse] for x in range(n_multi[mouse])]) for mouse in mouse_list] +
                                       [pd.concat([dict_df['s2'][tp][mouse] for x in range(n_multi[mouse])]) for mouse in mouse_list])
            else:
                new_df[tp] = pd.concat([pd.concat([dict_df[region_list[0]][tp][mouse] for x in range(n_multi[mouse])]) for mouse in mouse_list])
    if verbose:
        for mouse in mouse_list:
            print(f'Corrected number of trials for mouse {mouse}: {len(new_df[timepoints[0]][new_df[timepoints[0]]["mouse"] == mouse])}')
    return new_df

def difference_pre_post(ss, tt='hit', reg='s1', duration_window=1.2):
    """Compute difference df/f response between a post-stim window and a pre_stim
    baseline window average. Computes the average window acitivty per neuron and per
    trial, and then returns the average of the elementwise difference between
    all neurons and trials.
    #TODO: merge with dynamic equivalent
    Parameters:
    ---------------
        ss: Session
            session to evaluate
        xx: str, default='hit'
            trial type
        reg: str, default='s1'
            region
        duration_window_: float
            length of  window

    Returns:
    ---------------
        metric: float
            difference

    """
    inds_pre_stim = np.logical_and(ss.filter_ps_time < 0, ss.filter_ps_time >= -2) # hard-set 2 second pre stimulus baseline
    inds_post_stim = np.logical_and(ss.filter_ps_time < (ss.filter_ps_time[ss.filter_ps_time > 0][0] + duration_window),
                                    ss.filter_ps_time >= ss.filter_ps_time[ss.filter_ps_time > 0][0])  # post stimulus window

    if reg == 's1':
        reg_inds = ss.s1_bool
    elif reg == 's2':
        reg_inds = ss.s2_bool


    if tt == 'ur_hit':  # unrewarded hit
        general_tt = 'hit' # WILL be changed later, but this is more efficient I think with lines of code
        odd_tt_only = True
    elif tt == 'ar_miss':  # autorewarded miss
        general_tt = 'miss'
        odd_tt_only = True
    else:  # regular cases
        general_tt = tt
        if tt == 'hit' or tt == 'miss':
            odd_tt_only = False  # only use regular ones
        elif tt == 'fp' or tt == 'cr':
            odd_tt_only = None  # does not matter

    if odd_tt_only is None or general_tt == 'fp' or general_tt == 'cr': # if special tt do not apply
        pre_stim_act = ss.behaviour_trials[:, np.logical_and(ss.photostim < 2,
                                             ss.outcome==general_tt), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and(ss.photostim < 2,
                                             ss.outcome==general_tt), :][:, :, ss.filter_ps_array[inds_post_stim]][reg_inds, :, :]
    elif general_tt =='miss' and odd_tt_only is not None:  # if specified miss type (autorewarded or not autorewarded)
        pre_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.autorewarded==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.autorewarded==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_post_stim]][reg_inds, :, :]
    elif general_tt =='hit' and odd_tt_only is not None:  # if specified hit type (unrewarded or rewarded)
        if odd_tt_only is True:  # unrewarded hit
            general_tt = 'miss'   # unrewarded hits are registered as misssess
        pre_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.unrewarded_hits==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.unrewarded_hits==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_post_stim]][reg_inds, :, :]

    pre_met = np.mean(pre_stim_act, 2)
    post_met = np.mean(post_stim_act, 2)

    metric = np.mean(post_met - pre_met)
    if metric.shape == ():  # if only 1 element it is not yet an array, while the separate trial output can be, so change for consistency
        metric = np.array([metric]) #  pack into 1D array
    return metric

def difference_pre_post_dynamic(ss, tt='hit', reg='s1', duration_window_pre=1.2,
                                tp_post=1.0, odd_tt_only=None, return_trials_separate=False):
    """Compute difference df/f response between a post-stim timepoint tp_post and a pre_stim
    baseline window average. Returns the average of the elementwise difference between
    all neurons and trials.

    Parameters:
    ---------------
        ss: Session
            session to evaluate
        tt: str, default='hit'
            trial type
        reg: str, default='s1'
            region
        duration_window_pre: float
            lenght of baseline window, taken <= 0
        tp_post: float
            post stim time point
        odd_tt_only: bool or None, default=None
            if True; only eval unrew_hit / autorew miss; if False; only evaluate non UH/AM; if None; evaluate all
            i.e. bundles boolean for unrewarded_hits and autorewarded trial types.

    Returns:
    ---------------
        metric: float
            difference

    """
    inds_pre_stim = np.logical_and(ss.filter_ps_time <= 0, ss.filter_ps_time >= (-1 * duration_window_pre))
    frame_post = ss.filter_ps_array[np.where(ss.filter_ps_time == tp_post)[0]]  # corresponding frame
    if reg == 's1':
        reg_inds = ss.s1_bool
    elif reg == 's2':
        reg_inds = ss.s2_bool

    if tt == 'ur_hit':  # unrewarded hit
        general_tt = 'hit' # WILL be changed later, but this is more efficient I think with lines of code
        odd_tt_only = True
    elif tt == 'ar_miss':  # autorewarded miss
        general_tt = 'miss'
        odd_tt_only = True
    elif tt =='spont_rew':
        odd_tt_only = True
        general_tt = tt
    else:  # regular cases
        general_tt = tt
        if tt == 'hit' or tt == 'miss':
            odd_tt_only = False  # only use regular ones
        elif tt == 'fp' or tt == 'cr':
            odd_tt_only = None  # does not matter
    if odd_tt_only is None or general_tt == 'fp' or general_tt == 'cr': # if special tt do not apply
        pre_stim_act = ss.behaviour_trials[:, np.logical_and(ss.photostim < 2,
                                             ss.outcome==general_tt), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and(ss.photostim < 2,
                                             ss.outcome==general_tt), :][:, :, frame_post][reg_inds, :, :]
    elif general_tt =='miss' and odd_tt_only is not None:  # if specified miss type (autorewarded or not autorewarded)
        pre_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.autorewarded==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.autorewarded==odd_tt_only)), :][:, :, frame_post][reg_inds, :, :]
    elif general_tt =='hit' and odd_tt_only is not None:  # if specified hit type (unrewarded or rewarded)
        if odd_tt_only is True:  # unrewarded hit
            general_tt = 'miss'   # unrewarded hits are registered as misssess
        pre_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.unrewarded_hits==odd_tt_only)), :][:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = ss.behaviour_trials[:, np.logical_and.reduce((ss.photostim < 2,
                                                 ss.outcome==general_tt, ss.unrewarded_hits==odd_tt_only)), :][:, :, frame_post][reg_inds, :, :]

    elif tt == general_tt == 'spont_rew':
        flu_arr = ss.pre_rew_trials
        pre_stim_act = flu_arr[:, :, ss.filter_ps_array[inds_pre_stim]][reg_inds, :, :]
        post_stim_act = flu_arr[:, :, frame_post][reg_inds, :, :]

    else:
        raise ValueError('tt {} not understood'.format(tt))



    pre_met = np.squeeze(np.mean(pre_stim_act, 2))  # take mean over time points
    post_met = np.squeeze(post_stim_act)
    assert pre_met.shape == post_met.shape  # should now both be n_neurons x n_trials
    if return_trials_separate is False:
        metric = np.mean(post_met - pre_met)  # mean over neurons & trials,
    elif return_trials_separate:
        metric = np.mean(post_met - pre_met, 0)  # mean over neurons, not trials
    if metric.shape == ():  # if only 1 element it is not yet an array, while the separate trial output can be, so change for consistency
        metric = np.array([metric]) #  pack into 1D array
    return metric

def create_df_differences(sessions):
    ## Compute pre stim window vs post stim window
    dict_diff_wind = {name: np.zeros(8 * len(sessions), dtype='object') for name in ['diff_dff', 'region', 'trial_type', 'session']}
    ind_data = 0
    for _, sess in sessions.items():
        for reg in ['s1', 's2']:
            for tt in ['hit', 'fp', 'miss', 'cr']:
                mean_diff = difference_pre_post(ss=sess,
                        tt=tt, reg=reg, duration_window=1)
                if len(mean_diff) == 0:
                    pass
                else:
                    dict_diff_wind['diff_dff'][ind_data] = mean_diff[0]
                    dict_diff_wind['region'][ind_data] = reg.upper()
                    dict_diff_wind['trial_type'][ind_data] = tt
                    dict_diff_wind['session'][ind_data] = sess.signature
                    ind_data += 1

    df_differences = pd.DataFrame(dict_diff_wind)
    return df_differences

def create_df_dyn_differences(sessions, tp_dict):
    # list_tp = tp_dict['mutual'][np.where(np.logical_and(tp_dict['mutual'] >= -2, tp_dict['mutual'] <= 5))]
    list_tp = tp_dict['mutual'][np.where(tp_dict['mutual'] >= -2)[0]]
    list_tt = ['hit', 'fp', 'miss', 'cr', 'ur_hit', 'ar_miss']
    # dict_diff = {name: np.zeros(2 * len(list_tt) * len(sessions) *
    #                             len(list_tp), dtype='object') for name in ['diff_dff', 'region', 'trial_type', 'session', 'timepoint', 'new_trial_id']}
    ind_data = 0
    dict_diff = {name: np.array([]) for name in ['diff_dff', 'region', 'trial_type', 'session', 'timepoint', 'new_trial_id']}  # initiate empy dicts
    for _, sess in tqdm(sessions.items()):
        for tp in list_tp:
            for reg in ['s1', 's2']:
                for tt in list_tt:
    #                 dict_diff['diff_dff'][ind_data] = pof.difference_pre_post_dynamic(ss=sess,
    #                                           general_tt=tt, reg=reg, duration_window_pre=2,
    #                                           tp_post=tp, odd_tt_only=True)
    #                 dict_diff['region'][ind_data] = reg.upper()
    #                 dict_diff['trial_type'][ind_data] = tt
    #                 dict_diff['session'][ind_data] = sess.signature
    #                 dict_diff['timepoint'][ind_data] = tp.copy()
    #                 ind_data += 1
                    mean_trials = difference_pre_post_dynamic(ss=sess,
                                            tt=tt, reg=reg, duration_window_pre=2,
                                            tp_post=tp, return_trials_separate=True)
                    if len(mean_trials) == 0:
    #                     print(tp, reg, sess, tt ,'   no trials')
                        pass
                    else:  # add array of new values
                        dict_diff['diff_dff'] = np.concatenate((dict_diff['diff_dff'], mean_trials))
                        dict_diff['region'] = np.concatenate((dict_diff['region'], [reg.upper() for x in range(len(mean_trials))]))
                        dict_diff['trial_type'] = np.concatenate((dict_diff['trial_type'], [tt for x in range(len(mean_trials))]))
                        dict_diff['session'] = np.concatenate((dict_diff['session'], [sess.signature for x in range(len(mean_trials))]))
                        dict_diff['timepoint'] = np.concatenate((dict_diff['timepoint'], [tp.copy() for x in range(len(mean_trials))]))
                        dict_diff['new_trial_id'] = np.concatenate((dict_diff['new_trial_id'], [ind_data + x for x in range(1, len(mean_trials) + 1)]))  # continuing indices
                        ind_data = dict_diff['new_trial_id'][-1]
    dict_diff['timepoint'] = dict_diff['timepoint'].astype('float32')
    dict_diff['diff_dff'] = dict_diff['diff_dff'].astype('float32')
    df_dyn_differences = pd.DataFrame(dict_diff)
    return df_dyn_differences

def get_decoder_data_for_violin_plots(sessions, tp_list=[1.0, 4.0]):
    ## Which time points to include in violin plots:
      # in seconds

    region_list = ['s1', 's2']
    dict_df_test = {reg: {} for reg in region_list}
    for reg in region_list:
        for tp in tp_list:  # retrain (deterministic) decoders for these time points, and save detailed info
            _, dict_df_test[reg][tp], __, ___ = train_test_all_sessions(sessions=sessions, verbose=0,# n_split=n_split,
                                        trial_times_use=np.array([tp]), return_decoder_weights=False,
                                        include_autoreward=False, neurons_selection=reg,
                                        C_value=50, reg_type='l2', train_projected=False)

    ## turn into df that can be used for violin plots efficiently,
    ## normalised so that each animals is equally important in averaging
    violin_df_test = make_violin_df_custom(input_dict_df=dict_df_test,
                                            flat_normalise_ntrials=True, verbose=1)
    return violin_df_test

def label_urh_arm(sessions, verbose=1):
    for key, ss in sessions.items():
        ss.outcome[ss.autorewarded] = 'arm'
        ss.outcome[ss.unrewarded_hits] = 'urh'
    if verbose > 0:
        print('URH and ARM trials have been labelled')

def create_df_table_details(sessions, exclude_150stim=True, count_hit_miss_nstim=False):
    """Create Dataframe table with details of sessions."""
    n_sessions = len(sessions)
    if exclude_150stim:
            str_trials = 'Trials (excl 150)'
    else:
        str_trials = 'Trials (incl 150)'
    column_names = ['Mouse', 'Run', 'f (Hz)', #'# Imaging planes',
                    r"$N$" + 'S1', r"$N$" + 'S2',
                   str_trials, 'Hit', 'FP', 'Miss', 'CR', 'UR Hit', 'AR Miss', 'Too early', 'Spont']
    if count_hit_miss_nstim:
        column_names = column_names + [f'{tt}_{n_stim}' for tt in ['hit', 'miss'] for n_stim in [5, 10, 20, 30, 40, 50, 150]]
    dict_details = {cc: np.zeros(n_sessions, dtype='object') for cc in column_names}
    for key, ss in sessions.items():
        dict_details['Mouse'][key] = ss.mouse
        dict_details['Run'][key] = ss.run_number
        dict_details['f (Hz)'][key] = ss.frequency
#         dict_details['# Imaging planes'][key] = len(np.unique(ss.plane_number))
        dict_details[r"$N$" + 'S1'][key] = np.sum(ss.s1_bool)
        dict_details[r"$N$" + 'S2'][key] = np.sum(ss.s2_bool)
        if exclude_150stim:
            leave_out_150_inds = ss.photostim < 2
        else:
            leave_out_150_inds = np.ones(len(ss.photostim), dtype='bool')
        dict_details[str_trials][key] = len(ss.outcome[leave_out_150_inds])
        # print(ss.name, ss.n_trials, len(ss.outcome))
        dict_details['Hit'][key] = np.sum(ss.outcome[leave_out_150_inds] == 'hit')
        dict_details['FP'][key] = np.sum(ss.outcome[leave_out_150_inds] == 'fp')
        dict_details['Miss'][key] = np.sum(np.logical_and(ss.outcome[leave_out_150_inds] == 'miss',
                                                          ss.autorewarded[leave_out_150_inds] == False))
        dict_details['CR'][key] = np.sum(ss.outcome[leave_out_150_inds] == 'cr')
        dict_details['Too early'][key] = np.sum(ss.outcome[leave_out_150_inds] == 'too_')
        dict_details['UR Hit'][key] = np.sum(ss.unrewarded_hits[leave_out_150_inds])
        dict_details['AR Miss'][key] = np.sum(ss.autorewarded[leave_out_150_inds])
        dict_details['Spont'][key] = ss.pre_rew_trials.shape[1]
        if count_hit_miss_nstim:
            for tt in ['hit', 'miss']:
                for n_stim in [5, 10, 20, 30, 40, 50, 150]:
                    dict_details[f'{tt}_{n_stim}'][key] = np.sum(np.logical_and(ss.outcome[leave_out_150_inds] == tt,
                                                                                ss.trial_subsets[leave_out_150_inds] == n_stim))
        n_trials_summed_tt = [dict_details[xx][key] for xx in ['Hit', 'FP', 'Miss', 'CR', 'Too early', 'AR Miss', 'UR Hit']]
        if np.sum(ss.unrewarded_hits) > 0:
            assert np.unique(ss.outcome[ss.unrewarded_hits]) == ['urh'], f'urh not correctly labelled for {key, ss}: {np.unique(ss.outcome[ss.unrewarded_hits])}'
        if np.sum(ss.autorewarded) > 0:
            assert np.unique(ss.outcome[ss.autorewarded]) == ['arm'], f'arm not correctly labelled for {key, ss}: {np.unique(ss.outcome[ss.autorewarded])}'
        assert np.sum(n_trials_summed_tt) == dict_details[str_trials][key], f'total number of trials in {key, ss} is not correct: {n_trials_summed_tt, dict_details[str_trials][key]}'
    df_details = pd.DataFrame(dict_details)
    df_details = df_details.sort_values(by=['Mouse', 'Run'])
    df_details = df_details.reset_index()
    del df_details['index']
    return df_details

def perform_logreg_cv(sessions, c_value_array, reg_list=['s1', 's2']):
    """Takes max over regions if both are given"""
    max_acc_scores = {}
    for key, ss in sessions.items():
        print(ss)
        ## First without reg:
        max_dec_values = np.zeros((len(c_value_array) + 1, 2))
        (lick_acc, lick_acc_split, ps_acc, ps_acc_split, ps_pred_split, lick_half,
                 angle_dec, dec_weights) = compute_accuracy_time_array(sessions={0: ss}, time_array=tp_dict['cv_reg'],
                                                              projected_data=False, reg_type='none',
                                                              region_list=reg_list,
                                                              average_fun=class_av_mean_accuracy)
        assert len(lick_acc) == 1
        mr_name = list(lick_acc.keys())[0]
        max_lick_dec = np.max(lick_acc[mr_name])  #np.max([np.max(reg_acc[:, 0]) for _, reg_acc in lick_acc.items()])
        max_ps_dec = np.max(ps_acc[mr_name])  #np.max([np.max(reg_acc[:, 0]) for _, reg_acc in ps_acc.items()])
        max_dec_values[0, :] = max_lick_dec.copy(), max_ps_dec.copy()
        ## Then with varying reg strengths:
        for i_c, c_value in enumerate(c_value_array):
            ## Compute results
            (lick_acc, lick_acc_split, ps_acc, ps_acc_split, ps_pred_split, lick_half,
                 angle_dec, dec_weights) = compute_accuracy_time_array(sessions={0: ss}, time_array=tp_dict['cv_reg'],
                                                              projected_data=False,
                                                              reg_type='l2', regularizer=c_value,
                                                              region_list=reg_list,
                                                              average_fun=class_av_mean_accuracy)
            assert len(lick_acc) == 1
#             mr_name = list(lick_acc.keys())[0]
            max_lick_dec = np.max(lick_acc[reg][:, 0]) # np.max(lick_acc[mr_name])  #np.max([np.max(reg_acc[:, 0]) for _, reg_acc in lick_acc.items()])
            max_ps_dec = np.max(ps_acc[reg][:, 0]) #np.max(ps_acc[mr_name])  #np.max([np.max(reg_acc[:, 0]) for _, reg_acc in ps_acc.items()])
            max_dec_values[i_c + 1, :] = max_lick_dec.copy(), max_ps_dec.copy()
        ## Save:
        max_acc_scores[key] = max_dec_values.copy()
    return max_acc_scores

def create_tp_dict(sessions):
    ## Integrate different imaging frequencies to create an array of mutual (shared) time points:
    freqs = np.unique([ss.frequency for _, ss in sessions.items()])
    tp_dict = {}
    for ff in freqs:
        for _, ss in sessions.items():   # assume pre_seconds & post_seconds equal for all sessions
            if ss.frequency == ff:
                tp_dict[ff] = ss.filter_ps_time
    if len(freqs) == 2:  # for hard-coded bit next up
        tp_dict['mutual'] = np.intersect1d(ar1=tp_dict[freqs[0]], ar2=tp_dict[freqs[1]])
    elif len(freqs) == 1:
        tp_dict['mutual'] = tp_dict[freqs[0]]
    return tp_dict

def opt_leaf(w_mat, dim=0, link_metric='correlation'):
    '''create optimal leaf order over dim, of matrix w_mat.
    see also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html#scipy.cluster.hierarchy.optimal_leaf_ordering'''
    assert w_mat.ndim == 2
    if dim == 1:  # transpose to get right dim in shape
        w_mat = w_mat.T
    dist = scipy.spatial.distance.pdist(w_mat, metric=link_metric)  # distanc ematrix
    link_mat = scipy.cluster.hierarchy.ward(dist)  # linkage matrix
    if link_metric == 'euclidean':
        opt_leaves = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(link_mat, dist))
        # print('OPTIMAL LEAF SOSRTING AND EUCLIDEAN USED')
    elif link_metric == 'correlation':
        opt_leaves = scipy.cluster.hierarchy.leaves_list(link_mat)
    return opt_leaves, (link_mat, dist)


def get_percent_cells_responding(session, region='s1', direction='positive', prereward=False):

    # Haven't built this for 5 Hz data
    assert session.mouse not in ['J048', 'RL048']

    # 0.015 gives you 5% of cells responding (positive + negative)
    # on cr (for session 0)
    # Get me for 5% across all 
    fdr_rate = 0.015

    ## Get data:
    if not prereward:
        flu = session.behaviour_trials
    else:
        flu = session.pre_rew_trials
    times_use = session.filter_ps_time
    if region == 's1':
        flu = flu[session.s1_bool, :, :]
    elif region == 's2':
        flu = flu[session.s2_bool, :, :]
    
    percent_cells_responding = []
    magnitude = []

    for trial_idx in range(flu.shape[1]):
        trial = flu[:, trial_idx, :]

        ## 500 ms before the stim with a nice juicy buffer to the artifact just in case
        pre_idx = np.where(times_use < -0.15)[0][-15:]  

        ## You can dial this back closer to the artifact if you cut out 150
        post_idx = np.logical_and(times_use > 1, times_use <= 1.5)
        
        pre_array = trial[:, pre_idx]
        post_array = trial[:, post_idx]
        
        p_vals = [scipy.stats.wilcoxon(pre, post)[1] for pre, post in zip(pre_array, post_array)]
        p_vals = np.array(p_vals)
        
        sig_cells, correct_pval, _, _ = multitest.multipletests(p_vals, alpha=fdr_rate, method='fdr_bh',
                                                            is_sorted=False, returnsorted=False)
        
        ## This doesn't split by positive and negative percent_cells_responding.append(sum(sig_cells))
        positive = np.mean(post_array, 1) > np.mean(pre_array, 1)
        negative = np.logical_not(positive)        
        
        if direction == 'positive':
            percent_cells_responding.append(np.sum(np.logical_and(sig_cells, positive)))
            magnitude.append(np.sum(np.mean(post_array[positive, :], 1) - np.mean(pre_array[positive, :] , 1)))
        else:
            percent_cells_responding.append(np.sum(np.logical_and(sig_cells, negative)))
            magnitude.append(np.sum(np.mean(post_array[negative, :], 1) - np.mean(pre_array[negative, :] , 1)))
        
    if region == 's1':
        n = np.sum(session.s1_bool)
    elif region == 's2':
        n = np.sum(session.s2_bool)
        
    percent_cells_responding = np.array(percent_cells_responding) / n * 100
    
    assert len(percent_cells_responding) == flu.shape[1]
    assert len(magnitude) == flu.shape[1]
    return percent_cells_responding

def get_data_dict(lm_list, region, tt_plot=['hit', 'miss', 'cr', 'fp', 'spont']):
    ''' Gets the percent cells responding across all trials for individual sessions
        Hit and miss trials, only when n_cells_stimmed > 20 (50% behaviour 
        sigmoid threshold)
    '''
    data_dict = {k:[] for k in tt_plot}

    for session_idx in range(len(lm_list)):
        session = lm_list[session_idx].session

        n_responders =  get_percent_cells_responding(session, region, direction='positive')\
                        + get_percent_cells_responding(session, region, direction='negative')

        for tt in tt_plot:
            if tt == 'spont':
                continue

            tt_idx = session.outcome == tt
#             if tt in ['hit', 'miss']:
#                 tt_idx = np.logical_and(tt_idx, session.trial_subsets>20)
            data_dict[tt].append(np.mean(n_responders[tt_idx]))

        n_responders =  get_percent_cells_responding(session, region, direction='positive', prereward=True)\
                        + get_percent_cells_responding(session, region, direction='negative', prereward=True)

        data_dict['spont'].append(np.mean(n_responders))
    
    data_dict['Hit'] = data_dict.pop('hit')
    data_dict['Miss'] = data_dict.pop('miss')
    data_dict['CR'] = data_dict.pop('cr')
    data_dict['FP'] = data_dict.pop('fp')
    data_dict['Reward\nonly'] = data_dict.pop('spont')
    
    return {k:np.array(v) for k, v in data_dict.items()}
    
def interdep_corr_balance(dict_activ_hit, dict_activ_miss, reg='s2'):
    dict_activ_dict = {'hit': dict_activ_hit, 'miss': dict_activ_miss}
    dict_exc, dict_inh = {}, {}
    full_arr_all = {xx: {x: np.array([]) for x in ['hit', 'miss']} for xx in ['exc', 'inh']}
    for tt in ['hit', 'miss']:
        dict_exc[tt] = dict_activ_dict[tt][reg]['positive']
        dict_inh[tt] = dict_activ_dict[tt][reg]['negative']
        assert (dict_exc[tt].keys() == dict_inh[tt].keys())
        list_nstim = list(dict_exc[tt].keys())

    for n_stim in list_nstim:
        arr_exc, arr_inh = {}, {}
        for tt in ['hit', 'miss']:
            arr_exc[tt] = 1
            arr_exc[tt] = dict_exc[tt][n_stim].copy()
            arr_inh[tt] = dict_inh[tt][n_stim].copy()
        nn = np.where(np.logical_and(~np.isnan(arr_inh['hit']), ~np.isnan(arr_inh['miss'])))  # filter nans (when no trial present)
        
        for tt in ['hit', 'miss']:
            arr_exc[tt] = arr_exc[tt][nn]
            arr_inh[tt] = arr_inh[tt][nn]
            full_arr_all['exc'][tt] = np.concatenate((full_arr_all['exc'][tt], arr_exc[tt].copy()))
            full_arr_all['inh'][tt] = np.concatenate((full_arr_all['inh'][tt], arr_inh[tt].copy()))

    corr_dict = {}
    list_combi_tuples = [('hit', 'exc'), ('hit', 'inh'), ('miss', 'exc'), ('miss', 'inh')]
    for i_tup_1 in range(4):
        for i_tup_2 in range(i_tup_1, 4):
            tup_1 = list_combi_tuples[i_tup_1]
            tup_2 = list_combi_tuples[i_tup_2]
            tt_1, res_1 = tup_1
            tt_2, res_2 = tup_2
            corr_dict[f'{tt_1}_{res_1} vs {tt_2}_{res_2}'] = np.corrcoef(full_arr_all[res_1][tt_1], 
                                                                         full_arr_all[res_2][tt_2])[1, 0]

    return corr_dict


def transfer_dict(msm, region, direction='positive'):
    '''For each session, compute how many responding cells [in direction] there are 
    for both hit and miss trials. '''
    n_cells_list_of_lists = [[5], [10], [20], [30], [40], [50], [150]]
    # n_cells_list_of_lists = [[5,10], [20,30], [40,50], [150]]
    hit_mean_dict, miss_mean_dict = {}, {}
    hit_var_dict, miss_var_dict = {}, {}
    n_sessions = len(msm.linear_models)
    for session_idx in range(n_sessions):
        session = msm.linear_models[session_idx].session
        n_responders = get_percent_cells_responding(session, region=region, direction=direction)

        for n_cells in n_cells_list_of_lists:
            idx = np.isin(session.trial_subsets, n_cells)
            idx_miss = np.logical_and(idx, session.outcome == 'miss')
            idx_hit = np.logical_and(idx, session.outcome == 'hit')

            centre_cells = np.mean(n_cells)
            if centre_cells not in hit_mean_dict:
                hit_mean_dict[centre_cells] = np.zeros(n_sessions)
                miss_mean_dict[centre_cells] = np.zeros(n_sessions)
                hit_var_dict[centre_cells] = np.zeros(n_sessions)
                miss_var_dict[centre_cells] = np.zeros(n_sessions)

            hit_mean_dict[centre_cells][session_idx] = np.mean(n_responders[idx_hit])
            miss_mean_dict[centre_cells][session_idx] = np.mean(n_responders[idx_miss])
            hit_var_dict[centre_cells][session_idx] = np.var(n_responders[idx_hit])
            miss_var_dict[centre_cells][session_idx] = np.var(n_responders[idx_miss])
            
    return hit_mean_dict, miss_mean_dict, hit_var_dict, miss_var_dict

def baseline_subtraction(flu, lm):
    ''' Takes a cell averaged flu matrix [n_trials x time]
        and subtracts pre-stim activity of an individual trial 
        from every timepoint in that trial.
        '''
    baseline = np.mean(flu[:, lm.frames_map['pre']], 1)
    flu = np.subtract(flu.T, baseline).T
    return flu

def session_flu(lm, region, outcome, frames, n_cells, subtract_baseline=True):

    (data_use_mat_norm, data_use_mat_norm_s1, data_use_mat_norm_s2, data_spont_mat_norm, ol_neurons_s1, ol_neurons_s2, outcome_arr,
        time_ticks, time_tick_labels, time_axis) = pop.normalise_raster_data(session=lm.session, 
                            sort_neurons=False, filter_150_stim=False)
    
    assert (outcome_arr == lm.session.outcome).all()
    # print(np.where(lm.frames_map['pre'])[0])
    # print(lm.session.filter_ps_time[np.where(lm.frames_map['pre'])[0]])
    # print(np.where(lm.frames_map['post'])[0])
    # print(lm.session.filter_ps_time[np.where(lm.frames_map['post'])[0]])
    # Select region and trial outcomes
    if outcome != 'pre_reward':
        flu = data_use_mat_norm  # lm.flu
        # assert data_use_mat_norm.shape == lm.flu.shape, print(f'{data_use_mat_norm.shape}, {lm.flu.shape}')
        outcome_bool = lm.session.outcome == outcome
        
        if outcome in ['hit', 'miss']:
            n_stimmed_bool = np.isin(lm.session.trial_subsets, n_cells)
            outcome_bool = np.logical_and(outcome_bool, n_stimmed_bool)
        
        flu = flu[:, outcome_bool, :]
    else:
        flu = data_spont_mat_norm  # lm.pre_flu
    
    flu = flu[lm.region_map[region], :, :]

    # Mean across cells
    flu = np.mean(flu, 0)
    
    # if subtract_baseline:
    #     flu = baseline_subtraction(flu, lm)
        
    # Select desired frames
    if frames != 'all':
        flu = flu[:, lm.frames_map[frames]]
    
    return flu, time_axis

def select_cells_and_frames(lm, region='s1', frames='pre'):
    flu = lm.flu
    flu = flu[lm.region_map[region], :, :]
    flu = flu[:, :, lm.frames_map[frames]]
    return flu

def get_covariates(lm, region, match_tnums=False, prereward=False, hitmiss_only=False,
                    filter_150=False):
    
    covariate_dict, y = lm.prepare_data(frames='all', model='partial', n_comps_include=0,
                                        outcomes=(['hit', 'miss'] if hitmiss_only else np.unique(lm.session.outcome)), 
                                        prereward=prereward,
                                        region=region, return_matrix=False,
                                        remove_easy=filter_150)
    # if prereward is False:
    #     assert (np.where(np.isin(lm.session.outcome, ['hit', 'miss']))[0] == covariate_dict['trial_number_original']).all()
    covariate_dict['y'] = y
    
    if match_tnums:
        hit_idx = np.where(y==1)[0]
        miss_idx = np.where(y==0)[0]
        n_misses = len(miss_idx)
        assert False, 'ensure n_misses > n_hits, replace False in choice? '
        hit_idx = np.random.choice(hit_idx, size=n_misses)
        keep_idx = np.hstack((hit_idx, miss_idx))
        covariate_dict = {k:v[keep_idx] for k,v in covariate_dict.items()}
        y = y[keep_idx]
    
        assert sum(y==0) == sum(y==1)
    
    return covariate_dict

def add_vcr_to_lm(lm_list, hard_reset=True, zscore=True):
    for ilm, linear_model in tqdm(enumerate(lm_list)):
        if (hard_reset is True) or ((hard_reset is False) and (hasattr(linear_model.session, 'cov_dict') is False)):  # make dict:
            linear_model.session.cov_dict = {}
            linear_model.session.cov_dict_reward_only = {}
        for reg in ['s1', 's2']:
            ## Regular trials:
            cov_dict = get_covariates(linear_model, reg)
            tmp_arr = np.zeros(len(linear_model.session.outcome)) + np.nan 
            if zscore:
                tmp_arr[cov_dict['trial_number_original']] = scipy.stats.zscore(cov_dict['variance_cell_rates'])
            else:
                tmp_arr[cov_dict['trial_number_original']] = cov_dict['variance_cell_rates']
            linear_model.session.cov_dict['variance_cell_rates_' + reg] = copy.deepcopy(tmp_arr)

            ## Reward only trials:
            cov_dict = get_covariates(linear_model, reg, prereward=True)
            tmp_arr = np.zeros(linear_model.session.pre_rew_trials.shape[1]) + np.nan 
            if zscore:
                tmp_arr[cov_dict['trial_number_original']] = scipy.stats.zscore(cov_dict['variance_cell_rates'])
            else:
                tmp_arr[cov_dict['trial_number_original']] = cov_dict['variance_cell_rates']
            linear_model.session.cov_dict_reward_only['variance_cell_rates_' + reg] = copy.deepcopy(tmp_arr)

def create_df_from_cov_dicts(cov_dicts, zscore_list=[]):
    cov_dicts = copy.deepcopy(cov_dicts)
    n_sessions = len(cov_dicts)
    if len(zscore_list) > 0:
        for zscore_covar_name in zscore_list:
            if zscore_covar_name in cov_dicts[0].keys():
                for i_ss in range(n_sessions):
                    tmp_vcr = cov_dicts[i_ss][zscore_covar_name]
                    cov_dicts[i_ss][zscore_covar_name] = scipy.stats.zscore(tmp_vcr)

    list_dfs = [pd.DataFrame(v) for v in cov_dicts.values()]
    super_df = pd.concat(list_dfs, ignore_index=True)
    return super_df

def compute_density_hit_miss_covar(super_covar_df, cov_name='variance_cell_rates', 
                          include_150=True, n_bins_covar=7, #zscore_covar=False,
                          metric='fraction_hit', verbose=0):
    n_stim_arr = [5, 10, 20, 30, 40, 50]
    if include_150:
        n_stim_arr = n_stim_arr + [150]
    assert (np.unique(super_covar_df['n_cells_stimmed']) == n_stim_arr).all()
    assert include_150 is True, 'not implemented'
    if len(n_stim_arr) == n_bins_covar:   # if square matrix
        compute_collapsed_density = True 
    else:
        compute_collapsed_density = False 
    n_stim_arr = np.array(n_stim_arr)

    ## Create arrays with percentiles (bins) and stuff
    all_cov_arr = np.sort(super_covar_df[cov_name])
    percentile_arr = np.linspace(100 / n_bins_covar, 100, n_bins_covar)
    cov_perc_arr = np.array([np.percentile(all_cov_arr, x) for x in percentile_arr])
    median_cov_perc_arr = np.array([np.percentile(all_cov_arr, x - 5) for x in percentile_arr])
    mat_fraction = np.zeros((len(n_stim_arr), len(cov_perc_arr)))
    
    total_n = 0
    for i_nstim, n_stim in enumerate(n_stim_arr):  # select bin and compute fraction hit
        prev_perc = np.min(all_cov_arr) - 10
        for i_cov_perc, cov_perc in enumerate(cov_perc_arr):
            sub_df = super_covar_df.loc[(super_covar_df['n_cells_stimmed'] == n_stim) &
                                        (super_covar_df[cov_name] > prev_perc) &
                                        (super_covar_df[cov_name] <= cov_perc)]          
            total_n += len(sub_df)
            if len(sub_df['y']) > 0:
                fraction_hit_miss = np.sum(sub_df['y']) / len(sub_df['y'])
            else:
                fraction_hit_miss = np.nan
            if metric == 'fraction_hit':
                mat_fraction[i_nstim, i_cov_perc] = fraction_hit_miss
            elif metric == 'occupancy':
                mat_fraction[i_nstim, i_cov_perc] = len(sub_df['y'])
            prev_perc = cov_perc
    assert total_n == len(super_covar_df), f'not all rows have been used: {total_n}/{len(super_covar_df)}'
  
    if compute_collapsed_density and metric == 'fraction_hit':
        arr_diag_index = np.arange(-n_bins_covar + 1, n_bins_covar - 1)
        mean_mat_arr = np.zeros(len(arr_diag_index))
        ci_mat_arr = np.zeros(len(arr_diag_index))
        for i_diag_index, diag_index in enumerate(arr_diag_index):
            mat_inds = np.where(np.eye(n_bins_covar, k=diag_index) == 1)  # inds of diagonal with 'same SNR'
            tmp_n_stim_inds_arr, tmp_cov_perc_inds_arr = mat_inds 

            cov_perc_arr_incl_min = np.concatenate((np.array([np.min(all_cov_arr) - 10]), ## -10 to be sure its including the min of all_cov_arr
                                                    cov_perc_arr))
            collapsed_df = None
            for i_el in range(len(tmp_n_stim_inds_arr)):  # loop through bins that are in this diagonal
                i_nstim = tmp_n_stim_inds_arr[i_el]
                i_cov_perc = tmp_cov_perc_inds_arr[i_el]

                n_stim = n_stim_arr[i_nstim]
                lower_cov = cov_perc_arr_incl_min[i_cov_perc]
                upper_cov = cov_perc_arr_incl_min[i_cov_perc + 1]

                sub_df = super_covar_df.loc[(super_covar_df['n_cells_stimmed'] == n_stim) &
                                            (super_covar_df[cov_name] > lower_cov) &
                                            (super_covar_df[cov_name] <= upper_cov)]      
                # print('part', np.mean(sub_df['y']), len(sub_df['y']))
                if collapsed_df is None:  # concat bins into 1 df
                    collapsed_df = sub_df
                else:
                    collapsed_df = pd.concat([collapsed_df, sub_df], ignore_index=True)
                # print(len(sub_df), len(collapsed_df))
            if len(collapsed_df['y']) > 0:  # compute fraction hit trials
                fraction_hit_miss = np.sum(collapsed_df['y']) / len(collapsed_df['y'])
                ci = 1.96 * np.sqrt(fraction_hit_miss * (1 - fraction_hit_miss)) / np.sqrt(len(collapsed_df))  # ci of bernoullie distr
            else:
                fraction_hit_miss = np.nan
            # print('pool', fraction_hit_miss, len(collapsed_df['y']))
            mean_mat_arr[i_diag_index] = fraction_hit_miss
            ci_mat_arr[i_diag_index] = ci 
            if i_diag_index == 0:
                total_df = collapsed_df
            else:
                total_df = pd.concat([total_df, collapsed_df])
        hit_label = total_df['y']
        indep_var = total_df[cov_name]
        model = sm.GLM(hit_label, sm.add_constant(indep_var), family=sm.families.Binomial())
        results = model.fit() 
        if verbose:
            print('Summary GLM:')
            print(results.summary())
            print('P values:')
            print(results.pvalues)
    else:
        mean_mat_arr, ci_mat_arr = None, None 

    return (mat_fraction, median_cov_perc_arr, cov_perc_arr, n_stim_arr), (mean_mat_arr, ci_mat_arr) 

def get_subset_dprime(session):
    assert session.trial_subsets.shape == session.outcome.shape
    
    outcome_arr = session.outcome
    trial_subsets_arr = session.trial_subsets

    fp_rate = np.sum(outcome_arr == 'fp') / (np.sum(outcome_arr == 'fp') + np.sum(outcome_arr == 'cr'))
    subset_dprimes = []
#     for subset in [[5, 10], [20, 30], [40, 50], 1[50]]:
    for subset in [[5], [10], [20], [30], [40], [50], [150]]:
        idx = np.isin(trial_subsets_arr, subset)
        outcome = outcome_arr[idx]
        hit_rate = np.sum(outcome == 'hit') / (np.sum(outcome == 'hit') + np.sum(outcome == 'miss'))
        subset_dprimes.append(utils.utils_funcs.d_prime(hit_rate, fp_rate))
    
    return subset_dprimes

def get_alltrials_dprime(session, trial_inds=None):
    assert session.trial_subsets.shape == session.outcome.shape
    if trial_inds is None:
        trial_inds = np.arange(len(session.outcome))
    outcome_arr = session.outcome[trial_inds]
    trial_subsets_arr = session.trial_subsets[trial_inds]

    fp_rate = np.sum(session.outcome == 'fp') / (np.sum(session.outcome == 'fp') + np.sum(session.outcome == 'cr'))
    hit_rate = np.sum(outcome_arr == 'hit') / (np.sum(outcome_arr == 'hit') + np.sum(outcome_arr == 'miss'))
    dprime = utils.utils_funcs.d_prime(hit_rate, fp_rate)

    return dprime

def pf(x, max_value, alpha, beta):
    # psychometric function
    ''' Max value: max value of sigmoid
        alpha: x_axis midpoint
        beta: the growth rate
    '''
    return max_value / (1 + np.exp(-(x - alpha) / beta)) 

def trial_binner(arr):
    group_dict = {0: '0',
                    5: '5-10',
                    10: '5-10',
                    20: '20-30',
                    30: '20-30',
                    40: '40-50',
                    50: '40-50',
                    150: '150'}
    return np.array([group_dict[a] for a in arr])

def log_reg_covars(covar_dict, list_x_var=['mean_pre', 'corr_pre', 'variance_cell_rates'], 
                    region='s1', hard_balance=True, zscore_data=False):
    covar_dict = covar_dict[region]
    n_sessions = len(covar_dict)

    mean_pred_dict = {x: np.zeros(n_sessions) for x in list_x_var}
    var_pred_dict = {x: np.zeros(n_sessions) for x in list_x_var}

    for i_x_var, x_var in enumerate(list_x_var):
        for i_s in range(n_sessions):

            x_var_array = -1* copy.deepcopy(covar_dict[i_s][x_var])

            y_array = copy.deepcopy(covar_dict[i_s]['y'])
            
            if zscore_data:
                x_var_array = scipy.stats.zscore(x_var_array)
            if hard_balance:
                n_trials_per_tt = np.minimum(np.sum(y_array == 0), np.sum(y_array == 1))
                trial_inds_0 = np.random.choice(a=np.where(y_array == 0)[0], size=n_trials_per_tt, replace=False)
                trial_inds_1 = np.random.choice(a=np.where(y_array == 1)[0], size=n_trials_per_tt, replace=False)
                subsample_trial_inds = np.concatenate((trial_inds_0, trial_inds_1))
                x_var_array = x_var_array[subsample_trial_inds]
                y_array = y_array[subsample_trial_inds]
                
            print(x_var, np.mean(x_var_array[y_array == 1]) > np.mean(x_var_array[y_array ==0]))
            pred_arr = np.zeros(len(y_array))
            # y_array[x_var_array < 0] = 0
            # y_array[x_var_array > 0] = 1
            kfolds = sklearn.model_selection.StratifiedKFold(n_splits=4)
            
            for train_idx, test_idx in kfolds.split(np.zeros_like(x_var_array), y_array):

                x_var_array_train, x_var_array_test = x_var_array[train_idx].reshape(-1, 1), x_var_array[test_idx].reshape(-1, 1)
                y_array_train, y_array_test = y_array[train_idx], y_array[test_idx]
                
                # model = sklearn.linear_model.LogisticRegression(penalty='l1', C=0.5, solver='saga',
                                                                # class_weight='balanced')
                # model = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis() 
                # model.fit(X=x_var_array_train, y=y_array_train)
                                # print(x_var, i_s, model.score(x_var_array_test, y=y_array_test))
                # model_predictions = model.predict(x_var_array_test)
                
                if x_var == 'variance_cell_rates':
                    x_var_hit_mean = np.mean(x_var_array_train[y_array_train == 1])
                    x_var_miss_mean = np.mean(x_var_array_train[y_array_train == 0])

                    # print(x_var_hit_mean > x_var_miss_mean)
                    
                
            #     pred_arr[test_idx] = sklearn.metrics.accuracy_score(y_array_test, model_predictions)
            # mean_pred_dict[x_var][i_s] = np.mean(pred_arr)
            # var_pred_dict[x_var][i_s] = np.var(pred_arr)
            
    return mean_pred_dict, var_pred_dict
