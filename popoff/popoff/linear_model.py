import os
import sys
import pdb
import traceback
import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA, NMF, FactorAnalysis
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, f1_score, balanced_accuracy_score
from scipy import sparse
from scipy import stats
from average_traces import AverageTraces
from pop_off_functions import prob_correct, mean_accuracy, score_nonbinary
from Session import build_flu_array_single, SessionLite
from utils_funcs import build_flu_array
import copy
import pickle
from popoff import loadpaths
from IPython.display import HTML, display
from IPython.core.debugger import Pdb
ipdb = Pdb()

USER_PATHS_DICT = loadpaths.loadpaths()

COLORS = [

    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41',  # matrix green
]


def do_fa(X, n_components, plot=False):
    ''' Run FactorAnalysis on data matrix

    Parameters
    -----------
    X : neural_activity matrix. [n_cells x time]
           (transposed before fa)
    n_components: number of factors to compute
    plot : bool, default False
           Plot varience explained curve?
    '''
    model = FactorAnalysis(n_components=n_components)
    model.fit(X.T)
    components = model.components_
    noise = model.noise_variance_
    square_summed = np.sum(components**2, axis=1)

    varexp = []

    for comp in range(n_components):
        varexp.append(
            square_summed[comp] #/ (np.sum(square_summed) + np.sum(noise))
            )

    if plot:
        plt.plot(varexp, color='black', lw=4)
        plt.xlabel('Factor ID')
        plt.ylabel("% Variance explained\n(pre-stimulus)")
        plt.xticks(range(n_components), range(n_components))
    return varexp, components


def do_pca(X, n_components, plot=False):
    ''' Run PCA on data matrix

    Parameters
    -----------
    X : neural_activity matrix. [n_cells x time]
           (transposed before pca)
    n_components: number of principle components to compute
    plot : bool, default False
           Plot varience explained curve?
    '''

    model = PCA(n_components=n_components)
    model.fit(X.T)
    varexp = np.cumsum(model.explained_variance_ratio_)
    # Projection of neural activity on principal axes
    components = np.dot(model.components_, X)
    loading = model.components_
    assert components.shape == (n_components, X.shape[1])

    if plot:
        plt.plot(model.explained_variance_ratio_, color='black', lw=4)
        plt.xlabel('Principal Component')
        plt.ylabel("Variance explained\n(pre-stimulus)")
        plt.xticks(range(n_components), range(n_components))
    return varexp, components, loading


def pca_session(session, cells_include, n_components=100, plot=False,
                save_PC_matrix=False):
    ''' Appends comps attributes to session objects

    Runs PCA on the full fluoresence matrix then uses "backend" functions
    from Session.py to create 3d array in the same way as
    session.behaviour_trials is created.
    This requires a slightly awkward import and modifcation of
    build_flu_array_X as Session.py was not built to be used in this way

    Parameters
    -----------
    session : Session object
    n_components : Number of PCs to compute
    plot : bool, default False
        Plot varience explained curve?

    Appends
    ---------
    session.comps = [n_components, n_trials, n_frames] array
    session.comps_pre = [n_components, n_trials, n_frames] prereward array

    '''

    # Load in the full dff data
    session.load_data()
    run = session.run
    flu = run.flu[session.filtered_neurons, :]
    flu = flu[cells_include, :]

    _, components = do_pca(flu, n_components, plot=plot)
    # Hacky but straightforward, stick the components into the run object to send it into
    # build_flu_array_X
    run.comps = components

    # Need to build array [n_cells, n_trials, n_frames] from components in the same
    # way as session.behaviour_trials is built.

    if session.mouse in ['J048', 'RL048']:  # 5Hz 3 plane data
        # PCs are composed of multiple cells, so the don't have different frame
        # start times, so take the average start time across cells.
        # Mutating frames_ms is the most straightforward way to do this.
        # func = mode
        # run.frames_ms = np.tile(func(run.frames_ms, axis=0)[0], (n_components, 1))
        # run.frames_ms_pre = np.tile(func(run.frames_ms_pre, axis=0)[0], (n_components, 1))
        # Trying to take the average of frames_ms results in NaNs in the array, taking the
        # first n_components works though
        run.frames_ms = run.frames_ms[:n_components, :]
        run.frames_ms_pre = run.frames_ms[:n_components, :]

        arr = build_flu_array(run, session.galvo_ms, pre_frames=session.pre_frames,
                              post_frames=session.post_frames, use_comps=True)

        arr_pre = build_flu_array(run, session.run.pre_reward, pre_frames=session.pre_frames,
                              post_frames=session.post_frames, use_comps=True, is_prereward=True)

    else:  # 30Hz 1 plane
        arr = build_flu_array_single(run, pre_frames=session.pre_frames,
                                     post_frames=session.post_frames, use_comps=True, fs=30)

        arr_pre = build_flu_array_single(run, pre_frames=session.pre_frames,
                                     post_frames=session.post_frames, use_comps=True, fs=30,
                                     prereward=True)

    # Remove NaN trials
    session.comps = arr[:, session.nonnan_trials, :]
    nonnan_pre = np.unique(np.where(~np.isnan(arr_pre))[1])
    session.comps_pre = arr_pre[:, nonnan_pre, :]

    if not save_PC_matrix:
        session.clean_obj()  # "Garbage collection" to remove session.run

    return session


def mean_cross_correlation(flu, frames):

    ''' Takes the mean of the absolute off-diagonal
        values of the crosscorrelation matrix


    Parameters
    ----------
    flu : fluoresence array [n_cells x n_trials x n_frames]
    frames : indexing array, frames across which to compute correlation

    Returns
    -------
    trial_corr : vector of len n_trials ->
                 mean of correlation coefficient matrix matrix on each trial.

    '''

    trial_corr = []

    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        matrix = np.corrcoef(trial)
        matrix = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        matrix = np.abs(matrix)
        mean_cov = np.mean(matrix)
        trial_corr.append(mean_cov)

    return np.array(trial_corr)


def largest_singular_value(flu, frames, centre=False):

    singular_values = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        trial = trial.T
        if centre:
            mean_ = np.mean(trial, axis=0)
            trial -= mean_
        _, s, _ = np.linalg.svd(trial)
        singular_values.append(s[0])

    return np.array(singular_values)


def largest_PC_var(flu, frames):

    PC_vars = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        varexp, _, _ = do_pca(trial, 10, plot=False)
        PC_vars.append(varexp[0])

    return np.array(PC_vars)


def largest_factor_var(flu, frames):

    PC_vars = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        varexp, _ = do_fa(trial, 10, plot=False)
        PC_vars.append(varexp[0])

    return np.array(PC_vars)


def jonas_metric(flu, frames):

    jm = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        # mean across cells
        trial -= np.mean(trial)
        meaned = np.mean(trial, 0)
        jm.append(np.std(meaned))

    return np.array(jm)


def largest_PC_loading(flu, frames):

    PC_loading = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        pca = PCA(n_components=10)
        pca.fit(trial.T)

        # loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings = pca.components_.T
        PC_loading.append(loadings[:, 0])

    return np.array(PC_loading)


def largest_PC_trace(flu, frames):

    traces = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        _, trace = do_pca(trial, 10, plot=False)
        traces.append(trace[0, :])

    return traces


def flattened_variance(flu, frames):

    vars_ = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        vars_.append(np.var(trial))

    return np.array(vars_)


def variance_cell_rates(flu, frames): 

    vars_ = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        vars_.append(np.var(np.mean(trial, axis=1)))

    return np.array(vars_)


def mean_cell_variance(flu, frames):

    vars_ = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        vars_.append(np.mean(np.var(trial, axis=1)))

    return np.array(vars_)

def variance_pop_mean(flu, frames):

    vars_ = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        meaned = np.mean(trial, 0)
        vars_.append(np.var(meaned))

    return np.array(vars_)


def reward_history(session, window_size=5):

    binary_reward = (session.outcome == 'hit').astype('int')
    rolling_window = np.zeros(len(binary_reward))

    for trial_idx in range(len(binary_reward)):
        if trial_idx < window_size:
            rolling_window[trial_idx] = sum(binary_reward[:trial_idx])
        else:
            rolling_window[trial_idx] = sum(
                binary_reward[trial_idx-window_size:trial_idx])

    return rolling_window


class LabelEncoder():
    ''' Reimplement skearn.preprocessing.LabelEncoder for user defined label
        ordering

    Need to change sklearn's LabelEncoder as this object applies the sorted()
    function to label order, where I wish to define label order manually.
    The specific use case is to allow [miss=0, hit=1] and [cr=0, fp=1] when
    labels are encoded seperately.

    Parameters
    -----------
    y : array or list of labels to encode
    sorted_order : array or list specifying the order to sort the labels.
        elements of y are assigned labels based on their position in
        sorted_order.

    Attibutes
    -----------
    encoder : hash table used to convert y to labels
    inverse_encoder : encoder hash table with key / vals inverted.

    Methods
    --------
    fit : fit build encoder hash table from y based on sorted_order
    transform : encode y to integer labels
    inverse_transform : decode back to original y from integer labels

    '''

    def __init__(self, sorted_order):
        self.sorted_order = sorted_order

    def __str__(self):
        if hasattr(self, 'encoder'):
            return f'Labels will be encoded by hash table {self.encoder}'
        else:
            return 'FIT ME!!'

    def fit(self, y):

        labels = sorted(set(y), key=lambda v: self.sorted_order.index(v))
        self.encoder = {}
        self.inverse_encoder = {}

        for idx, label in enumerate(labels):
            self.encoder[label] = idx
            self.inverse_encoder[idx] = label

    def transform(self, y):
        return np.array([self.encoder[elem] for elem in y])

    def inverse_transform(self, y):
        return np.array([self.inverse_encoder[elem] for elem in y])


class LinearModel():

    def __init__(self, session, times_use, remove_targets=False, use_spks=False, 
                 remove_toosoon=False):
        ''' Perform logistic regression on Session object

        Attributes
        ----------
        session : individual session object
        times_use : inherited from AverageTraces object, times common
                    across sessions with different frame rates

        '''

        self.session = session
        if use_spks:
            self.session.behaviour_trials = self.session.spks_behaviour_trials
            print('WARNING: behaviour trials switched to spks')

        # times_use is inherited from AverageTraces, specify as an arugment
        # rather than using class inheritence so that this class does
        # not need to load every session
        self.times_use = times_use
        self.remove_targets = remove_targets

        self.setup_flu()
        self.target_info()

        if use_spks:
            self.trim_deconvolved()

        if self.remove_targets:
            self.remove_targets_from_data()

        # Init encoder with required sort order
        self.encoder = LabelEncoder(['miss', 'hit', 'cr', 'fp'])

        self.session.outcome = self.nan_removal(self.session.outcome)
        self.session.trial_subsets = self.nan_removal(
            self.session.trial_subsets)

        if remove_toosoon:
            self.too_sooner()

    def nan_removal(self, arr):
        ''' Sometimes nans are not removed from e.g. self.session.outcome
            like they should be in Session
            '''

        try:
            arr = arr[self.session.nonnan_trials]
        except IndexError:  # Already been non-nanned
            pass
        return arr

    def too_sooner(self):

        for trial in range(self.session.n_trials):

            lick = self.session.first_lick[trial]

            if lick is None:
                continue

            if self.session.outcome[trial] == 'hit' and lick < 250:
                self.session.outcome[trial] = 'too_soon'

    def setup_flu(self):
        ''' Setup self.flu data array [n_cells x n_trials x [n_frames]
            for use in subsequent functions.

            Also setup pre, post and remove_artifact attributes which can be
            used to index the n_frames dimension of flu array
            '''

        self.flu = self.session.behaviour_trials
        # session.frames_use is used to match imaging rates across sessions
        self.flu = self.flu[:, :, self.session.frames_use]

        self.pre_flu = self.session.pre_rew_trials
        self.pre_flu = self.pre_flu[:, :, self.session.frames_use]

        # Split the trace into pre=frames-before-stim and post=frames-after-stim
        # times_use inherited from AverageTraces
        self.pre = np.logical_and(self.times_use < -0.07, self.times_use > -2)

        long_post = True
        if long_post:
            self.post = self.times_use > 0.8
            print('long post time')
        else:
            self.post = np.logical_and(self.times_use > 0.8, self.times_use < 1.8)
            print('short post time')
        self.remove_artifact = np.logical_or(self.pre, self.post)

        # Allows future functions to use 'frames' argument as a string
        # to be mapped to matching indexing array
        self.frames_map = {'pre': self.pre,
                           'post': self.post,
                           'all': self.remove_artifact}

        self.region_map = {'all': np.repeat(True, self.session.n_cells),
                           's1': self.session.s1_bool,
                           's2': self.session.s2_bool
                           }

    def prepare_data(self, frames='all', model='full',
                     outcomes=['hit', 'miss', 'cr', 'fp'], region='all',
                     n_comps_include=0, prereward=False, remove_easy=False,
                     return_matrix=True):
        ''' Prepare fluoresence data in Session object for regression

        Parameters
        ----------
        session : Session object to get data from

        frames : {'pre', 'post', 'all'}, default='all'
            Which trial frames (relative to photostim) to return?

        model : {'full', 'partial'}, default='full'
            full = include mean activity of all cells in model
            partial = include only 'network features' e.g. PCS in model

        outcomes : {['hit', 'miss', 'fp', 'cr']}, default=['hit', 'miss' 'fp', 'cr']
            Which trial types do you want to decode on?

        region : {['s1', 's2', 'both']}, default='both'
            Include cells from which region in regression?

        n_comps_include : How many PCs to include in the partial model
            (requires model='partial')

        prereward : bool, default=False
            Just include prereward trials? Negates outcomes argument


        Returns
        --------
        X : data matrix for use as independent variable [n_samples x n_features]
        y : vector for use as dependent variable [n_samples]

        '''

        if prereward:
            flu = self.pre_flu
            y = np.zeros(flu.shape[1])
            trial_bool = y.astype('bool')

        else:
            outcome = self.session.outcome

            trial_bool = np.isin(outcome, outcomes)

            if remove_easy:
                test_and_catch = self.session.photostim != 2
                trial_bool = np.logical_and(trial_bool, test_and_catch)

            outcome = outcome[trial_bool]
            flu = self.flu[:, trial_bool, :]

            # Fit encoder on data that is independent of the trial
            # order in the session
            self.encoder.fit(outcome)
            print(self.encoder)
            y = self.encoder.transform(outcome)

        # Select cells from required region
        flu = flu[self.region_map[region], :, :]

        # For the full model the independent variables are the mean activity
        # across the whole trial for every cell
        if model == 'full':
            X = self.covariates_full(flu=flu, frames=frames)
        elif model == 'partial':
            covariates_dict = self.covariates_partial(flu=flu, frames=frames,
                                        trial_bool=trial_bool, region=region,
                                        n_comps_include=n_comps_include,
                                        prereward=prereward)

            X = self.dict2matrix(covariates_dict)
        else:
            raise ValueError(f'model {model} not recognised')

        if return_matrix:
            return X, y
        else:
            return covariates_dict, y

    def transform_data(self, X):
        # Get input matrix to (n_samples, n_features)
        X = X.T
        # Demean and scale to unit varience
        scaler = sklearn.preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        return X

    def dict2matrix(self, dict_):
        X = np.vstack([v for v in dict_.values()])
        return self.transform_data(X)

    def covariates_full(self, flu, frames):

        # Use this to check that PC performance
        # was not ~= full model as PCs are subbed but cells are
        # not. Confirmed that performance for full subbed is roughly
        # the same
        if frames == 'subbed':
            pre = flu[:, :, self.frames_map['pre']]
            post = flu[:, :, self.frames_map['post']]
            X = np.mean(post, 2) - np.mean(pre, 2)
        else:
            flu_frames = flu[:, :, self.frames_map[frames]]
            X = np.mean(flu_frames, 2)

        return self.transform_data(X)

    def covariates_partial(self, flu, frames, trial_bool, n_comps_include,
                            region='all', prereward=False):

        # Function to subtract the mean of pre frames from the
        # mean of post frames -> [n_cells x n_trials]
        sub_frames = lambda arr: np.mean(
            arr[:, :, self.post], 2) - np.mean(arr[:, :, self.pre], 2)

        covariates_dict = {}

        # Mean population activity on every trial
        covariates_dict['trial_mean'] = np.mean(
            flu[:, :, self.remove_artifact], (0, 2))

        # Average post - pre across all cells
        covariates_dict['delta_f'] = np.mean(sub_frames(flu), 0)

        # Mean network activity just before the stim
        covariates_dict['mean_pre'] = np.mean(flu[:, :, self.pre], (0, 2))
        # Mean network activity just after the stim
        covariates_dict['mean_post'] = np.mean(flu[:, :, self.post], (0, 2))

        # Mean trace correlation pre stim
        covariates_dict['corr_pre'] = np.log((mean_cross_correlation(flu, 
                                                    self.frames_map['pre'])))
        # Mean trace correlation post stim
        covariates_dict['corr_post'] = np.log((mean_cross_correlation(flu, 
                                                     self.frames_map['post'])))

        covariates_dict['largest_singular_value'] = np.log(largest_singular_value(
                                                           flu, self.pre))

        covariates_dict['largest_PC_var'] = np.log(largest_PC_var(flu, self.pre))
        # covariates_dict['largest_PC_var'] = largest_PC_var(flu, self.pre)

        covariates_dict['largest_factor_var'] = np.log(largest_factor_var(flu, self.pre))

        covariates_dict['flat'] = np.ones(*covariates_dict['mean_pre'].shape)

        # The log of the timescales is taken further down
        covariates_dict['ts_s1_pre'] = (np.abs(self.session.tau_dict['S1_pre'][trial_bool]))
        covariates_dict['ts_s2_pre'] = (np.abs(self.session.tau_dict['S2_pre'][trial_bool]))
        covariates_dict['ts_both_pre'] = (np.abs(self.session.tau_dict['all_pre'][trial_bool]))

        covariates_dict['trial_number'] = np.arange(*covariates_dict['mean_pre'].shape)

        covariates_dict['flattened_variance'] = flattened_variance(flu, self.pre)
        covariates_dict['variance_pop_mean'] = variance_pop_mean(flu, self.pre)
        covariates_dict['variance_cell_rates'] = variance_cell_rates(flu, self.pre)
        covariates_dict['mean_cell_variance'] = mean_cell_variance(flu, self.pre)


        covariates_dict['reward_history'] = reward_history(self.session)[trial_bool]

        covariates_dict['n_cells_stimmed'] = self.session.trial_subsets[trial_bool]

        covariates_dict['lick'] = self.session.decision[trial_bool]
        covariates_dict['reward'] = (self.session.outcome == 'hit')\
                                     .astype('int')[trial_bool]

        covariates_dict['jonas_metric'] = jonas_metric(flu, self.pre)


        for key in ['ts_s1_pre', 'ts_s2_pre', 'ts_both_pre']:
            val = covariates_dict[key]
            val[np.logical_or(val < 30, val > 3000)] = np.nan
            covariates_dict[key] = np.log(val)

        # if prereward:
            # if region != 'all':
                # raise NotImplementedError('prereward comps do not yet have '
                                           # 'region dependency')
            # PCs = self.session.comps_pre
        # else:
            # print(f' Cell included from region {region}')
            # PCs = self.session.pca_dict[region]
            # PCs = PCs[:, trial_bool, :]

        # PCs = PCs[:, :, self.session.frames_use]

        # assert n_comps_include <= PCs.shape[0]
        # PCs = PCs[0:n_comps_include, :, :]

        # if frames == 'all':
            # covariates_dict['PCs'] = sub_frames(PCs)
        # elif frames == 'pre':
            # covariates_dict['PCs'] = np.mean(PCs[:, :, self.pre], 2)
        # elif frames == 'post':
            # covariates_dict['PCs'] = np.mean(PCs[:, :, self.post], 2)

        return covariates_dict

    def trim_deconvolved(self, pre_trim=5, post_trim=5, plot=False):
        ''' Give the linear_model.frames_map dict a haircut to
            avoid the weird peak where the NaNs are '''

        # Remove 5 frames + the 3 from ML_session, so 8 frames total = 267 ms
        self.pre[np.where(self.pre)[0][-pre_trim:]] = False

        # Remove 5 extra frames from the start
        self.post[np.where(self.post)[0][:post_trim]] = False

        self.frames_map['pre'] = self.pre
        self.frames_map['post'] = self.post
        self.frames_map['all'] = np.logical_or(self.pre, self.post)

        if plot:
            # I haven't aligned this to a proper trace but you get
            # the idea
            plt.plot(np.mean(self.flu, (0, 1))[self.frames_map['pre']])
            plt.plot(np.mean(self.flu, (0, 1))[self.frames_map['post']])
            plt.ylim(0, 0.01)

    def build_confusion_matrix(self, y_true, y_pred):
        ''' Builds a "3d" confusion matrix, allowing you to stack
            mutliple confusion matrices by multiple calls to this function
            designed to later be summed across 3rd dimension
            '''

        C = sklearn.metrics.confusion_matrix(y_true, y_pred)

        if C.shape != (4, 4):
            return

        if not hasattr(self, 'confusion_matrix'):
            self.confusion_matrix = C
        else:
            self.confusion_matrix = np.dstack((self.confusion_matrix, C))

    def logistic_regression(self, X, y, penalty, C, solver='lbfgs', n_folds=5,
                            digital_score=True, compute_confusion=False,
                            random_state=None, filter_models=False,
                            return_results=False, stratified_kfold=True):

        ''' Perform cross validated logistic regression on data
            Driver function for sklearn class https://tinyurl.com/sklearn-logistic

        Parameters
        -----------

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix, where n_samples is the number of samples and
            n_features is the number of trials.

        y : array-like of shape (n_samples,)
            Target vector relative to X (trial outcome)

        penalty : {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}

        C: float, Inverse of regularization strength

        solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
            Algorithm to use in the optimization problem.

        n_folds : int, default 5, number of cross validation folds

        digital_score : bool, default True
            Whether to score model performance digitally, if True each test
            case is scored as right or wrong. If False, performance is scored
            as model confidence.

        random_state : {None, int, numpy.random.RandomState instance}, default=None
            see sklearn docs, seed of random number generator used for kfold
            splits. None will cause different kfold splits with each function
            call. Int will seed and provide the same splits on each calls

        compute_confusion : bool, default False
            Whether to add results to self.confusion_matrix

        filter_models : bool, default False
            Only return models with good classification performance


        Returns
        --------
        means : mean performance across folds
        stds : std of performance across folds
        models : list of len n_folds.
            Contains each fit LogisticRegression object

        '''

        results = []
        models = []

        if stratified_kfold:
            kfold = sklearn.model_selection.StratifiedKFold
        else:
            kfold = sklearn.model_selection.KFold

        folds = kfold(n_splits=n_folds, shuffle=True,
                      random_state=random_state)

        for train_idx, test_idx in folds.split(X, y):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = sklearn.linear_model.LogisticRegression(penalty=penalty, C=C,
                                                            class_weight='balanced', solver=solver,
                                                            random_state=random_state)

            model.idx_dict = {'train': train_idx,
                              'test': test_idx}

            model.fit(X=X_train, y=y_train)

            if not filter_models or model.score(X_test, y_test) > 0.55:
                models.append(model)
            else:
                print('model filtering on')

            if compute_confusion:
                # Add this model performance to the running confusion matrix
                self.build_confusion_matrix(y_test, model.predict(X_test))

            if digital_score:
                # results.append(model.score(X_test, y_test))
                # results.append(roc_auc_score(y_test, model.predict(X_test)))
                # results.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
                # print(y_test)
                # print(model.predict(X_test))
                # print('\n')
                results.append(balanced_accuracy_score(
                               y_test, model.predict(X_test)))
            else:
                results.append(score_nonbinary(model, X_test, y_test))

        if return_results:
            return results, models
        else:
            return np.mean(results), np.std(results), models

    def repeated_crossfold(self, X, y, penalty, C, solver='lbfgs', n_repeats=10, n_folds=5,
                            digital_score=True, compute_confusion=False,
                            random_state=None, filter_models=False,
                            stratified_kfold=True):

        all_results = []
        for _ in range(n_repeats):

            results, _ = self.logistic_regression(X, y, penalty, C, 
                                               solver=solver, n_folds=n_folds,
                                               digital_score=digital_score, 
                                               compute_confusion=compute_confusion,
                                               random_state=random_state,
                                               return_results=True,
                                               filter_models=filter_models,
                                               stratified_kfold=stratified_kfold)
            all_results.append(results)

        return np.array(all_results).ravel()


    def performance_vs_reg(self, X, y, penalty, solvers):
        ''' Model performance as a function of C.
            C values currently hardcoded
        '''

        Cs = np.logspace(-4, 3, 8)

        for idx, solver in enumerate(solvers):

            means = []
            stds = []

            for C in Cs:

                # Plotting performance as a function of regularisation strength
                # when there is no regularisation doesnt make sense, but it is
                # the easiest way to eyeball the plots
                if penalty == 'none':
                    C = 0

                mean_acc, std_acc, _ = self.logistic_regression(X, y, penalty,
                                                                C, solver)
                means.append(mean_acc)
                stds.append(std_acc)

            means = np.array(means)
            stds = np.array(stds)
            sems = stds / 5

            plt.plot(Cs, means, label=solver,
                     color=COLORS[idx])
            plt.xscale('log')
            plt.fill_between(Cs, means-sems, means +
                             sems, color=COLORS[idx], alpha=0.3)
            plt.axhline(1/len(set(y)), linestyle=':')

        plt.legend()
        plt.title(penalty.upper())
        plt.xlabel('C (Inverse Regularisation Strength)')
        plt.ylabel('Classifier Performance')

    def model_params_plot(self, frames='all', n_comps_in_partial=10,
                          outcomes=['hit', 'miss']):
        ''' Plot to quantify performance of different solvers, penalties and
            regularisation strengths.

            Parameters
            -----------
            frames : which frames relative to photostim to use for regression
            n_comps_in_partial : How many PCs to use if using partial model

            '''

        for idx, model in enumerate(['full', 'partial']):

            plt.figure(figsize=(9, 4))
            plt.suptitle(model, fontsize=16)

            X, y = self.prepare_data(frames=frames, model=model,
                                     outcomes=outcomes,
                                     n_comps_include=n_comps_in_partial)

            solvers_dict = {
                # 'l2': [None],
                # 'none': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                'l1': ['liblinear', 'saga'],
                # 'l2': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                'l2': ['newton-cg', 'lbfgs', 'liblinear']
            }

            n_plots = 0
            for penalty, solvers in solvers_dict.items():
                n_plots += 1
                plt.subplot(1, len(solvers_dict), n_plots)
                self.performance_vs_reg(X, y, penalty, solvers)
                plt.ylim(0, 1)

    def plot_betas(self, frames, model, n_comps_in_partial=10, multiclass=False,
                   plot=True, region='all'):
        ''' Plot the beta values of each covariate in the model
            '''

        penalty = 'l1'
        C = 0.5
        solver = 'saga'

        if multiclass:
          outcomes = ['hit', 'miss', 'fp', 'cr']
        else:
          outcomes = ['hit', 'miss']

        X, y = self.prepare_data(frames, model,
                                 n_comps_include=n_comps_in_partial, region=region,
                                 outcomes=outcomes, return_matrix=True)

        acc, std_acc, models = self.logistic_regression(X, y, penalty=penalty, C=C,
                                                        solver=solver,
                                                        filter_models=False)
        coefs = []
        for model in models:
            coef = np.squeeze(model.coef_)
            coefs.append(coef)

            if not plot:
                continue

            # Plot each fold's betas as points
            for idx, c in enumerate(coef):
                label = self.encoder.inverse_transform([idx])[0]
                plt.plot(c, '.', color=COLORS[idx], label=label, markersize=9)

        # labels = ['Mean Activity', r'Population $\Delta$F',
                  # 'Mean activity pre', 'Mean activity post',
                  # 'Mean noise correlation pre', 'Mean noise correlation post',
                  # 'Largest SV', 'flat', 'ts_pre_s1', 'ts_pre_s2', 'ts_pre_both']
        labels = ['Mean activity pre', 'mean noise corrlelation pre', 'largest_singular_value pre',
                  'flat', 'ts_pre_s1', 'ts_pre_s2', 'ts_pre_both']

        [labels.append(f'PC{i}') for i in range(n_comps_in_partial)]

        # Useful for full model, how sparse is B vector -> how many cells important?
        if model == 'full': print(sum(coef == 0) / len(coef))

        if plot:
            # Legend with duplicates removed
            # https://stackoverflow.com/a/13589144/10723511
            handles, leg_labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(leg_labels, handles))
            plt.legend(by_label.values(), by_label.keys(),
                       bbox_to_anchor=(1.04, 1))
            plt.axhline(0, linestyle=':')
            plt.ylabel(r'$\beta$')
            plt.xlabel('Covariate')
            xt = plt.xticks(np.arange(len(labels)), labels,
                          rotation=90, fontsize=14)

        return coefs, labels

    def partial_model_performance(self, frames, n_comps_in_partial=10,
                                  plot=True, multiclass=False):
        ''' Plot partial model performance relative to full model
            performance as a function of number of partial covariates
            included '''

        # These hyperparams give the best performance across sessions
        penalty = 'l2'
        solver = 'lbfgs'
        C = 0.5

        if multiclass:
          outcomes = ['hit', 'miss', 'fp', 'cr']
        else:
          outcomes = ['hit', 'miss']

        X, y = self.prepare_data('all', 'full', outcomes=outcomes)
        mean_full, std_full, _ = self.logistic_regression(X, y, penalty=penalty,
                                                          C=C, solver=solver)

        self.add_partial_map = {'Mean activity only': ['trial_mean'],
                        r'+ Population $\Delta$F': ['trial_mean', 'delta_f'],
                         '+ Mean pre & post':  ['trial_mean', 'delta_f',
                                                'mean_pre', 'mean_post'],
                         '+ Correlations pre & post': ['trial_mean', 'delta_f',
                                                      'mean_pre', 'mean_post',
                                                      'corr_pre', 'corr_post'],
                        f'+ {n_comps_in_partial} PCs': ['trial_mean', 'delta_f',
                                                        'mean_pre', 'mean_post',
                                                        'corr_pre', 'corr_post',
                                                        'PCs']
                                                        }

        if n_comps_in_partial == 0:
            del self.add_partial_map[f'+ {n_comps_in_partial} PCs']

        covariates_dict, y = self.prepare_data('all', 'partial',
                                 n_comps_include=n_comps_in_partial,
                                 outcomes=outcomes, return_matrix=False)

        means = []
        stds = []

        for covs in self.add_partial_map.values():
            X = self.dict2matrix(
                {k: v for k, v in covariates_dict.items() if k in covs})

            mean, std, _ = self.logistic_regression(X, y, penalty=penalty,
                                                    C=C, solver=solver)
            means.append(mean)
            stds.append(std)

        means.append(mean_full)
        stds.append(std_full)

        if plot:
            plt.errorbar(range(len(means)), means, yerr=stds, fmt='o',
                         color=COLORS[1], ecolor='lightgray', elinewidth=3, capsize=5)
            tick_labels = list(self.add_partial_map.keys())
            tick_labels.append('Full model')
            plt.xticks(range(len(means)), tick_labels,
                       rotation=80, fontsize=18)

            if not multiclass:
                plt.axhline(0.5, linestyle=':')
            else:
                plt.axhline(0.25, linestyle=':')

            plt.ylabel('Mean classification accuracy')

        return np.array(means), np.array(stds)

    def performance_vs_ncomps(self, frames):
        ''' Plot partial model performance as a function of the
            number of components
            '''

        accs = []
        std_accs = []
        C = 0.5

        comps_test = range(1, 50, 5)
        for n_comps in comps_test:

            X, y = self.prepare_data('all', 'partial', n_comps_include=n_comps)
            acc, std_acc, _ = self.logistic_regression(X, y, 'l1', C, 'saga')
            accs.append(acc)
            std_accs.append(std_acc)

        accs = np.array(accs)
        std_accs = np.array(std_accs)

        # Full model performance (repeated n_components times)
        X, y = self.prepare_data('all', 'full')
        mean_full, std_full, _ = self.logistic_regression(
            X, y, 'l1', C, 'saga')
        mean_full = np.repeat(mean_full, len(accs))
        std_full = np.repeat(std_full, len(accs))

        plt.plot(comps_test, accs, label='Partial Model')
        plt.fill_between(comps_test, accs-std_accs, accs+std_accs, alpha=0.3)

        plt.plot(comps_test, mean_full, label='Full Model')
        plt.fill_between(comps_test, mean_full-std_full,
                         mean_full+std_full, alpha=0.3)

        plt.legend(fontsize=18)
        plt.ylabel('Classification Accuracy')
        plt.xlabel('Number of Components Included')

    def target_info(self):
        ''' Gets information about targets for the session

        Attributes
        -----------
        n_times_targetted: int vector, how many times was each cell targetted
        ever_targetted: bool vector, was the cell every targetted

        '''

        # Don't need the frame dimension here
        is_target = self.session.is_target[:, :, 0]
        is_target = is_target[:, self.session.photostim == 1]
        self.n_times_targetted = np.sum(is_target, 1)
        self.ever_targetted = np.any(is_target, axis=1)

    def remove_targets_from_data(self):
        self.flu = self.flu[~self.ever_targetted, :, :]
        self.pre_flu = self.pre_flu[~self.ever_targetted, :, :]
        self.region_map = {keys: values[~self.ever_targetted]
                           for keys, values in self.region_map.items()}

    def targets_histogram(self):
        ''' Plot n_times_targetted histogram '''

        self.target_info()
        plt.hist(n_times_targetted)
        plt.xlabel('Number of times targeted')
        plt.title('S1')

    def beta_targets_correlation(self, region='all'):
        ''' Plots a scatter plot to show correlation between the number
            of times a cell is taretted and its model beta coef

        N.b. Currently only works with the first model of the cross val

        '''

        X, y = self.prepare_data('all', 'full',
                                 outcomes=['hit', 'miss', 'cr', 'fp'])

        acc, std_acc, models = self.logistic_regression(X, y, 'l1', 0.5,
                                                        'saga')

        # The index of the coef that corresponds to hit trials
        hit_idx = self.encoder.transform(['hit'])[0]
        # Project cells onto hit axis
        self.hit_coef = models[0].coef_[hit_idx, :]

        # Can we reject the null that the distribution of the target and
        # non-target betas is the same?
        _, p_val = ks_2samp(self.hit_coef[self.ever_targetted],
                            self.hit_coef[~self.ever_targetted])

        if p_val < 0.05:
            print('NULL REJECTED!!')

        coef = self.hit_coef[self.region_map[region]]
        n_times_targetted = self.n_times_targetted[self.region_map[region]]

        plt.plot(n_times_targetted, coef, '.')
        plt.xlabel('Number of times targeted')
        plt.ylabel(r'$\beta$')

        print(f'{round(sum(coef!=0) / len(coef) * 100, 2)}% of cells have non-0 coefs')

    def target_proba(self):
        ''' Plot the probabilty of being 0 for targets and non-targets '''

        X, y = self.prepare_data('all', 'full')
        acc, std_acc, models = self.logistic_regression(
            X, y, 'l1', 0.5, 'saga')
        for model in models:
            coef = np.squeeze(model.coef_)[self.session.s1_bool]
            # P(non0)
            P = 1 - sum(coef == 0) / len(coef)
            # P(non0|ever_target)
            P1 = 1 - sum(coef[self.ever_targetted] == 0) / \
                len(coef[self.ever_targetted])
            plt.plot([0, 1], [P, P1], color=COLORS[0],
                     marker='.', markersize=10)

            plt.xlim(-0.5, 1.5)


    def performance_covariate_correlation(self, covariate, frames='subbed', plot=False):


        # These hyperparams give the best performance across sessions
        penalty = 'l1'
        solver = 'saga'
        C = 0.1

        # Prepare data for a hit vs fp classifier
        X, y = self.prepare_data(frames=frames, model='full', region='s2',
                                 outcomes=['hit', 'fp'],
                                 n_comps_include=10)

        # Train n_folds stim classifers
        _, _, models = self.logistic_regression(
                       X, y, penalty=penalty, C=C, solver=solver, digital_score=True)

        # X_hit = X[hit_idx, :]

        # Get the partial covariates pre in S1
        X_partial, y = self.prepare_data(frames='pre', model='partial',
                                 outcomes=['hit', 'fp'],
                                 region='s1',
                                 n_comps_include=10,
                                 return_matrix=False)

        # Pull just the covariate you're correlating out of the 
        # partial dictionary
        # X_partial = X_partial[covariate]

        # hit_cov = X[covariate][hit_idx]
        

        # Just want to look at hit trials
        hit_idx = np.where(y==0)[0]
        # If you wana remove easy trials
        not_easy = np.where(np.isin(X_partial['n_cells_stimmed'], [5,10,20,30,40,50]))[0]

        X_partial = X_partial[covariate]
        hit_confidence = []
        hit_cov = []

        for model in models:
            # The indexs of the test set for this model
            test_idx = model.idx_dict['test']
            # Indexes of test hit trials
            test_hit = np.intersect1d(hit_idx, test_idx)
            test_hit = np.intersect1d(test_hit, not_easy)
            
            confidence = model.predict_proba(X[test_hit, :])
            hit_confidence.append(confidence[:, 0])
            hit_cov.append(X_partial[test_hit])
            
        # hit_confidence = np.mean(np.array(hit_confidence), 0)

        hit_confidence = np.concatenate(hit_confidence)
        hit_cov = np.concatenate(hit_cov)

        
        # assert len(hit_cov) == len(hit_idx)


        if plot:

            plt.plot(hit_cov, hit_confidence, '.')

            slope, intercept, r_value, p_value, std_err = stats.linregress(hit_cov, hit_confidence)
            x_lims = (min(hit_cov), max(hit_cov))
            y_lims = (min(hit_confidence), max(hit_confidence))


            corr, p = stats.spearmanr(hit_cov, hit_confidence)


            # plt.text((x_lims[0] + x_lims[1]) / 2, (y_lims[1] + y_lims[0])/2, 
                     # f'r^2 = {round(r_value**2, 2)}\ngradient = {round(slope, 4)}'
                     # f'\np={p_value:.2e}\nSR={stats.spearmanr(hit_cov, hit_confidence)}',
                     # fontsize=16)
            plt.text((x_lims[0] + x_lims[1]) / 2, (y_lims[1] + y_lims[0])/2,
                     f'{round(corr, 2)} {round(p, 3)}',
                     fontsize=16)

            plt.plot(hit_cov, intercept + slope * hit_cov, '-', color='black')

        return hit_confidence, hit_cov


    def project_model(self, frames='all', model='full', region='both', plot=False, 
                      digital_score=True):
        ''' Train a model to classify hit and miss and then test
            it on catch and prereward trials.

        Parameters
        ----------
        As in prepare_data

        Returns
        ---------
        results : tuple (2 x 3)
            Mean and Std for hit vs miss, cr vs fp, prereward

        '''

        # These hyperparams give the best performance across sessions
        penalty = 'l1'
        solver = 'saga'
        C =  0.1

        region = 's2'

        # Prepare data for hit and fp trials
        X, y = self.prepare_data(frames=frames, model=model,
                                 region=region,
                                 outcomes=['hit', 'fp'],
                                 n_comps_include=10)


        # y[np.isin(y, [0,1])] = 1
        # y[np.isin(y, [2,3])] = 0

        # Prepare data for hit and miss trials
        # X, y = self.prepare_data(frames=frames, model=model, n_comps_include=10)
        # Prepare data for fp vs cr trials
        X_catch, y_catch = self.prepare_data(frames=frames, model=model, region=region,
                                             outcomes=['miss', 'cr'],
                                             n_comps_include=10)

        # Prepare data for prereward trials
        X_pre, y_pre = self.prepare_data(frames=frames, model=model, region=region,
                                         n_comps_include=10,
                                         prereward=True)

        # X = np.vstack((X, X_pre))
        # y = np.hstack((y, y_pre))

        # Cross validated full model accuracy
        mean_test, std_test, models = self.logistic_regression(
            X, y, penalty=penalty, C=C, solver=solver, n_folds=5, digital_score=digital_score)

        # Test different trial types on each of the 5 models fit on hit vs miss
        accs_miss = []
        accs_pre = []
        for model in models:
            if digital_score:
                accs_miss.append(model.score(X_catch, y_catch))
                accs_pre.append(model.score(X_pre, y_pre))
                # print(model.predict(X_pre))
                # print(y_pre)
                # accs_pre.append(balanced_accuracy_score(model.predict(X_pre), y_pre))
            else:

                accs_miss.append(score_nonbinary(model, X_catch, y_catch))
                accs_pre.append(score_nonbinary(model, X_pre, y_pre))

        if plot:
            plt.errorbar(0, mean_test, std_test, marker='o',
                         capsize=10, color=COLORS[0])
            # DOESNT MAKE SENSE WITH THE PROPAGATION MODEL
            plt.errorbar(1, np.mean(accs_miss), np.std(accs_miss),
                         marker='o', capsize=10, color=COLORS[1])
            plt.errorbar(2, np.mean(accs_pre), np.std(accs_pre),
                         marker='o', capsize=10, color=COLORS[2])

            plt.xticks([0, 1, 2], ['Hit vs FP', 'Miss vs Cr', 'Spont'],
                       rotation=45)

            plt.axhline(1/len(set(y)), linestyle=':')

        return ((mean_test, std_test),
                (np.mean(accs_miss), np.std(accs_miss)),
                (np.mean(accs_pre), np.std(accs_pre)))

    def compare_regions(self, frames='all', outcomes=['hit', 'miss'], plot=True):

        penalty = 'l1'
        C = 0.5
        solver = 'saga'

        regions = ['s1', 's2', 'all']
        regions = ['all']

        mean_accs = []
        std_accs = []
        coefs = []

        for idx, region in enumerate(regions):

            X, y = self.prepare_data(frames=frames, model='full',
                                     outcomes=outcomes,
                                     region=region,
                                     n_comps_include=0)

            mean_acc, std_acc, models = self.logistic_regression(X, y, penalty,
                                                            C, solver)
            mean_accs.append(mean_acc)
            std_accs.append(std_acc)
            all_coefs = np.array([np.squeeze(model.coef_) for model in models])
            coefs.append(np.mean(all_coefs, axis = 0))


            if plot:
                plt.errorbar(idx, mean_acc, yerr=std_acc, fmt='o',
                             color=COLORS[1], ecolor='lightgray', elinewidth=3, capsize=5)

        if plot:
            plt.ylim(0, 1)
            plt.axhline(0.25)
            plt.xticks(range(3), regions)

        return mean_accs, std_accs, coefs[0]  # coefs will break if you have >1 region

    def pca_regions(self, n_components=100, save_PC_matrix=False):
        ''' Driver function for pca_session to build PCs based on cells in
            s1 and s2 seperately

            Returns:
            '''

        self.session.pca_dict = {}
        for region_name, cells_include in self.region_map.items():

            if self.remove_targets:
                # Awkward af, need to "add back in" previously removed targets
                # to the cells_include boolean but set them to False
                # could cause future bugs
                temp = np.repeat(False, self.ever_targetted.shape)
                temp[~self.ever_targetted] = cells_include
                cells_include = temp

            session = pca_session(self.session, cells_include, n_components=n_components,
                                  plot=False, save_PC_matrix=save_PC_matrix)
            # Weird way of building a dictionary from session attributes but avoids
            # changing the structure of pca_session to split by region
            self.session.pca_dict[region_name] = session.comps

    def dropout(self, region='s1', return_results=True, plot=True):

        X, y = self.prepare_data(frames='all', model='partial',
                                 outcomes=['hit', 'miss'],
                                 region=region,
                                 n_comps_include=0,
                                 return_matrix=False,)

        # covs_keep = ['mean_pre', 'corr_pre', 'largest_PC_var',
                     # f'ts_{region}_pre', 'reward_history', 'trial_number', 'n_cells_stimmed']

        covs_keep = ['mean_pre', 'variance_cell_rates', 'largest_PC_var',
                     'reward_history', 'trial_number', 'n_cells_stimmed']

        X = {k: v for k, v in X.items() if k in covs_keep}

        penalty = 'l1'
        C = 0.5
        solver = 'saga'

        results_dict = {}

        # Full
        results, models = self.logistic_regression(self.dict2matrix(X), y,
                                                              penalty=penalty, C=C,
                                                              solver=solver,
                                                              n_folds=5,
                                                              random_state=0,
                                                              filter_models=False,
                                                              return_results=True,
                                                              stratified_kfold=True)
        results_dict['all_covs'] = results

        if plot: plt.figure(figsize=(12, 6))
        n_points = 0

        if plot:
            plt.plot([n_points]*len(results), results, '.', color=COLORS[0])
        # plt.errorbar(n_points, full_acc, yerr=full_std, capsize=20)

        n_points += 1

        for label, cov in X.items():
            temp_dict = copy.deepcopy(X)
            temp_dict.pop(label, None)

            results, _ = self.logistic_regression(self.dict2matrix(temp_dict), y,
                                                  penalty=penalty, C=C, solver=solver,
                                                  random_state=0,
                                                  n_folds=5,
                                                  filter_models=False,
                                                  return_results=True,
                                                  stratified_kfold=True)
            results_dict[label] = results

            # plt.errorbar(n_points, acc, yerr=std_acc, capsize=20)
            if plot:
                plt.plot([n_points]*len(results),
                         results, '.', color=colors[1])
            n_points += 1

        np.random.shuffle(y)
        # this shouldn't be repeated three times
        results, _ = self.logistic_regression(self.dict2matrix(X), y,
                                              penalty=penalty, C=C, solver=solver,
                                              random_state=0,
                                              n_folds=5,
                                              filter_models=False,
                                              return_results=True,
                                              stratified_kfold=True)

        results_dict['shuffled_null'] = results

        if plot:
            labels = list(X.keys())

            labels.insert(0, 'No Covariate dropped')

            plt.xticks(np.arange(n_points), labels, rotation=90)
            plt.ylabel('Classification Accuracy')
            plt.xlabel('Covariate Dropped')
            plt.title(region.upper())
            plt.ylim((0.4, 1))
            plt.axhline(0.5, ls=':')

            # For the beta plot
            plt.figure(figsize=(12, 6))

        coefs = []
        for model in models:
            coef = np.squeeze(model.coef_)
            coefs.append(coef)
            # Plot each fold's betas as points
            if plot: plt.plot(coef, '.', color=COLORS[0], markersize=9)

        if plot:
            labels.pop(0)
            plt.xticks(np.arange(n_points), labels, rotation=90)
            plt.axhline(0)

        return results_dict, coefs

    def single_covariate(self, region='s1', plot=True):

        n_comps_include = 5
        X, y = self.prepare_data(frames='pre', model='partial',
                                 outcomes=['hit', 'miss'],
                                 region=region,
                                 n_comps_include=n_comps_include,
                                 remove_easy=True,
                                 return_matrix=False)

        # for i in range(n_comps_include):
            # X[f'PC{i}'] = X['PCs'][i, :]

        covs_keep = ['mean_pre', 'corr_pre',
                    'largest_PC_var', f'ts_{region}_pre', 'reward_history', 'trial_number',
                    'n_cells_stimmed']  # 'PC1', 'PC2', 'PC3',

        covs_keep = ['mean_pre', 'variance_pre', 'flat',
                     f'ts_{region}_pre', 'reward_history', 'trial_number', 'n_cells_stimmed']


        covs_keep = ['mean_pre', 'variance_cell_rates', 'largest_PC_var',
                     'reward_history', 'trial_number', 'n_cells_stimmed',
                     'largest_singular_value']

        X = {k: v for k, v in X.items() if k in covs_keep}

        penalty = 'l1'
        C = 0.5
        solver = 'saga'

        n_points = 0

        if plot:
            plt.figure(figsize=(12, 6))

        means_dict = {}
        stds_dict = {}
        for label, cov in X.items():

            cov = np.expand_dims(cov, axis=1)
            acc, std_acc, models = self.logistic_regression(cov, y, penalty=penalty, C=C,
                                                            solver=solver,
                                                            filter_models=False,
                                                            stratified_kfold=True)

            means_dict[label] = acc
            stds_dict[label] = std_acc

            if plot:
                plt.errorbar(n_points, acc, yerr=std_acc, capsize=20)
            n_points += 1

        # Include all covariates
        acc, std_acc, models = self.logistic_regression(self.dict2matrix(X), y, penalty=penalty, C=C,
                                                        solver=solver,
                                                        filter_models=False,
                                                        stratified_kfold=True)

        means_dict['all_covariates'] = acc
        stds_dict['all_covariates'] = std_acc

        np.random.shuffle(y)
        acc, std_acc, models = self.logistic_regression(cov, y, penalty=penalty, C=C,
                                                        solver=solver,
                                                        filter_models=False,
                                                        stratified_kfold=True)

        means_dict['shuffled_null'] = acc
        stds_dict['shuffled_null'] = std_acc

        if plot:
            plt.errorbar(n_points, acc, yerr=std_acc, capsize=20)
            n_points += 1

            labels = list(X.keys())
            labels.append('All covariates')

            plt.xticks(np.arange(n_points), labels, rotation=90)
            plt.axhline(0.5, linestyle=':')
            plt.ylabel('Classification Accuracy')

        return means_dict, stds_dict

    def shuffled_model(self, region='s1'):

        penalty = 'l1'
        C = 0.5
        solver = 'saga'

        X, y = self.prepare_data(frames='all', model='partial',
                                 outcomes=['hit', 'miss'],
                                 region=region,
                                 n_comps_include=0,
                                 return_matrix=True,)

        X = X[:, 0]
        X = np.expand_dims(X, axis=1)

        n_trials = len(y)
        train_idx = np.repeat(False, n_trials)
        train_idx[:int(0.8*n_trials)] = True
        test_idx = np.logical_not(train_idx)

        np.random.shuffle(y)

        model = sklearn.linear_model.LogisticRegression(penalty=penalty, C=C,
                                                        class_weight='balanced', solver=solver,
                                                        random_state=None)
        model.fit(X[train_idx, :], y[train_idx])

        print(f' Actual class = {y[test_idx]}')
        print(f' Binarised Prediction = {model.predict(X[test_idx])}')
        print(f' Analog Prediction = {model.predict_proba(X[test_idx])[:, 1]}')
        print(
            f' BAS = {balanced_accuracy_score(y[test_idx], model.predict(X[test_idx]))}')
        print(
            f' AUC(ROC) = {roc_auc_score(y[test_idx], model.predict_proba(X[test_idx])[:, 1])}')
        print(
            f' F1 score = {f1_score(y[test_idx], model.predict(X[test_idx]))}')
        print(f' Naive Score = {model.score(X[test_idx], y[test_idx])}')


class PoolAcrossSessions(AverageTraces):

    def __init__(self, save_PCA=False, remove_targets=False, subsample_sessions=True, remove_toosoon=False,
                 remove_too_few_cells=True):
        ''' Build object to pool across multiple LinearModel objects

        Allows you to build the useful attributes and make the plots
        contained in LinearModel methods, but across multiple Sessions
        Inherits from AverageTraces to use the load_sessions method and
        times_use attribute.

        Parameters
        -----------

        save_PCA : bool, default False.
            Do you want to compute PCs on session object?
            comps attributes are appended to the Session objects after saving,
            so only need to set to True the first time you run this class.
            Requires access to run objects.

        Attributes
        -----------
        linear_models : list of len n_sessions containing LinearModel objects

        Methods
        -----------
        As in LinearModel but pooled across sessions

        '''

        # Inherit here to get the times_use variable from AverageTraces and to
        # keep linear model to a single session
        super().__init__('dff')

        self.remove_targets = remove_targets
        self.remove_toosoon = remove_toosoon

        idxs_remove = []
        for idx, session in self.sessions.items():    

            if session.n_cells < 200:
                idxs_remove.append(idx)

        for idx in idxs_remove:
            self.sessions.pop(idx, None)

        self.linear_models = [LinearModel(session, self.times_use,
                                          remove_targets=remove_targets,
                                          remove_toosoon=remove_toosoon)
                              for session in self.sessions.values()]

        # Add PCA attributes to session if they are not already saved
        # for idx, session in self.sessions.items():
        for idx, linear_model in enumerate(self.linear_models):

            # Components already computed and saved
            if hasattr(linear_model.session, 'comps') or not save_PCA:
                continue
            else:
                linear_model.pca_regions(n_components=20, save_PC_matrix=False)
                self.sessions[idx] = linear_model.session
                # self.sessions[idx] = pca_session(session, n_components=100,
                                                 # plot=False)

        # Cache the PCA components to the Session object so we do not need to
        # recalculate every time this class is initialised
        if save_PCA:
            save_path = os.path.expanduser(
                f'{USER_PATHS_DICT["base_path"]}/sessions_lite_dff.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(self.sessions, f)

        timescales_pkl = 'OASIS_TAU_dffDetrended_60Pre60PostStim_sessions_liteNoSPKS3_flu.pkl'
        timescales_pkl_path = os.path.join(
            USER_PATHS_DICT['base_path'], timescales_pkl)

        with open(timescales_pkl_path, 'rb') as f:
            timescale_sessions = pickle.load(f)

        # Subsample sessions to make a training set
        # Now done only for timescale sessions
        # self.sessions = {key:value for key, value in
                         # self.sessions.items() if key in [2,5,14]}

        # Indexs of the timescales sessions to keep as a training set
        keep_sessions = [0, 3, 7]
        # Match it to the daddy sessions using the __repr__ string that contains
        # mouse id and session number

        if subsample_sessions:
            timescale_sessions = {key: session for key, session in timescale_sessions.items()
                                  if key in keep_sessions}

        # reprs of the timescales sessions, switch key value order to lookup key later
        session_stamps = {session.__repr__(): key for key,
                          session in timescale_sessions.items()}

        # [print(s) for s in timescale_sessions.values()]
        # print('\n')
        # [print(s) for s in self.sessions.values()]


        # Subsample self.sessions to get the same as timescale sessions
        temp = {}
        for key, session in self.sessions.items():
            if session.__repr__() in session_stamps.keys():
                temp[session_stamps[session.__repr__()]] = session

        if subsample_sessions:
            self.sessions = temp
        else:
            # self.sessions = temp
            print('ALERT SESSIONS NOT SUBSAMPLED')

        # Get the tau_dict into the daddy session Need to adjust this so it
        # only loops through 30 Hz session
        for key in self.sessions.keys():
            try:
                self.sessions[key].tau_dict = timescale_sessions[key].tau_dict
            except KeyError:
                continue

        # ipdb.set_trace()
        # # This is a shitty fix redefining this variable but it allows for caching
        # # of the pca_dict variable
        self.linear_models = [LinearModel(session, self.times_use,
                                          remove_targets=remove_targets)
                              for session in self.sessions.values()]

    def project_model(self, frames='all', model='full'):


        results = []

        for linear_model in self.linear_models:
            try:
                res = linear_model.project_model(frames=frames,
                                           model=model, region='s2', 
                                           plot=False)
                results.append(res)
            except ValueError:
                print(linear_model.session)


        results = np.array(results)
        means = results[:, :, 0]
        stds = results[:, :, 1]

        grand_mean = np.mean(means, 0)
        grand_std = self.combine_stds(stds)

        plt.errorbar(0, grand_mean[0], grand_std[0], marker='o',
                     capsize=10, color=COLORS[0])
        plt.errorbar(1, grand_mean[1], grand_std[1],
                     marker='o', capsize=10, color=COLORS[1])
        plt.errorbar(2, grand_mean[2], grand_std[2],
                     marker='o', capsize=10, color=COLORS[2])

        plt.xticks([0, 1, 2], ['Hit vs FP', 'Miss vs cr', 'Spont'],
                   rotation=45)

        plt.axhline(0.5, linestyle=':')

        return grand_mean, grand_std

    def combine_stds(self, stds):

        # I am combining STDs across sessions based on this post
        # https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
        variences = np.square(stds)
        return np.sqrt(np.mean(variences, 0))

    def model_params_plot(self):
        ''' Churns out lots of figures on top of each other
            for eyeballing in jupyter
            '''

        for linear_model in self.linear_models:

            linear_model.setup_flu()  # Should we call this is __init__?
            linear_model.model_params_plot()

    def plot_betas(self, frames, model, n_comps_in_partial=10, multiclass=False,
                   region='all'):
        ''' Currently only works with multiclass model '''

        all_coefs = []
        for linear_model in self.linear_models:
            coefs, labels = linear_model.plot_betas(frames, model, n_comps_in_partial,
                                                  region=region, multiclass=multiclass,
                                                  plot=False)
            all_coefs.append(np.array(coefs))

        all_coefs = np.vstack(all_coefs)

        # Use this to label the legend if you want to put all boxplots
        # on the same plot
        legend_labels = {}

        for trial_idx in range(all_coefs.shape[1]):

            fig, ax = plt.subplots(figsize=(16, 4))
            trial_coefs = all_coefs[:, trial_idx, :]
            box = ax.boxplot(trial_coefs, showfliers=True)

            plt.setp(box['fliers'], markeredgecolor=COLORS[trial_idx])
            for _, line_list in box.items():
                for line in line_list:
                    line.set_color(COLORS[trial_idx])

            label = self.linear_models[0].encoder.inverse_transform([trial_idx])[
                                                                    0]
            # Keep track of the boxes and their labels for the legend
            legend_labels[label] = box['boxes'][0]

            plt.title(label)

            plt.axhline(0, linestyle=':')
            plt.ylabel(r'$\beta$')
            plt.xlabel('Covariate')
            # Boxplot defaults to range(1, N+1)
            xt = plt.xticks(np.arange(1, len(labels)+1), labels,
                          rotation=90, fontsize=14)

    def partial_model_performance(self, frames, n_comps_in_partial=10, multiclass=False):

        means = []
        stds = []
        for linear_model in self.linear_models:
            mean, std = linear_model.partial_model_performance(frames,
                                                               n_comps_in_partial,
                                                               plot=False,
                                                               multiclass=multiclass)
            means.append(mean)
            stds.append(std)

        means = np.array(means)
        stds = np.array(stds)

        grand_mean = np.mean(means, 0)
        grand_std = self.combine_stds(stds)

        plt.figure(figsize=(8, 6))
        plt.errorbar(range(len(grand_mean)), grand_mean, yerr=grand_std, fmt='o',
                     color=COLORS[1], ecolor='lightgray', elinewidth=3, capsize=5)

        tick_labels = list(linear_model.add_partial_map.keys())
        tick_labels.append('Full model')
        plt.xticks(range(len(grand_mean)), tick_labels,
                   rotation=80, fontsize=18)
        plt.ylabel('Mean classification accuracy')
        if multiclass:
          plt.ylim(0, 1)
          plt.axhline(0.25, linestyle=':')
        else:
          plt.ylim(0.4, 1)
          plt.axhline(0.5, linestyle=':')

    def compare_regions(self, frames='all'):

        all_means = []
        all_stds = []
        all_coefs = []

        for linear_model in self.linear_models:
           mean_accs, std_accs, coefs = linear_model.compare_regions(
               frames=frames, plot=False)
           all_means.append(mean_accs)
           all_stds.append(std_accs)
           all_coefs.append(coefs)

        all_means = np.array(all_means)
        all_stds = np.array(all_stds)

        grand_mean = np.mean(all_means, axis=0)
        grand_std = self.combine_stds(all_stds)

        plt.errorbar(range(len(grand_mean)), grand_mean, yerr=grand_std, fmt='o',
                     color=COLORS[1], ecolor='lightgray', elinewidth=3, capsize=5)

        regions = ['Mean cell activity\nprior to stimulation']

        plt.ylim(0, 1)
        plt.axhline(0.5, linestyle=':')
        plt.xticks(range(len(grand_mean)), regions)
        return all_coefs

    def build_confusion_matrix(self):
        ''' Builds a "3d" confusion matrix, allowing you to stack
            mutliple confusion matrices by multiple calls to this function
            designed to later be summed across 3rd dimension
            '''
        self.confusion_matrix = None

        for session in self.sessions.values():
            linear_model = LinearModel(session, self.times_use)

            X, y = linear_model.prepare_data(model='full',
                                   outcomes=['hit', 'miss', 'fp', 'cr'])

            acc, std_acc, models = linear_model.logistic_regression(X, y, 'l2', 0.5,
                                            'lbfgs', compute_confusion=True)

            C = np.sum(linear_model.confusion_matrix, 2)

            # Inefficient but clean way of stacking matrices to 3d for summing
            # after loop.
            if self.confusion_matrix is None:
                self.confusion_matrix = C
            else:
                self.confusion_matrix = np.dstack((self.confusion_matrix, C))

        cmd = ConfusionMatrixDisplay(np.sum(self.confusion_matrix, 2),
                  display_labels=linear_model.encoder.inverse_transform([0, 1, 2, 3]))

        cmd.plot(cmap='Blues')

    def dropout(self, region='s1'):

        # This is so shit
        mean_accs = {}
        std_accs = {}
        all_betas = []
        for linear_model in self.linear_models:
            results_dict, betas = linear_model.dropout(
                region=region, plot=False)
            all_betas.append(betas)
            for k, v in results_dict.items():
                try:
                    mean_accs[k] = np.append(np.mean(v), mean_accs[k])
                    std_accs[k] = np.append(np.std(v), std_accs[k])
                except KeyError:
                    mean_accs[k] = np.array(np.mean(v))
                    std_accs[k] = np.array(np.std(v))

        all_betas = np.array(all_betas)
        all_betas = all_betas.reshape(all_betas.shape[0] * all_betas.shape[1],
                                      all_betas.shape[-1])

        n_points = 0
        for (label, means), (label2, stds) in zip(mean_accs.items(), std_accs.items()):
            assert label == label2  # Make sure the dicts are not in wrong order
            plt.errorbar(n_points, np.mean(means), self.combine_stds(stds)/5, marker='o',
                         capsize=10, color=COLORS[0])
            n_points += 1

        labels = list(mean_accs.keys())
        xt = plt.xticks(np.arange(len(labels)), labels,
                        rotation=90, fontsize=14)
        # plt.ylim(0.4,0.75)
        plt.ylabel('Balanced Accuracy Score')
        plt.xlabel('Dropped covariate')
        plt.title(region)
        plt.axhline(0.5, linestyle=':')

        plt.figure(figsize=(12, 6))
        labels.pop(0)
        labels.pop(-1)
        plt.errorbar(np.arange(all_betas.shape[1]), np.mean(all_betas, 0),
                     np.std(all_betas, 0), fmt='o',
                     capsize=10, color=COLORS[0])
        xt = plt.xticks(np.arange(len(labels)), labels,
                        rotation=90, fontsize=14)
        plt.axhline(0, linestyle=':')
        plt.title(region)

    def single_covariate(self, region='s1'):

        for idx, linear_model in enumerate(self.linear_models):
            mean_acc, std_acc = linear_model.single_covariate(
                region=region, plot=False)

            if idx == 0:
                all_means = mean_acc
                all_stds = std_acc
            else:
                for key in mean_acc.keys():
                    all_means[key] = np.append(all_means[key], mean_acc[key])
                    all_stds[key] = np.append(all_stds[key], std_acc[key])

        n_points = 0
        for (label, means), (label2, stds) in zip(all_means.items(), all_stds.items()):
            assert label == label2
            plt.errorbar(n_points, np.mean(means), self.combine_stds(stds)/5, fmt='o',
                         color=COLORS[1], ecolor='lightgray', elinewidth=3, capsize=5)
            n_points += 1

        labels = all_means.keys()
        xt = plt.xticks(np.arange(len(labels)), labels,
                      rotation=90, fontsize=14)

        plt.ylabel('Balanced Accuracy Score')
        plt.xlabel('Single Covariate in model')
        # plt.ylim(0.4,0.7)
        plt.title(region)
        plt.axhline(0.5, linestyle=':')

    def performance_covariate_correlation(self, covariate, frames='subbed'):
        
        x = []
        y = []
        for linear_model in self.linear_models:
            try:
                hit_confidence, hit_cov = linear_model.\
                                    performance_covariate_correlation(covariate,
                                                                   frames=frames)
                x.append(hit_cov)
                y.append(hit_confidence)
            except ValueError:
                print(linear_model.session)

        x = np.concatenate(x)
        y = np.concatenate(y)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_lims = (min(x), max(x))
        y_lims = (min(y), max(y))

        plt.plot(x, y, '.')
        plt.plot(x, intercept + slope * x, '-', color='black')
        plt.text(x_lims[1], (y_lims[1] + y_lims[0])/2, 
                 f'r^2 = {round(r_value**2, 2)}\ngradient = {round(slope, 4)}\np={p_value:.2e}',
                 fontsize=16)
         


        

    def summary_table(self):

        metadata = []

        for s in self.sessions.values():

            if (s.mouse == 'RL070' and s.run_number == 29 or
                s.mouse == 'J064' and s.run_number == 10 or
                s.mouse == 'J065' and s.run_number == 10):

                in_current = 'True'
            else:
                in_current = 'False'

            if s.mouse in ['J048', 'RL048']:
                fs = '5 Hz'
            else:
                fs = '30 Hz'

            mouse_info = [s.mouse, s.run_number, s.n_cells, fs, in_current]
            metadata.append(mouse_info)

        headers = ['Mouse Name', 'Run Number', 'Number of Cells',
            'Imaging Frequency', 'In Current Plots?']
        metadata.insert(0, headers)

        PoolAcrossSessions.display_table(metadata)

    def plot_string(self, is_plot=True, additional_strings=None):
        lm = self.linear_models[0]
        string = (f'Hello Adam\n'
                 f'Remove targets is {self.remove_targets}.\n'
                 f'Remove too soon is {self.remove_toosoon}\n'
                 f'Here are the sessions in play:\n' 
                 f'{[s.__str__() for s in self.sessions.values()]}\n'
                 f'pre-frames spans {round(np.min(lm.times_use[lm.pre]), 2)} to '
                 f'{round(np.max(lm.times_use[lm.pre]), 2)} inclusive\n'
                 f'post-frames spans {round(np.min(lm.times_use[lm.post]), 2)} to '
                 f'{round(np.max(lm.times_use[lm.post]), 2)} inclusive'
                 )

        if additional_strings is not None:
            for s in additional_strings:
                string = string + '\n' + s
            
                    
        if is_plot:
            xlims = plt.gca().get_xlim()
            ylims = plt.gca().get_ylim()
            plt.text(xlims[1], ylims[0], string, fontsize=20)
        else:
            return string

    @staticmethod
    def display_table(data):

        html = '<table style="table-layout:auto, width:180px">'

        for idx, row in enumerate(data):
            if idx == 0:
                style = 'h3'
            else:
                style = 'br'
            html += "<tr>"
            for field in row:
                html += f'<td style="width:100px"><{style}>{field}</{style}><td>'
            html += "</tr>"

        html += "</table>"

        display(HTML(html))


class MultiSessionModel(PoolAcrossSessions):
    def __init__(self, remove_targets, subsample_sessions, remove_toosoon):
        ''' Build models that include covariates from multiple sessions '''
        super().__init__(save_PCA=False, remove_targets=remove_targets,
                        subsample_sessions=subsample_sessions, remove_toosoon=remove_toosoon)


        self.remove_targets = remove_targets
        self.remove_toosoon = remove_toosoon
        # Hijack functions from from one of the 
        # LinearModel objects where the initialisation isn't important
        self.logistic_regression = self.linear_models[0].logistic_regression
        self.repeated_crossfold = self.linear_models[0].repeated_crossfold
        self.dict2matrix = self.linear_models[0].dict2matrix

        self.penalty = 'l1'
        self.C = 0.5
        self.solver = 'saga'

    def across_session_covariates(self, region='s1', norm='zscore'):

        covs_keep = ['mean_pre', 'corr_pre', 'largest_singular_value', 'largest_PC_var', 
                    'largest_factor_var', f'ts_{region}_pre', 'reward_history', 'trial_number',
                    'n_cells_stimmed', 'jonas_metric', 'variance_pre']

        covs_keep = [
                    'reward_history', 'trial_number',
                    'n_cells_stimmed',
                    'variance_cell_rates'
                    ]

        to_norm =   ['mean_pre', 'corr_pre', 'largest_singular_value', 'largest_PC_var',
                     'largest_factor_var', 'variance_pre', ]
                    
        all_X = None
        all_y = None

        if norm == 'zscore':
            norm_func = stats.zscore
        else:
            # No normalisation
            norm_func = lambda x: x

        for lm in self.linear_models:

            X, y = lm.prepare_data(frames='pre', model='partial',
                                     outcomes=['hit', 'miss'],
                                     region=region,
                                     n_comps_include=5,
                                     return_matrix=False)

            # Subsample to the covariates you want
            X = {k: v for k, v in X.items() if k in covs_keep}

            # Apply the norm function for the required covariates
            for k, v in X.items():
                if k in to_norm:
                    X[k] = norm_func(v)


            if all_X is None:
                all_X = X
                all_y = y
            else:
                all_X = self.dict_merger(all_X, X)
                all_y = np.concatenate((all_y, y))

        return all_X, all_y


    def single_covariate(self, region='s1'):


        n_points = 0

        X, y = self.across_session_covariates(region=region)

        # Every covaraite
        results, _ = self.logistic_regression(self.dict2matrix(X), y,
                                                              penalty=self.penalty, C=self.C,
                                                              solver=self.solver,
                                                              n_folds=5,
                                                              filter_models=False,
                                                              stratified_kfold=True,
                                                              return_results=True)

        plt.plot([n_points]*len(results), results, '.', color='blue')
        n_points += 1

        for label, cov in X.items():

            cov = np.expand_dims(cov, axis=1)
            results, _ = self.logistic_regression(cov, y, penalty=self.penalty, C=self.C,
                                                            solver=self.solver,
                                                            filter_models=False,
                                                            stratified_kfold=True,
                                                            return_results=True)

            # plt.errorbar(n_points, acc, yerr=std_acc, capsize=20)
            plt.plot([n_points]*len(results), results, '.', color='blue')
            n_points += 1

        labels = list(X.keys())
        labels.insert(0, 'All covariates')

        plt.xticks(np.arange(n_points), labels, rotation=90)
        plt.axhline(0.5, linestyle=':')


    def plot_betas(self, region='s1'):


        X, y = self.across_session_covariates(region=region)

        # Every covaraite
        results, models = self.logistic_regression(self.dict2matrix(X), y,
                                                              penalty=self.penalty, C=self.C,
                                                              solver=self.solver,
                                                              n_folds=5,
                                                              filter_models=False,
                                                              stratified_kfold=True,
                                                              return_results=True)

        labels = X.keys()

        x_axis = np.arange(len(labels))
        for model in models:
            coefs = model.coef_.ravel()
            plt.plot(x_axis, coefs, '.', color='blue')

        plt.xticks(x_axis, labels, rotation=90)
        plt.axhline(0, linestyle=':')


    def dropout(self, region='s1'):

        
        X, y = self.across_session_covariates(region=region)

        n_points = 0

        # Every covaraite
        results = self.repeated_crossfold(self.dict2matrix(X), y,
                                         penalty=self.penalty, C=self.C,
                                         solver=self.solver,
                                         n_folds=5,
                                         n_repeats=30,
                                         filter_models=False,
                                         stratified_kfold=True)

        plt.plot([n_points] * len(results), results, '.', color='blue')
        plt.errorbar(n_points, np.mean(results), np.std(results), marker='o',
                     capsize=10, color=COLORS[0])
        labels = ['All covariate']
        n_points += 1

        
        for label, cov in X.items():

            temp_dict = copy.deepcopy(X)
            temp_dict.pop(label, None)

            # Every covaraite
            results = self.repeated_crossfold(self.dict2matrix(temp_dict), y,
                                              penalty=self.penalty, C=self.C,
                                              solver=self.solver,
                                              n_folds=5,
                                              n_repeats = 30,
                                              filter_models=False,
                                              stratified_kfold=True)

            plt.plot([n_points] * len(results), results, '.', color='blue')
            plt.errorbar(n_points, np.mean(results), np.std(results), marker='o',
                         capsize=10, color=COLORS[0])
            labels.append(label)
            n_points += 1

        plt.xticks(np.arange(n_points), labels, rotation=90)

        plt.ylim(0.45, 1)
        plt.axhline(0.5, linestyle=':')


    def plot_string(self, region, is_plot=True, additional_strings=None):
        lm = self.linear_models[0]
        string = (f'Hello Adam\n'
                 f'Remove targets is {self.remove_targets}.\n'
                 f'Remove too soon is {self.remove_toosoon}\n'
                 f'Here are the sessions in play:\n' 
                 f'{[s.__str__() for s in self.sessions.values()]}\n'
                 f'Cells in region {region} are included' 
                 f'pre-frames spans {round(np.min(lm.times_use[lm.pre]), 2)} to '
                 f'{round(np.max(lm.times_use[lm.pre]), 2)} inclusive\n'
                 f'post-frames spans {round(np.min(lm.times_use[lm.post]), 2)} to '
                 f' {round(np.max(lm.times_use[lm.post]), 2)} inclusive'
                 )

        if additional_strings is not None:
            for s in additional_strings:
                string = string + '\n' + s
            
        if is_plot:
            xlims = plt.gca().get_xlim()
            ylims = plt.gca().get_ylim()
            plt.text(xlims[1], ylims[0], string, fontsize=20)
        else:
            return string


    @staticmethod
    def dict_merger(a, b):

        ''' Merges a and b: dictionaries of lists / array
        by concatenating their attributes.
        a and b need the same keys
        '''

        assert a.keys() == b.keys()

        c ={}

        for key, a_val, b_val in zip(a.keys(), a.values(), b.values()):
            c[key]=np.concatenate((a_val, b_val))

        return c
