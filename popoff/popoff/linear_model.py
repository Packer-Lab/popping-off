import os
import sys
import pdb, traceback, sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA, NMF
from scipy import sparse
from scipy.stats import ks_2samp, mode
from average_traces import AverageTraces
from pop_off_functions import prob_correct, mean_accuracy, score_nonbinary
from Session import build_flu_array_single
from utils_funcs import build_flu_array
import pickle
from popoff import loadpaths

USER_PATHS_DICT = loadpaths.loadpaths()

COLORS = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41',  # matrix green
]


def do_pca(data, model, plot=False):

    ''' Run PCA on data matrix
        
    Parameters
    -----------
    data : matrix to do PCA on
    model: PCA object
    plot : bool, default False
        Plot varience explained curve?
    '''

    X = data
    model.fit(X)
    varexp = np.cumsum(model.explained_variance_ratio_)
    components = model.components_

    if plot:
        plt.plot(varexp, label="dff", color=COLORS[0])
        plt.legend()
        plt.xlabel("Num. of components")

        plt.ylabel("Variance explained")
    return varexp, components


def pca_session(session, n_components=100, plot=False):

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
    flu = session.run.flu

    _, components = do_pca(flu, PCA(n_components=n_components), plot=plot)
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
    nonnan = np.unique(np.where(~np.isnan(arr))[1])
    nonnan_pre = np.unique(np.where(~np.isnan(arr_pre))[1])
    session.comps = arr[:, session.nonnan_trials, :]
    # session.comps = arr[:, nonnan, :]
    session.comps_pre = arr_pre[:, nonnan_pre, :]

    print(np.isnan(session.comps).any())
    session.clean_obj()  # "Garbage collection" to remove session.run

    return session


def trace_correlation(flu, frames):

    ''' Compute trial-wise trace correlation for fluoresence array 

    Parameters
    ----------
    flu : fluoresence array [n_cells x n_trials x n_frames]
    frames : indexing array, frames across which to compute correlation

    Returns
    -------
    trial_corr : vector of len n_trials ->
                 mean of covariance matrix on each trial.

    '''

    trial_corr = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        mean_cov = np.mean(np.cov(trial), (0, 1))
        trial_corr.append(mean_cov)

    return np.array(trial_corr)


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

    def __init__(self, session, times_use):
        
        ''' Perform logistic regression on Session object

        Attributes
        ----------
        session : individual session object
        times_use : inherited from AverageTraces object, times common
                    across sessions with different frame rates

        '''

        self.session = session
        # times_use is inherited from AverageTraces, specify as an arugment
        # rather than using class inheritence so that this class does
        # not need to load every session
        self.times_use = times_use
        self.setup_flu()


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
        self.pre = self.times_use < 0  # times_use inherited from AverageTraces
        self.post = self.times_use > 1
        self.remove_artifact = np.logical_or(self.pre, self.post)

        # Allows future functions to use 'frames' argument as a string
        # to be mapped to matching indexing array
        self.frames_map = {'pre': self.pre,
                           'post': self.post,
                           'all': self.remove_artifact}


    def prepare_data(self, frames='all', model='full', trial_type='test',
                     outcomes=['hit', 'miss', 'cr', 'fp'], n_comps_include=0, 
                     prereward=False, return_matrix=True):

        ''' Prepare fluoresence data in Session object for regression

        Parameters
        ----------
        session : Session object to get data from

        frames : {'pre', 'post', 'all'}, default='all'
            Which trial frames (relative to photostim) to return?

        model : {'full', 'partial'}, default='full'
            full = include mean activity of all cells in model
            partial = include only 'network features' e.g. PCS in model

        n_comps_include : How many PCs to include in the partial model 
            (requires model='partial')

        outcomes : {['hit', 'miss', 'fp', 'cr']}, default=['hit', 'miss']
                   which trial types do you want to decode on?

        TODO: Currently partial regression parameters are added in a hacky, code repeating way
              Make this much more flexible so the user can specify an argument of which
              parameters to include

        Returns
        --------
        X : data matrix for use as independent variable [n_samples x n_features]
        y : vector for use as dependent variable [n_samples]

        '''

        # Select the trial type of interest
        tt_map = {'catch': 0,
                  'test': 1,
                  'easy': 2}

        if prereward:
            flu = self.pre_flu
            y = np.ones(flu.shape[1])
            trial_bool = y.astype('bool')

        else:
            outcome = self.session.outcome[self.session.nonnan_trials]

            trial_bool = np.isin(outcome, outcomes)

            # Change 100 to 2 if you want only test trials
            remove_easy = self.session.photostim != 100
            trial_bool = np.logical_and(trial_bool, remove_easy)

            outcome = outcome[trial_bool]

            self.encoder = LabelEncoder(['miss', 'hit', 'cr', 'fp'])
            # Fit encoder on data that is independent of the trial
            # order in the session
            self.encoder.fit(outcome)
            y = self.encoder.transform(outcome)
            flu = self.flu[:, trial_bool, :]

        # For the full model the independent variables are the mean activity across the whole trial
        # for every cell
        if model == 'full':
            X = self.covariates_full(flu=flu, frames=frames)
        elif model == 'partial':
            covariates_dict = self.covariates_partial(flu=flu, frames=frames,
                                        trial_bool=trial_bool,
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

        flu_frames = flu[:, :, self.frames_map[frames]]
        X = np.mean(flu_frames, 2)
        return self.transform_data(X)

    def covariates_partial(self, flu, frames, trial_bool, n_comps_include, prereward=False):

        # Function to subtract the mean of pre frames from the
        # mean of post frames -> [n_cells x n_trials]
        sub_frames = lambda arr: np.mean(arr[:, :, self.post], 2) - np.mean(arr[:, :, self.pre], 2)

        covariates_dict = {}

        # Mean population activity on every trial
        covariates_dict['trial_mean'] = np.mean(flu[:, :, self.remove_artifact], (0, 2))

        # Average post - pre across all cells
        covariates_dict['delta_f'] = np.mean(sub_frames(flu), 0)

        # Mean network activity just before the stim
        covariates_dict['mean_pre'] = np.mean(flu[:, :, self.pre], (0, 2))
        # Mean network activity just after the stim
        covariates_dict['mean_post'] = np.mean(flu[:, :, self.post], (0, 2))

        # Mean trace correlation pre stim
        covariates_dict['corr_pre'] = trace_correlation(flu, self.pre)
        # Mean trace correlation post stim
        covariates_dict['corr_post'] = trace_correlation(flu, self.post)

        if prereward:
            PCs = self.session.comps_pre
        else:
            PCs = self.session.comps
            PCs = PCs[:, trial_bool, :]

        PCs = PCs[:, :, self.session.frames_use]

        assert n_comps_include <= PCs.shape[0]
        PCs = PCs[0:n_comps_include, :, :]

        if frames == 'all':
            covariates_dict['PCs'] = sub_frames(PCs)
        elif frames == 'pre':
            covariates_dict['PCs'] = np.mean(PCs[:, :, self.pre], 2)
        elif frames == 'post':
            covariates_dict['PCs'] = np.mean(PCs[:, :, self.post], 2)

        # Use this if you don't want to subtract but rather mean whole trace (worse performance)
        #x3 = np.mean(PCs[:, :, remove_artifact], 2)

        if frames == 'pre':
            # Set this at the start of the function? Remove code repetition?
            pre_keys = ['mean_pre', 'corr_pre', 'PCs']
            covariates_dict = {k:covariates_dict[k] for k in pre_keys}
            print('hi')
        elif frames == 'post':
            print('hi')
            pre_keys = ['mean_post', 'corr_post', 'PCs']
            covariates_dict = {k:covariates_dict[k] for k in post_keys}

        return covariates_dict

    def covariates_partial_bak(self, flu, frames, trial_bool, n_comps_include, prereward=False):

        # Function to subtract the mean of pre frames from the
        # mean of post frames -> [n_cells x n_trials]
        sub_frames = lambda arr: np.mean(arr[:, :, self.post], 2) - np.mean(arr[:, :, self.pre], 2)

        # Mean population activity on every trial
        x1 = np.mean(flu[:, :, self.remove_artifact], (0, 2))

        # Average post - pre across all cells
        x2 = np.mean(sub_frames(flu), 0)

        # Mean network activity just before the stim
        x3 = np.mean(flu[:, :, self.pre], (0, 2))
        # Mean network activity just after the stim
        x4 = np.mean(flu[:, :, self.post], (0, 2))

        # Mean trace correlation pre stim
        x5 = trace_correlation(flu, self.pre)
        # Mean trace correlation post stim
        x6 = trace_correlation(flu, self.post)

        if prereward:
            PCs = self.session.comps_pre
        else:
            PCs = self.session.comps
            PCs = PCs[:, trial_bool, :]

        PCs = PCs[:, :, self.session.frames_use]

        assert n_comps_include <= PCs.shape[0]
        PCs = PCs[0:n_comps_include, :, :]

        if frames == 'all':
            x7 = sub_frames(PCs)
        elif frames == 'pre':
            x7 = np.mean(PCs[:, :, self.pre], 2)
        elif frames == 'post':
            x7 = np.mean(PCs[:, :, self.post], 2)

        # Use this if you don't want to subtract but rather mean whole trace (worse performance)
        #x3 = np.mean(PCs[:, :, remove_artifact], 2)

        if frames == 'all':
            X = np.vstack((x1, x2, x3, x4, x5, x6, x7))
        elif frames == 'pre':
            X = np.vstack((x3, x5, x7))
        elif frames == 'post':
            X = np.vstack((x4, x6, x7))

        return X

    def build_confusion_matrix(self, y_true, y_pred):
        
        ''' Builds a "3d" confusion matrix, allowing you to stack
            mutliple confusion matrices by multiple calls to this function
            designed to later be summed across 3rd dimension
            '''

        C = sklearn.metrics.confusion_matrix(y_true, y_pred)
        
        if not hasattr(self, 'confusion_matrix'):
            self.confusion_matrix = C
        else:
            self.confusion_matrix = np.dstack((self.confusion_matrix, C))


    def logistic_regression(self, X, y, penalty, C, solver='lbfgs', n_folds=5, 
                            digital_score=True, compute_confusion=False,
                            filter_models=False):
        
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

        folds = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=True)

        for train_idx, test_idx in folds.split(X, y):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model =  sklearn.linear_model.LogisticRegression(penalty=penalty, C=C,
                                                            class_weight='balanced', solver=solver)
            model.fit(X=X_train, y=y_train)


            if not filter_models or model.score(X_test, y_test) > 0.7:
                models.append(model)
            else:
                print('model filtering on')

            if compute_confusion:
                # Add this model performance to the running confusion matrix
                self.build_confusion_matrix(y_test, model.predict(X_test))

            if digital_score:
                results.append(model.score(X_test, y_test))
            else:
                results.append(score_nonbinary(model, X_test, y_test))

        return np.mean(results), np.std(results), models


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

            plt.figure(figsize=(18, 8))
            plt.suptitle(model, fontsize=30)

            X, y = self.prepare_data(frames=frames, model=model,
                                     trial_type='test',
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
                plt.ylim(0,1)


    def plot_betas(self, frames, model, n_comps_in_partial=10):
        
        ''' Plot the beta values of each covariate in the model 
            TODO: Make compatible with mutliclass classifier
            '''

        X, y = self.prepare_data(frames, model, 
                                 n_comps_include=n_comps_in_partial,
                                 outcomes=['hit', 'miss'])

        acc, std_acc, models = self.logistic_regression(X, y, 'l1', 0.5,
                                                        'saga', filter_models=False)
        coefs = []
        for model in models:
            coef = np.squeeze(model.coef_)
            coefs.append(coef)
            plt.plot(coef, '.', markersize=10, color=COLORS[0])

        labels = ['Mean Activity', r'Population $\Delta$F', 'Mean activity pre',
                  'Mean activity post', 'Mean trace correlation pre', 'Mean trace correlation post']

        [labels.append(f'PC{i}') for i in range(n_comps_in_partial)]

        plt.axhline(0, linestyle=':')
        plt.ylabel(r'$\beta$')
        plt.xlabel('Covariate')
        xt = plt.xticks(np.arange(len(labels)), labels,
                        rotation=90, fontsize=14)

        if model == 'full':
            # Useful for full model, how sparse is B vector -> how many cells important?
            print(sum(coef == 0) / len(coef))


    def partial_model_performance(self, frames, n_comps_in_partial=10, plot=True):

        ''' Plot partial model performance relative to full model 
            performance as a function of number of partial covariates
            included '''


        # These hyperparams give the best performance across sessions
        penalty = 'l2'
        solver = 'lbfgs'
        C = 0.5

        # Full model performance
        outcomes = ['hit', 'miss']
        # outcomes = ['hit', 'miss', 'fp', 'cr']

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
            X = self.dict2matrix({k:v for k, v in covariates_dict.items() if k in covs})

            # mean, std, _ = self.logistic_regression(X, y, 'l1', 0.5, 'saga')

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
            plt.xticks(range(len(means)), tick_labels, rotation=80, fontsize=18)
            plt.axhline(0.5, linestyle=':')
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

        X, y = self.prepare_data('all', 'full')
        acc, std_acc, models = self.logistic_regression(X, y, 'l1', 0.5,
                                                        'saga')

        # The index of the coef that corresponds to hit trials
        hit_idx = self.encoder.transform(['hit'])[0]
        coef = models[0].coef_[hit_idx, :]

        # Can we reject the null that the distribution of the target and 
        # non-target betas is the same?
        _, p_val = ks_2samp(coef[self.ever_targetted],
                            coef[~self.ever_targetted])
        if p_val < 0.05:
            print('NULL REJECTED!!')

        region_map = {'all': np.repeat(True, self.session.n_cells),
                      's1': self.session.s1_bool,
                      's2': self.session.s2_bool
                      }

        coef = coef[region_map[region]]
        n_times_targetted = self.n_times_targetted[region_map[region]]

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

    def project_model(self, frames='all', model='full', plot=False, digital_score=True):
        
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
        penalty = 'l2'
        solver = 'lbfgs'
        C = 0.5
        
        # Prepare data for hit and miss trials
        X, y = self.prepare_data(frames=frames, model=model, 
                                 outcomes=['hit', 'miss'],
                                 n_comps_include=10)

        # Prepare data for hit and miss trials X, y = self.prepare_data(frames=frames, model=model, n_comps_include=10) 
        # Prepare data for fp vs cr trials
        X_catch, y_catch = self.prepare_data(frames=frames, model=model,
                                             trial_type='catch', outcomes=['fp', 'cr'],
                                             n_comps_include=10)

        # Prepare data for prereward trials
        X_pre, y_pre = self.prepare_data(frames=frames, model=model,
                                         n_comps_include=10, outcomes=['hit', 'miss'],
                                         prereward=True)

        # Cross validated full model accuracy
        mean_test, std_test, models = self.logistic_regression(
            X, y, penalty=penalty, C=C, solver=solver, digital_score=digital_score)


        # Test different trial types on each of the 5 models fit on hit vs miss
        accs_catch = []
        accs_pre = []
        for model in models:
            if digital_score:
                accs_catch.append(model.score(X_catch, y_catch))
                accs_pre.append(model.score(X_pre, y_pre))
            else:
                accs_catch.append(score_nonbinary(model, X_catch, y_catch))
                accs_pre.append(score_nonbinary(model, X_pre, y_pre))

        if plot:
            plt.errorbar(0, mean_test, std_test, marker='o',
                         capsize=10, color=COLORS[0])
            plt.errorbar(1, np.mean(accs_catch), np.std(accs_catch),
                         marker='o', capsize=10, color=COLORS[1])
            plt.errorbar(2, np.mean(accs_pre), np.std(accs_pre),
                         marker='o', capsize=10, color=COLORS[2])

            plt.xticks([0, 1, 2], ['Hit vs Miss', 'FP vs CR', 'Spont'],
                       rotation=45)

            plt.axhline(0.5, linestyle=':')

        return ((mean_test, std_test),
                (np.mean(accs_catch), np.std(accs_catch)),
                (np.mean(accs_pre), np.std(accs_pre)))


class PoolAcrossSessions(AverageTraces):

    def __init__(self, save_PCA=False):
        
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

        for idx, session in self.sessions.items():

            # Components already computed and saved
            if hasattr(session, 'comps') and not save_PCA:  
                continue
            else: 
                self.sessions[idx] = pca_session(session, n_components=100,
                                                 plot=False)

        # Cache the PCA components to the Session object so we do not need to
        # recalculate every time this class is initialised
        if save_PCA:
            save_path = os.path.expanduser(
                f'{USER_PATHS_DICT["base_path"]}/sessions_lite_flu.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(self.sessions, f)

        # 
        self.linear_models = [LinearModel(
            session, self.times_use) for session in self.sessions.values()]


    def project_model(self, frames='all', model='full'):

        results = [linear_model.project_model(frames=frames,
                                              model=model, 
                                              plot=False) 
                  for linear_model in self.linear_models]

        results = np.array(results)
        means = results[:,:,0]
        stds = results[:,:,1]

        grand_mean = np.mean(means, 0)
        grand_std = self.combine_stds(stds)

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

    def plot_betas(self, frames, model, n_comps_in_partial=10):

        plt.figure(figsize=(10,5))
        for linear_model in self.linear_models:
            # Will probably want to parse the coefs variable later
            coefs = linear_model.plot_betas(frames, model, n_comps_in_partial)

    def partial_model_performance(self, frames, n_comps_in_partial=10):
        
        means = []
        stds = []
        for linear_model in self.linear_models:
            mean, std = linear_model.partial_model_performance(frames,
                                                                 n_comps_in_partial,
                                                                 plot=False)
            means.append(mean)
            stds.append(std)

            
        means = np.array(means)
        stds = np.array(stds)

        grand_mean = np.mean(means, 0)
        grand_std = self.combine_stds(stds)
        
        plt.errorbar(range(len(grand_mean)), grand_mean, yerr=grand_std, fmt='o', 
                     color=COLORS[1], ecolor='lightgray', elinewidth=3, capsize=5)

        tick_labels = list(linear_model.add_partial_map.keys())
        tick_labels.append('Full model')
        plt.xticks(range(len(grand_mean)), tick_labels, rotation=80, fontsize=18)
        plt.axhline(0.5, linestyle=':')
        plt.ylabel('Mean classification accuracy')
        plt.ylim(0.4, 1)


