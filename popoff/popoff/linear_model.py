import os
import sys
import pdb, traceback, sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import sparse
from scipy.stats import ks_2samp, mode
from average_traces import AverageTraces
from pop_off_functions import prob_correct, mean_accuracy, score_nonbinary
from Session import build_flu_array_single, SessionLite
from utils_funcs import build_flu_array
import pickle
from popoff import loadpaths
from IPython.core.debugger import Pdb
ipdb = Pdb()

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
    flu = session.run.flu[session.filtered_neurons, :]
    flu = flu[cells_include, :]

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
    session.comps = arr[:, session.nonnan_trials, :]
    nonnan_pre = np.unique(np.where(~np.isnan(arr_pre))[1])
    session.comps_pre = arr_pre[:, nonnan_pre, :]

    if not save_PC_matrix:
        session.clean_obj()  # "Garbage collection" to remove session.run

    return session

def noise_correlation(flu, signal_trials, frames):

    ''' Compute trial-wise noise correlation for fluoresence array ->
        pairwise correlations minus mean
        

    Parameters
    ----------
    flu : fluoresence array [n_cells x n_trials x n_frames]
    signal_trials : which trials do you want to calculate the 
                    mean from for subtraction
    frames : indexing array, frames across which to compute correlation

    Returns
    -------
    trial_corr : vector of len n_trials ->
                 mean of correlation coefficient matrix matrix on each trial.
                 TODO: remove the diagonal (won't make much difference)

    '''

    trial_corr = []
    # mean_to_subtract = np.mean(flu[:, signal_trials, :], 1)
    # mean_to_subtract = mean_to_subtract[:, frames]

    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        # trial = trial - mean_to_subtract
        mean_cov = np.mean(np.corrcoef(trial), (0, 1))
        trial_corr.append(mean_cov)

    return np.array(trial_corr)


def largest_singular_value(flu, frames):

    singular_values = []
    for t in range(flu.shape[1]):
        trial = flu[:, t, :]
        trial = trial[:, frames]
        _, s, _ = np.linalg.svd(trial)
        singular_values.append(s[0])

    return np.array(singular_values)

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

    def __init__(self, session, times_use, remove_targets=False):
        
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
        self.remove_targets = remove_targets

        self.setup_flu()
        self.target_info()

        if self.remove_targets:
            self.remove_targets_from_data()

        # Init encoder with required sort order
        self.encoder = LabelEncoder(['miss', 'hit', 'cr', 'fp'])


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

        self.region_map = {'all': np.repeat(True, self.session.n_cells),
                           's1': self.session.s1_bool,
                           's2': self.session.s2_bool
                           }


    def prepare_data(self, frames='all', model='full',
                     outcomes=['hit', 'miss', 'cr', 'fp'], region='all',
                     n_comps_include=0, prereward=False, return_matrix=True):

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
            y = np.ones(flu.shape[1])
            trial_bool = y.astype('bool')

        else:
            outcome = self.session.outcome[self.session.nonnan_trials]
            trial_bool = np.isin(outcome, outcomes)
            
            # Change 100 to 2 if you want only test trials
            # (REMOVE THIS HACK)
            remove_easy = self.session.photostim != 100
            trial_bool = np.logical_and(trial_bool, remove_easy)
            outcome = outcome[trial_bool]
            flu = self.flu[:, trial_bool, :]

            # Fit encoder on data that is independent of the trial
            # order in the session
            self.encoder.fit(outcome)
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
        covariates_dict['corr_pre'] = noise_correlation(flu, self.session.photostim==1, self.pre)
        # Mean trace correlation post stim
        covariates_dict['corr_post'] = noise_correlation(flu, self.session.photostim==1, self.post)

        covariates_dict['largest_singular_value'] = largest_singular_value(flu, self.pre)

        covariates_dict['flat'] = np.ones(*covariates_dict['delta_f'].shape)


        if prereward:
            if region != 'all': 
                raise NotImplementedError('prereward comps do not yet have '
                                           'region dependency')
            PCs = self.session.comps_pre
        else:
            # PCs = self.session.comps
            print(f' Cell included from region {region}')
            PCs = self.session.pca_dict[region]
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
        elif frames == 'post':
            pre_keys = ['mean_post', 'corr_post', 'PCs']
            covariates_dict = {k:covariates_dict[k] for k in post_keys}

        return covariates_dict

    def build_confusion_matrix(self, y_true, y_pred):
        
        ''' Builds a "3d" confusion matrix, allowing you to stack
            mutliple confusion matrices by multiple calls to this function
            designed to later be summed across 3rd dimension
            '''

        C = sklearn.metrics.confusion_matrix(y_true, y_pred)

        if C.shape != (4,4):
            return
        
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


            if not filter_models or model.score(X_test, y_test) > 0.55:
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
                plt.ylim(0,1)


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


        labels = ['Mean Activity', r'Population $\Delta$F',
                  'Mean activity pre', 'Mean activity post',
                  'Mean noise correlation pre', 'Mean noise correlation post',
                  'Largest SV', 'flat']

        [labels.append(f'PC{i}') for i in range(n_comps_in_partial)]

        # Useful for full model, how sparse is B vector -> how many cells important?
        if model == 'full': print(sum(coef == 0) / len(coef))


        if plot:
            # Legend with duplicates removed 
            # https://stackoverflow.com/a/13589144/10723511
            handles, leg_labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(leg_labels, handles))
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04,1))
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
            X = self.dict2matrix({k:v for k, v in covariates_dict.items() if k in covs})

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
        self.region_map = {keys:values[~self.ever_targetted] 
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
        penalty = 'l1'
        solver = 'saga'
        C = 0.5
        
        # Prepare data for hit and miss trials
        X, y = self.prepare_data(frames=frames, model=model, 
                                 outcomes=['hit', 'miss'],
                                 n_comps_include=10)

        # Prepare data for hit and miss trials 
        # X, y = self.prepare_data(frames=frames, model=model, n_comps_include=10) 
        # Prepare data for fp vs cr trials
        X_catch, y_catch = self.prepare_data(frames=frames, model=model,
                                             outcomes=['fp', 'cr'],
                                             n_comps_include=10)

        # Prepare data for prereward trials
        X_pre, y_pre = self.prepare_data(frames=frames, model=model,
                                         n_comps_include=10,
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




    def compare_regions(self, frames='all', plot=True):
        
        penalty = 'l2'
        C = 0.5
        solver = 'lbfgs'
        
        regions = ['s1', 's2', 'all']

        mean_accs = []
        std_accs = []

        for idx, region in enumerate(regions):

            X, y = self.prepare_data(frames=frames, model='full',
                                     outcomes=['hit', 'miss', 'fp', 'cr'], 
                                     region=region,
                                     n_comps_include=0)

            mean_acc, std_acc, _ = self.logistic_regression(X, y, penalty,
                                                            C, solver)
            mean_accs.append(mean_acc)
            std_accs.append(std_acc)

            if plot:
                plt.errorbar(idx, mean_acc, yerr=std_acc, fmt='o', 
                             color=COLORS[1], ecolor='lightgray', elinewidth=3, capsize=5)

        if plot:
            plt.ylim(0,1)
            plt.axhline(0.25)
            plt.xticks(range(3), regions)

        return mean_accs, std_accs


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

            session = pca_session(self.session, cells_include, n_components=n_components, plot=False, 
                        save_PC_matrix=save_PC_matrix)
            # Weird way of building a dictionary from session attributes but avoids
            # changing the structure of pca_session to split by region
            self.session.pca_dict[region_name] = session.comps

class PoolAcrossSessions(AverageTraces):

    def __init__(self, save_PCA=False, remove_targets=False):
        
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


        self.linear_models = [LinearModel(session, self.times_use,
                                          remove_targets=remove_targets)
                              for session in self.sessions.values()]

        # Add PCA attributes to session if they are not already saved
        # for idx, session in self.sessions.items():
        for idx, linear_model in enumerate(self.linear_models):

            # Components already computed and saved
            if hasattr(linear_model.session, 'comps') and not save_PCA:  
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
                f'{USER_PATHS_DICT["base_path"]}/sessions_lite_flu.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(self.sessions, f)

        # Subsample sessions to make a training set
        self.sessions = {key:value for key, value in 
                         self.sessions.items() if key in [2,5,14]}

        # This is a shitty fix redefining this variable but it allows for caching
        # of the pca_dict variable
        self.linear_models = [LinearModel(session, self.times_use,
                                          remove_targets=remove_targets)
                              for session in self.sessions.values()]

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

            fig, ax = plt.subplots(figsize=(16,4))
            trial_coefs = all_coefs[:, trial_idx, :]
            box = ax.boxplot(trial_coefs, showfliers=True)
                
            plt.setp(box['fliers'], markeredgecolor=COLORS[trial_idx])
            for _, line_list in box.items():
                for line in line_list:
                    line.set_color(COLORS[trial_idx])
        
            label = self.linear_models[0].encoder.inverse_transform([trial_idx])[0]
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
        
        plt.figure(figsize=(8,6))
        plt.errorbar(range(len(grand_mean)), grand_mean, yerr=grand_std, fmt='o', 
                     color=COLORS[1], ecolor='lightgray', elinewidth=3, capsize=5)

        tick_labels = list(linear_model.add_partial_map.keys())
        tick_labels.append('Full model')
        plt.xticks(range(len(grand_mean)), tick_labels, rotation=80, fontsize=18)
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

        for linear_model in self.linear_models: 
           mean_accs, std_accs = linear_model.compare_regions(frames=frames, plot=False)
           all_means.append(mean_accs)
           all_stds.append(std_accs)

        all_means = np.array(all_means)
        all_stds = np.array(all_stds)

        grand_mean = np.mean(all_means, axis=0)
        grand_std = self.combine_stds(all_stds)

        plt.errorbar(range(3), grand_mean, yerr=grand_std, fmt='o', 
                     color=COLORS[1], ecolor='lightgray', elinewidth=3, capsize=5)

        regions = ['s1', 's2', 'all']

        plt.ylim(0,1)
        plt.axhline(0.25, linestyle=':')
        plt.xticks(range(3), regions)
    
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
                  display_labels=linear_model.encoder.inverse_transform([0,1,2,3]))

        cmd.plot(cmap ='Blues')





