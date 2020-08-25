import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA, NMF
from scipy.stats import ks_2samp, mode
from average_traces import AverageTraces
from pop_off_functions import prob_correct, mean_accuracy, score_nonbinary
from Session import build_flu_array_single
from utils_funcs import build_flu_array
import pickle
from popoff import loadpaths
USER_PATHS_DICT = loadpaths.loadpaths()



def do_pca(data, model, plot=False):

    X = data
    model.fit(X)
    varexp = np.cumsum(model.explained_variance_ratio_)
    components = model.components_

    if plot:
        plt.plot(varexp, label="dff", color=colors[0])
        plt.legend()
        plt.xlabel("Num. of components")

        plt.ylabel("Variance explained")
    return varexp, components


def pca_session(session, n_components=100, plot=False):


    ''' Takes a session object creates [n_components, n_trials, n_frames] array
    Creation of the array is achieved by running PCA on the full fluoresence matrix
    then using "backend" functions from Session.py to create this array in the same way
    as session.behaviour_trials is created. This requires a slightly awkward import and modifcation
    of build_flu_array_X as Session.py was not built to be used in this way

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
        run.frames_ms = np.tile(mode(run.frames_ms, axis=0)[0], (n_components, 1))
        run.frames_ms_pre = np.tile(mode(run.frames_ms_pre, axis=0)[0], (n_components, 1))

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
    nonnan_pre = np.unique(np.where(~np.isnan(arr_pre))[1])
    session.comps = arr[:, session.nonnan_trials, :]
    session.comps_pre = arr_pre[:, nonnan_pre, :]

    session.clean_obj()  # Garbage collection to remove session.run

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


class LinearModel():

    def __init__(self, session, times_use):

        self.session = session
        # times_use is inherited from AverageTraces, specify as an arugment
        # rather than using class inheritence so that this class does
        # not need to load every session
        self.times_use = times_use
        self.setup_flu()

        self.colors = [
            '#08F7FE',  # teal/cyan
            '#FE53BB',  # pink
            '#F5D300',  # yellow
            '#00ff41',  # matrix green
        ]

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

        ### henlo

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
                     n_comps_include=0, prereward=False):

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
            trial_bool = self.session.photostim == tt_map[trial_type]
            # Subsample flu attribute to just trials of interest
            flu = self.flu[:, trial_bool, :]
            # Decode lick vs no lick
            y = self.session.decision
            y = y[trial_bool]

            # Make sure binary classification
            assert len(set(y)) == 2

        # For the full model the independent variables are the mean activity across the whole trial
        # for every cell
        if model == 'full':
            X = self.covariates_full(flu=flu, frames=frames)
        elif model == 'partial':
            X = self.covariates_partial(flu=flu, frames=frames,
                                        trial_bool=trial_bool,
                                        n_comps_include=n_comps_include,
                                        prereward=prereward)
        else:
            raise ValueError(f'model {model} not recognised')

        # Get input matrix to (n_samples, n_features)
        X = X.T
        # Demean and scale to unit varience
        scaler = sklearn.preprocessing.StandardScaler()
        X = scaler.fit_transform(X)

        return X, y

    def covariates_full(self, flu, frames):

        flu_frames = flu[:, :, self.frames_map[frames]]
        X = np.mean(flu_frames, 2)

        return X

    def covariates_partial(self, flu, frames, trial_bool, n_comps_include, prereward=False):

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


    @staticmethod
    def logistic_regression(X, y, penalty, reg_strength, solver, n_folds=5, binary_classify=False):

        results = []
        models = []

        folds = sklearn.model_selection.StratifiedKFold(
            n_splits=n_folds, shuffle=True)

        for train_idx, test_idx in folds.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = sklearn.linear_model.LogisticRegression(penalty=penalty, C=reg_strength,
                                                            class_weight='balanced', solver=solver)
            model.fit(X=X_train, y=y_train)

            models.append(model)

            if binary_classify:
                results.append(model.score(X_test, y_test))
            else:
                results.append(score_nonbinary(model, X_test, y_test))

        return np.mean(results), np.std(results), models

    def performance_vs_reg(self, X, y, penalty, solvers):

        reg_strengths = np.logspace(-4, 3, 20)

        for idx, solver in enumerate(solvers):

            means = []
            stds = []

            for reg_strength in reg_strengths:

                # Plotting performance as a function of regularisation strength
                # when there is no regularisation doesnt make sense, but it is
                # the easiest way to eyeball the plots
                if penalty == 'none':
                    reg_strength = 0

                mean_acc, std_acc, _ = self.logistic_regression(
                    X, y, penalty, reg_strength, solver)
                means.append(mean_acc)
                stds.append(std_acc)

            means = np.array(means)
            stds = np.array(stds)
            sems = stds / 5

            plt.plot(reg_strengths, means, label=solver,
                     color=self.colors[idx])
            plt.xscale('log')
            plt.fill_between(reg_strengths, means-sems, means +
                             sems, color=self.colors[idx], alpha=0.3)
            plt.axhline(0.5, linestyle=':')
            plt.ylim((0, 0.5))

        plt.legend()
        plt.title(penalty.upper())
        plt.ylim(0.35, 1)
        plt.xlabel('C (Inverse Regularisation Strength)')
        plt.ylabel('Classifier Performance')

    def model_params_plot(self, frames='all', n_comps_in_partial=10):

        for idx, model in enumerate(['full', 'partial']):

            if model == 'partial':
                continue

            plt.figure(figsize=(18, 8))
            plt.suptitle(model, fontsize=30)

            X, y = self.prepare_data(
                frames=frames, model=model, trial_type='test', n_comps_include=n_comps_in_partial)

            solvers_dict = {
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

    def plot_betas(self, frames, model, n_comps_in_partial=10):

        X, y = self.prepare_data(
            frames, model, n_comps_include=n_comps_in_partial)
        acc, std_acc, models = self.logistic_regression(
            X, y, 'l1', 0.5, 'saga')
        plt.figure(figsize=(10, 5))

        for model in models:
            coef = np.squeeze(model.coef_)
            plt.plot(coef, '.', markersize=10, color=self.colors[0])

        labels = ['Mean Activity', r'Population $\Delta$F', 'Mean activity pre',
                  'Mean activity post', 'Mean trace correlation pre', 'Mean trace correlation post']

        [labels.append(f'PC{i}') for i in range(n_comps_in_partial)]

        plt.axhline(0, linestyle=':')
        plt.ylabel(r'$\beta$')
        plt.xlabel('Covariate')
        xt = plt.xticks(np.arange(len(labels)), labels,
                        rotation=90, fontsize=14)

        # Useful for full model, how sparse is B vector -> how many cells important?
        print(sum(coef == 0) / len(coef))


    def partial_model_performance(self, frames, n_comps_in_partial):

        # Full model performance
        X, y = self.prepare_data('all', 'full')
        mean_full, std_full, _ = self.logistic_regression(
            X, y, 'none', 0.5, 'saga')

        # Partial model performance as a function of number of covariates
        X, y = self.prepare_data(
            'all', 'partial', n_comps_include=n_comps_in_partial)

        if n_comps_in_partial > 0:
            idx_covariates = [[0], [0, 1], [0, 1, 2, 3],
                              [0, 1, 2, 3, 4, 5], range(X.shape[1])]
            labels = ['Mean activity only', r'+ Population $\Delta$F', '+ Mean pre & post',
                      '+ Correlations pre & post', f'+ {n_comps_in_partial} PCs', 'full model']
        else:
            idx_covariates = [[0], [0, 1], [0, 1, 2, 3], range(X.shape[1])]
            labels = ['Mean activity only', r'+ Population $\Delta$F', '+ Mean pre & post',
                      '+ Correlations pre & post', 'full model']

        partial_means = []
        partial_stds = []
        for idx in idx_covariates:

            partial_mean, partial_std, _ = self.logistic_regression(
                X[:, idx], y, 'l1', 0.5, 'saga')
            partial_means.append(partial_mean)
            partial_stds.append(partial_std)

        plot_mean = np.array(partial_means + [mean_full])
        plot_std = np.array(partial_stds + [std_full])

        plt.plot(plot_mean, marker='o')
        plt.fill_between(np.arange(len(plot_mean)), plot_mean -
                         plot_std, plot_mean + plot_std, alpha=0.2)
        xt = plt.xticks(np.arange(len(labels)), labels,
                        rotation=90, fontsize=18)
        plt.ylabel('Mean classification accuracy')

    def performance_vs_ncomps(self, frames):

        # Partial model performance as a function of n_components
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
        plt.ylim(0.4, 1)

    def target_info(self):

        # Don't need the frame dimension here
        is_target = self.session.is_target[:, :, 0]
        is_target = is_target[self.session.s1_bool, :]
        is_target = is_target[:, self.session.photostim == 1]
        self.n_times_targetted = np.sum(is_target, 1)
        self.ever_targetted = np.any(is_target, axis=1)

    def targets_histogram(self):

        plt.hist(n_times_targetted)
        plt.xlabel('Number of times targeted')
        plt.title('S1')

    def beta_targets_correlation(self):

        X, y = self.prepare_data('all', 'full')
        acc, std_acc, models = self.logistic_regression(
            X, y, 'l1', 0.5, 'saga')
        coef = np.squeeze(models[0].coef_)[self.session.s1_bool]
        plt.plot(self.n_times_targetted, abs(coef), '.')
        plt.xlabel('Number of times targeted')
        plt.ylabel(r'|$\beta$|')

        # Can we reject the null that the distribution of the target and non-target betas is the same?
        _, p_val = ks_2samp(coef[self.ever_targetted],
                            coef[~self.ever_targetted])
        if p_val < 0.05:
            print('NULL REJECTED!!')

    def target_proba(self):

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
            plt.plot([0, 1], [P, P1], color=self.colors[0],
                     marker='.', markersize=10)

            plt.xlim(-0.5, 1.5)

    def project_model(self, frames='all', model='full', plot=False, binary_classify=False):

        # These hyperparams give the best performance across sessions
        penalty = 'l2'
        solver = 'lbfgs'
        C = 0.5

        # Prepare data for hit and miss trials
        X, y = self.prepare_data(frames=frames, model=model, n_comps_include=10)


        # Prepare data for fp vs cr trials
        X_catch, y_catch = self.prepare_data(frames=frames, model=model,
                                             trial_type='catch', n_comps_include=10)

        # Prepare data for prereward trials
        X_pre, y_pre = self.prepare_data(frames=frames, model=model,
                                         n_comps_include=10, prereward=True)

        # Cross validated full model accuracy
        mean_test, std_test, models = self.logistic_regression(
            X, y, penalty=penalty, reg_strength=C, solver=solver, binary_classify=binary_classify)


        # Test different trial types on each of the 5 models fit on hit vs miss
        accs_catch = []
        accs_pre = []
        for model in models:
            if binary_classify:
                accs_catch.append(model.score(X_catch, y_catch))
                accs_pre.append(model.score(X_pre, y_pre))
            else:
                accs_catch.append(score_nonbinary(model, X_catch, y_catch))

                accs_pre.append(score_nonbinary(model, X_pre, y_pre))

        if plot:
            plt.errorbar(0, mean_test, std_test, marker='o',
                         capsize=10, color=self.colors[0])
            plt.errorbar(1, np.mean(accs_catch), np.std(accs_catch),
                         marker='o', capsize=10, color=self.colors[1])
            plt.errorbar(2, np.mean(accs_pre), np.std(accs_pre),
                         marker='o', capsize=10, color=self.colors[2])

            plt.ylim(0.4, 1)
            plt.xticks([0, 1, 2], ['Hit vs Miss', 'FP vs CR', 'Spont'],
                       rotation=45)

            plt.axhline(0.5, linestyle=':')

        return ((mean_test, std_test),
                (np.mean(accs_catch), np.std(accs_catch)),
                (np.mean(accs_pre), np.std(accs_pre)))


class PoolAcrossSessions(AverageTraces):

    def __init__(self, save_PCA=True):

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

        self.linear_models = [LinearModel(
            session, self.times_use) for session in self.sessions.values()]

    def project_model(self, frames='all', model='full'):

        results = [linear_model.project_model(frames=frames,
                                                         model=model, 
                                                         plot=False) 
                              for linear_model in self.linear_models]

        results = np.array(results)

        means = np.mean(results[:,:,0], 0)
        # I am combining STDs across sessions based on this post
        # https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation
        variences = np.square(results[:,:,1])
        stds = np.sqrt(np.mean(variences, 0))

        return means, stds

    def model_params_plot(self):

        # Makes sure plot that can be eyeballed in jupyter notebook
        for linear_model in self.linear_models:
            linear_model.setup_flu()
            linear_model.model_params_plot()









