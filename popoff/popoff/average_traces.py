import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import pandas as pd
import seaborn as sns
import math
import copy
from scipy import stats
from statsmodels.stats import multitest


from Session import SessionLite, Session
from loadpaths import loadpaths

## Wes Anderson color palette
# sys.path.append(os.path.expanduser('~/Documents/code'))
# import wampl.wes as wes
# wes.set_palette('FantasticFox1')
# plt.rcParams['pdf.fonttype'] = 42

# WES_COLS = []

# for ii, x in enumerate(plt.rcParams['axes.prop_cycle']()):
    # if ii%5 == 0:
        # WES_COLS.append(x['color'])
    # if ii > 100:
        # break

class AverageTraces():
    
    def __init__(self, flu_flavour):

        """ Class to load and process Session objects from Session.py 
            and create trial by trial arrays across multiple sessions
            sorted by trial types and brain area.

            Input Arguments
            -----------------
            flu_flavour -- which flu data type to load, valid inputs:
                           [dff, denoised_flu, spks, raw]
            mask -- [n_cells x n_trials x n_timepoints]. Cells / trials
                    you want to mask

            Methods
            -----------------
            load_sessions    -- loads the sessions pkls built by Session.py
                                and defined by flu flavour. Called on __init__
            match_framerates -- patch frames_use attribute to Session objects
                                allowing for simultaneous analysis of 5 Hz and
                                30 Hz data.
            build_trace_dict -- builds a dictionary with keys 's1' and 's2'. Each
                                val contains a tuple of arrays [n_trials x n_frames] containing
                                cell averaged data for every trial recorded across all
                                sessions. Array 0 in tuple == behaviour data; array 2 in tuple ==
                                prereward data.
            tt_raveled       -- get the indexs of trial types matched to arrays in trace_dict.
                                Inputs:
                                trial_type -> [easy, test, nogo, all]
                                trial_outcome -> [hit, miss, fp, cr, ar_miss, ur_hit, all]
                                Returns:
                                boolean list of trials of both trial_type and trial_outcome
            s1s2_plot        -- plot trial averaged traces of trial types across brain areas
                                see average_traces.ipynb

            """


        self.flu_flavour = flu_flavour
        self.user_paths = loadpaths()
        self.load_sessions()
        self.match_framerates()
        self.n_plots=0


    def load_sessions(self, ts='2021-10-22'):

        base_path = self.user_paths['base_path']
        # if self.flu_flavour == 'denoised_flu':
        #     sessions_file = f'sessions_lite_denoised_flu_{ts}.pkl'
        # elif self.flu_flavour == 'dff':
        #     sessions_file = f'sessions_lite_flu_{ts}.pkl'
        # elif self.flu_flavour == 'spks':
        #     sessions_file = f'sessions_lite_spks_{ts}.pkl'
        # elif self.flu_flavour == 'raw':
        #     sessions_file = f'sessions_lite_flu_raw_{ts}.pkl'
        # else:
        #     raise ValueError('flu_flavour not recognised')
            
        sessions_file = 'sessions_lite_flu_2022-08-11.pkl'
        sessions_path = os.path.join(base_path, sessions_file)
                                 
        with open(sessions_path, 'rb') as f:
            self.sessions = pickle.load(f)


    @property
    def tp_dict(self):

        ''' Calculate the imaging frames to use that match across different fs
        Returns a dictionary tp_dict which has key 'mutual'.
        This contains a list of times that are shared between different 
        frame rates. 
        This can be used to index session.filter_ps_time, allowing combining of 
        sessions recorded with different frame rates.
        The returned dictionary is parsed by match_framerates
            
        '''

        ## Integrate different imaging frequencies:
        freqs = np.unique([ss.frequency for _, ss in self.sessions.items()])
        tp_dict = {}
        for ff in freqs:
            # assume pre_seconds & post_seconds equal for all sessions
            for _, ss in self.sessions.items():   
                if ss.frequency == ff:
                    tp_dict[ff] = ss.filter_ps_time

        if len(freqs) == 2:  # for hard-coded bit next up
            tp_dict['mutual'] = np.intersect1d(ar1=tp_dict[freqs[0]], 
                                               ar2=tp_dict[freqs[1]])
        elif len(freqs) == 1:
            tp_dict['mutual'] = tp_dict[freqs[0]]
        
        return tp_dict


    def match_framerates(self):

        ''' For each session, patch a attribute 'frames_use'
        which based on tp_dict allows for each combinbing of sessions
        with different frame rates '''

        # Find the frames to use that match across all sessions
        # This is used to fix matthias' crazy long trials
        baseline_start = -8
        trial_end = 8
        self.times_use = self.tp_dict['mutual']
        self.times_use = self.times_use[self.times_use >= baseline_start]
        self.times_use = self.times_use[self.times_use <= trial_end]

        for idx, session in self.sessions.items():

            session.frames_use = [session.filter_ps_array
                                 [np.where(session.filter_ps_time == tt)[0][0]] 
                                 for tt in self.times_use]

            session.frames_use = np.array(session.frames_use)
                
            assert len(self.times_use) == len(session.frames_use) 
    

    def build_trace_dict(self):

        self.trace_dict = {

        's1': self.trace_tuple('s1'),
        's2': self.trace_tuple('s2')

        }


    def trace_tuple(self, region):

        ''' Calls session stacker for an individual region in order to build a trace dict '''
        
        behaviour = self.session_stacker(region, prereward=False, 
                                         sub_baseline=True)
        prereward = self.session_stacker(region, prereward=True,
                                         sub_baseline=True)
        
        return behaviour, prereward


    def session_stacker(self, cells_include='s1', prereward=False,
                        sub_baseline=True, mask=None):


        ''' Function underpinning self.trace_dict
            Returns stacked_trials  [n_trials x n_frames] (possibly tranpose of this)
            the cell averaged trace (of cells specified in cells_include for each trial of each session
            '''

        for idx, session in enumerate(self.sessions.values()):

            if prereward:
                behaviour_trials = session.pre_rew_trials
            else:
                behaviour_trials = session.behaviour_trials

            if mask == 'targets' and not prereward:
                behaviour_trials = np.ma.array(behaviour_trials, mask=~session.is_target)
            elif mask == 'followers' and not prereward:
                behaviour_trials = np.ma.array(behaviour_trials, mask=session.is_target)

            # Cells include is a list of boolean lists of len(n_sesions)
            if type(cells_include) == list:
                cell_bool = cells_include[idx]
            elif cells_include == 's1':
                cell_bool = session.s1_bool
            elif cells_include == 's2':
                cell_bool = session.s2_bool
            else:
                cell_bool = np.repeat(True, session.n_neurons)

            # Bug in Session.py kept NaN in prereward, think ive fixed but keep an eye
            if np.isnan(behaviour_trials).any():
                print('NaN ahoy')

            behaviour_trials = behaviour_trials[cell_bool, :, :]

            baseline_frames = np.where((session.filter_ps_time>=-2) & 
                                       (session.filter_ps_time<-1))[0]

            baseline = np.mean(behaviour_trials[:, :, baseline_frames], 2)
            
            if sub_baseline:
                baseline_subbed = behaviour_trials[:, :, session.frames_use] \
                                  - baseline[:,:,np.newaxis]
            else:
                baseline_subbed = behaviour_trials[:, :, session.frames_use]

            
            ##### HERE WE ARE DOING THE BASELINE SUBTRACTION AGAIN CHECK THAT THIS WORKS IT 
            ##### USED TO BE DONE ABOVE BUT I CHANGED IT AND I DONT KNOW WHY 

            # pre_range, post_range = self.get_range(session)

            # subbed = np.mean(baseline_subbed[:, :, self.times_use<0], 2)\
                     # - np.mean(baseline_subbed[:, :, self.times_use>0], 2)

            # subbed = np.repeat(subbed[:, :, np.newaxis], baseline_subbed.shape[2], axis=2)
            # baseline_subbed = np.ma.array(baseline_subbed, mask=subbed>0)
            
            # This is an inefficient way of making arrays
            if idx == 0:
                stacked_trials = np.mean(baseline_subbed, 0)
            else:
                stacked_trials = np.vstack((stacked_trials, 
                                            np.mean(baseline_subbed, 0)))
            
        return stacked_trials


    def tt_idxs(self, session, trial_type='all', trial_outcome='all'):
    
        assert len(session.photostim) == len(session.decision)

        if trial_type == 'nogo':
            type_use = session.photostim == 0
        elif trial_type == 'test':
            type_use = session.photostim == 1
        elif trial_type == 'easy':
            type_use = session.photostim == 2
        elif trial_type == 'all':
            type_use = np.repeat(True, len(session.photostim))
            
        if trial_outcome == 'hit' or trial_outcome=='fp':
            outcome_use = np.logical_and(session.decision == 1, 
                                         session.unrewarded_hits==False)
        elif trial_outcome == 'miss' or trial_outcome=='cr':
            outcome_use = np.logical_and(session.decision == 0,
                                         session.autorewarded==False)
        elif trial_outcome == 'ar_miss':
            outcome_use = session.autorewarded
        elif trial_outcome == 'ur_hit':
            outcome_use = session.unrewarded_hits
        elif trial_outcome == 'all':
            outcome_use = np.repeat(True, len(session.decision)) 
            
        return np.logical_and(type_use, outcome_use)


    def balanced_hitmiss(self, session, hits, misses):

        balanced_hits = np.full_like(hits, False)
        balanced_misses = np.full_like(misses, False)

        subsets = np.unique(session.trial_subsets)

        for subset in subsets:

            trial_idxs = session.trial_subsets == subset
            subset_hits = np.logical_and(trial_idxs, hits)
            subset_misses = np.logical_and(trial_idxs, misses)

            subset_hits, subset_misses = match_booleans(subset_hits, 
                                                        subset_misses)

            balanced_hits[np.where(subset_hits)[0]] = True
            balanced_misses[np.where(subset_misses)[0]] = True

        assert sum(balanced_hits) == sum(balanced_misses)
        
        return balanced_hits, balanced_misses

        
    def tt_raveled(self, trial_type='all', trial_outcome='all', balance=False):

        trials_use = []
        
        for _, session in self.sessions.items():
            
            if balance and trial_type=='test':
                
                hits = self.tt_idxs(session, trial_type, 'hit')
                misses = self.tt_idxs(session, trial_type, 'miss')
                balanced_hits, balanced_misses = self.balanced_hitmiss(session, 
                                                                       hits,
                                                                       misses)
                if trial_outcome == 'hit':
                    trials_use.append(balanced_hits)
                elif trial_outcome == 'miss':
                    trials_use.append(balanced_misses)
                    
            else:
                trials_use.append(self.tt_idxs(session, 
                                               trial_type,
                                               trial_outcome))
                
        return np.concatenate(trials_use)


    @staticmethod
    def match_booleans(arr1, arr2):
    
        ''' match two boolean arrays to have the same number of Trues 
            without moving position of existing Trues '''
        
        s1 = sum(arr1)
        s2 = sum(arr2)
        
        if s1 == s2:
            pass
        elif s1 > s2:
            arr1[np.random.choice(np.where(arr1)[0], s1-s2, replace=False)] = False
        elif s2 > s1:
            arr2[np.random.choice(np.where(arr2)[0], s2-s1, replace=False)] = False
            
        assert sum(arr1) == sum(arr2), f'{sum(arr1)} {sum(arr2)}; {s1} {s2}'
        
        return arr1, arr2


    def average_trace_plotter(self, df_plot, tt, manual=True):
        
        ''' Responsible for actual plotting. 
            Arguments:
            df_plot -- pandas dataframe 

            '''
        color_tt = {'hit': 'green', 'miss': 'grey', 'fp': 'magenta', 
                    'cr': 'brown', 'ur_hit': '#7b85d4', 'ar_miss': '#e9d043',
                    'spont_rew': 'darkorange'}

        self.n_plots+=1
        if not manual:
            # sns.lineplot(data=df_plot, x='timepoint', 
                         # y='dff', linewidth=3, #color=WES_COLS[self.n_plots], 
                         # label=tt, ci=95) 

            sns.lineplot(data=df_plot[df_plot['timepoint'] <= 0], x='timepoint', 
                         y='dff', linewidth=3, label=tt, ci=95) 

            sns.lineplot(data=df_plot[df_plot['timepoint'] >= 0.7], x='timepoint', 
                         y='dff', linewidth=3, label=None, ci=95)

        else:
            # Manual control over plots while playing around
            timepoints = np.unique(np.array(df_plot['timepoint']))
            num_timepoints = len(self.times_use)
            
            data = np.array(df_plot['dff']) 
            num_datapoints = len(data)
            data = data.reshape(int(num_datapoints/num_timepoints), num_timepoints)

            # Removes that one trials with nans at the end (I believe resulting from the
            # aosisNan function
            # count = 0
            # for i, d in enumerate(data):
                # if sum(np.isnan(d)) > 7:
                    # count+=1
                    # data = np.delete(data, i, axis=0)

            self.data = data

            mean_data = np.mean(data, 0)
            sem = np.std(data, 0) / math.sqrt(data.shape[0])
            
            plt.plot(timepoints, mean_data, color=color_tt[tt], label=tt)
            plt.fill_between(timepoints, mean_data-sem, mean_data+sem, alpha=0.1, color=color_tt[tt])


    def plotting_df(self, stacked_trials, stacked_prereward=None, outcomes=['hit', 'miss'],
                    stim_type='test', balance=False, show_plot=True, do_rob_mean=False, label=None):

        ''' Builds a pandas dataframe for an individual trace to be plotted / analysed
            This allows to call sns.lineplot on the dataframe for bootstrapped error bars and
            mean traces. Dataframe includes all trials.
        
            '''

        
        tt_mapper = {

            'hit': stim_type,
            'miss': stim_type,
            'ur_hit': stim_type,
            'ar_miss': stim_type,
            'spont_rew': None,
            'cr': 'nogo',
            'fp': 'nogo'

        }

        df_plots = []

        for outcome in outcomes:
            
            if outcome=='spont_rew':
                assert stacked_prereward is not None
                dff = stacked_prereward
            else:
                trials_use = self.tt_raveled(tt_mapper[outcome], outcome, 
                                             balance=balance)

                dff = stacked_trials[trials_use, :]

            if do_rob_mean:
                # Mean within sessions and then across sessions, rather than across trials (grand mean)
                # Ignore this chunk if doing the classical mean across all trials (overall mean)

                # Still need to do spont_rew
                if outcome in ['fp', 'cr', 'spont_rew']:
                    ps = 0
                elif stim_type == 'easy':
                    ps = 2
                else:
                    ps = 1

                rob_mean = []
                running_n = 0

                for session in self.sessions.values():
                    
                    if outcome == 'spont_rew':
                        n_trial_in_session = session.pre_rew_trials.shape[1]
                    else:
                        n_trial_in_session = sum(np.logical_and(session.photostim==ps, session.outcome==outcome))
                    boi = np.mean(dff[running_n:n_trial_in_session+running_n, :], 0)
                    rob_mean.append(boi)
                    
                    running_n += n_trial_in_session
                    
                dff = np.array(rob_mean)
            
            d = {name: np.array([]) for name in ['dff', 'timepoint']}  
            d['dff'] = dff.ravel() 
            d['timepoint'] = np.tile(self.times_use, dff.shape[0])

            df_plot = pd.DataFrame(d) 
            
            if show_plot:
                self.average_trace_plotter(df_plot, outcome, manual=False)
            df_plots.append(df_plot)

        return df_plots


    def s1s2_plot(self, ylims, grid, stim_type='test', balance=False):

        ''' Master plotting function to plot all trial types for s1 and s2 
            in neighbouring plots.

            Arguments -- 
            ylims - y axis limit of both plots (tuple / list of lenght 2)
            grid - Gridspec object upon which to plot (TODO: make this optional)
            stim_type - The go trial stimulus type to plot ['easy', 'test']
            balance - Balance the number of each trial type in the plot (bool)

            '''

        # global plotting params
        plt.style.use('ggplot')
        # gg plot colors are nice but want seaborn style
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        sns.set_style('white')

        # PREREWRAD S2 DIFFERNET?
        plt.subplots_adjust(wspace=0.4,hspace=0.4)
        
        gs = gridspec.GridSpecFromSubplotSpec(1, 2, grid, width_ratios=[1, 1]) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        
        # Which trial outcomes to plot
        outcomes = ['hit', 'miss', 'ur_hit', 'ar_miss', 'spont_rew', 'cr', 'fp']
        outcomes = ['hit', 'miss', 'spont_rew', 'cr', 'fp']

        for idx, region in enumerate(['s1', 's2']):
            plt.subplot(gs[idx])
            d = self.plotting_df(*self.trace_dict[region], outcomes=outcomes, 
                                 stim_type=stim_type, balance=balance)
            
            plt.ylim(ylims)
            plt.title('Average {} response'.format(region))
            plt.axhline(0)
            plt.xlabel('Time (s)')
            
            plt.ylabel(self.flu_flavour)
            if idx == 0:
                plt.legend(fontsize=14)
            else:
                plt.legend().set_visible(False)


class SingleCells(AverageTraces):

    def __init__(self, flu_flavour='dff'):

        ''' This class is used to generate average trace plots split by responder / non-responder etc.
            very messy and needs rewriting'''

        super().__init__(flu_flavour)

        del_idx = []
        for idx, session in self.sessions.items():
            if session.behaviour_trials.shape[0] < 200:
                del_idx.append(idx)

        for d in del_idx:
            del self.sessions[d]

    @property
    def flat_trace(self):
        d = {name: np.array([]) for name in ['dff', 'timepoint']}  
        d['dff'] = np.zeros((len(self.times_use), 10)).ravel() * np.nan
        d['timepoint'] = np.tile(self.times_use, 10)
        return pd.DataFrame(d)


    def build_trace_dict(self, region='s1', stim_type='test'):

        trace_dict = {}

        # Will fill the trace_dict with three columns: ['Targets', 'Background', 'Responders']
        # containing keys of trial_types, containing tupes? of s1 and s2
        # This is confusing but the returned dictionary is very easy to work with

        trial_types = ['hit', 'miss', 'fp', 'cr', 'spont_rew']

        #############

        col = 'Targets'
        col_dict = {}

        targets = self.session_stacker(cells_include='s1', prereward=False, mask='targets')

        for tt in trial_types:
            if tt not in ['hit', 'miss'] or region == 's2':
                col_dict[tt] = self.flat_trace
            else:
                col_dict[tt] = self.plotting_df(targets, outcomes=[tt], stim_type=stim_type, 
                                                show_plot=False, label='Targets')[0]

        trace_dict[col] = col_dict

        ##############

        col = 'Background'
        col_dict = {}

        background = self.session_stacker(cells_include=region, prereward=False, mask='followers')
        background_pre = self.session_stacker(cells_include=region, prereward=True, mask='followers')

        for tt in trial_types:
            if tt != 'spont_rew':
                col_dict[tt] = self.plotting_df(background, outcomes=[tt], 
                                                stim_type=stim_type, show_plot=False)[0]
            else:
                col_dict[tt] = self.plotting_df(None, background_pre, outcomes=[tt],
                                                stim_type=stim_type, show_plot=False)[0]

        trace_dict[col] = col_dict

        ###############

        col = 'Responders'
        col_dict = {}

        for tt in trial_types:

            if tt == 'spont_rew':

                responders_sessions = [self.get_sig_pass(session, 0.1, prereward=True, cells=region) 
                                                  for session in self.sessions.values()]

                responders = self.session_stacker(cells_include=responders_sessions, prereward=True, mask='followers')
                col_dict[tt] = self.plotting_df(None, responders, stim_type=stim_type, outcomes=[tt], show_plot=False)[0]
                continue

            elif tt in ['hit', 'miss']:

                responders_sessions = [self.get_sig_pass(session, 0.1, subset=[20,30,40,50], cells=region) 
                                      for session in self.sessions.values()]

            elif tt in ['fp', 'cr']:

                responders_sessions = [self.get_sig_pass(session, 0.1, subset=[0], cells=region) 
                                      for session in self.sessions.values()]
            

            responders = self.session_stacker(cells_include=responders_sessions, prereward=False, mask='followers')
            col_dict[tt] = self.plotting_df(responders, outcomes=[tt], stim_type=stim_type, show_plot=False)[0]
                

        trace_dict[col] = col_dict

        ####################

        return trace_dict

    @property
    def s1s2_dict(self):
        
        s1s2_dict = {}
            
        self.stim_type = 'test'

        s1s2_dict['s1'] = self.build_trace_dict(region='s1', stim_type=self.stim_type)
        s1s2_dict['s2'] = self.build_trace_dict(region='s2', stim_type=self.stim_type)

        return s1s2_dict

    def responder_target_plot(self, stim_type='test'):


        plt.figure(figsize=(10,5))
            
        plt.subplot(1,2, idx+1)
        plt.title(outcome)
        plt.ylim(-0.05, 0.2)

        # Plot the targets
        targets = self.session_stacker(cells_include='s1', prereward=False, mask='targets')
        self.plotting_df(targets, outcomes=[outcome], show_plot=True, label='Targets')

        for region in ['s1', 's2']:

            responders = [self.get_sig_pass(session, 0.1, subset=[20,30,40,50], cells=region) 
                          for session in self.sessions.values()]

            followers = self.session_stacker(cells_include=responders, prereward=False, mask='followers')
            # followers = self.session_stacker(cells_include=region, prereward=False, mask='followers')
            self.plotting_df(followers, stim_type=stim_type, outcomes=[outcome], show_plot=True, label=f'Background Responders {region}')

            prereward = self.session_stacker(cells_include=region, prereward=True,
                                             sub_baseline=True)
                                                
            self.plotting_df(None, prereward, stim_type=stim_type, outcomes=['spont_rew'], label=f'Spontaneous Reward Responders {region}')

        if idx ==0:
            plt.gca().legend().set_visible(False)
        else:
            plt.legend(fontsize='medium')


    def get_sig_pass(self, session, fdr_rate, subset=[150], cells='all', prereward=False):
    
        
        if prereward:
            arr = session.pre_rew_trials
        else:
            if len(subset) == 1:
                trials_use = session.trial_subsets == subset[0]
            else:
                trial_bools = [session.trial_subsets==sub for sub in subset]
                trials_use = np.logical_or.reduce(trial_bools)
                
            arr = session.behaviour_trials[:, trials_use, :]

        pre_frames, post_frames = self.get_range(session, plot=False)

        pre_array = self.prepare_population(copy.deepcopy(arr), pre_frames)
        post_array = self.prepare_population(copy.deepcopy(arr), post_frames)

        p_vals = [stats.wilcoxon(pre, post)[1] for pre, post in zip(pre_array, post_array)]
        p_vals = np.array(p_vals)

        sig_cells, correct_pval, _, _ = multitest.multipletests(p_vals, alpha=fdr_rate, method='fdr_bh',
                                                                is_sorted=False, returnsorted=False)

        if cells == 'all':
            return sig_cells
        elif cells == 's1':
            return np.logical_and(sig_cells, session.s1_bool)
        elif cells == 's2':
            return np.logical_and(sig_cells, session.s2_bool)
            

    def prepare_population(self, arr, frames, mean=True):
    
        arr_frames = arr[:, :, range(*frames)] 
        if mean:
            return np.mean(arr_frames, 2)
        else:
            return np.reshape(arr_frames, (arr_frames.shape[0], arr_frames.shape[1] * arr_frames.shape[2]))


    def get_range(self, session, plot=False):

        if session.mouse == 'RL048' or session.mouse == 'J048':
            # Need to be same length to do wilcoxon
            # Rob does 500ms  so 3 ish frames
            n_frames = 3
            pre_range = [16-n_frames, 16]
            post_range = [30, 30+n_frames]
        else:
            # 3 frames above so 15 frames here
            n_frames = 15
            pre_range = [100-n_frames, 100]
            post_range = [160, 160+n_frames]

        if plot:
            plt.figure()
            plt.plot(np.mean(session.behaviour_trials, (0,1)))

            for pre, post in zip(post_range, pre_range):
                plt.axvline(pre, color='blue')
                plt.axvline(post, color='red')
        
        return pre_range, post_range


