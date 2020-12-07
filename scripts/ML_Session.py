## General imports (also for subsequent analysis notebooks)
import sys, os
import json
import mrestimator as mre
import math
#from help_functions import fit_tau, oasis_deconvolve_Fc
from scipy.optimize import curve_fit
from IPython.core.debugger import Pdb
ipdb = Pdb()

sys.path.append('/home/loidolt/Projects/OASIS')

import scipy.signal
import oasis
from oasis import oasis_nan

# shut up mr. estimator
mre.ut._logstreamhandler.setLevel('ERROR')

# fix popoff import error
sys.path.append("/home/loidolt/Projects/popping-off/popoff")

from popoff import loadpaths

print(loadpaths.__file__)

user_paths_dict = loadpaths.loadpaths()

path_to_vape = user_paths_dict['vape_path']

sys.path.append(path_to_vape)
sys.path.append(os.path.join(path_to_vape, 'jupyter'))
sys.path.append(os.path.join(path_to_vape, 'utils'))

import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
import utils_funcs as utils 
import run_functions as rf
from subsets_analysis import Subsets
import pickle
import sklearn.decomposition
from cycler import cycler

# OASIS gives a useless warning
import warnings
warnings.simplefilter("ignore", UserWarning)

# Taken out to prevent need to clone OASIS to use SessionLite
#from oasis.functions import deconvolve

plt.rcParams['axes.prop_cycle'] = cycler(color=sns.color_palette('colorblind'))


def get_trial_frames_single(clock, start, pre_frames, post_frames, fs=30, paq_rate=20000):
    """Get frames that indicate trial start.

    Parameters:
    ----------------
        clock: ??
            ??
        start: ??
            ??
        pre_frames: int
            number of frames before stim to include in trial
        post_frames: int
            number of frames after stim to include in trial
        fs: int, default=30
            frame rate of imaging
        paq_rate: int, default=20000
            ??

    Returns:
    ---------------
        trial_frames: ??
            ??
    """
    # the frames immediately preceeding stim
    frame_idx = utils.closest_frame_before(clock, start)
    trial_frames = np.arange(frame_idx-pre_frames, frame_idx+post_frames)

    # is the trial outside of the frame clock
    is_beyond_clock = np.max(trial_frames) >= len(clock) or np.min(trial_frames) < 0

    if is_beyond_clock:
        return None

    frame_to_start = (start - clock[frame_idx]) / paq_rate  # time (s) from frame to trial_start
    frame_time_diff = np.diff(clock[trial_frames]) / paq_rate  # ifi (s)

    # did the function find the correct frame
    is_not_correct_frame = clock[frame_idx+1]  < start or clock[frame_idx] > start
    # the nearest frame to trial start was not during trial
    trial_not_running = frame_to_start > 1/fs
    #
    frames_not_consecutive = np.max(frame_time_diff) > 1/(fs-1)

    if trial_not_running or frames_not_consecutive:
        return None
    return trial_frames


def build_flu_array_single(run, use_spks=False, prereward=False, pre_frames=30, post_frames=80, fs=30):
    ''' Build an trial by trial fluoresence array of shape [n_cells x n_frames x n_trials].

    Parameters:
    ---------------
        run: run object
            run object containing data
        pre_frames: int, default=30
            number of frames before stim to include in trial
        post_frames: int, default=80
            number of frames after stim to include in trial
        fs: int, default=30
            frame rate of imaging

    Returns:
    ---------------
        flu_array: 3D np.array
            matrix of fluorescence data
    '''

    if not use_spks:
        flu = run.flu
    else:
        flu = run.spks

    # the frames that were actually imaged and the time (samples) that they occured
    clock = run.paqio_frames

    if prereward:
        # Time of prereward trial starts in paq samples
        trial_start = run.prereward_aligner.A_to_B(run.pre_reward)
    else:
        # Times of main behaviour trial start in paq samples
        trial_start = run.spiral_start
        # check that the number of trial starts detected by x galvo thresholding
        # matches the number of trials reported by pycontrol
        assert len(trial_start) == len(run.trial_start)

    for i, start in enumerate(trial_start):

        trial_frames = get_trial_frames_single(clock, start, pre_frames, post_frames)

        if trial_frames is None:
            flu_trial = np.full([flu.shape[0], pre_frames+post_frames], np.nan)
        else:
            flu_trial = flu[:, trial_frames]

        if i == 0:
            flu_array = flu_trial
        else:

            flu_array = np.dstack((flu_array, flu_trial))


    return np.swapaxes(flu_array,2,1)

def get_bad_frames(run, fs=30):    

    # Give it a three frame buffer pre-stim
    pre_frames = 3
    post_frames = math.ceil(1*fs)


    paqio_frames = utils.tseries_finder(run.num_frames, run.frame_clock)
    trial_start = utils.get_spiral_start(run.x_galvo_uncaging, run.paq_rate*6)

    bad_frames = []
    trial_starts = []
    for i, start in enumerate(trial_start):
        
        frames, start_idx = utils.get_trial_frames(paqio_frames, start, 
                                                   pre_frames, post_frames, 
                                                   paq_rate=run.paq_rate, 
                                                   fs=fs)
        trial_starts.append(start_idx)
        bad_frames.append(frames)


    flattened = [bf for bf in bad_frames if bf is not None]
    flattened = [item for sublist in flattened for item in sublist]


    return trial_starts, bad_frames

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B

def artifact_suppress(run, set_to=0, copy_run=False, interpolate=False, plot=False):

    if interpolate:
        set_to = np.nan

    if run.mouse_id == 'J048' or run.mouse_id == 'RL048':

        print('got a 5 here boi')

        fs = 5
        #ssf = uf.stim_start_frame_mat(run.trial_start, run.frames_ms)
        ssf = uf.stim_start_frame_mat(run.aligner.B_to_A(run.spiral_start), run.frames_ms)
        stim_start = np.min(ssf, 0)
        stim_start = stim_start[~np.isnan(stim_start)]

        pre_frames = 2 #math.ceil(1*fs)
        post_frames = 6 #math.ceil(1*fs)

        bad_frames = []

        for stim in stim_start:
            bad_frames.append(np.arange(stim-pre_frames, stim+post_frames))

        bad_frames = np.concatenate(bad_frames).astype('int')

        if not copy_run:
            run.flu[:, bad_frames] = set_to

            if interpolate:
                run.flu = fill_nan(run.flu)

            arr = uf.build_flu_array(run, run.trial_start)

        else:
            run_copy = copy.deepcopy(run)
            run_copy.flu[:, bad_frames] = set_to

            if interpolate:
                run_copy.flu = np.array([fill_nan(f) for f in run_copy.flu])

            #arr = uf.build_flu_array(run_copy, run_copy.trial_start)
            arr = uf.build_flu_array(run_copy, run_copy.aligner.B_to_A(run.spiral_start))

    else:
        print('oooh 30 here we go')
        fs = 30
        trial_starts, bad = get_bad_frames(run)
        bad = [b for b in bad if b is not None]
        bad = np.concatenate(bad)

        run.flu[:, bad] = set_to

        if interpolate:
            run.flu = np.array([fill_nan(f) for f in run.flu])

        arr  = build_flu_array_single(run)

    if plot:
        plt.plot(np.nanmean(arr, (0,1)), '.')
        #plt.ylim((-0.01, 0.15))

    return run.flu

def calc_spikeTriggeredAverage(dff, spks, theta, tau, fs, window=5*30):
    spks_idx = np.where(spks > theta)[0]
    n_spks = spks_idx.shape[0]
    triggered_dff = np.zeros((n_spks,window))
    for i in range(n_spks):
        if spks_idx[i]+window < dff.shape[0]:
            triggered_dff[i] = dff[spks_idx[i]:spks_idx[i]+window]
    triggered_average = np.nan_to_num(np.average(triggered_dff, axis=0), nan=0, 
            posinf=0, neginf=0)
    
    fit_func = lambda t,A,c: A*np.exp(-t/(fs*tau/1000)) + c
    try:
        popt, pcov = curve_fit(fit_func, np.arange(window), triggered_average)
    except:
        popt, pcov = (np.zeros(2), np.zeros((2,2)))
    return triggered_average, popt, np.diagonal(pcov)


def fit_tau(act, fs, numboot = 100, k_arr = None, method='sm'):
    print("act")
    print(act)
    print("k arr")
    print(k_arr) 
    try:
        print("I'm the new fit_tau!")
        coeff_res = mre.coefficients(act, method, k_arr, dt=(1/fs) * 1000, 
                numboot=100)
        print(coeff_res)
        print("coeff_res worked")
        tau_res = mre.fit(coeff_res, fitfunc='exponentialoffset', 
                quantiles=[.125, .25, .375, .5, .625, .75, .875], numboot=10)
        print(tau_res)
        #print(tau_res.quantiles)
        #print(numboot)
        #print(tau_res.tauquantiles)
        #print("tau_res worked")
        return (tau_res.tau, tau_res.tauquantiles, coeff_res.coefficients)
    except:
        print("coeff fit failed")
        return (0, np.zeros(7), np.zeros(k_arr.shape))        


class Session:
    """Class containing all info and data of 1 imaging session, as saved in a run.pkl file."""
    def __init__(self, mouse, run_number, pkl_path, remove_nan_trials=True,
                pre_seconds=4, post_seconds=6, pre_gap_seconds=0.2, post_gap_seconds=0.6,
                verbose=1, filter_threshold=10):
        """Initialize parameters and call all Class methods (except shuffle_labels()) to construct attributes.

        Parameters
        ----------
        mouse : str
            name mouse.
        run_number : int
            run number (i.e.session id of this mouse).
        pkl_path : str
            path to pickle file.
        remove_nan_trials : bool, default=True
            Remove trials with Nan Df/f data.
        pre_seconds : float, default=4
            time pre-stimulus to load into trial.
        post_seconds : float, default=6
            time post-stimulus to load into trial.
        pre_gap_seconds : float, default=0.2
            skip these seconds before PS stimulus onset
        post_gap_seconds : float, default=0.6
            skip these seconds after PS stimulus onset
        verbose : int, default=1
            verbosiness;  1: only important info (i.e. user end); 2: all info (debug end)
        filter_threshold : int, default=10 filter neurons with mean(abs(df/f)) > filter_threshold
        """
        self.mouse = mouse
        self.run_number = run_number
        self.signature = f'{self.mouse}_R{self.run_number}' # signature to use whenever
        self.pkl_path = pkl_path
        self.name = f'Mouse {mouse}, run {run_number}'
        self.run = None
        self.pre_seconds = pre_seconds  # pre stimulation
        self.post_seconds = post_seconds  # post stimulation
        self.pre_gap_seconds = pre_gap_seconds
        self.post_gap_seconds = post_gap_seconds
        self.verbose = verbose  # 1: only important info (i.e. user end); 2: all info (debug end)
        self.shuffled_s1s2_labels_indicator = False
        self.shuffled_trial_labels_indicator = False

        self.load_data(vverbose=self.verbose)  # load data from pkl path
        if not self.has_flu:
            return

        if mouse=='J048' or mouse=='RL048':
            self.frequency = 5
            self.build_time_gap_array()  # requires frequency def, construct time arrat with PS gap
            self.build_trials_multi(vverbose=self.verbose)  # create Df/f matrix (index per trial )
        else:
            self.frequency = 30
            self.build_time_gap_array()
            self.build_trials_single(vverbose=self.verbose)
        self.filter_neurons(vverbose=self.verbose, abs_threshold=filter_threshold)  #filter neurons based on mean abs values
        self.define_s1_s2()   # label s1 and s2 identity of neurons
        self.label_trials(vverbose=self.verbose)  # label trial outcomes
        self.remove_nan_trials_inplace(vverbose=self.verbose)  # remove nan traisl
        delattr(self.run, 'x_galvo_uncaging')   # to free memory

    def __str__(self):
        """Define name"""
        return self.name

    def __repr__(self):
        """Define representation"""
        return f'instance {self.name} of Session class'

    def load_data(self, vverbose=1):
        """Load run.pkl file that contains all data about a session."""
        if vverbose >= 1:
            print(f'Now loading mouse {self.mouse}, run {self.run_number}')
        run_path = os.path.join(self.pkl_path, self.mouse, f'run{self.run_number}.pkl')
        print(run_path)
        run_path = str(run_path)
        with open(run_path, 'rb') as f:  # load data
            r = pickle.load(f)
            self.run = r
        ## Start preprocessing:
        try:
            self.flu = self.run.flu
            self.has_flu = True
        except AttributeError:  # Not yet processed the fluoresence for this run
            self.has_flu = False
            return
        self.tstart_galvo = utils.threshold_detect(self.run.x_galvo_uncaging, 0)
        self.tstart_galvo = self.run.spiral_start
        self.trial_start = self.run.trial_start
        assert len(self.trial_start) == len(self.tstart_galvo)
        self.galvo_ms = self.run.aligner.B_to_A(self.tstart_galvo)
        if vverbose >= 1:
            print('microcontroller trial starts occur on average {} ms from galvo trial starts'
              .format(round(np.mean(self.trial_start - self.galvo_ms), 2)))

        ## Info about PS & outcome
        ### a different number of cells were stimulated on each trial
        ### need to create a Subsets object to get this info (future code refinement will
        ### include this info directly in the run object
        self.subsets = Subsets(self.run)
        self.trial_subsets = self.subsets.trial_subsets
        self.n_stim_arr = np.unique(self.trial_subsets)
        self.outcome = self.run.outcome
        self.outcome_arr = np.unique(self.outcome)

    def define_s1_s2(self, im_size=1024):  # define border (which is hard-defined at middle of image)
        """Define S1/S2 region borders and label all neurons accordingly.

        Parameters:
        -------------------
            im_size: int, default=1024
                size of imaging window in pixels. This is required because multi-plane data is saved by transposing planes +im_size pixels.
        """
        if self.run is None:
            self.load_data()
        self.n_cells = self.run.stat.shape[0]
        assert self.n_cells == self.behaviour_trials.shape[0]
        self.av_ypix = np.zeros(self.n_cells)
        self.av_xpix = np.zeros(self.n_cells)
        self.plane_number = np.zeros(self.n_cells)
        for neuron_id in range(self.n_cells):# get mean coords of pixels per cell
            self.av_xpix[neuron_id] = np.mean(self.run.stat[neuron_id]['xpix']) % im_size  # modulo 1024 because different planes are transposed by image size
            self.av_ypix[neuron_id] = np.mean(self.run.stat[neuron_id]['ypix']) % im_size
            # JR addition to deal with single plane data that has no iplane
            try:
                self.plane_number[neuron_id] = self.run.stat[neuron_id]['iplane']
            except KeyError:
                self.plane_number[neuron_id] = 0

        with open('/home/jrowland/Documents/code/Vape/s2_position.json') as json_file:
            s1s2_border_json = json.load(json_file)
    
        self.s1s2_border = s1s2_border_json[self.mouse][str(self.run_number)]
        
        self.s2_bool = self.av_xpix > self.s1s2_border
        self.s1_bool = np.logical_not(self.s2_bool)


    def run_oasis(self):
        """ Build spks array using Oasis deconvolution https://github.com/j-friedrich/OASIS """

        flu = self.flu.astype('float64')

        denoised_flu = np.empty_like(flu)
        deconved = np.empty_like(flu)

        for idx, cell in enumerate(tqdm(flu)):
            c, s, b, g, lam = deconvolve(cell, penalty=0)
            denoised_flu[idx, :] = c
            deconved[idx, :] = s

        self.run.flu = deconved  # Come up with a more elegant solution to this
        self.deconved = deconved
        self.denoised_flu = denoised_flu


    def build_time_gap_array(self):
        """Filter frames around PS due to laser artefact, build relevant new time/frame arrays etc."""
        self.pre_frames = int(np.round(self.pre_seconds * self.frequency))  # convert seconds to frame
        self.post_frames = int(np.round(self.post_seconds * self.frequency))
        print(self.pre_seconds)
        print(self.post_seconds)
        print("GAP FRAMES")
        print(self.pre_frames)
        print(self.post_frames)
        self.art_gap_start = self.pre_frames - int(np.round(self.pre_gap_seconds * self.frequency))  # end of pre-stim frames
        self.final_pre_gap_tp = np.arange(self.art_gap_start)[-1]
        self.art_gap_stop = self.pre_frames + int(np.round(self.post_gap_seconds * self.frequency))  # start of post-stim frames
        self.filter_ps_array = np.concatenate((np.arange(self.art_gap_start),
                                  np.arange(self.art_gap_stop, self.pre_frames + self.post_frames))) # frame array of remaining frames
        self.filter_ps_time = (self.filter_ps_array - self.art_gap_start + 1) / self.frequency  # time array for remaining frames wrt stim onset

    def build_trials_multi(self, use_spks=False, vverbose=1):
        """Construct 3D matrix of neural data (n_cells x n_trials x n_frames) for multi-plane data sets."""
        # array of fluoresence through behavioural trials (n_cells x n_trials x n_frames)
        # with e.g. the first trials spanning (galvo_ms[0] - pre_frames) : (galvo_ms[0] + post_frames)
        self.behaviour_trials = utils.build_flu_array(self.run, self.galvo_ms,
                                                      pre_frames=self.pre_frames, post_frames=self.post_frames, use_spks=use_spks)
#         self.behaviour_trials = self.behaviour_trials - np.nanmean(self.behaviour_trials, (1, 2))[:, np.newaxis, np.newaxis]
        if vverbose >= 2:
            print(f'Shape new array : {self.behaviour_trials.shape}')
        assert self.behaviour_trials.shape[1] == self.outcome.shape[0]

        self.pre_rew_trials = utils.build_flu_array(self.run, self.run.pre_reward, self.post_frames,
                                                     self.pre_frames, is_prereward=True, use_spks=use_spks)

        nan_trials = np.any(np.isnan(self.pre_rew_trials), axis=(0,2))
        self.pre_rew_trials = self.pre_rew_trials[:, ~nan_trials, :]

        assert np.sum(np.isnan(self.pre_rew_trials)) == 0

        if vverbose >= 2:
            print(self.behaviour_trials.shape, self.pre_rew_trials.shape)

    def build_trials_single(self, use_spks=False, vverbose=1):
        """Construct 3D matrix of neural data (n_cells x n_trials x n_frames) for single-plane data sets."""
        # array of fluoresence through behavioural trials (n_cells x n_trials x n_frames)
        # with e.g. the first trials spanning (galvo_ms[0] - pre_frames) : (galvo_ms[0] + post_frames)
        if not use_spks:
            self.behaviour_trials = build_flu_array_single(self.run, use_spks=use_spks, prereward=False,
                                                       pre_frames=self.pre_frames,
                                                       post_frames=self.post_frames, fs=30)
        elif use_spks:
            self.spks_behaviour_trials = build_flu_array_single(self.run, use_spks=use_spks, prereward=False,
                                                       pre_frames=self.pre_frames,
                                                       post_frames=self.post_frames, fs=30)
 
        if vverbose >= 2:
            print(f'Shape new array : {self.behaviour_trials.shape}')
        assert self.behaviour_trials.shape[1] == self.outcome.shape[0],\
               '{} {}'.format(self.behaviour_trials.shape[1], self.outcome.shape[0])

        if not use_spks:
            self.pre_rew_trials = build_flu_array_single(self.run, use_spks=use_spks, prereward=True,
                                                     pre_frames=self.pre_frames,
                                                     post_frames=self.post_frames, fs=30)
        elif use_spks:
            self.spks_pre_rew_trials = build_flu_array_single(self.run, use_spks=use_spks, prereward=True,
                                                     pre_frames=self.pre_frames,
                                                     post_frames=self.post_frames, fs=30)
            print("pre prewards included")


    def filter_neurons(self, vverbose=1, abs_threshold=10):
        """Filter neurons with surreal stats

        Parameters:
        -----------------
            abs_threshold, float, default=10
                upper bound on mean(abs(df/f))"""
        mean_abs_df = np.mean(np.abs(self.run.flu), 1)
        self.unfiltered_n_cells = self.run.flu.shape[0]
        self.filtered_neurons = np.where(mean_abs_df < abs_threshold)[0]
        self.behaviour_trials = self.behaviour_trials[self.filtered_neurons, :, :]
        self.pre_rew_trials = self.pre_rew_trials[self.filtered_neurons, :, :]
        self.run.flu = self.run.flu[self.filtered_neurons, :]
        self.run.flu_raw = self.run.flu_raw[self.filtered_neurons, :]
        self.run.stat = self.run.stat[self.filtered_neurons]
        if vverbose >= 1:
            if len(self.filtered_neurons < self.unfiltered_n_cells):
                print(f'{self.unfiltered_n_cells - len(self.filtered_neurons)} / {self.unfiltered_n_cells} cells filtered')
            else:
                print('No neurons filtered')

    def find_unrewarded_hits(self):
        """Find unrewarded hit trials that are defined as
        (registered as miss) & (lick before 1000ms) & (not an autorewarded trials )"""
        spiral_lick = self.run.spiral_licks  # [self.run.spiral_licks[x] for x in self.nonnan_trials]
        lick_trials = np.zeros(len(spiral_lick))
        for x in range(len(spiral_lick)):
            if len(spiral_lick[x] > 0):  # if licks present
                lick_trials[x] = (spiral_lick[x] < 1000)[0]  # True if first lick < 1000ms
            else:
                lick_trials[x] = False  # if no licks, always False
        mismatch = self.decision - lick_trials.astype('int')
        assert len(mismatch) == len(self.autorewarded)
        ind_unrew_hits = np.where(np.logical_and(mismatch == -1, self.autorewarded == False))[0]
        self.unrewarded_hits = np.zeros_like(self.decision, dtype='bool')
        self.unrewarded_hits[ind_unrew_hits] = True

    def label_trials(self, vverbose=1):
        """Construct the trial labels (PS & lick), and occurence table."""
        self.decision = np.logical_or(self.outcome == 'hit', self.outcome == 'fp').astype('int')
        self.photostim = np.ones_like(self.trial_subsets)  # ones = 5-50
        self.photostim[self.trial_subsets == 0] = 0
        self.photostim[self.trial_subsets == 150] = 2
        self.photostim_occ = {x: np.sum(self.photostim == x) for x in list(np.unique(self.photostim))}
        if vverbose >= 1:
            print(f'photo stim occurences: {self.photostim_occ}')
        self.autorewarded = np.array(rf.autoreward(self.run))  # array of bools whether an autoreward (after 3 consecutive misses) has occurred
        self.find_unrewarded_hits()  # adds unrewarded hits, requires self.decision to be defined

        assert self.photostim.shape == self.decision.shape
        self.n_unique_stims = len(np.unique(self.photostim))
        self.n_neurons = self.behaviour_trials.shape[0]
        self.n_times = self.behaviour_trials.shape[2]
        self.n_trials = self.behaviour_trials.shape[1]
        self.n_unique_dec = len(np.unique(self.decision))
        self.occ_table = np.zeros((self.n_unique_stims, 2))  # stim x dec
        for dec in range(self.n_unique_dec):
            for stim in range(self.n_unique_stims):
                self.occ_table[stim, dec] = np.sum(np.logical_and(self.decision == dec, self.photostim == stim))
        self.n_com_trials = np.max(self.occ_table).astype('int')
        if vverbose >= 1:
            print('Occurence table:')
            print(self.occ_table)

    def remove_nan_trials_inplace(self, vverbose=1, spks=False):
        """Identify trials for which NaN values occur in the neural activity and remove those."""
        if not spks:
            self.nonnan_trials = np.unique(np.where(~np.isnan(self.behaviour_trials))[1])
            self.behaviour_trials = self.behaviour_trials[:, self.nonnan_trials, :]
            self.photostim = self.photostim[self.nonnan_trials]
            self.decision = self.decision[self.nonnan_trials]
            self.trial_subsets = self.trial_subsets[self.nonnan_trials]
            self.outcome = self.outcome[self.nonnan_trials]
            self.autorewarded = self.autorewarded[self.nonnan_trials]
            self.unrewarded_hits = self.unrewarded_hits[self.nonnan_trials]

            self.is_target = self.is_target[:, self.nonnan_trials, :]
            self.n_trials = len(self.nonnan_trials)

        if spks:
            self.spks_behaviour_trials = self.spks_behaviour_trials[:, self.nonnan_trials, :]

        if vverbose >= 1:
            print(f'{len(self.nonnan_trials)} / {self.behaviour_trials.shape[1]} non nan trials identified')
            print(f'Numbers of PS cells: {np.unique(self.trial_subsets)}')  # exact amount of PS neurons
        if vverbose >= 2:
            print(f'Time array: {self.filter_ps_array}')  # time points outside of laser artefact

    def shuffle_trial_labels(self):
        """Shuffle all trial labels (in place)"""
        n_trials = len(self.photostim)
        random_inds = np.random.choice(a=n_trials, size=n_trials, replace=False)
        self.photostim = self.photostim[random_inds]
        self.decision = self.decision[random_inds]
        self.trial_subsets = self.trial_subsets[random_inds]
        self.outcome = self.outcome[random_inds]
        self.autorewarded = self.autorewarded[random_inds]
        self.unrewarded_hits = self.unrewarded_hits[random_inds]
        self.shuffled_trial_labels_indicator = True

    def shuffle_s1s2_labels(self):
        """Shuffle s1/s2 labels for all neurons"""
        n_s1 = np.sum(self.s1_bool)
        n_cells = len(self.s1_bool)
        random_inds = np.random.choice(a=n_cells, size=n_s1, replace=False)
        self.s1_bool = np.zeros(n_cells, dtype='bool')
        self.s1_bool[random_inds] = True
        self.s2_bool = np.logical_not(self.s1_bool)
        self.shuffled_s1s2_labels_indicator = True

    def get_targets(self):
        
        gt = rf.GetTargets(self.run)
        self.is_target = gt.is_target
        self.is_target = np.repeat(self.is_target[:, :, np.newaxis], self.n_times, axis=2)
        # Was a cell targeted on any trial?
        ever_targeted = np.any(self.is_target, axis=(1,2))
        # Check that all targets are in s1
        for target, s1 in zip(ever_targeted, self.s1_bool):
            if target:
                assert s1


class SessionLite(Session):
    ''' Does the same job as Session, using inheritence out of laziness to not combine 
        todo -- combine classes '''

    def __init__(self, mouse, run_number, pkl_path, flu_flavour, remove_nan_trials=True,
                pre_seconds=4, post_seconds=6, pre_gap_seconds=0.2, post_gap_seconds=0.6,
                verbose=1, filter_threshold=10):

        self.mouse = mouse
        self.run_number = run_number
        self.flu_flavour = flu_flavour
        self.pkl_path = pkl_path

        # Hard-coded for timescales change post-hoc
        self.pre_seconds = 61
        self.post_seconds = 61
        self.pre_gap_seconds = pre_gap_seconds
        self.post_gap_seconds = post_gap_seconds
        self.verbose = verbose
        self.filter_threshold = filter_threshold
        self.name = f'Mouse {mouse}, run {run_number}'

        self.load_data()

        if not self.has_flu:
            return

        if self.flu_flavour == 'flu':
            use_spks = False
        elif self.flu_flavour == 'spks':
            use_spks = True
        elif self.flu_flavour == 'denoised_flu':
            use_spks = False
            self.run.dff = copy.deepcopy(self.run.flu)
            assert self.run.flu.shape == self.run.denoised_flu.shape
            # Inelegant reassignment but saves reworking downstream functions
            self.run.flu = self.run.denoised_flu
        elif self.flu_flavour == 'flu_raw':
            use_spks = False
            self.run.dff = copy.deepcopy(self.run.flu)
            assert self.run.flu.shape == self.run.flu_raw.shape
            self.run.flu = self.run.flu_raw
        else:
            raise ValueError('self.flu_flavour not recognised')

        if mouse=='J048' or mouse=='RL048':
            self.frequency = 5
            self.build_time_gap_array()  # requires frequency def, construct time arrat with PS gap
            # create Df/f matrix (index per trial )
            self.flu = self.build_trials_multi(vverbose=self.verbose, use_spks=use_spks)
        else:
            self.frequency = 30
            self.build_time_gap_array()  # requires frequency def, construct time arrat with PS gap
            self.flu = self.build_trials_single(vverbose=self.verbose, use_spks=use_spks)
            print("FRAMES")
            print(self.pre_frames)
            print(self.post_frames)
            #print(null)
            print(self.run.flu.shape)
        if self.flu_flavour == 'dff':
            self.filter_neurons(vverbose=self.verbose)  #filter neurons based on mean abs values
        self.define_s1_s2()   # label s1 and s2 identity of neurons
        self.label_trials(vverbose=self.verbose)  # label trial outcomes
        self.get_targets()
        self.remove_nan_trials_inplace(vverbose=self.verbose)  # remove nan traisl

       
        # ML change starts
        print("hey ML")
        print(self.photostim)
        #self.flu_spont = np.copy(self.flu[:,10*60*30])
        self.mre_oasis()
        self.spont_spks = np.copy(self.run.spks[:,:10*60*30])
        print("self.spont_spks.shape")
        print(self.spont_spks.shape)
        self.build_trials_single(vverbose=self.verbose, use_spks=True)
        # Don't actually filter but run the function for backwards
        # compatibility
        self.filter_neurons(vverbose=self.verbose)  
        print(self.spks_behaviour_trials)
        self.remove_nan_trials_inplace(vverbose=self.verbose, spks=True)  # remove nan traisl
        print("nan trials removed")
        self.mre_spks()
        # ML change stops

        self.clean_obj()
        print("is this going through?")

    def filter_neurons(self, vverbose=1, abs_threshold_df=np.inf, abs_threshold_spks=np.inf):
        """Filter neurons with surreal stats
           Overwritten here by subclass to remove cells with 'surreal' flu and/or spks

        Parameters:
        -----------------
            abs_threshold, float, default=10
                upper bound on mean(abs(df/f)) or mean(spks)"""

        # run.flu is reassigned above, so make sure the same neurons are filtered
        # regardless of flu_flavour
        if self.flu_flavour == 'denoised_flu' or self.flu_flavour == 'flu_raw':
            mean_abs_df = np.mean(np.abs(self.run.dff), 1)
        else:
            mean_abs_df = np.mean(np.abs(self.run.flu), 1)

        mean_abs_spks = np.nanmean(np.abs(self.run.spks), 1)
        self.unfiltered_n_cells = self.run.flu.shape[0]
        self.filtered_neurons = np.where((mean_abs_df < abs_threshold_df) &
                                         (mean_abs_spks < abs_threshold_spks))[0]
        self.behaviour_trials = self.behaviour_trials[self.filtered_neurons, :, :]
        self.pre_rew_trials = self.pre_rew_trials[self.filtered_neurons, :, :]
        self.run.flu = self.run.flu[self.filtered_neurons, :]
        self.run.flu_raw = self.run.flu_raw[self.filtered_neurons, :]
        self.run.stat = self.run.stat[self.filtered_neurons]
        self.is_target = self.is_target[self.filtered_neurons, :, :]

        if vverbose >= 1:
            if len(self.filtered_neurons < self.unfiltered_n_cells):
                print(f'{self.unfiltered_n_cells - len(self.filtered_neurons)} / {self.unfiltered_n_cells} cells filtered')
            else:
                print('No neurons filtered')

    def clean_obj(self):

        attrs_remove = ['run', 'flu']

        for attr in attrs_remove:
            print(attr)
            delattr(self, attr)
            print("deleted")

    
    def mre_oasis(self):
        fs = 30
        n_cells = self.run.flu.shape[0]
        print("n_cells from self.run.flu")
        print(n_cells)

        self.g_raw = np.zeros(n_cells)
        self.tau_raw = np.zeros(n_cells)
        self.tau_raw_quantiles = np.zeros((n_cells,7))
        self.raw_coefficients = np.zeros((n_cells,299))
        self.spike_triggered_average = np.zeros((n_cells,5*30))
        self.PCOV = np.zeros((n_cells,2)) 
        self.POPT = np.zeros((n_cells,2)) 

        # calculate tau_df/F on 10 min of spont activity
        # we want to do this prior to masking the spont_rewards!
        # (because mr. estimator cannot handle nans)
        for i_cell in range(n_cells):
            dff_detrended = scipy.signal.detrend(self.run.flu[i_cell,:10*60*30],
                                                 type='constant', axis=-1)
            reshaped_dff = dff_detrended.reshape((10,60*30)) 
            fit_res = fit_tau(reshaped_dff, fs=fs, k_arr=np.arange(1,300),
                    numboot=100)
            #print(fit_res)
            self.tau_raw[i_cell] = fit_res[0]
            self.tau_raw_quantiles[i_cell] = fit_res[1]
            self.raw_coefficients[i_cell] = fit_res[2]
            self.g_raw[i_cell] = np.exp(-1 / (fs* self.tau_raw[i_cell]/1000)) 

        masked_dff = artifact_suppress(self.run, set_to = np.nan, plot=True, 
                                       interpolate=False, copy_run=True)
        masked_dff = masked_dff.astype('float64')
        self.run.spks = np.zeros_like(self.run.flu)
        for i_cell in range(n_cells):
            c,s = oasis_nan.oasisAR1(masked_dff[i_cell], self.g_raw[i_cell])
            self.run.spks[i_cell] = s
            sta, popt, pcov = calc_spikeTriggeredAverage(np.nan_to_num(dff_detrended), 
                                                   np.nan_to_num(s[:10*60*30]),
                                                   0.01, self.tau_raw[i_cell], 
                                                   30, window=5*30)
            self.spike_triggered_average[i_cell] = sta
            self.PCOV[i_cell] = pcov
            self.POPT[i_cell] = popt
        self.save_me = {}
        self.save_me['masked_dff'] = masked_dff
        self.save_me['spks'] = self.run.spks

    def mre_spks(self):
        fps = 30
        pre_frames = np.arange(0*fps,60*fps)
        post_frames = np.arange(61*fps, 120*fps)
        n_trials = self.spks_behaviour_trials.shape[1]
        n_frames = pre_frames.shape[0]

        spks = np.nan_to_num(self.spks_behaviour_trials)

        print("SPKS SHAPE")
        print(spks.shape)
        print(pre_frames)
        print(post_frames)

        spks_pre_all = np.sum(spks[:,:,pre_frames], axis=0)
        print(spks_pre_all[0].shape)
        tau_pre_all = np.zeros(n_trials)
        rate_pre_all = np.zeros(n_trials)
        #print(spks_pre_all[i_trial-window:i_trial+window].shape)
        for i_trial in range(n_trials):
            trial_spks = np.reshape(spks_pre_all[i_trial], (1,n_frames))
            tau_res = fit_tau(trial_spks, fps, k_arr=np.arange(1,5*fps))
            print("tau res")
            print(tau_res)
            tau_pre_all[i_trial] = tau_res[0]
            rate_pre_all[i_trial] = np.sum(trial_spks) / n_frames * fps
        
        spks_post_all = np.sum(spks[:,:,post_frames], axis=0)
        tau_post_all = np.zeros(n_trials)
        rate_post_all = np.zeros(n_trials)
        for i_trial in range(n_trials):
            tau_res = fit_tau(spks_post_all[i_trial], fps, k_arr=np.arange(1,5*fps))
            tau_post_all[i_trial] = tau_res[0]
            rate_post_all[i_trial] = np.sum(spks_post_all[i_trial]) / n_frames * fps

        # S1 only

        print("S1 bool shape")
        print(self.s1_bool.shape)
        print(spks[self.s1_bool][:,:,pre_frames].shape)
        spks_pre_S1 = np.sum(spks[self.s1_bool][:,:,pre_frames], axis=0)
        tau_pre_S1 = np.zeros(n_trials)
        rate_pre_S1 = np.zeros(n_trials)

        for i_trial in range(n_trials):
            tau_res = fit_tau(spks_pre_S1[i_trial], fps, k_arr=np.arange(1,5*fps))
            tau_pre_S1[i_trial] = tau_res[0]
            rate_pre_S1[i_trial] = np.sum(spks_pre_S1[i_trial]) / n_frames * fps

        spks_post_S1 = np.sum(spks[self.s1_bool][:,:,post_frames], axis=0)
        tau_post_S1 = np.zeros(n_trials)
        rate_post_S1 = np.zeros(n_trials)

        for i_trial in range(n_trials):
            tau_res = fit_tau(spks_post_S1[i_trial], fps, k_arr=np.arange(1,5*fps))
            tau_post_S1[i_trial] = tau_res[0]
            rate_post_S1[i_trial] = np.sum(spks_post_S1[i_trial]) / n_frames * fps

        # S2 only
        spks_pre_S2 = np.sum(spks[~self.s1_bool][:,:,pre_frames], axis=0)
        tau_pre_S2 = np.zeros(n_trials)
        rate_pre_S2 = np.zeros(n_trials)

        for i_trial in range(n_trials):
            tau_res = fit_tau(spks_pre_S2[i_trial], fps, k_arr=np.arange(1,5*fps))
            tau_pre_S2[i_trial] = tau_res[0]
            rate_pre_S2[i_trial] = np.sum(spks_pre_S2[i_trial]) / n_frames * fps

        spks_post_S2 = np.sum(spks[~self.s1_bool][:,:,post_frames], axis=0)
        tau_post_S2 = np.zeros(n_trials)
        rate_post_S2 = np.zeros(n_trials)

        for i_trial in range(n_trials):
            tau_res = fit_tau(spks_post_S2[i_trial], fps, k_arr=np.arange(1,5*fps))
            tau_post_S2[i_trial] = tau_res[0]
            rate_post_S2[i_trial] = np.sum(spks_post_S2[i_trial]) / n_frames * fps
       
        print(tau_pre_all)
        print(tau_post_all)
        self.tau_dict = {"all_pre":tau_pre_all, "all_post":tau_post_all,
                         "S1_pre":tau_pre_S1, "S1_post":tau_post_S1,
                         "S2_pre":tau_pre_S2, "S2_post":tau_post_S2}
        self.rate_dict = {"all_pre":rate_pre_all, "all_post":rate_post_all,
                         "S1_pre":rate_pre_S1, "S1_post":rate_post_S1,
                         "S2_pre":rate_pre_S2, "S2_post":rate_post_S2}


def only_numerics(seq):
    seq_type= type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

def load_files(save_dict, data_dict, folder_path, flu_flavour):
    total_ds = 0
    debug = False
    for mouse in data_dict.keys():
        if mouse != 'RL070' and debug:
            continue
        if mouse in ['RL048', 'J048']:
            continue
        for run_number in data_dict[mouse]:

            if run_number != 29 and debug:
                continue

            session = SessionLite(mouse, run_number, folder_path, 
                                  flu_flavour=flu_flavour, pre_gap_seconds=0,
                                  post_gap_seconds=0, post_seconds=8)

            if session.has_flu:
                print("session lite created")
                save_dict[total_ds] = session
                total_ds += 1
                print(f'succesfully loaded mouse {mouse}, run {run_number}')
            else:
                print(f'{mouse}, run {run_number} flu not yet processed')
                
    return save_dict, total_ds


if __name__ == '__main__':

    flu_flavour = input('Enter a flu_flavour, valid entries are: [dff, raw, denoised, spks]\n\n')

    if flu_flavour == 'dff':
        flu_flavour = 'flu'
    elif flu_flavour == 'denoised':
        flu_flavour = 'denoised_flu'
    elif flu_flavour == 'raw':
        flu_flavour = 'flu_raw'
    elif flu_flavour == 'spks':
        flu_flavour = 'spks'
    else:
        print('Invalid flu_flavour')
        time.sleep(2)
        raise ValueError


    pkl_path = user_paths_dict['pkl_path']  #'/home/jrowland/Documents/code/Vape/run_pkls'

    if not os.path.exists(pkl_path):
        raise FileNotFoundError('pkl_path directory not found, did you update data_path.json?')

    ## Load data
    sessions = {}

    all_mice = [x for x in os.listdir(pkl_path) if x[-4:] != '.pkl']

    run_dict = {m: list(np.unique([int(only_numerics(x))
               for x in os.listdir(pkl_path + f'/{m}')]))
               for m in all_mice}

    if 'J065' in run_dict.keys() and 14 in run_dict['J065']:
        run_dict['J065'].remove(14)

    sessions, total_ds = load_files(save_dict=sessions, data_dict=run_dict,
                                    folder_path=pkl_path, flu_flavour=flu_flavour)

    save_path = os.path.expanduser(f'{user_paths_dict["base_path"]}/sessions_lite_spks.pkl')

    with open(save_path, 'wb') as f:
        pickle.dump(sessions, f, protocol=4)

