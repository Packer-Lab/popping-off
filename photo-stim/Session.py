## general imports (also for subsequent analysis notebooks)
import sys
import os
path_to_vape = os.path.expanduser('~/repos/Vape')
sys.path.append(path_to_vape)
sys.path.append(os.path.join(path_to_vape, 'jupyter'))
sys.path.append(os.path.join(path_to_vape, 'utils'))

oasis_path = os.path.expanduser('~/Documents/code/OASIS')
sys.path.append(oasis_path)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import utils_funcs as utils
import run_functions as rf
from subsets_analysis import Subsets
import pickle
import sklearn.decomposition
from cycler import cycler
from oasis.functions import deconvolve
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


def build_flu_array_single(run, prereward=False, pre_frames=30, post_frames=80, fs=30):
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

    flu = run.flu
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
        filter_threshold : int, default=10
            filter neurons with mean(abs(df/f)) > filter_threshold
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
        if use_spks:
            self.get_spks()

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
        with open(run_path, 'rb') as f:  # load data
            r = pickle.load(f)
            self.run = r
        ## Start preprocessing:
        self.flu = self.run.flu
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
        self.s2_bool = self.av_xpix > 512  # images were manually aligned to be half s1, half s2
        self.s1_bool = np.logical_not(self.s2_bool)

    def get_spks(self):
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
        self.art_gap_start = self.pre_frames - int(np.round(self.pre_gap_seconds * self.frequency))  # end of pre-stim frames
        self.final_pre_gap_tp = np.arange(self.art_gap_start)[-1]
        self.art_gap_stop = self.pre_frames + int(np.round(self.post_gap_seconds * self.frequency))  # start of post-stim frames
        self.filter_ps_array = np.concatenate((np.arange(self.art_gap_start),
                                  np.arange(self.art_gap_stop, self.pre_frames + self.post_frames))) # frame array of remaining frames
        self.filter_ps_time = (self.filter_ps_array - self.art_gap_start + 1) / self.frequency  # time array for remaining frames wrt stim onset

    def build_trials_multi(self, vverbose=1):
        """Construct 3D matrix of neural data (n_cells x n_trials x n_frames) for multi-plane data sets."""
        # array of fluoresence through behavioural trials (n_cells x n_trials x n_frames)
        # with e.g. the first trials spanning (galvo_ms[0] - pre_frames) : (galvo_ms[0] + post_frames)
        self.behaviour_trials = utils.build_flu_array(self.run, self.galvo_ms,
                                                      pre_frames=self.pre_frames, post_frames=self.post_frames)
#         self.behaviour_trials = self.behaviour_trials - np.nanmean(self.behaviour_trials, (1, 2))[:, np.newaxis, np.newaxis]
        if vverbose >= 2:
            print(f'Shape new array : {self.behaviour_trials.shape}')
        assert self.behaviour_trials.shape[1] == self.outcome.shape[0]

        self.pre_rew_trials = utils.build_flu_array(self.run, self.run.pre_reward, self.post_frames,
                                                     self.pre_frames, is_prereward=True) 
    
        nan_trials = np.any(np.isnan(self.pre_rew_trials), axis=(0,2))
        self.pre_rew_trials = self.pre_rew_trials[:, ~nan_trials, :]
                                                        
        assert np.sum(np.isnan(self.pre_rew_trials)) == 0

        if vverbose >= 2:
            print(self.behaviour_trials.shape, self.pre_rew_trials.shape)

    def build_trials_single(self, vverbose=1):
        """Construct 3D matrix of neural data (n_cells x n_trials x n_frames) for single-plane data sets."""
        # array of fluoresence through behavioural trials (n_cells x n_trials x n_frames)
        # with e.g. the first trials spanning (galvo_ms[0] - pre_frames) : (galvo_ms[0] + post_frames)
        self.behaviour_trials = build_flu_array_single(self.run, prereward=False,
                                                       pre_frames=self.pre_frames,
                                                       post_frames=self.post_frames, fs=30)
        if vverbose >= 2:
            print(f'Shape new array : {self.behaviour_trials.shape}')
        assert self.behaviour_trials.shape[1] == self.outcome.shape[0],\
               '{} {}'.format(self.behaviour_trials.shape[1], self.outcome.shape[0])

        self.pre_rew_trials = build_flu_array_single(self.run, prereward=True,
                                                     pre_frames=self.pre_frames,
                                                     post_frames=self.post_frames, fs=30)
    
        #self.pre_rew_trials = self.pre_rew_trials[:, 1:9, :]
        #assert np.sum(np.isnan(self.pre_rew_trials)) == 0

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

    def remove_nan_trials_inplace(self, vverbose=1):
        """Identify trials for which NaN values occur in the neural activity and remove those."""
        self.nonnan_trials = np.unique(np.where(~np.isnan(self.behaviour_trials))[1])
        self.behaviour_trials = self.behaviour_trials[:, self.nonnan_trials, :]
        self.photostim = self.photostim[self.nonnan_trials]
        self.decision = self.decision[self.nonnan_trials]
        self.trial_subsets = self.trial_subsets[self.nonnan_trials]
        self.outcome = self.outcome[self.nonnan_trials]
        self.autorewarded = self.autorewarded[self.nonnan_trials]
        self.unrewarded_hits = self.unrewarded_hits[self.nonnan_trials]
        self.n_trials = len(self.nonnan_trials)

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





class Session_lite(Session):
    def __init__(self):
        pass



