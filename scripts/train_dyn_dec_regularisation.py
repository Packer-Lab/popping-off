## This script performs the regularisation sweep of the dynamic decoders. Results are saved.

import popoff
from Session import SessionLite, build_flu_array_single
from linear_model import PoolAcrossSessions, LinearModel, pca_session, LabelEncoder, largest_PC_trace, largest_PC_loading, do_pca
import numpy as np
import sys, os, pickle, copy
from tqdm import tqdm
import pop_off_functions as pof
import pop_off_plotting as pop
from datetime import datetime

## Parameters:
store_folder = '/home/tplas/repos/popping-off/notebooks/regularisation_optimisation_dyn_dec'  # where to store regularisation sweep results
reg_strength_array = np.logspace(-3, 7, 11)  # Set array of regularisation strength
## Define which decoders to train:
dict_tt_train = {'hit/cr': ['hit', 'cr'],
                 'hit/miss': ['hit', 'miss'],
                 'miss/cr': ['miss', 'cr'],
                 'spont/cr': ['spont', 'cr'],
                 'hit/cr 10 trials': ['hit', 'cr']}  # to run 10 trials; set hard_set_10_trials to True
subsample_timepoints = False  # if True, subsample time points for speed-up
add_pop_var = True  # if True, add pop var (to enable split by pop var). takes a few minutes to add.

## Load data in form of Jimmy's PAS object
print('Loading data\n', '-----------\n')
remove_targets = False
pas = PoolAcrossSessions(save_PCA=False, subsample_sessions=False,
                         remove_targets=remove_targets, remove_toosoon=True)
# print(pas.sessions)

## Create sessions object from PAS:
try:  # ensure sessions doesn't exist yet 
    sessions
    assert type(sessions) is dict
except NameError:
    pass

sessions = {}
int_keys_pas_sessions = pas.sessions.keys()
# print(int_keys_pas_sessions)
i_s = 0
for ses in pas.sessions.values():  # load into sessions dict (in case pas skips an int as key)
    ses.signature = f'{ses.mouse}_R{ses.run_number}'
    sessions[i_s] = ses
    i_s += 1
print(sessions)
assert len(sessions) == 11
pof.label_urh_arm(sessions=sessions)  # label arm and urh
   
print('------------------------------------')
print(f'{len(sessions)} sessions are loaded')
if add_pop_var:
    print('Now adding population variance metrics to all sessions')
print('------------------------------------')
tp_dict = pof.create_tp_dict(sessions=sessions)

## Add VCR measurements to session objects
if add_pop_var:
    pof.add_vcr_to_lm(lm_list=pas.linear_models)
    list_save_covs = ['variance_cell_rates_s1']  # to be passed to the training function
    print('-----\nPopulation variance added\n-----------')
else:
    list_save_covs = []

## Define time points for training
tp_dict['decoders'] = tp_dict['mutual']  # use all time points (resolution) that are shared between all sessions
tp_dict['decoders'] = tp_dict['decoders'][np.logical_and(tp_dict['decoders'] >- 2,  # time window wherein decoders are trainined
                                                         tp_dict['decoders'] <= 4)]
if subsample_timepoints:
    print('WARNING: SUBSAMPLING TIME POINTS')
    tp_dict['decoders'] = tp_dict['decoders'][::20]  # optional: subsample to speed up 

pre_stim_art_time = -0.07  # define ps artefact gap
post_stim_art_time = 0.35  # assuming 150-cells-targeted trials are excluded
time_array_plot = copy.deepcopy(tp_dict['decoders'])
time_array_full = copy.deepcopy(tp_dict['decoders'])
time_array_plot[np.logical_and(time_array_plot >= pre_stim_art_time, 
                               time_array_plot < post_stim_art_time)] = np.nan
time_array_full -= pre_stim_art_time
time_array_plot -= pre_stim_art_time

print('Training the following decoders: ', dict_tt_train.keys())

## Make empty dicts to save results:
all_lick_pred_split_tt, all_lick_pred_split_tt_nstim, all_lick_pred_split_tt_covar = {}, {} ,{}
all_ps_pred_split_tt, all_ps_pred_split_tt_nstim, all_ps_pred_split_tt_covar = {}, {}, {}

print('Regularisation strength: ', reg_strength_array)
print('List covariates: ', list_save_covs)
print('---------\nStart of training\n-----------')

## Train: 
## Compute results decoders (note: CV of regularisation is down below in the notebook)
for i_reg, reg_strength in enumerate(reg_strength_array):
    (all_lick_pred_split_tt[reg_strength], all_lick_pred_split_tt_nstim[reg_strength], 
     all_lick_pred_split_tt_covar[reg_strength], all_ps_pred_split_tt[reg_strength], 
     all_ps_pred_split_tt_nstim[reg_strength], 
     all_ps_pred_split_tt_covar[reg_strength]) = {}, {}, {}, {}, {}, {}
    for key, list_tt_train in dict_tt_train.items():
        print(f'Now training {key} decoder of regularisation iteration {i_reg + 1}/{len(reg_strength_array)}')

        (all_lick_pred_split_tt[reg_strength][key], all_lick_pred_split_tt_nstim[reg_strength][key], 
         all_lick_pred_split_tt_covar[reg_strength][key], all_ps_pred_split_tt[reg_strength][key], 
         all_ps_pred_split_tt_nstim[reg_strength][key], 
         all_ps_pred_split_tt_covar[reg_strength][key]) = pof.compute_prediction_time_array_average_per_mouse_split(sessions=sessions, 
                                                      time_array=tp_dict['decoders'],
                                                      projected_data=False, 
                                                      reg_type='l2', regularizer=reg_strength, 
                                                      average_fun=pof.class_av_mean_accuracy,
                                                      list_tt_training=list_tt_train,
                                                      concatenate_sessions_per_mouse=False,
                                                      hard_set_10_trials=(True if key == 'hit/cr 10 trials' else False),
                                                      list_save_covs=list_save_covs)

## Save:
dt = datetime.now()
timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)
with open(os.path.join(store_folder, timestamp + '__all_lick_pred_split_tt.pickle'), 'wb') as handle:
    pickle.dump(all_lick_pred_split_tt, handle)

with open(os.path.join(store_folder, timestamp + '__all_lick_pred_split_tt_nstim.pickle'), 'wb') as handle:
    pickle.dump(all_lick_pred_split_tt_nstim, handle)

with open(os.path.join(store_folder, timestamp + '__all_lick_pred_split_tt_covar.pickle'), 'wb') as handle:
    pickle.dump(all_lick_pred_split_tt_covar, handle)

with open(os.path.join(store_folder, timestamp + '__all_ps_pred_split_tt.pickle'), 'wb') as handle:
    pickle.dump(all_ps_pred_split_tt, handle)

with open(os.path.join(store_folder, timestamp + '__all_ps_pred_split_tt_nstim.pickle'), 'wb') as handle:
    pickle.dump(all_ps_pred_split_tt_nstim, handle)

with open(os.path.join(store_folder, timestamp + '__all_ps_pred_split_tt_covar.pickle'), 'wb') as handle:
    pickle.dump(all_ps_pred_split_tt_covar, handle)
