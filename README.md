# Photostimulation Of Population Obviously Fuels Firing

Code for rotation in Packer lab.

## Instructions:

- Clone repository
- To install the correct Python packages, please build a conda environment from the `pop_off_env.yml` file.
- Change photo-stim/data_paths.json to your local paths
- Please set the data path configuration file to 'assume unchanged' in get settings, such its changes are not suggested as commit. Use: `git update-index --assume-unchanged photo-stim/data_paths.json` 
- To build sessions.pkl files run 'python Session.py' from command line and enter flu_flavour through cli. A new .pkl file will be built for each flu_flavour
- Each .pkl contains a dictionary of SessionLite objects.

## SessionLite Attributes
- behaviour_trials (float64): 3d array of imaging data as defined by flu_flavour [n_cells x n_trials x n_frames].
- outcome (str): what was the behavioural response to the trial?
- decision (bool): did the animal lick or not?
- photostim (int): 0 = no stim; 1 = test trial; 2 = easy trial.
- trial_subsets (int): how many cells were stimulated on each trial?
- s1_bool (bool): is the cell in s1?
- s2_bool (bool): is the cell in s2?

## info folders:

photo-stim: Jimmy's data, vape preprocessing, demixing PCA, decoding analysis

subspaces: Sarah's data, preprocessing, subspace analysis

miscellaneous: replicate Marshel ea 2019, PCA stuff

## Additional info:

To use code in the photo-stim folder, one requires the data pre-processing repo VAPE. Furthermore some routines in `photo-stim/Session.py` were taken from VAPE. VAPE can be cloned here: https://github.com/neuromantic99/Vape

-------------

To use OASIS for spike deconvolution, you must install their package. To do so, clone this repo: 
https://github.com/j-friedrich/OASIS
and follow their python installation instructions.

------------

To use demixing PCA, you must install their package. To do so, clone this repo: https://github.com/machenslab/dPCA/tree/master/python/
and follow their python installation instructions. 

