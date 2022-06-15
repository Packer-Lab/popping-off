# Photostimulation Of Population Obviously Fuels Firing

Code for analysis of all-optical neural data obtained in Packer lab, Oxford. 
This is the data of the following preprint:

https://www.biorxiv.org/content/10.1101/2021.12.28.474343v1


## Instructions for installation repo:

- Clone repository
- To install the correct Python packages, please build a conda environment from the `pope.yml` file.
- Additionally; run `pip install google-api-python-client` and `pip install google-auth-oauthlib`
- Add a profile to `data_paths.json`, with links to your local paths. (`base_path` should be the path that the `.pkl` data is in. `Vape_path` should be the directory the Vape repository is in. The other two are not needed for data analysis (only for pre-processing).
- Install pop-off by going to /popping-off/popoff/ and running `python setup.py develop`


# Additional detail (not necessary for getting started)

## Data:
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

This repo _attempts_ to follow the directory structure as recommmended by: https://drivendata.github.io/cookiecutter-data-science/ . Most of the code is wrapped in objects (functions and classes), which are called in notebooks to plot results. In summary, there are four main folders:
- figures (saved figures (preferably pdf or svg))
- notebooks (Jupyter notebooks that run the functions)
- popoff (contains all relevant modules for the notebooks)
- scripts (code that is not relevant for notebooks, but is used in other stages of the project (e.g. data pre-processing)). 

## Additional info:

One requires the repo VAPE for data pre-processing. Furthermore, some routines in `scripts/Session.py` were taken from VAPE. VAPE can be cloned here: https://github.com/neuromantic99/Vape

-------------

To use OASIS for spike deconvolution, one needs to install their package. To do so, clone this repo: 
https://github.com/j-friedrich/OASIS
and follow their python installation instructions.

------------

To use demixing PCA, one must install their package. To do so, clone this repo: https://github.com/machenslab/dPCA/tree/master/python/
and follow their python installation instructions..

