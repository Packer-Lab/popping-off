# TO DO LISTS

### TODO JR
- [ ] Neuropil signal pre Get Josh to do this?
- [ ] Participation ratio
- [ ] Does largest SV correspond to any of the variances? 
- [ ] Fix the model and run on all the data
- [ ] 9th Jan
- [x] Add the other variances
- [x] Add a note to the table about whether there's an effect
- [x] More PCs? as many as the number of neurons?
- [x] Thick lines for the PCs. Generally improve graphics
- [x] Dropout repeated cross-folds
- [x] Same number of it and miss trials in the firing rate plot
- [x] Distribution of all neuron firing rates hit vs miss
- [x] Distribution of all neuron correlations etc
- [x] Make the plot matrix (some e.g. populations metrics wont be possible)
- [x] A flag for S1 and S2
- [x] The hit and miss eigenspectrum plots
- [x] Check how the churchlands measure variance
- [x] Make a function to print flags and sessions included etc
- [x] Does the variance predict propagation?
- [x] Distribution plots of different variance flavours
- [x] Classifier plot of different variance flavours
- [x] Discard licks 250ms 
- [x] Churchland 2010 natneuro (Do our results match?)
- [x] Log the covariates that are better fit by the logs
- [x] RERUN WITH NEW PCA Viola's PC plot -> trace of the first PC before hit and miss
- [x] Factor analysis
- [x] Merge multiple sessions for the logistic classifier
- [x] Fix markdown checklist
- [x] Make the IO plot to Saxey's recommendation 
- [x] Show the distributions of PC loadings before hit and before miss 
- [x] Cross-correlation: take the absolute value of each element of cov matrix 

### TODO ML
- [ ] Email Johannas about the oasis nan
- [ ] Do fun stuff with the PCs
- [ ] Put the deconvolved spike data through the pipeline
- [ ] Photostim period length

# Glossary 

### Neural activity matrix 
* *symbol*: $X$
* *size* ($n_{neurons}$ x $n_{times}$)
* *defined by:* neural recordings

#### Synonyms:
* The activity of 1 neuron $i$ is row $i$: $x_i(t)$
* Neural dynamics

### ------

### Covariance matrix
* *symbol*: $C$
* *size*: ($n_{neurons}$ x $n_{neurons}$)
* *defined by*: covariance of activity matrix $X$

#### Synonyms:
* pairwise covariance

### ------


### Principal directions
* *symbol*: $V$
* *size matrix*: ($n_{comps}$ x $n_{neurons}$)
* *defined by*: eigendecomposition $C = V L V^T$, where $L$ is the (diagonal) matrix with eigenvalues

#### Synonyms:
* Loading matrix
* principal axes
* Eigenvectors
* right singular vectors

### ------


### Eigenvalues of Covariance matrix
* *symbol*: $L$
* *size*: ($n_{comps}$, $n_{comps}$) = ($n_{neurons}$, $n_{neurons}$) (equal in case of full eigendecomposition)
* *defined by*: eigendecomposition $ = V L V^T$, where $V$ is the matrix of eigenvectors

#### Synonyms:
* eigenvalues $\lambda_k$ are on the diagonal 
* variance explained = eigenvalues / sum(eigenvalues) = $\frac{\lambda_k}{\sum_k \lambda_k}$

### ------


### Principal Component (Dynamic Activity)
* *symbol*: $Z$
* *size matrix*: (n_comps x n_times)
* *defined by*: $Z = V \cdot X$ (Principal directions _dot_ Neural activity)

#### Synonyms:
* The activity of one PC $k$ is row $k$: $z_k(t)$
* Neural activity projected onto Principal axes
* Data projected on Principal axes
* Principal components
* PC scores
* Latent activity
* Latent components
* left singular vector _dot_ (diagonal) singular value matrix

### ------

### Variances
* variance_pop_mean: take the population mean across cells -> [time]. What is the variance of this vector?
* variance_cell_rates: take the mean across time for all cells ->  [n_cells]. What is the variance of this vector? 
* mean_cell_variance: take the variance of each cell through time -> [n_cells]. What is the mean of all the cell variances?


### References:
* https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
* https://stats.stackexchange.com/questions/311908/what-is-pca-components-in-sk-learn
* https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
