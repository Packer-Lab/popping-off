# TO DO LISTS

### TODO JR
- [ ] Discard licks 250ms 
- [ ] Churchland 2010 natneuro (Do our results match?)
- [ ] Log the covariates that are better fit by the logs
- [ ] RERUN WITH NEW PCA Viola's PC plot -> trace of the first PC before hit and miss
- [ ] Factor analysis
- [ ] Merge multiple sessions for the logistic classifier
- [ ] Please plot hit trials traces split by num cells
- [ ] Fix the model and run on all the data
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


### Principal Ccomponent (Dynamic Activity)
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



### References:
* https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
* https://stats.stackexchange.com/questions/311908/what-is-pca-components-in-sk-learn
* https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
