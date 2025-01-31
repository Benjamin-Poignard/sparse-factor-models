# Sparse factor models of high-dimension

Matlab implementation of sparse factor models based on the paper:

*Sparse factor models of high-dimension* by Benjamin Poignard and Yoshikazu Terada.

Link: https://arxiv.org/abs/2307.05952

# Overview

The code in this replication includes:

- The different sparsity patterns for sparse factor loading matrix considered in the simulated experiments: the replicator should execute program *simulations.m*.
- The real data experiment for the MSCI and S&P 100 portfolios: the replicator should execute program *real_data_analysis.m*.
- The real data experiment for diffusion index data: the replicator should refer to *multilagforecast.m*.

# Data availability

The MSCI and S&P 100 data used to support the findings of this study are publicly available. The MSCI data were collected from the link: https://www.msci.com. The S&P 100 data were collected from the link https://finance.yahoo.com. The S&P 500 data were downloaded (license required) from the link: https://macrobond.com.

The full sample period of the MSCI data is: 12/31/1998 - 03/12/2018. The full sample period of the S&P 100 data is: 18/02/2010 - 01/23/2020. The S&P 100 indices contains the 94 assets considered in the paper: AbbVie Inc., Dow Inc., General Motors, Kraft Heinz, Kinder Morgan and PayPal Holdings are excluded from the original dataset. 

The raw data file for the MSCI indices is *MSCI.xls*. The raw data file for the S&P 100 indices is *SP100.xls* and the S&P 100 data used in the paper (excluding AbbVie Inc., Dow Inc., General Motors, Kraft Heinz, Kinder Morgan and PayPal Holdings) are stored in *data_SP.mat*. The replicator can access both MSCI and S&P 100 datasets.

The diffusion index data can be found in *lndata.mat*. The code *multilagforecast.m* is a direct extension of the paper of Bai and Liao (2016): https://econweb.rutgers.edu/yl1114/papers/factor3/factor3.html

# Software requirements

The Matlab code was run on a Mac-OS Apple M1 Ultra with 20 cores and 128 GB Memory. The version of the Matlab software on which the code was run is a follows: 9.12.0.1975300 (R2022a) Update 3.

The following toolboxes should be installed:

- Statistics and Machine Learning Toolbox, Version 12.3.
- Parallel Computing Toolbox, Version 7.6.

The Parallel Computing Toolbox is highly recommended to run the code to speed up the cross-validation procedure employed to select the optimal tuning parameter. 

# Description of the code

The main function to conduct the joint estimation of the factor model parameters $\Lambda^\ast$ and $\Psi^\ast$, while penalizing $\Lambda^\ast$ only, is *sparse_factor.m*.

**sparse_factor.m**:

Purpose of the function: jointly estimate $(\Lambda,\Psi)$ under the sparsity constraint for $\Lambda$ and the diagonal constraint for $\Psi$, for a given loss function (Gaussian or least squares) and a given penalty function (SCAD or MCP). The implementation is based on the LQ-algorithm described in the paper (see the section "Implementations"). A K-fold cross-validation procedure is employed to select the optimal tuning parameter $\gamma_n$ when the data are i.i.d.

<p align="center">
[Lambda,gamma_opt,Psi] = sparse_factor(X,m,loss,gamma,method,K,Lambda_init,Psi_init)
</p>

Inputs:
- X: n x p matrix of observations
- m: number of factors (a priori set by the user)
- loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
- gamma: tuning parameter (grid of candidates set by the user)
- method: SCAD or MCP penalization (a_scad = 3.7, b_mcp = 3.5)
- K (optional input): number of folds for cross-validation; K must be larger strictly than 2
- Lambda_init (optional input): inital parameter value for the factor loading matrix
- Psi_init (optional input): inital parameter value for the variance-covariance matrix (diagonal) of the idiosyncratic errors, jointly obtained with Lambda_init

Outputs:
- Lambda: estimated sparse loading matrix
- gamma_opt: optimal tuning parameter selected by the K-fold cross-validation procedure
- Psi: variance-covariance matrix (diagonal) of the idiosyncratic errors

**sparse_factor_TS.m**:

In the context of time dependent data, the function *sparse_factor_TS.m* should be used by the replicator as it employs a different procedure for selecting the optimal tuning parameter. This function is run in *real_data_analysis.m*. The purpose of the function is the same as *sparse_factor.m*.

<p align="center">
[Lambda,gamma_opt,Psi] = sparse_factor_TS(X,m,loss,gamma,method,Lambda_init,Psi_init)
</p>

Inputs:
Inputs:
- X: n x p matrix of observations
- m: number of factors (a priori set by the user)
- loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
- gamma: tuning parameter (grid of candidates set by the user)
- method: SCAD or MCP penalization (a_scad = 3.7, b_mcp = 3.5)
- Lambda_init (optional input): inital parameter value for the factor loading matrix
- Psi_init (optional input): inital parameter value for the variance-covariance matrix (diagonal) of the idiosyncratic errors, jointly obtained with Lambda_init

Outputs:
- Lambda: estimated sparse loading matrix
- gamma_opt: optimal tuning parameter selected by the out-of-sample cross-validation procedure
- Psi: variance-covariance matrix (diagonal) of the idiosyncratic errors

The different sparsity patterns in $\Lambda^\ast$ considered in the experiments based on simulated data, i.e., perfect simple structure, perfect simple structure with overlaps and non-sparse blocks, perfect simple structure with overlaps and sparse blocks, arbitrary sparse structure, are generated by the functions: *simulate_perfect_structure_overlap_full_block.m* for the two first cases; *simulate_perfect_structure_overlap.m* for the perfect simple structure with overlaps and sparse blocks case; *simulate_general_structure.m* for the arbitrary sparse structure. The case of an arbitrary sparse structure on $\Lambda^\ast$ with $\Psi^\ast$ sparse non-diagonal is also provided. The file *simulations.m* contains the sections describing how to generate these sparse $\Lambda^\ast$ matrices and how to estimate the factor model in the i.i.d. case and in the time-series case.

The code for estimating the sparse approximate factor model of Bai and Liao (2016) directly builds upon the Matlab code of Professor Liao: https://econweb.rutgers.edu/yl1114/papers/factor3/factor3.html

The code for estimating the DCC model using the composited likelihood method is provided in the replication package: the replicator should refer to *dcc_mvgarch_for.m*. To be precise, both full likelihood and composite-likelihood methods are implemented in the second-step objective function. The latter method is based on contiguous overlapping pairs, which builds upon C. Pakel, N. Shephard, K. Sheppard and R.F. Engle (2021) and should be used when the dimension is large (i.e., larger than 200, 300, 400). The DCC-GARCH code builds upon the MFE toolbox of K. Sheppard, https://www.kevinsheppard.com/code/matlab/mfe-toolbox/

The functions *GIS.m* and *covMarket.m* for estimating the geometric-inverse shrinkage estimator (nonlinear shrinkage based on the symmetrized Kullback-Leibler loss) and the estimator shrunk towards a one-factor market, respectively, build upon the code of O.Ledoit: https://github.com/oledoit/covShrinkage

# On the tuning parameter $\gamma_n$

The selection of an optimal tuning parameter for SCAD/MCP relies on a K-fold cross-validation (K=5 by default) with parallelization, when the data are i.i.d.: in that case, *sparse_factor.m* should be employed. If the data are time series data, *sparse_factor_TS.m* should be considered: it employs an out-of-sample approach for cross-validation, where 75% of the data are used for training and the remaining 25% is used as the test sample. For both cases, the cross-validation score function is minimized over the grid of $\gamma_n$ candidates defined as $c\sqrt{\log(p)/n}$, where $c$ a user-specified grid of values. The computation time will also depend on the size of the set of values c selected for cross-validation to choose the optimal $\gamma_n$. To save run-time computation, the user may set a smaller grid. 
