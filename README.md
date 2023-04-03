# Sparse factor models of high-dimension

Matlab implementation of sparse factor models based on the paper:

Sparse factor models of high-dimension, by Benjamin Poignard and Yoshikazu Terada.

Link:

# Overview

The code in this replication includes:

- The different sparsity patterns for sparse factor loading matrix considered in the simulated experiments: the replicator should execute program *simulations.m*.
- The real data experiment for the MSCI and S&P 100 portfolios: the replicator should execute program *real_data_analysis.m*.

# Data availability

The MSCI and S&P 100 data used to support the findings of this study are publicly available. The MSCI data were collected from the link: https://www.msci.com. The S&P 100 data were collected from the link https://finance.yahoo.com. The S&P 500 data were downloaded (with license) from the link: https://macrobond.com.

The raw data file for the MSCI indices is MSCI.xls. The raw data file for the S&P 100 indices is SP100.xls and can found under the . Both are provided in the replication package. 

The full sample period of the MSCI data is: 12/31/1998 - 03/12/2018. The full sample period of the S&P 100 data is: 18/02/2010 - 01/23/2020. The S&P 100 indices contains the 94 assets considered in the paper: AbbVie Inc., Dow Inc., General Motors, Kraft Heinz, Kinder Morgan and PayPal Holdings are excluded from the original dataset. 

# Software requirements

The Matlab code was run on a Mac-OS Apple M1 Ultra with 20 cores and 128 GB Memory. The version of the Matlab software on which the code was run is a follows: 9.12.0.1975300 (R2022a) Update 3.

The following toolboxes should be installed:

- Statistics and Machine Learning Toolbox, Version 12.3.
- Parallel Computing Toolbox, Version 7.6. Parallel Computing Toolbox is highly recommended to run the code to speed up the cross-validation procedure employed to select the optimal tuning parameter. All the run-time requirements displayed below are reported when the code is run with the Parallel Computing Toolbox.

# Description of the code

The main function to conduct the joint estimation of the factor model parameters $\Lambda$ and $\Psi$, while penalizing $\Lambda$ only, is *sparse_factor.m*.

**sparse_factor.m**:

Purpose of the function: jointly estimate $(\Lambda,\Psi)$ under the sparsity constraint for $\Lambda$ and the diagonal constraint for $\Psi$, for a given loss function (Gaussian or least squares) and a given penalty function (SCAD or MCP). The implementation is based on the LQ-algorithm described in the paper (see the section "Implementations"). A K-fold cross-validation procedure is employed to select the optimal tuning parameter $\gamma_n$ when the data are i.i.d. 

<p align="center">
[Lambda,gamma_opt,Psi] = sparse_factor(X,m,loss,gamma,method,K,Lambda_first,Psi_first)
</p>

Inputs:
- X: n x p matrix of observations
- m: number of factors (a priori set by the user)
- loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
- gamma: tuning parameter (grid of candidates set by the user)
- method: SCAD or MCP penalization (a_scad = 3.7, b_mcp = 3.5): see *lambda_penalized.m* to modify a_scad and b_mcp
- K (optional input): number of folds for cross-validation; K must be larger strictly than 2. K = 5 by default if not specified
- Lambda_first (optional input): inital parameter value for the factor loading matrix
- Psi_first (optional input): inital parameter value for the variance-covariance matrix (diagonal) of the idiosyncratic errors, jointly obtained with Lambda_first

Outputs:
- Lambda: sparse factor loading matrix
- gamma_opt: optimal tuning parameter select by the K-fold cross-validation procedure
- Psi: variance-covariance matrix (diagonal) of the idiosyncratic errors

In the context of time dependent data, the function *sparse_factor_TS.m* should be used by the replicator as it employs a different procedure for selecting the optimal tuning parameter.
<p align="center">
[Lambda,gamma_opt,Psi] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_first,Psi_first)
</p>

Inputs:
- X: n x p matrix of observations
- m: number of factors (a priori set by the user)
- loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
- gamma: tuning parameter (grid of candidates set by the user)
- method: SCAD or MCP penalization (a_scad = 3.7, b_mcp = 3.5): see *lambda_penalized.m* to modify a_scad and b_mcp
- Lambda_first (optional input): inital parameter value for the factor loading matrix
- Psi_first (optional input): inital parameter value for the variance-covariance matrix (diagonal) of the idiosyncratic errors, jointly obtained with Lambda_first

Outputs:
- Lambda: sparse factor loading matrix
- gamma_opt: optimal tuning parameter select by the K-fold cross-validation procedure
- Psi: variance-covariance matrix (diagonal) of the idiosyncratic errors

The four different sparsity patterns in $\Lambda$ considered in the experiments based on simulated data, i.e., perfect simple structure, perfect simple structure with overlaps and non-sparse blocks, perfect simple structure with overlaps and sparse blocks and arbitrary sparse structure, are generated by the functions: *simulate_perfect_structure_overlap_full_block.m* for the two first cases; *simulate_perfect_structure_overlap.m* for the perfect simple structure with overlaps and sparse blocks case; *simulate_general_structure.m* for the arbitrary sparse structure. The file *simulations.m* contains four main sections describing how to generate these sparse $\Lambda$ matrices and how to estimate the factor model (i.i.d. case).

# On the tuning parameter $\gamma_n$

