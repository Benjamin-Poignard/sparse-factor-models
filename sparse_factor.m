function [Lambda,gamma_opt,Psi] = sparse_factor(X,m,loss,gamma,method,K,Lambda_init,Psi_init)

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization (a_scad = 3.7, b_mcp = 3.5):
%          see lambda_penalized.m to modify a_scad and b_mcp
%          - K (optional input): number of folds for cross-validation; K
%          must be larger strictly than 2
%          - Lambda_init (optional input): inital parameter value for the
%          factor loading matrix
%          - Psi_init (optional input): inital parameter value for the
%          variance-covariance matrix (diagonal) of the idiosyncratic
%          errors, jointly obtained with Lambda_init
% Outputs:
%          - Lambda: sparse factor loading matrix
%          - gamma_opt: optimal tuning parameter selected by the K-fold
%          cross-validation procedure
%          - Psi: variance-covariance matrix (diagonal) of the
%          idiosyncratic errors

% if no cross-validation number is specified, then K = 5 by default
if nargin < 6
    K = 5;
end
% if no first step estimator for Lambda and Psi are provided, then the
% initial values run the following function to get an initial point
if nargin < 7
    [Lambda_init,Psi_init] = non_penalized_factor(X,m,loss);
end

% Initialization
[Lambda_rotated,~] = penalized_factor_Qstep(Lambda_init,m,gamma,method);

% Iteration to obtain (Lambda,Psi)
[Lambda,gamma_opt,Psi] = cv_sfm(X,m,Lambda_rotated,Psi_init,loss,gamma,method,K);

if max(abs(vec(Lambda)))>20
    Lambda = Lambda_init; Psi = Psi_init;
end

switch loss
    case 'Gaussian'
        switch method
            case 'scad'
                fprintf(1,'Estimation with scad-penalized Gaussian loss completed \n')
            case 'mcp'
                fprintf(1,'Estimation with mcp-penalized Gaussian loss completed \n')
        end
    case 'LS'
        switch method
            case 'scad'
                fprintf(1,'Estimation with scad-penalized LS loss completed \n')
            case 'mcp'
                fprintf(1,'Estimation with mcp-penalized LS loss completed \n')
        end
end