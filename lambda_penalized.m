function [param_psi,param_l,param_q] = lambda_penalized(X,m,Lambda_first,Psi_first,loss,gamma,method)

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - Lambda_first: inital parameter value for the
%          factor loading matrix (satisfying IC5 condition)
%          - Psi_first: inital parameter value for the variance-covariance
%          matrix (diagonal) of the idiosyncratic errors, jointly obtained
%          with Lambda_first
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization
% Outputs:
%          - param_psi: estimated parameters of the variance-covariance
%          matrix (diagonal) of the idiosyncratic errors
%          - param_l: estimated parameters of the L matrix from the
%          LQ-decomposition
%          - param_q: estimated parameters of the Q matrix from the
%          LQ-decomposition satisfying QxQ' = I_m with I_m the m x m
%          identity matrix

max_iter = 10^8; grid = 100; p = size(X,2);

% The user may modify the following a_scad and b_mcp values
a_scad = 3.7; b_mcp = 3.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q_first = penalized_factor_Qstep_sampling(Lambda_first,m,grid,gamma,method,a_scad,b_mcp);

% L-step
[L_step,l_step] = penalized_factor_Lstep(cov(X),Lambda_first,Q_first,Psi_first,m,loss,gamma,method,a_scad,b_mcp);

% Q-step
Q_step = penalized_factor_Qstep(L_step,Q_first,m,gamma,method,a_scad,b_mcp);

% Psi-step
Psi_step = psi_estimation(cov(X),L_step*Q_step,Psi_first,loss);

param_psi_update = diag(Psi_step); param_update = [l_step;vec(Q_step)];

dim_l = m*(m+1)/2+(p-m)*m;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Iterate until convergence %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Tol = eps; count = 0;
while count < max_iter
    
    count = count+1;
    param_psi = param_psi_update; param = param_update;
    
    % L-step
    param_l = param(1:dim_l); L = [tril(dvech(param_l(1:m*(m+1)/2),m),0);reshape(param_l(m*(m+1)/2+1:end),p-m,m)];
    param_q = param(dim_l+1:end); Q = reshape(param_q,m,m);
    [L_step,l_step] = penalized_factor_Lstep(cov(X),L,Q,Psi_first,m,loss,gamma,method,a_scad,b_mcp);
    
    % Q-step
    Q_step = penalized_factor_Qstep(L_step,Q,m,gamma,method,a_scad,b_mcp);
    
    % Psi-step
    Psi_step = psi_estimation(cov(X),L_step*Q_step,diag(param_psi),loss);
    
    param_psi_update = diag(Psi_step); param_update = [l_step;vec(Q_step)];
    
    if (norm([param_psi_update;param_update] - [param_psi;param])/max([1,norm([param_psi_update;param_update]),norm([param_psi_update;param])]) <= Tol)
        break
    end
    
end
param_psi = param_psi_update; param_l = param_update(1:dim_l); param_q = param_update(dim_l+1:end);