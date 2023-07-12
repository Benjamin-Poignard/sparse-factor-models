function [L,param_l] = penalized_factor_Lstep(S,L,Q,Psi,m,loss,gamma,method,a_scad,b_mcp)

% Inputs:
%          - S: sample variance-covariance matrix of the n x p observations
%          - L: L matrix from the LQ-decomposition
%          - Q: Q matrix from the LQ-decomposition satisfying QxQ' = I_m
%          with I_m the m x m identity matrix
%          - Psi: variance-covariance matrix (diagonal) of the
%          idiosyncratic errors
%          - m: number of factors (a priori set by the user)
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization
%          - a_scad: SCAD parameter
%          - b_mcp: MCP parameter
% Output:
%          - L: estimated L matrix from the LQ-decomposition
%          - param_l: vec(L) (vector of the parameters entering in L)

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 500000;
optimoptions.MaxFunEvals = 500000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 500000;
optimoptions.Jacobian = 'off';
optimoptions.Display = 'off';

param_init = [vech(L(1:m,1:m));vec(L(m+1:end,:))]; p = size(S,2);
Loss= L_penalized_objective(S,param_init,Q,Psi,m,loss,gamma,method,a_scad,b_mcp);
if (isnan(Loss)&&abs(Loss)>10^3)
    lambda1_init = 0.1+(0.3-0.1)*rand(m*(m+1)/2,1);
    lambda2_init = vec(0.1+(0.3-0.1)*rand(p-m,m));
    param_init = [lambda1_init;lambda2_init];
end
[param_l,~,~,~,~,~]=fmincon(@(x)L_penalized_objective(S,x,Q,Psi,m,loss,gamma,method,a_scad,b_mcp),param_init,[],[],[],[],[],[],[],optimoptions);
L = [tril(dvech(param_l(1:m*(m+1)/2),m),0);reshape(param_l(m*(m+1)/2+1:end),p-m,m)];