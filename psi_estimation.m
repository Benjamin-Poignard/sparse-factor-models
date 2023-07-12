function Psi = psi_estimation(S,Lambda,Psi_init,loss)

% Inputs:
%          - S: sample variance-covariance matrix of the n x p observations
%          - Lambda: factor loading matrix
%          - Psi_init: inital value for the variance-covariance matrix
%          (diagonal) of the idiosyncratic errors
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
% Output:
%          - Psi: estimated variance-covariance matrix (diagonal) of the
%          idiosyncratic errors

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 500000;
optimoptions.MaxFunEvals = 500000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 500000;
optimoptions.Jacobian = 'off';
optimoptions.Display = 'off';

[param_est,~,~,~,~,~]=fmincon(@(x)psi_objective(S,Lambda,x,loss),diag(Psi_init),[],[],[],[],[],[],@(x)constr_psi(x),optimoptions);
Psi = diag(param_est);