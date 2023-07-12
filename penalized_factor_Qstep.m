function Q = penalized_factor_Qstep(L_step,Q,m,gamma,method,a_scad,b_mcp)

% Inputs:
%          - L_step: L matrix from the LQ-decomposition
%          - Q: Q matrix (to be updated) from the LQ-decomposition
%          satisfying QxQ' = I_m with I_m the m x m identity matrix
%          - m: number of factors (a priori set by the user)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization
%          - a_scad: SCAD parameter
%          - b_mcp: MCP parameter
% Output:
%          - Q: estimated Q matrix from the LQ-decomposition satisfying
%          Q'xQ = I_m with I_m the m x m identity matrix

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 20000;
optimoptions.MaxFunEvals = 300000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 10000;
optimoptions.Jacobian = 'off';
optimoptions.Display = 'off';

param_init = vec(Q);

[param_est,~,~,~,~,~]=fmincon(@(x)Q_penalized_objective(x,L_step,m,gamma,method,a_scad,b_mcp),param_init,[],[],[],[],[],[],@(x)constr_q(x,m),optimoptions);
Q = reshape(param_est,m,m);