function Q = penalized_factor_Qstep_sampling(L_step,m,sampling_size,gamma,method,a_scad,b_mcp)

% Function for initializing the Q matrix
% Inputs:
%          - L_step: L matrix from the LQ-decomposition
%          - m: number of factors (a priori set by the user)
%          - sampling_size: grid size of inital trials to get an optimal
%          starting value for the Q matrix
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization
%          - a_scad: SCAD parameter
%          - b_mcp: MCP parameter
% Output:
%          - Q: estimated Q matrix from the LQ-decomposition satisfying
%          Q'xQ = I_m with I_m the m x m identity matrix

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 500000;
optimoptions.MaxFunEvals = 500000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 500000;
optimoptions.Jacobian = 'off';
optimoptions.Display = 'off';

p = size(L_step,1);
loss_eval = zeros(sampling_size,1); Q_sampling = zeros(m,m,sampling_size);
for kk = 1:sampling_size
    [Q,~] = qr(rand(p,m)'); Q = Q';
    param_init = vec(Q);
    [param_est,FVAL,~,~,~,~]=fmincon(@(x)Q_penalized_objective(x,L_step,m,gamma,method,a_scad,b_mcp),param_init,[],[],[],[],[],[],@(x)constr_q(x,m),optimoptions);
    Q_sampling(:,:,kk) = reshape(param_est,m,m); loss_eval(kk) = 100*FVAL;
end
[~,index] = min(loss_eval);
Q = Q_sampling(:,:,index);