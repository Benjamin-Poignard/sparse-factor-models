function Loss_pen = Lambda_penalized_objective(S,x,Psi,m,loss,gamma,method,a_scad,b_mcp)

% Inputs:
%          - S: sample variance-covariance matrix of the n x p observations
%          - x: vector of the parameters entering in the L matrix
%          - Q: Q matrix from the LQ-decomposition satisfying  QxQ' = I_m
%          with I_m the m x m identity matrix
%          - m: number of factors (a priori set by the user)
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization
%          - a_scad: SCAD parameter
%          - b_mcp: MCP parameter
% Output:
%          - Loss_pen: loss function of the L-step evaluated at vec(Lambda)
%          with Lambda = L*Q and penalty function

p = size(S,2);
Lambda = reshape(x,p,m); Sigma = Lambda*Lambda' + Psi;

switch method
    case 'scad'
        pen = scad(vec(Lambda),gamma,a_scad);
    case 'mcp'
        pen = mcp(vec(Lambda),gamma,b_mcp);
end
switch loss
    case 'Gaussian'
        Loss =  ( log(det(Sigma))+trace(S/Sigma) );
    case 'LS'
        Loss = norm(S-Sigma,'fro')^2/p;
end
Loss_pen = Loss + pen;