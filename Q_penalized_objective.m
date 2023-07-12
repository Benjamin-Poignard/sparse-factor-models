function Loss = Q_penalized_objective(x,L_step,m,gamma,method,a_scad,b_mcp)

% Inputs: 
%          - x: vector of the parameters entering in the Q matrix
%          - L_step: L matrix from the LQ-decomposition
%          - m: number of factors (a priori set by the user)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization 
%          - a_scad: SCAD parameter
%          - b_mcp: MCP parameter
% Output:
%          - Loss: loss function of the Q-step evaluated at vec(Lambda)
%          with Lambda = L*Q

Q = reshape(x,m,m);
Lambda = L_step*Q; 
switch method
    case 'scad'
        Loss = scad(vec(Lambda),gamma,a_scad);
    case 'mcp'
        Loss = mcp(vec(Lambda),gamma,b_mcp);
end


