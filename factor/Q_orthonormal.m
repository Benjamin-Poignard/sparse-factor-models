function [Loss,G] = Q_orthonormal(x,L_step,m,gamma,method)

% Inputs:
%          - x: vector of the parameters entering in the Q matrix
%          - L_step: L matrix from the LQ-decomposition
%          - m: number of factors (a priori set by the user)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization
% Output:
%          - Loss: loss function of the Q-step evaluated at vec(Lambda)
%          with Lambda = L*Q
%          - G: gradient function (vector) of the objective function

Q = reshape(x,m,m);
Lambda = L_step*Q;
switch method
    case 'scad'
        Loss = scad(vec(Lambda),gamma,3.7);
        G = Grad(@(x)scad(vec(L_step*reshape(x,m,m)),gamma,3.7),vec(Q));
        G = reshape(G,m,m);
    case 'mcp'
        Loss = mcp(vec(Lambda),gamma,3.5);
        G = Grad(@(x)mcp(vec(L_step*reshape(x,m,m)),gamma,3.5),vec(Q));
        G = reshape(G,m,m);
end
end

function penalty = scad(param,lambda,a)
penalty = 0;
for ii = 1:length(param)
    if (abs(param(ii))<=lambda)
        shrinkage = lambda*abs(param(ii));
    elseif ((lambda<abs(param(ii))) && (abs(param(ii)) <=a*lambda))
        shrinkage = -(param(ii)^2-2*a*lambda*abs(param(ii))+lambda^2)/(2*(a-1));
    elseif (abs(param(ii))>a*lambda)
        shrinkage = (a+1)*lambda^2/2;
    end
    penalty = penalty + shrinkage;
end
end

function penalty = mcp(param,lambda,b)
penalty = 0;
for ii = 1:length(param)
    if (abs(param(ii))<=b*lambda)
        shrinkage = lambda*abs(param(ii)) - (param(ii)^2)/(2*b);
    elseif (abs(param(ii))>b*lambda)
        shrinkage = b*lambda^2/2;
    end
    penalty = penalty + shrinkage;
end
end