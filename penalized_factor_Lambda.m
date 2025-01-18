function [Lambda,param_lambda] = penalized_factor_Lambda(X,m,Lambda_init,Psi,loss,gamma,method)

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - Lambda_init: p x m inital parameter value
%          for the factor loading matrix
%          - Psi: diagonal variance-covariance matrix  of the idiosyncratic
%          errors obtained in Psi-step
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization
% Outputs:
%          - Lambda: estimated sparse loading matrix
%          - param_lambda: estimated sparse loading matrix in vector form

p = size(X,2); dim = p*m;

switch method
    case 'scad'
        a_scad = 3.7;
        mu = 1/(a_scad-1);
    case 'mcp'
        b_mcp = 3.5;
        mu = 1/b_mcp;
    case 'lasso'
        mu = 0;
end
eta = 200;
% calibration of the step size
nu = (1/eta)/(1+(mu/eta));
maxIt = 80e3; crit = 10^(-8); i = 0;

Lambda = Lambda_init; Lambda_old = Lambda;
error_check = zeros(maxIt,1);

while i<maxIt
    
    i = i+1;
    Sigma = Lambda*Lambda'+Psi;
    
    switch loss
        case 'Gaussian'
            gradient = 2*vec((Sigma)\(Sigma-cov(X))/(Sigma)*Lambda);
        case 'LS'
            gradient = 4*vec((Sigma-cov(X))*Lambda)/p;
    end
    gradient_modified = gradient - mu*vec(Lambda);
    Z = (1/(1+(mu/eta)))*(vec(Lambda)-gradient_modified/eta);
    param_est = zeros(dim,1);
    
    switch method
        case 'scad'
            for ii = 1:dim
                if (0 <= abs(Z(ii)) && abs(Z(ii)) <= nu*gamma)
                    param_est(ii) = 0;
                elseif (nu*gamma <= abs(Z(ii)) && abs(Z(ii)) <= (nu+1)*gamma)
                    param_est(ii) = Z(ii)-(sign(Z(ii))*nu*gamma);
                elseif ((nu+1)*gamma <= abs(Z(ii)) && abs(Z(ii)) <= a_scad*gamma)
                    param_est(ii) = (Z(ii)-((sign(Z(ii))*a_scad*nu*gamma)/(a_scad-1)))/(1-(nu/(a_scad-1)));
                elseif (a_scad*gamma <= abs(Z(ii)))
                    param_est(ii) = Z(ii);
                end
            end
            Lambda = reshape(param_est,p,m);
            
        case 'mcp'
            for ii = 1:dim
                if (0 <= abs(Z(ii)) && abs(Z(ii)) <= nu*gamma)
                    param_est(ii) = 0;
                elseif (nu*gamma <= abs(Z(ii)) && abs(Z(ii)) <= b_mcp*gamma)
                    param_est(ii) = (Z(ii)-(sign(Z(ii))*nu*gamma))/(1-nu/b_mcp);
                elseif (b_mcp*gamma <= abs(Z(ii)))
                    param_est(ii) = Z(ii);
                end
            end
            Lambda = reshape(param_est,p,m);
            
        case 'lasso'
            for ii = 1:dim
                param_est(ii) = sign(Z(ii))*subplus(abs(Z(ii))-gamma/eta);
            end
            Lambda = reshape(param_est,p,m);
            
    end
    
    error = (norm(vec(Lambda - Lambda_old))/max([1,norm(vec(Lambda)),norm(vec(Lambda_old))]));
    error_check(i) = error;
    if (error<crit)
        break;
    end
    
    Lambda_old = Lambda;
    
end
param_lambda = vec(Lambda);