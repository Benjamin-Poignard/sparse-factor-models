function [param_lambda,param_psi] = sfm(X,m,Lambda_init,Psi_init,loss,gamma,method,iter)

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - Lambda_init: p x m x length(gamma) inital parameter values
%          for the factor loading matrix; one Lambda_init for each gamma
%          - Psi_init: diagonal variance-covariance matrix  of the
%          idiosyncratic errors obtained in Psi-step
%          - gamma: tuning parameter (single value)
%          - method: SCAD or MCP penalization
%          - iter: maximum number of iterations between Lambda and Psi

% Outputs:
%          - param_lambda: estimated sparse loading matrix (vectorized)
%          - param_psi: estimated diagonal elements of covariance matrix of
%          idosyncratic errors

param_psi_update = diag(Psi_init); param_lambda_update = vec(Lambda_init);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Iterate until convergence %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Tol = 10^(-6); kk = 0; error_track = zeros(iter,1);
while kk<iter
    
    kk = kk+1;
    
    param_psi = param_psi_update; param_lambda = param_lambda_update;
    
    % Lambda-step
    Psi = diag(param_psi); Lambda = reshape(param_lambda,size(Lambda_init,1),m);
    [Lambda_update,param_lambda_update] = penalized_factor_Lambda(X,m,Lambda,Psi,loss,gamma,method);
    
    % Psi-step
    [~,param_psi_update] = psi_estimation(X,m,Lambda_update,Psi,loss);
    
    error = norm([param_psi_update;param_lambda_update] - [param_psi;param_lambda])/max([1,norm([param_psi_update;param_lambda_update]),norm([param_psi;param_lambda])]);
    error_track(kk) = error;
    
    if (error <= Tol)
        break
    end
    
    if kk>10&&abs(error_track(kk)-error_track(kk-1))<10^(-4)
        break
    end
end
