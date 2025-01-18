function [Lambda,gamma_opt,Psi] = cv_sfm_ts(X,m,Lambda_init,Psi_init,loss,gamma,method)

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - Lambda_init: p x m x length(gamma) inital parameter values
%          for the factor loading matrix; one Lambda_init for each gamma
%          - Psi_init: diagonal variance-covariance matrix  of the
%          idiosyncratic errors obtained in Psi-step
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization
% Outputs:
%          - Lambda: estimated sparse loading matrix
%          - gamma_opt: optimal tuning parameter
%          - Psi: estimated covariance (diagonal) of idiosyncratic errors 

[T,p] = size(X);
if length(gamma)>1
    dim = p*m; len_in = round(0.75*T);
    X_in = X(1:len_in,:); X_out = X(len_in+1:T,:);
    theta_lambda_fold = zeros(dim,length(gamma)); theta_psi_fold = zeros(p,length(gamma));
    parfor jj = 1:length(gamma)
        [theta_lambda_fold(:,jj),theta_psi_fold(:,jj)] = sfm(X_in,m,Lambda_init(:,:,jj),Psi_init,loss,gamma(jj),method,101);
    end
    
    count = zeros(length(gamma),1);
    for ii = 1:length(gamma)
        Lambda = reshape(theta_lambda_fold(:,ii),p,m);
        Psi = diag(theta_psi_fold(:,ii));
        S = cov(X_out); Sigma = Lambda*Lambda'+Psi;
        switch loss
            case 'Gaussian'
                L =  ( log(abs(det(Sigma)))+trace(S/Sigma) );
            case 'LS'
                L = norm(S-Sigma,'fro')^2;
        end
        count(ii) = L;
    end
    clear ii kk
    ii = count==min(min(count)); gamma_opt = gamma(ii); Lambda_init_opt = Lambda_init(:,:,ii);
    if length(gamma_opt)>1
        gamma_opt = gamma(1); Lambda_init_opt = Lambda_init(:,:,1);
    end
    [param_lambda,param_psi] = sfm(X,m,Lambda_init_opt,Psi_init,loss,gamma_opt,method,101);
    Lambda = reshape(param_lambda,p,m); Psi = diag(param_psi);
else
    % If gamma is a scalar, no cross-validation
    gamma_opt = gamma;
    [param_lambda,param_psi] = sfm(X,m,Lambda_init,Psi_init,loss,gamma_opt,method,101); p = size(X,2);
    Lambda = reshape(param_lambda,p,m); Psi = diag(param_psi);
end