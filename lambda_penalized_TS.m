function [Lambda,gamma_opt] = lambda_penalized_TS(X,m,Lambda_init,Psi,loss,gamma,method)

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - Lambda_init: inital parameter value for the
%          factor loading matrix (satisfying IC5 condition)
%          - Psi: diagonal variance-covariance matrix  of the idiosyncratic
%          errors obtained in Psi-step
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization
% Outputs:
%          - Lambda: estimated sparse loading matrix
%          - gamma_opt: optimal tuning parameter selected by CV

[T,p] = size(X);
if length(gamma)>1
    % If gamma is a vector, then out-of-sample cross-validation is 
    % performed to select the optimal tuning parameter
    dim = p*m;
    len_in = round(0.75*T); X_in = X(1:len_in,:); X_out = X(len_in+1:T,:);
    theta_lambda_fold = zeros(dim,length(gamma));
    parfor jj = 1:length(gamma)
        [~,theta_lambda_fold(:,jj)] = penalized_factor_Lambda_TS(X_in,m,Lambda_init(:,:,jj),Psi,loss,gamma(jj),method);
    end
    count = zeros(length(gamma),1);
    for ii = 1:length(gamma)
        Lambda = reshape(theta_lambda_fold(:,ii),p,m);
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
    [~,param_lambda] = penalized_factor_Lambda_TS(X,m,Lambda_init_opt,Psi,loss,gamma_opt,method);
    Lambda = reshape(param_lambda,p,m);
else
    % If gamma is a scalar, then no cross-validation is performed
    gamma_opt = gamma;
    [~,param_lambda] = penalized_factor_Lambda_TS(X,m,Lambda_init,Psi,loss,gamma_opt,method); p = size(X,2);
    Lambda = reshape(param_lambda,p,m);
end