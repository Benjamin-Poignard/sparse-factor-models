function [Lambda,gamma_opt] = lambda_penalized(X,m,Lambda_init,Psi,loss,gamma,method,K)

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - Lambda_init: p x m x length(gamma) inital parameter values
%          for the factor loading matrix; one Lambda_init for each gamma
%          - Psi: diagonal variance-covariance matrix  of the idiosyncratic
%          errors obtained in Psi-step
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization
%          - K: number of fold for CV
% Outputs:
%          - Lambda: estimated sparse loading matrix
%          - gamma_opt: optimal tuning parameter selected by CV

[n,p] = size(X); len = round(n/K);
if length(gamma)>1
    % If gamma is a vector, then a 5-fold cross-validation is performed to
    % select the optimal tuning parameter
    dim = p*m;
    X_f = zeros(len,p,K); theta_lambda_fold = zeros(dim,length(gamma),K);
    X_f(:,:,1) = X(1:len,:); X_temp = X; X_temp(1:len,:)=[];
    parfor jj = 1:length(gamma)
        [~,theta_lambda_fold(:,jj,1)] = penalized_factor_Lambda(X_temp,m,Lambda_init(:,:,jj),Psi,loss,gamma(jj),method);
    end
    for kk = 2:K-1
        X_f(:,:,kk) = X((kk-1)*len+1:kk*len,:); X_temp = X; X_temp((kk-1)*len+1:kk*len,:)=[];
        parfor jj = 1:length(gamma)
            [~,theta_lambda_fold(:,jj,kk)] = penalized_factor_Lambda(X_temp,m,Lambda_init(:,:,jj),Psi,loss,gamma(jj),method);
        end
    end
    X_f(:,:,K) = X(end-len+1:end,:); X_temp = X; X_temp(end-len+1:end,:) = [];
    parfor jj = 1:length(gamma)
        [~,theta_lambda_fold(:,jj,K)] = penalized_factor_Lambda(X_temp,m,Lambda_init(:,:,jj),Psi,loss,gamma(jj),method);
    end
    count = zeros(length(gamma),1);
    for ii = 1:length(gamma)
        for kk = 1:K
            Lambda = reshape(theta_lambda_fold(:,ii,kk),p,m);
            S = cov(X_f(:,:,kk)); Sigma = Lambda*Lambda'+Psi;
            switch loss
                case 'Gaussian'
                    L =  ( log(det(Sigma))+trace(S/Sigma) );
                case 'LS'
                    L = norm(S-Sigma,'fro')^2;
            end
            count(ii) = count(ii) + L;
        end
        count(ii) = count(ii)/K;
    end
    clear ii kk
    ii = count==min(min(count)); gamma_opt = gamma(ii); Lambda_init_opt = Lambda_init(:,:,ii);
    if length(gamma_opt)>1
        gamma_opt = gamma(1); Lambda_init_opt = Lambda_init(:,:,1);
    end
    [~,param_lambda] = penalized_factor_Lambda(X,m,Lambda_init_opt,Psi,loss,gamma_opt,method);
    Lambda = reshape(param_lambda,p,m);
else
    % If gamma is a scalar, no cross-validation
    gamma_opt = gamma;
    [~,param_lambda] = penalized_factor_Lambda(X,m,Lambda_init,Psi,loss,gamma_opt,method); p = size(X,2);
    Lambda = reshape(param_lambda,p,m);
end