function [Lambda,gamma_opt,Psi] = sparse_factor(X,m,loss,gamma,method,K,Lambda_first,Psi_first)

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization (a_scad = 3.7, b_mcp = 3.5):
%          see lambda_penalized.m to modify a_scad and b_mcp
%          - Lambda_first (optional input): inital parameter value for the
%          factor loading matrix
%          - Psi_first (optional input): inital parameter value for the
%          variance-covariance matrix (diagonal) of the idiosyncratic
%          errors, jointly obtained with Lambda_first
%          - K (optional input): number of folds for cross-validation; K
%          must be larger strictly than 2
% Outputs:
%          - Lambda: sparse factor loading matrix
%          - gamma_opt: optimal tuning parameter select by the K-fold
%          cross-validation procedure
%          - Psi: variance-covariance matrix (diagonal) of the
%          idiosyncratic errors

[n,p] = size(X);
% if no cross-validation number is specified, then K = 5 by default
if nargin < 6
    K = 5;
end
% if no first step estimator for Lambda and Psi are provided, then the
% initial values are set as the estimators satisfying IC5 conditions of Bai
% and Li (2012), 'Statistical analysis of factor models of high
%         dimension', the Annals of Statistics, 40(1): 436-465
if nargin < 7
    [Lambda_first,Psi_first] = non_penalized_factor(cov(X),m,loss);
end

thresholding = 0.001; len = round(n/K);
if length(gamma)>1
    % If gamma is a vector, then a 5-fold cross-validation is performed to
    % select the optimal tuning parameter
    dim_l = m*(m+1)/2+(p-m)*m; dim_q = m^2;
    X_f = zeros(len,p,K); theta_psi_fold = zeros(p,length(gamma),K); theta_l_fold = zeros(dim_l,length(gamma),K); theta_q_fold = zeros(dim_q,length(gamma),K);
    X_f(:,:,1) = X(1:len,:); X_temp = X; X_temp(1:len,:)=[];
    parfor jj = 1:length(gamma)
        [theta_psi_fold(:,jj,1),theta_l_fold(:,jj,1),theta_q_fold(:,jj,1)] = lambda_penalized(X_temp,m,Lambda_first,Psi_first,loss,gamma(jj),method);
    end
    fprintf(1,'Estimation for fold 1 completed \n')
    for kk = 2:K-1
        X_f(:,:,kk) = X((kk-1)*len+1:kk*len,:); X_temp = X; X_temp((kk-1)*len+1:kk*len,:)=[];
        parfor jj = 1:length(gamma)
            [theta_psi_fold(:,jj,kk),theta_l_fold(:,jj,kk),theta_q_fold(:,jj,kk)] = lambda_penalized(X_temp,m,Lambda_first,Psi_first,loss,gamma(jj),method);
        end
        fprintf(1,'Estimation for fold %d completed \n',kk)
    end
    X_f(:,:,K) = X(end-len+1:end,:); X_temp = X; X_temp(end-len+1:end,:) = [];
    parfor jj = 1:length(gamma)
        [theta_psi_fold(:,jj,K),theta_l_fold(:,jj,K),theta_q_fold(:,jj,K)] = lambda_penalized(X_temp,m,Lambda_first,Psi_first,loss,gamma(jj),method);
    end
    fprintf(1,'Estimation for fold %d completed \n',K)
    count = zeros(length(gamma),1);
    for ii = 1:length(gamma)
        for kk = 1:K
            L = [tril(dvech(theta_l_fold(1:m*(m+1)/2,ii,kk),m),0);reshape(theta_l_fold(m*(m+1)/2+1:end,ii,kk),p-m,m)];
            Q = reshape(theta_q_fold(:,ii,kk),m,m);
            Lambda = L*Q; Lambda(abs(Lambda)<thresholding)=0; Psi = diag(theta_psi_fold(:,ii,kk));
            S = cov(X_f(:,:,kk));
            Sigma = Lambda*Lambda'+Psi;
            switch loss
                case 'Gaussian'
                    L =  ( log(det(Sigma))+trace(S/Sigma) )/(2*p);
                case 'LS'
                    L = norm(S-Sigma,'fro')^2/(2*p);
            end
            count(ii) = count(ii) + L;
        end
        count(ii) = count(ii)/K;
    end
    clear ii kk
    ii = count==min(min(count)); gamma_opt = gamma(ii);
    if length(gamma_opt)>1
        gamma_opt = gamma(1);
    end
    fprintf(1,'Final step: estimation with the optimal tuning parameter \n')
    [param_psi,param_l,param_q] = lambda_penalized(X,m,Lambda_first,Psi_first,loss,gamma_opt,method);
    L = [tril(dvech(param_l(1:m*(m+1)/2),m),0);reshape(param_l(m*(m+1)/2+1:end),p-m,m)];
    Q = reshape(param_q,m,m); Psi = diag(param_psi);
    Lambda = L*Q; Lambda(abs(Lambda)<thresholding)=0;
    fprintf(1,'Estimation completed \n')
else
    % If gamma is a scalar, then no cross-validation is performed
    fprintf(1,'One tuning parameter was provided. No cross-validation is performed \n')
    gamma_opt = gamma;
    [param_psi,param_l,param_q] = lambda_penalized(X,m,Lambda_first,Psi_first,loss,gamma_opt,method); p = size(X,2);
    L = [tril(dvech(param_l(1:m*(m+1)/2),m),0);reshape(param_l(m*(m+1)/2+1:end),p-m,m)];
    Q = reshape(param_q,m,m); Psi = diag(param_psi);
    Lambda = L*Q; Lambda(abs(Lambda)<thresholding)=0;
    fprintf(1,'Estimation completed \n')
end