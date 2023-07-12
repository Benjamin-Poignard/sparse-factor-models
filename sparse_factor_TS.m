function [Lambda,gamma_opt,Psi] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_first,Psi_first)

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
% Outputs:
%          - Lambda: sparse factor loading matrix
%          - gamma_opt: optimal tuning parameter select by the K-fold
%          cross-validation procedure
%          - Psi: variance-covariance matrix (diagonal) of the
%          idiosyncratic errors

[T,p] = size(X);
if nargin < 6
    [Lambda_first,Psi_first] = non_penalized_factor(cov(X),m,loss);
end

thresholding = 0.001;
if length(gamma)>1
    dim_l = m*(m+1)/2+(p-m)*m; dim_q = m^2;
    len_in = round(0.75*T); X_in = X(1:len_in,:); X_out = X(len_in+1:T,:);
    theta_psi_fold = zeros(p,length(gamma)); theta_l_fold = zeros(dim_l,length(gamma)); theta_q_fold = zeros(dim_q,length(gamma));
    parfor jj = 1:length(gamma)
        [theta_psi_fold(:,jj,1),theta_l_fold(:,jj,1),theta_q_fold(:,jj,1)] = lambda_penalized(X_in,m,Lambda_first,Psi_first,loss,gamma(jj),method);
    end
    fprintf(1,'In-sample estimation completed \n')
    count = zeros(length(gamma),1);
    for ii = 1:length(gamma)
        L = [tril(dvech(theta_l_fold(1:m*(m+1)/2,ii),m),0);reshape(theta_l_fold(m*(m+1)/2+1:end,ii),p-m,m)];
        Q = reshape(theta_q_fold(:,ii),m,m);
        Lambda = L*Q; Lambda(abs(Lambda)<thresholding)=0; Psi = diag(theta_psi_fold(:,ii));
        S = cov(X_out);
        Sigma = Lambda*Lambda'+Psi;
        switch loss
            case 'Gaussian'
                L =  ( log(det(Sigma))+trace(S/Sigma) )/(2*p);
            case 'LS'
                L = norm(S-Sigma,'fro')^2/(2*p);
        end
        count(ii) = count(ii) + L;
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