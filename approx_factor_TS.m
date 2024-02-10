function [Lambda,gamma_opt,Psi] = approx_factor_TS(X,m,gamma)

% Sparse approximate factor estimator of Bai and Liao (2016), 'Efficient 
% estimation of approximate factor models via penalized maximum
% likelihood', Journal of Econometrics, 191 (1), 1-18

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - gamma: tuning parameter (grid of candidates set by the user)
% Outputs:
%          - Lambda: factor loading matrix satisfying
%          Lambda' x inv(Psi) x Lambda diagonal
%          - gamma_opt: optimal tuning parameter selected by the K-fold
%          cross-validation procedure
%          - Psi: sparse approximate variance-covariance matrix of the
%          idiosyncratic errors obtained by the EM algorithm

[T,N]=size(X);

% Calcualte  PCA
[V,D]=eig(X*X');
[~,I]=sort(diag(D));
for i=1:m
    F(:,i)=sqrt(T)*V(:,I(T-i+1));
end
LamPCA=X'*F/T;
uhat=X'-LamPCA*F';
SuPCA=uhat*uhat'/T;
LPCA=LamPCA;
SigPCA=SuPCA;

%%%%%%%%%%%% Calculate SFM (diagonal Max Likelihood of Bai and Li 2012) as initial value
%%%%%%%%%% SFM estimate uses PCA as initial value
Sy=X'*X/T;
kk=1;
Lambda0=ones(N,m)*10;
Lambda=LPCA;
Su=SigPCA;
Psi=diag(diag(Su));
Psi_old=eye(N)*100;

while likelihoodlambda(Sy,Lambda0,Psi_old,'Gaussian')-likelihoodlambda(Sy,Lambda,Psi,'Gaussian')>10^(-7)&&kk<4000
    Psi_old=Psi;
    Lambda0=Lambda;
    A=inv(Lambda0*Lambda0'+Psi);
    C=Sy*A*Lambda0;
    Eff=eye(m)-Lambda0'*A*Lambda0+Lambda0'*A*C;
    Lambda=C/Eff;
    M=Sy-Lambda*Lambda0'*A*Sy;
    Psi=diag(diag(M));
    kk=kk+1 ;
end
Lambda_init = Lambda; Psi_init = Psi;

[T,p] = size(X);
if length(gamma)>1
    % If gamma is a vector, then a 5-fold cross-validation is performed to
    % select the optimal tuning parameter
    dim = p*m;
    len_in = round(0.75*T); X_in = X(1:len_in,:); X_out = X(len_in+1:T,:);
    theta_lambda_fold = zeros(dim,length(gamma)); psi_fold = zeros(p*(p+1)/2,length(gamma));
    parfor jj = 1:length(gamma)
        [theta_lambda_fold(:,jj),psi_fold(:,jj)] = EM_algorithm(X_in,m,0.08,gamma(jj),Lambda_init,Psi_init);
    end
    count = zeros(length(gamma),1);
    for ii = 1:length(gamma)
        Lambda = reshape(theta_lambda_fold(:,ii),p,m); Psi = dvech(psi_fold(:,ii),p);
        S = cov(X_out); Sigma = Lambda*Lambda'+Psi;
        L =  ( log(det(Sigma))+trace(S/Sigma) );
        count(ii) = count(ii) + L;
    end
    clear ii kk
    ii = count==min(min(count)); gamma_opt = gamma(ii);
    if length(gamma_opt)>1
        gamma_opt = gamma(1);
    end
    [param_lambda,param_psi] = EM_algorithm(X,m,0.08,gamma_opt,Lambda_init,Psi_init);
    Lambda = reshape(param_lambda,p,m); Psi = dvech(param_psi,p);
else
    % If gamma is a scalar, then no cross-validation is performed
    gamma_opt = gamma;
    [param_lambda,param_psi] = EM_algorithm(X,m,0.08,gamma_opt,Lambda_init,Psi_init);
    Lambda = reshape(param_lambda,p,m); Psi = dvech(param_psi,p);
end