function [Lambda,Psi] = non_penalized_factor(X,m,loss)

% Inputs:
%         - X: n x p vector of observations
%         - m: number of factors
%         - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
% Outputs:
%         - Lambda: factor loading matrix satisfying
%         Lambda' x inv(Psi) x Lambda diagonal
%         - Psi: variance-covariance matrix (diagonal) of the idiosyncratic
%         errors

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

while likelihoodlambda(Sy,Lambda0,Psi_old,loss)-likelihoodlambda(Sy,Lambda,Psi,loss)>10^(-7)&&kk<4000
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

switch loss
    case 'Gaussian'
        fprintf('First step estimation with Gaussian loss completed \n')
    case 'LS'
        fprintf('First step estimation with LS loss completed \n')
end