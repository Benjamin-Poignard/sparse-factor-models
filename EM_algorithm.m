function [lambda_sol,psi_sol] = EM_algorithm(X,m,lambda,gamma,Lambda_init,Psi_init)

% EM algorithm for joint estimation of Lambda and Psi (sparse non-diagonal)

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - lambda: step size
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - Lambda_init: initial value of the factor loading matrix satisfying
%          Lambda_init' x inv(Psi_init) x Lambda_init diagonal
%          - Psi_init: variance-covariance matrix (diagonal) of the idiosyncratic
%          errors
% Outputs:
%          - lambda_sol: vector form of the factor loading matrix satisfying
%          Lambda' x inv(Psi) x Lambda diagonal
%          - psi_sol: vector form of the sparse approximate
%          variance-covariance matrix of the idiosyncratic errors

t = 0.1;
Sy=cov(X);
Lambda=Lambda_init;
Su=Psi_init;
Sigmaold=Su;
Lambda0=Lambda;
A=inv(Lambda0*Lambda0'+Sigmaold);
C=Sy*A*Lambda0;
Eff=eye(m)-Lambda0'*A*Lambda0+Lambda0'*A*C;
Lambda=C*inv(Eff);
Su=Sy-C*Lambda'-Lambda*C'+Lambda*Eff*Lambda';
KML=Sigmaold-t*(inv(Sigmaold)-inv(Sigmaold)*Su*inv(Sigmaold));
P=Pmatrix(Su,gamma);
B=lambda*t*P;
Sigma1=soft(KML,B);
kk=1;
while  likelihoodTrue(Sy,P,Sigmaold,Lambda0,lambda)- likelihoodTrue(Sy,P,Sigma1,Lambda,lambda)>10^(-7)&kk<5000
    Sigmaold=Sigma1;
    Lambda0=Lambda;
    A=inv(Lambda0*Lambda0'+Sigma1);
    C=Sy*A*Lambda0;
    Eff=eye(m)-Lambda0'*A*Lambda0+Lambda0'*A*C;
    Lambda=C*inv(Eff);
    Su=Sy-C*Lambda'-Lambda*C'+Lambda*Eff*Lambda';
    KML=Sigmaold-t*(inv(Sigmaold)-inv(Sigmaold)*Su*inv(Sigmaold));
    P=Pmatrix(Su,gamma);
    B=lambda*t*P;
    Sigma1=soft(KML,B);
    kk=kk+1;
end
lambda_sol=vec(Lambda0);
psi_sol=vech(Sigmaold);