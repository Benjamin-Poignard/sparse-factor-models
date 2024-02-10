function [ED,FN] = ranking_metric(data,Sigma,scale)

% Inputs:
%          - data: n x p matrix of observations
%          - Sigma: variance-covariance estimator
%          - scale: scaling parameter of the data (scalar value)
% Outputs:
%          - ED: Euclidean distance
%          - FN: Frobenius norm

[T,~] = size(data); ED = zeros(T,1); FN = zeros(T,1);

if (size(Sigma,3)<2)
    for t = 1:T
        ED(t) = vech((data(t,:)'*data(t,:))-Sigma)'*vech((data(t,:)'*data(t,:))-Sigma);
        FN(t) = trace(((data(t,:)'*data(t,:))-Sigma)'*((data(t,:)'*data(t,:))-Sigma));
    end
else
    for t = 1:T
        ED(t) = vech((data(t,:)'*data(t,:))-Sigma(:,:,t))'*vech((data(t,:)'*data(t,:))-Sigma(:,:,t));
        FN(t) = trace(((data(t,:)'*data(t,:))-Sigma(:,:,t))'*((data(t,:)'*data(t,:))-Sigma(:,:,t)));
    end
end
ED = ED./scale^4; FN = FN./scale^4;


