function L = psi_objective(S,Lambda,param,loss)

% Inputs: 
%          - S: sample variance-covariance matrix of the n x p observations
%          - Lambda: factor loading matrix
%          - param: parameters of the variance-covariance matrix
%          (diagonal) of the idiosyncratic errors
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
% Output:
%          - L: loss function evaluated at param (so Psi matrix)

p = size(S,2); 
Psi = diag(param);
Sigma = Lambda*Lambda' + Psi;
switch loss
    case 'Gaussian'
        L =  ( log(det(Sigma))+trace(S/Sigma) )/p;
    case 'LS'
        L = norm(S-Sigma,'fro')^2/p;
end