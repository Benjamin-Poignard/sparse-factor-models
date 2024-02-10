function L = non_penalized_objective(S,param,m,loss)

% Inputs: 
%         - S: sample variance-covariance matrix of the vector of 
%         observations
%         - param: parameter of the factor model (Lambda and Psi), where
%         the parameters of the factor loading matrix satisfy IC5 condition
%         - m: number of factors
%         - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
% Output:
%         - L: value of the loss function evaluated at param


p = size(S,2);
dim_lambda1 = m*(m+1)/2; dim_lambda2 = (p-m)*m;
Lambda1 = tril(dvech(param(1:dim_lambda1),m));
Lambda2 = reshape(param(dim_lambda1+1:dim_lambda1+dim_lambda2),p-m,m);
Lambda = [Lambda1;Lambda2];
Psi = diag(param(dim_lambda1+dim_lambda2+1:end));
Sigma = Lambda*Lambda' + Psi;
switch loss
    case 'Gaussian'
        L =  (log(det(Sigma))+trace(S/Sigma))/p;
    case 'LS'
        L = norm(S-Sigma,'fro')^2/p;
end
