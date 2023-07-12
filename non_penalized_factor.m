function [Lambda,Psi] = non_penalized_factor(S,m,loss)

% Inputs:
%         - S: sample variance-covariance matrix
%         - m: number of factors
%         - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
% Outputs:
%         - Lambda: factor loading matrix satisfying IC5 condition of Bai
%         and Li (2012), 'Statistical analysis of factor models of high
%         dimension', the Annals of Statistics, 40(1): 436-465
%         - Psi: variance-covariance matrix (diagonal) of the idiosyncratic
%         errors

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 500000;
optimoptions.MaxFunEvals = 500000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 500000;
optimoptions.Jacobian = 'off';
optimoptions.Display = 'off';

p = size(S,2); dim_lambda1 = m*(m+1)/2; dim_lambda2 = (p-m)*m;
sampling_size = 10;
loss_eval = zeros(sampling_size,1); param = zeros(dim_lambda1+dim_lambda2+p,sampling_size);
parfor kk = 1:sampling_size
    lambda1_init = 0.1+(0.3-0.1)*rand(m*(m+1)/2,1);
    lambda2_init = vec(0.1+(0.3-0.1)*rand(p-m,m));
    psi_init = eye(p);
    param_init = [lambda1_init;lambda2_init;diag(psi_init)];
    [param_est,FVAL,~,~,~,~]=fmincon(@(x)non_penalized_objective(S,x,m,loss),param_init,[],[],[],[],[],[],@(x)constr(x,p,m),optimoptions);
    param(:,kk) = param_est; loss_eval(kk) = 100*FVAL;
end

[~,index] = min(loss_eval); param_est = param(:,index);
Lambda1 = tril(dvech(param_est(1:dim_lambda1),m));
Lambda2 = reshape(param_est(dim_lambda1+1:dim_lambda1+dim_lambda2),p-m,m);
Lambda = [Lambda1;Lambda2];
Psi = diag(param_est(dim_lambda1+dim_lambda2+1:end));
fprintf('First step estimation completed \n')