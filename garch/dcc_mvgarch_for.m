function [parameters,Rt_oos,H_oos]=dcc_mvgarch_for(data,method,n_period)

% Inputs:
%        - data: n x p vector of returns
%        - method: estimation method: 'full' or 'CL', with 'full' the full
%        the full likelihood-based estimation and 'CL' the composite
%        likelihood-based estimation
%        - n_period: the date which splits the data into the in-sample part
%        and out-of-sample part: [1,n_period] = in-sample part and
%        [n_period+1,end] = out-of-sample part
% Outputs:
%        - parameters: (Number of univariateparameters)*N+2 x 1 vector,
%        the two last ones are the scalar DCC parameters
%        - Rt_oos: N x N x Toos out-of-sample correlation matrix process,
%        with Toos the lenght of [n_period+1,end]
%        - H_oos: N x N x Toos out-of-sample matrix of univariate variance 
%        processes, with Toos the lenght of [n_period+1,end]


X = data(1:n_period,:); % in-sample data
X_out = data(n_period+1:end,:); % out-of-sample data

% in-sample estimation of the scalar DCC
[parameters,~,H]=dcc_mvgarch(X,method);

% out-of-sample univariate GARCH(1,1) processes
h_oos=zeros(size(X_out,1),size(X_out,2));
index = 1;
for jj=1:size(X_out,2)
    univariateparameters=parameters(index:index+1+1);
    [~,h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,X_out(:,jj));
    index=index+1+1+1;
end
h_oos = sqrt(h_oos);

% scalar DCC out-of-sample correlation process
[~,Rt_oos,~,~]=dcc_mvgarch_generate_oos(parameters,X_out,X,H);

% scalar DCC-based out-of-sample covariance process
H_oos = zeros(size(X,2),size(X,2),size(X_out,1));
for t = 1:size(X_out,1)
    H_oos(:,:,t) = diag(h_oos(t,:))*Rt_oos(:,:,t)*diag(h_oos(t,:));
end