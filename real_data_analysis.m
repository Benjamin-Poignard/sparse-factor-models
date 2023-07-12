% The following code replicates the real data experiment with the two
% following portfolios: MSCI and S&P 100
% The details can be found in the paper, Section "Empirical analysis based
% on real data"
% The competing models are:
% - scalar DCC (full and composite likelihood methods can be employed)
% - sample variance-covariance
% - factor model: estimator deduced from the Gaussian and least squares
%   loss functions and SCAD/MCP penalty functions, for different numbers of
%   factors
% - 1/p strategy

% The out-of-sample GMVP performances (AVG, SD, IR) are stored in
% Results_1, Results_2, Results_3, Results_4
%% MSCI portfolio, out-of-sample period: 04/01/2016 -- 03/12/2018
clear
clc

% load the MSCI country stock indices: 23 assets
Table = readtable('MSCI.xls');

data_MSCI = Table{1:end,[2:end]};
% transform into log-returns
returns = 100*(log(data_MSCI(2:end,:))-log(data_MSCI(1:end-1,:)));
p = size(returns,2);
n_period = 4500;

X = returns(1:n_period,:); % in-sample data
X_out = returns(n_period+1:end,:); % out-of-sample data

factor = [2 3 5]; % specify the number of factors
K = length(factor);

w_scad_g = zeros(p,K); w_mcp_g = zeros(p,K);
w_scad_ls = zeros(p,K); w_mcp_ls = zeros(p,K);

for j = 1:K
    
    % selection of the number of factors (user-specified in factor)
    m = factor(j);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% First step under IC5 %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % First step estimation for initialization (factor model under IC5
    % constraint for Lambda and diagonal variance-covariance Psi)
    [Lambda_g,Psi_g] = non_penalized_factor(cov(X),m,'Gaussian');
    
    [Lambda_ls,Psi_ls] = non_penalized_factor(cov(X),m,'LS');
    
    
    grid = m*[0.1 0.15:0.05:10]; gamma = grid*sqrt(log(p*m)/n_period);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'Gaussian';
    
    method = 'scad';
    [Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_g,Psi_g);
    Sigma_scad_g = Lambda_scad_g*Lambda_scad_g'+ Psi_scad_g;
    w_scad_g(:,j) = GMVP(Sigma_scad_g);
    
    method = 'mcp';
    [Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_g,Psi_g);
    Sigma_mcp_g = Lambda_mcp_g*Lambda_mcp_g'+ Psi_mcp_g;
    w_mcp_g(:,j) = GMVP(Sigma_mcp_g);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'LS';
    
    method = 'scad';
    [Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_ls,Psi_ls);
    Sigma_scad_ls = Lambda_scad_ls*Lambda_scad_ls'+ Psi_scad_ls;
    w_scad_ls(:,j) = GMVP(Sigma_scad_ls);
    
    method = 'mcp';
    [Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_ls,Psi_ls);
    Sigma_mcp_ls = Lambda_mcp_ls*Lambda_mcp_ls'+ Psi_mcp_ls;
    w_mcp_ls(:,j) = GMVP(Sigma_mcp_ls);
    
end

% sample variance-covariance estimator
Sample_VC = cov(X);
w_sample = GMVP(Sample_VC);

% in-sample estimation of the scalar DCC
[parameters_dcc,~,H_in]=dcc_mvgarch(X,'full');

% out-of-sample univariate GARCH(1,1) processes
h_oos=zeros(size(X_out,1),size(X_out,2));
index = 1;
for jj=1:size(X_out,2)
    univariateparameters=parameters_dcc(index:index+1+1);
    [simulatedata, h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,X_out(:,jj));
    index=index+1+1+1;
end
h_oos = sqrt(h_oos);

% scalar DCC out-of-sample correlation process
[~,Rt_dcc_oos,~,~]=dcc_mvgarch_generate_oos(parameters_dcc,X_out,X,H_in);

% scalar DCC out-of-sample covariance process
Hdcc = zeros(p,p,size(X_out,1));
for t = 1:size(X_out,1)
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

% out-of-sample portfolio weights and returns for the fixed estimators
W = [w_scad_g w_mcp_g w_scad_ls w_mcp_ls w_sample ones(p,1)/p];
T = size(X_out,1); n_candidates = size(W,2);
e = zeros(T,n_candidates);
for t = 1:T
    for j = 1:n_candidates
        e(t,j) = W(:,j)'*X_out(t,:)';
    end
end
% out-of-sample portfolio weights and returns for DCC estimator
wdcc = zeros(p,T); e_dcc = zeros(T,1);
for t = 1:T
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    e_dcc(t) = wdcc(:,t)'*X_out(t,:)';
end

% out-of-sample returns
e = [e e_dcc];

% out-of-sample average portfolio returns, standard deviations and
% information ratios
Results_1 = [252*mean(e);sqrt(252)*std(e);(252*mean(e))./(sqrt(252)*std(e))]';
%% MSCI portfolio, out-of-sample period: 06/01/2012 -- 03/12/2018
clear
clc
% load the MSCI country stock indices: 23 assets

Table = readtable('MSCI.xls');

data_MSCI = Table{1:end,[2:end]};
% transform into log-returns
returns = 100*(log(data_MSCI(2:end,:))-log(data_MSCI(1:end-1,:)));
p = size(returns,2);
n_period = 3500;

X = returns(1:n_period,:);
X_out = returns(n_period+1:end,:);

factor = [2 3 5]; K = length(factor);

w_scad_g = zeros(p,K); w_mcp_g = zeros(p,K);
w_scad_ls = zeros(p,K); w_mcp_ls = zeros(p,K);

for j = 1:K
    
    % selection of the number of factors (user-specified in factor)
    m = factor(j);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% First step under IC5 %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % First step estimation for initialization (factor model under IC5
    % constraint for Lambda and diagonal variance-covariance Psi)
    [Lambda_g,Psi_g] = non_penalized_factor(cov(X),m,'Gaussian');
    
    [Lambda_ls,Psi_ls] = non_penalized_factor(cov(X),m,'LS');
    
    
    grid = m*[0.1 0.15:0.05:10]; gamma = grid*sqrt(log(p*m)/n_period);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'Gaussian';
    
    method = 'scad';
    [Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_g,Psi_g);
    Sigma_scad_g = Lambda_scad_g*Lambda_scad_g'+ Psi_scad_g;
    w_scad_g(:,j) = GMVP(Sigma_scad_g);
    
    method = 'mcp';
    [Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_g,Psi_g);
    Sigma_mcp_g = Lambda_mcp_g*Lambda_mcp_g'+ Psi_mcp_g;
    w_mcp_g(:,j) = GMVP(Sigma_mcp_g);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'LS';
    
    method = 'scad';
    [Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_ls,Psi_ls);
    Sigma_scad_ls = Lambda_scad_ls*Lambda_scad_ls'+ Psi_scad_ls;
    w_scad_ls(:,j) = GMVP(Sigma_scad_ls);
    
    method = 'mcp';
    [Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_ls,Psi_ls);
    Sigma_mcp_ls = Lambda_mcp_ls*Lambda_mcp_ls'+ Psi_mcp_ls;
    w_mcp_ls(:,j) = GMVP(Sigma_mcp_ls);
    
end

% sample variance-covariance estimator
Sample_VC = cov(X);
w_sample = GMVP(Sample_VC);

% in-sample estimation of the scalar DCC
[parameters_dcc,~,H_in]=dcc_mvgarch(X,'full');

% out-of-sample univariate GARCH(1,1) processes
h_oos=zeros(size(X_out,1),size(X_out,2));
index = 1;
for jj=1:size(X_out,2)
    univariateparameters=parameters_dcc(index:index+1+1);
    [simulatedata, h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,X_out(:,jj));
    index=index+1+1+1;
end
h_oos = sqrt(h_oos);

% scalar DCC out-of-sample correlation process
[~,Rt_dcc_oos,~,~]=dcc_mvgarch_generate_oos(parameters_dcc,X_out,X,H_in);

% scalar DCC out-of-sample covariance process
Hdcc = zeros(p,p,size(X_out,1));
for t = 1:size(X_out,1)
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

% out-of-sample portfolio weights and returns for the fixed estimators
W = [w_scad_g w_mcp_g w_scad_ls w_mcp_ls w_sample ones(p,1)/p];
T = size(X_out,1); n_candidates = size(W,2);
e = zeros(T,n_candidates);
for t = 1:T
    for j = 1:n_candidates
        e(t,j) = W(:,j)'*X_out(t,:)';
    end
end
% out-of-sample portfolio weights and returns for DCC estimator
wdcc = zeros(p,T); e_dcc = zeros(T,1);
for t = 1:T
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    e_dcc(t) = wdcc(:,t)'*X_out(t,:)';
end

% out-of-sample returns
e = [e e_dcc];

% out-of-sample average portfolio returns, standard deviations and
% information ratios
Results_2 = [252*mean(e);sqrt(252)*std(e);(252*mean(e))./(sqrt(252)*std(e))]';
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% S&P 100 portfolio, out-of-sample period: 01/30/2018 -- 01/23/2020
clear
clc
% load the S&P 100 stock indices: 94 assets
% the data are under the .mat format
% they can also be found in SP100.xls
load data_SP.mat

% transform into log-returns
returns = 100*(log(data(2:end,:))-log(data(1:end-1,:)));
p = size(returns,2);
n_period = 2000;

X = returns(1:n_period,:); X_out = returns(n_period+1:end,:);

factor = [2 3 5 7]; K = length(factor);

w_scad_g = zeros(p,K); w_mcp_g = zeros(p,K);
w_scad_ls = zeros(p,K); w_mcp_ls = zeros(p,K);

for j = 1:K
    
    % selection of the number of factors (user-specified in factor)
    m = factor(j);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% First step under IC5 %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % First step estimation for initialization (factor model under IC5
    % constraint for Lambda and diagonal variance-covariance Psi)
    [Lambda_g,Psi_g] = non_penalized_factor(cov(X),m,'Gaussian');
    
    [Lambda_ls,Psi_ls] = non_penalized_factor(cov(X),m,'LS');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    grid = m*[1 1.05:0.05:5]; gamma = grid*sqrt(log(p*m)/n_period);
    loss = 'Gaussian';
    
    method = 'scad';
    [Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_g,Psi_g);
    Sigma_scad_g = Lambda_scad_g*Lambda_scad_g'+ Psi_scad_g;
    w_scad_g(:,j) = GMVP(Sigma_scad_g);
    
    method = 'mcp';
    [Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_g,Psi_g);
    Sigma_mcp_g = Lambda_mcp_g*Lambda_mcp_g'+ Psi_mcp_g;
    w_mcp_g(:,j) = GMVP(Sigma_mcp_g);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    grid = sqrt(m)*[0.2 0.25:0.05:5]; gamma = grid*sqrt(log(p*m)/n_period);
    loss = 'LS';
    
    method = 'scad';
    [Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_ls,Psi_ls);
    Sigma_scad_ls = Lambda_scad_ls*Lambda_scad_ls'+ Psi_scad_ls;
    w_scad_ls(:,j) = GMVP(Sigma_scad_ls);
    
    method = 'mcp';
    [Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_ls,Psi_ls);
    Sigma_mcp_ls = Lambda_mcp_ls*Lambda_mcp_ls'+ Psi_mcp_ls;
    w_mcp_ls(:,j) = GMVP(Sigma_mcp_ls);
    
end

% sample variance-covariance estimator
Sample_VC = cov(X);
w_sample = GMVP(Sample_VC);

% in-sample estimation of the scalar DCC
[parameters_dcc,~,H_in]=dcc_mvgarch(X,'full'); % set 'CL' instead of 'full' to employ the composite likelihood method

% out-of-sample univariate GARCH(1,1) processes
h_oos=zeros(size(X_out,1),size(X_out,2));
index = 1;
for jj=1:size(X_out,2)
    univariateparameters=parameters_dcc(index:index+1+1);
    [simulatedata, h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,X_out(:,jj));
    index=index+1+1+1;
end
h_oos = sqrt(h_oos);

% scalar DCC out-of-sample correlation process
[~,Rt_dcc_oos,~,~]=dcc_mvgarch_generate_oos(parameters_dcc,X_out,X,H_in);

% scalar DCC out-of-sample covariance process
Hdcc = zeros(p,p,size(X_out,1));
for t = 1:size(X_out,1)
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

% out-of-sample portfolio weights and returns for the fixed estimators
W = [w_scad_g w_mcp_g w_scad_ls w_mcp_ls w_sample ones(p,1)/p];
T = size(X_out,1); n_candidates = size(W,2);
e = zeros(T,n_candidates);
for t = 1:T
    for j = 1:n_candidates
        e(t,j) = W(:,j)'*X_out(t,:)';
    end
end
% out-of-sample portfolio weights and returns for DCC estimator
wdcc = zeros(p,T); e_dcc = zeros(T,1);
for t = 1:T
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    e_dcc(t) = wdcc(:,t)'*X_out(t,:)';
end

% out-of-sample returns
e = [e e_dcc];

% out-of-sample average portfolio returns, standard deviations and
% information ratios
Results_3 = [252*mean(e);sqrt(252)*std(e);(252*mean(e))./(sqrt(252)*std(e))]';
%% S&P 100 portfolio, out-of-sample period: 02/04/2016 -- 01/23/2020
clear
clc
% load the S&P 100 stock indices: 94 assets
% the data are under the .mat format
% they can also be found in SP100.xls
load data_SP.mat

% transform into log-returns
returns = 100*(log(data(2:end,:))-log(data(1:end-1,:)));
p = size(returns,2);
n_period = 1500;

X = returns(1:n_period,:); X_out = returns(n_period+1:end,:);

factor = [2 3 5 7]; K = length(factor);

w_scad_g = zeros(p,K); w_mcp_g = zeros(p,K);
w_scad_ls = zeros(p,K); w_mcp_ls = zeros(p,K);

for j = 1:K
    
    % selection of the number of factors (user-specified in factor)
    m = factor(j);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% First step under IC5 %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % First step estimation for initialization (factor model under IC5
    % constraint for Lambda and diagonal variance-covariance Psi)
    [Lambda_g,Psi_g] = non_penalized_factor(cov(X),m,'Gaussian');
    
    [Lambda_ls,Psi_ls] = non_penalized_factor(cov(X),m,'LS');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    grid = m*[1 1.05:0.05:5]; gamma = grid*sqrt(log(p*m)/n_period);
    loss = 'Gaussian';
    
    method = 'scad';
    [Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_g,Psi_g);
    Sigma_scad_g = Lambda_scad_g*Lambda_scad_g'+ Psi_scad_g;
    w_scad_g(:,j) = GMVP(Sigma_scad_g);
    
    method = 'mcp';
    [Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_g,Psi_g);
    Sigma_mcp_g = Lambda_mcp_g*Lambda_mcp_g'+ Psi_mcp_g;
    w_mcp_g(:,j) = GMVP(Sigma_mcp_g);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    grid = sqrt(m)*[0.2 0.25:0.05:5]; gamma = grid*sqrt(log(p*m)/n_period);
    loss = 'LS';
    
    method = 'scad';
    [Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_ls,Psi_ls);
    Sigma_scad_ls = Lambda_scad_ls*Lambda_scad_ls'+ Psi_scad_ls;
    w_scad_ls(:,j) = GMVP(Sigma_scad_ls);
    
    method = 'mcp';
    [Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor_TS(X,m,gamma,loss,method,Lambda_ls,Psi_ls);
    Sigma_mcp_ls = Lambda_mcp_ls*Lambda_mcp_ls'+ Psi_mcp_ls;
    w_mcp_ls(:,j) = GMVP(Sigma_mcp_ls);
    
end

% sample variance-covariance estimator
Sample_VC = cov(X);
w_sample = GMVP(Sample_VC);

% in-sample estimation of the scalar DCC
[parameters_dcc,~,H_in]=dcc_mvgarch(X,'full'); % set 'CL' instead of 'full' to employ the composite likelihood method

% out-of-sample univariate GARCH(1,1) processes
h_oos=zeros(size(X_out,1),size(X_out,2));
index = 1;
for jj=1:size(X_out,2)
    univariateparameters=parameters_dcc(index:index+1+1);
    [simulatedata, h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,X_out(:,jj));
    index=index+1+1+1;
end
h_oos = sqrt(h_oos);

% scalar DCC out-of-sample correlation process
[~,Rt_dcc_oos,~,~]=dcc_mvgarch_generate_oos(parameters_dcc,X_out,X,H_in);

% scalar DCC out-of-sample covariance process
Hdcc = zeros(p,p,size(X_out,1));
for t = 1:size(X_out,1)
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

% out-of-sample portfolio weights and returns for the fixed estimators
W = [w_scad_g w_mcp_g w_scad_ls w_mcp_ls w_sample ones(p,1)/p];
T = size(X_out,1); n_candidates = size(W,2);
e = zeros(T,n_candidates);
for t = 1:T
    for j = 1:n_candidates
        e(t,j) = W(:,j)'*X_out(t,:)';
    end
end
% out-of-sample portfolio weights and returns for DCC estimator
wdcc = zeros(p,T); e_dcc = zeros(T,1);
for t = 1:T
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    e_dcc(t) = wdcc(:,t)'*X_out(t,:)';
end

% out-of-sample returns
e = [e e_dcc];

% out-of-sample average portfolio returns, standard deviations and
% information ratios
Results_4 = [252*mean(e);sqrt(252)*std(e);(252*mean(e))./(sqrt(252)*std(e))]';