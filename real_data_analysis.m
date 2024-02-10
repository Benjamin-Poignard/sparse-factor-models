clear
clc
% Select the portfolio: 'MSCI' or 'SP100'
portfolio = 'MSCI'; scale = 100;

switch portfolio
    
    case 'MSCI'
        
        % MSCI portfolio, out-of-sample period:
        Table = readtable('MSCI.xls');
        data_MSCI = Table{1:end,[2:end]};
        % transform into log-returns
        returns = scale*(log(data_MSCI(2:end,:))-log(data_MSCI(1:end-1,:)));
        p = size(returns,2);
        n_period = 3000;
        % estimation method for DCC
        method_dcc = 'full';
        
    case 'SP100'
        
        % S&P 100 portfolio, out-of-sample period:
        % load the S&P 100 stock indices: 94 assets
        % the data are under the .mat format
        % they can also be found in SP100.xls
        load data_SP.mat
        % transform into log-returns
        returns = scale*(log(data(2:end,:))-log(data(1:end-1,:)));
        p = size(returns,2);
        n_period = 1100;
        % estimation method for DCC
        method_dcc = 'full';
        
end

X = returns(1:n_period,:); % in-sample data
X_out = returns(n_period+1:end,:); % out-of-sample data
T_out = size(X_out,1);

% Selection of the number of factors following Onatski's method
number_factors = factor_selection(X,6);

% Specification of the number of factors for estimation
factor = [1 2 3 4 5]; 

K = length(factor);

w_scad_g = zeros(p,K); w_mcp_g = zeros(p,K);
w_scad_ls = zeros(p,K); w_mcp_ls = zeros(p,K);
w_saf = zeros(p,K);

ranking_scad_g_ed = zeros(T_out,K); ranking_mcp_g_ed = zeros(T_out,K);
ranking_scad_ls_ed = zeros(T_out,K); ranking_mcp_ls_ed = zeros(T_out,K);
ranking_saf_ed = zeros(T_out,K);

ranking_scad_g_fn = zeros(T_out,K); ranking_mcp_g_fn = zeros(T_out,K);
ranking_scad_ls_fn = zeros(T_out,K); ranking_mcp_ls_fn = zeros(T_out,K);
ranking_saf_fn = zeros(T_out,K);

for j = 1:K
    
    % selection of the number of factors (user-specified in factor)
    m = factor(j);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% First step under IC5 %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % First step estimation for initialization (factor model under IC5
    % constraint for Lambda and diagonal variance-covariance Psi)
    [Lambda_g,Psi_g] = non_penalized_factor(X,m,'Gaussian');
    
    [Lambda_ls,Psi_ls] = non_penalized_factor(X,m,'LS');
    
    grid = m*[0.1 0.15:0.05:10]; gamma = grid*sqrt(log(p*m)/n_period);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'Gaussian';
    
    method = 'scad';
    [Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor_TS(X,m,loss,gamma,method,Lambda_g,Psi_g);
    Sigma_scad_g = Lambda_scad_g*Lambda_scad_g'+ Psi_scad_g;
    w_scad_g(:,j) = GMVP(Sigma_scad_g);
    [ED,FN] = ranking_metric(X_out,Sigma_scad_g,scale); ranking_scad_g_ed(:,j) = ED; ranking_scad_g_fn(:,j) = FN;
    
    method = 'mcp';
    [Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor_TS(X,m,loss,gamma,method,Lambda_g,Psi_g);
    Sigma_mcp_g = Lambda_mcp_g*Lambda_mcp_g'+ Psi_mcp_g;
    w_mcp_g(:,j) = GMVP(Sigma_mcp_g);
    [ED,FN] = ranking_metric(X_out,Sigma_mcp_g,scale); ranking_mcp_g_ed(:,j) = ED; ranking_mcp_g_fn(:,j) = FN;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = 'LS';
    
    method = 'scad';
    [Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor_TS(X,m,loss,gamma,method,Lambda_ls,Psi_ls);
    Sigma_scad_ls = Lambda_scad_ls*Lambda_scad_ls'+ Psi_scad_ls;
    w_scad_ls(:,j) = GMVP(Sigma_scad_ls);
    [ED,FN] = ranking_metric(X_out,Sigma_scad_ls,scale); ranking_scad_ls_ed(:,j) = ED; ranking_scad_ls_fn(:,j) = FN;
    
    method = 'mcp';
    [Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor_TS(X,m,loss,gamma,method,Lambda_ls,Psi_ls);
    Sigma_mcp_ls = Lambda_mcp_ls*Lambda_mcp_ls'+ Psi_mcp_ls;
    w_mcp_ls(:,j) = GMVP(Sigma_mcp_ls);
    [ED,FN] = ranking_metric(X_out,Sigma_mcp_ls,scale); ranking_mcp_ls_ed(:,j) = ED; ranking_mcp_ls_fn(:,j) = FN;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SAF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    grid = m*[0.001 0.002:0.005:10]; gamma = grid*sqrt(log(p*(p+1)/2)/n_period);
    
    [Lambda_saf,gamma_opt_saf,Psi_saf] = approx_factor_TS(X,m,gamma);
    Sigma_saf = Lambda_saf*Lambda_saf'+ Psi_saf;
    w_saf(:,j) = GMVP(Sigma_saf);
    [ED,FN] = ranking_metric(X_out,Sigma_saf,scale); ranking_saf_ed(:,j) = ED; ranking_saf_fn(:,j) = FN;
    
end

% sample variance-covariance estimator-based GMVP
w_sample = GMVP(cov(X)); [ranking_sample_ed,ranking_sample_fn] = ranking_metric(X_out,cov(X),scale);
% geometric-inverse shrinkage (GIS) estimator-based GMVP
w_gis = GMVP(GIS(X)); [ranking_gis_ed,ranking_gis_fn] = ranking_metric(X_out,GIS(X),scale);
% one-factor market model shrinkage estimator-based GMVP
w_covMarket = GMVP(covMarket(X)); [ranking_covM_ed,ranking_covM_fn] = ranking_metric(X_out,covMarket(X),scale); ranking_covM = [ED,FN];

% scalar DCC variance-covariance process
[~,~,Hdcc] = dcc_mvgarch_for(returns,method_dcc,n_period); [ranking_dcc_ed,ranking_dcc_fn] = ranking_metric(X_out,Hdcc,scale);

% out-of-sample portfolio weights and returns for the fixed estimators
W = [w_scad_g w_mcp_g w_scad_ls w_mcp_ls w_saf w_sample w_gis w_covMarket];
T = size(X_out,1); n_candidates = size(W,2);
e_gmvp = zeros(T,n_candidates);
for t = 1:T
    for j = 1:n_candidates
        e_gmvp(t,j) = W(:,j)'*X_out(t,:)';
    end
end
% out-of-sample portfolio weights and returns for DCC estimator
wdcc = zeros(p,T); e_dcc = zeros(T,1);
for t = 1:T
    wdcc(:,t)= GMVP(Hdcc(:,:,t));
    e_dcc(t) = wdcc(:,t)'*X_out(t,:)';
end

% out-of-sample GMVP returns
e_gmvp = [e_gmvp e_dcc];

% out-of-sample average portfolio returns, standard deviations and
% information ratios
Results = [252*mean(e_gmvp);sqrt(252)*std(e_gmvp);(252*mean(e_gmvp))./(sqrt(252)*std(e_gmvp))]';
E_gmvp = e_gmvp.^2;

E_ed = [ranking_scad_g_ed ranking_mcp_g_ed ranking_scad_ls_ed ranking_mcp_ls_ed...
    ranking_saf_ed ranking_sample_ed ranking_gis_ed ranking_covM_ed];
E_ed_av = 252*mean(E_ed);

E_fn = [ranking_scad_g_fn ranking_mcp_g_fn ranking_scad_ls_fn ranking_mcp_ls_fn...
    ranking_saf_fn ranking_sample_fn ranking_gis_fn ranking_covM_fn];
E_fn_av = 252*mean(E_fn);

% Model confidence set (MCS) at the 10% significance level with block
% bootstrap (see Hansen et al. (2003)) with 10,000 replications

% Model Confidence Test GMVP loss
[includedR, pvalsR_gmvp, excluded] = mcs(E_gmvp,0.1,10000,12);
excl_select_model_gmvp = [excluded ;includedR];
[excl_select_model_gmvp pvalsR_gmvp]

% Model Confidence Test Euclidean distance loss
[includedR, pvalsR_ed, excluded] = mcs(E_ed,0.1,10000,12);
excl_select_model_ed = [excluded ;includedR];
[excl_select_model_ed pvalsR_ed]

% Model Confidence Test Frobenius norm loss
[includedR, pvalsR_fn, excluded] = mcs(E_fn,0.1,10000,12);
excl_select_model_fn = [excluded ;includedR];
[excl_select_model_fn pvalsR_fn]