% The following code replicates the simulation experiment with the five
% different sparsity patterns in the loading matrix Lambda:
% (i) perfect simple structure
% (ii) perfect simple structure with overlaps and non-sparse blocks
% (iii) perfect simple structure with overlaps and sparse blocks
% (iv) general arbitrary sparse structure
% The variance-covariance Psi of the idiosyncratic errors is assumed
% diagonal in cases (i)-(iv)
% (v) general arbitrary sparse structure with non-diagonal sparse variance-
% covariance Psi
% ==> the observations are drawn based on an iid DGP. The vector of
% observations X are simulated in the multivariate Gaussian distribution,
% centered and with variance-covariance matrix Sigma deduced from the
% factor model structure, i.e. Sigma = Lambda x Lambda' + Psi

% The code also replicates the simulation experiment when the observations
% are drawn from a times-series-based DGP, where Lambda and Psi satisfy (v)
%% On the tuning parameter gamma
% the optimal tuning parameter for the penalization of Lambda is selected
% by a K-fold cross-validation (when the data are i.i.d.; K = 5 in the
% simulation experiments).
% gamma is specified as the vector:
%           gamma = grid * sqrt( log(dimension) / n )
% where n is the sample size, dimension is the number of parameters to be
% penalized and grid is a grid of values specified by the user.
% gamma is a user-specified vector; any other vector of values may be
% specified

% in the time-series case, an out-of-sample cross-validation procedure is
% performed
%% Case (i) Sparsity pattern in Lambda: Perfect simple structure
clear
clc

% Select the sample size:
n = 250; % n = 500;

% Select the dimension and number of factors
p = 60; m = 3; set = 20; set2 = 0;
% set2: controls for the size of overlaps: for the perfect simple
% structure, no overlaps so set2 = 0

% p = 60; m = 3; set = 15; set2 = 0;

% p = 120; m = 3; set = 40; set2 = 0;

% p = 120; m = 4; set = 30; set2 = 0;

% p = 180; m = 3; set = 60; set2 = 0;

% Generate a true sparse loading matrix satisfying the perfect simple
% structure with overlaps and non-sparse blocks and verify the number of
% true zero and true non-zero entries
Lambda_check = simulate_perfect_structure_overlap_full_block(p,m,set,set2);
Nzero = check_NZ(vec(Lambda_check),vec(Lambda_check)); sparsity = check_zero(vec(Lambda_check),vec(Lambda_check));

% Simulation of the true sparse loading matrix
Lambda_true = simulate_perfect_structure_overlap_full_block(p,m,set,set2);

% Simulation of the true variance-covariance matrix (diagonal) of the
% idiosyncratic variables
Psi_true = diag(diag((0.5+0.5*rand(p))));

% Construct the true variance-covariance matrix
Sigma_true = Lambda_true*Lambda_true' + Psi_true;

% Generate the observations from a centered normal distribution with
% variance-covariance Sigma_true
X = zeros(n,p);
for ii = 1:n
    X(ii,:) = mvnrnd(zeros(p,1),Sigma_true);
end

% Select the number of folds for the cross-validation to select the optimal
% tuning parameter
fold = 5; grid = (0.1:0.1:4); gamma = grid*sqrt(log(p*m)/n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'Gaussian';

% First step estimation for initialization (factor model under IC5
% constraint for Lambda and diagonal variance-covariance Psi)
[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_scad_g = transform_factor(Lambda_true,Lambda_scad_g); Lambda_scad_g = lambda_modified(Lambda_true,Lambda_scad_g);
Nz_scad_g = check_NZ(vec(Lambda_true),vec(Lambda_scad_g));

% gamma_opt_scad_g: optimal tuning parameter selected by cross-validation
% for the SCAD and Gaussian loss function
% Lambda_scad_g, Psi_scad_g: SCAD penalized sparse factor loading estimator
% and diagonal covariance of idiosyncratic variables, respectively, jointly
% estimated by Gaussian loss function

method = 'mcp';
[Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_mcp_g = transform_factor(Lambda_true,Lambda_mcp_g); Lambda_mcp_g = lambda_modified(Lambda_true,Lambda_mcp_g);
Nz_mcp_g = check_NZ(vec(Lambda_true),vec(Lambda_mcp_g));
% gamma_opt_mcp_g: optimal tuning parameter selected by cross-validation
% for the MCP and Gaussian loss function
% Lambda_mcp_g, Psi_mcp_g: MCP penalized sparse factor loading estimator
% and diagonal covariance of idiosyncratic variables, respectively, jointly
% estimated by Gaussian loss function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'LS';

[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_scad_ls = transform_factor(Lambda_true,Lambda_scad_ls); Lambda_scad_ls = lambda_modified(Lambda_true,Lambda_scad_ls);
Nz_scad_ls = check_NZ(vec(Lambda_true),vec(Lambda_scad_ls));
% gamma_opt_scad_ls: optimal tuning parameter selected by cross-validation
% for the SCAD and least squares loss function
% Lambda_scad_ls, Psi_scad_ls: SCAD penalized sparse factor loading
% estimator and diagonal covariance of idiosyncratic variables,
% respectively, jointly estimated by least squares loss function

method = 'mcp';
[Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_mcp_ls = transform_factor(Lambda_true,Lambda_mcp_ls); Lambda_mcp_ls = lambda_modified(Lambda_true,Lambda_mcp_ls);
Nz_mcp_ls = check_NZ(vec(Lambda_true),vec(Lambda_mcp_ls));
% gamma_opt_mcp_ls: optimal tuning parameter selected by cross-validation
% for the MCP and least squares loss function
% Lambda_mcp_ls, Psi_mcp_ls: MCP penalized sparse factor loading
% estimator and diagonal covariance of idiosyncratic variables,
% respectively, jointly estimated by least squares loss function

% Compute the performance metrics
check = [check_zero(vec(Lambda_true),vec(Lambda_scad_g)) check_zero(vec(Lambda_true),vec(Lambda_mcp_g))  ...
    check_zero(vec(Lambda_true),vec(Lambda_scad_ls)) check_zero(vec(Lambda_true),vec(Lambda_mcp_ls)) ];

% check_prop: proportion of true zero entries correctly recovered
check_prop = check./sparsity;

% check_prop: proportion of true non-zero entries correctly recovered
check_prop2 = [Nz_scad_g Nz_mcp_g Nz_scad_ls Nz_mcp_ls]./Nzero;

% mean-square error defined as \|vec(Lambda_true) - vec(Lambda_est)\|^2_2
% with Lambda_est an estimator of Lambda_true
MSE = [ mse(Lambda_true,Lambda_scad_g) mse(Lambda_true,Lambda_mcp_g) ...
    mse(Lambda_true,Lambda_scad_ls) mse(Lambda_true,Lambda_mcp_ls) ];

%% Case (ii) Sparsity pattern in Lambda: Perfect simple structure with overlaps and non-sparse blocks

clear
clc

% Select the sample size:
n = 250;

% Select the dimension and number of factors
p = 60; m = 3; set = 20; set2 = round(0.5*p/m);
% set2: controls for the size of overlaps: for the perfect simple

% p = 60; m = 4; set = 15; set2 = round(0.5*p/m);
%
% p = 120; m = 3; set = 40; set2 = round(0.5*p/m);
%
% p = 120; m = 4; set = 30; set2 = round(0.5*p/m);
%
% p = 180; m = 3; set = 60; set2 = round(0.5*p/m);

% generate a true sparse loading matrix satisfying the perfect simple
% structure with overlaps and non-sparse blocks to verify the number of
% true zero and true non-zero entries
Lambda_check = simulate_perfect_structure_overlap_full_block(p,m,set,set2);
Nzero = check_NZ(vec(Lambda_check),vec(Lambda_check)); sparsity = check_zero(vec(Lambda_check),vec(Lambda_check));

% Simulation of the true sparse loading matrix
Lambda_true = simulate_perfect_structure_overlap_full_block(p,m,set,set2);

% Simulation of the true variance-covariance matrix (diagonal) of the
% idiosyncratic variables
Psi_true = diag(diag((0.5+0.5*rand(p))));

% Construct the true variance-covariance matrix
Sigma_true = Lambda_true*Lambda_true' + Psi_true;

% Generate the observations from a centered normal distribution with
% variance-covariance Sigma_true
X = zeros(n,p);
for ii = 1:n
    X(ii,:) = mvnrnd(zeros(p,1),Sigma_true);
end

% Select the number of folds for the cross-validation to select the optimal
% tuning parameter
fold = 5; grid = (0.1:0.1:4); gamma = grid*sqrt(log(p*m)/n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'Gaussian';

[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_scad_g = transform_factor(Lambda_true,Lambda_scad_g); Lambda_scad_g = lambda_modified(Lambda_true,Lambda_scad_g);
Nz_scad_g = check_NZ(vec(Lambda_true),vec(Lambda_scad_g));

method = 'mcp';
[Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_mcp_g = transform_factor(Lambda_true,Lambda_mcp_g); Lambda_mcp_g = lambda_modified(Lambda_true,Lambda_mcp_g);
Nz_mcp_g = check_NZ(vec(Lambda_true),vec(Lambda_mcp_g));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'LS';

[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_scad_ls = transform_factor(Lambda_true,Lambda_scad_ls); Lambda_scad_ls = lambda_modified(Lambda_true,Lambda_scad_ls);
Nz_scad_ls = check_NZ(vec(Lambda_true),vec(Lambda_scad_ls));

method = 'mcp';
[Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_mcp_ls = transform_factor(Lambda_true,Lambda_mcp_ls); Lambda_mcp_ls = lambda_modified(Lambda_true,Lambda_mcp_ls);
Nz_mcp_ls = check_NZ(vec(Lambda_true),vec(Lambda_mcp_ls));

% Compute the performance metrics
check = [check_zero(vec(Lambda_true),vec(Lambda_scad_g)) check_zero(vec(Lambda_true),vec(Lambda_mcp_g))  ...
    check_zero(vec(Lambda_true),vec(Lambda_scad_ls)) check_zero(vec(Lambda_true),vec(Lambda_mcp_ls)) ];

% check_prop: proportion of true zero entries correctly recovered
check_prop = check./sparsity;

% check_prop: proportion of true non-zero entries correctly recovered
check_prop2 = [Nz_scad_g Nz_mcp_g Nz_scad_ls Nz_mcp_ls]./Nzero;

% mean-square error defined as \|vec(Lambda_true) - vec(Lambda_est)\|^2_2
% with Lambda_est an estimator of Lambda_true
MSE = [ mse(Lambda_true,Lambda_scad_g) mse(Lambda_true,Lambda_mcp_g) ...
    mse(Lambda_true,Lambda_scad_ls) mse(Lambda_true,Lambda_mcp_ls) ];

%% Case (iii) Sparsity pattern in Lambda: Perfect simple structure with overlaps and sparse blocks

clear
clc

% Select the sample size:
n = 250;

% Select the dimension and number of factors
p = 60; m = 3; dimension = p*m; set = 20; set2 = round(0.5*p/m);
sparsity = round(dimension*0.7); threshold = 0.5;
% set2: controls for the size of overlaps: for the perfect simple

% p = 60; m = 4; dimension = p*m; set = 15; set2 = round(0.5*p/m);
% sparsity = round(dimension*0.7); threshold = 0.3;


% p = 120; m = 3; dimension = p*m; set = 40; set2 = round(0.5*p/m);
% sparsity = round(dimension*0.7); threshold = 0.3;


% p = 120; m = 4; dimension = p*m; set = 30; set2 = round(0.5*p/m);
% sparsity = round(dimension*0.7); threshold = 0.2;


% p = 180; m = 3; dimension = p*m; set = 60; set2 = round(0.5*p/m);
% sparsity = round(dimension*0.7); threshold = 0.3;

Nzero = dimension - sparsity;

% Simulation of the true sparse loading matrix
Lambda_true = simulate_perfect_structure_overlap(p,m,set,set2,threshold,sparsity);

% Simulation of the true variance-covariance matrix (diagonal) of the
% idiosyncratic variables
Psi_true = diag(diag((0.5+0.5*rand(p))));

% Construct the true variance-covariance matrix
Sigma_true = Lambda_true*Lambda_true' + Psi_true;

% Generate the observations from a centered normal distribution with
% variance-covariance Sigma_true
X = zeros(n,p);
for ii = 1:n
    X(ii,:) = mvnrnd(zeros(p,1),Sigma_true);
end

% Select the number of folds for the cross-validation to select the optimal
% tuning parameter
fold = 5; grid = (0.1:0.1:4); gamma = grid*sqrt(log(p*m)/n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'Gaussian';

[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_scad_g = transform_factor(Lambda_true,Lambda_scad_g); Lambda_scad_g = lambda_modified(Lambda_true,Lambda_scad_g);
Nz_scad_g = check_NZ(vec(Lambda_true),vec(Lambda_scad_g));

method = 'mcp';
[Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_mcp_g = transform_factor(Lambda_true,Lambda_mcp_g); Lambda_mcp_g = lambda_modified(Lambda_true,Lambda_mcp_g);
Nz_mcp_g = check_NZ(vec(Lambda_true),vec(Lambda_mcp_g));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'LS';

[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_scad_ls = transform_factor(Lambda_true,Lambda_scad_ls); Lambda_scad_ls = lambda_modified(Lambda_true,Lambda_scad_ls);
Nz_scad_ls = check_NZ(vec(Lambda_true),vec(Lambda_scad_ls));

method = 'mcp';
[Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_mcp_ls = transform_factor(Lambda_true,Lambda_mcp_ls); Lambda_mcp_ls = lambda_modified(Lambda_true,Lambda_mcp_ls);
Nz_mcp_ls = check_NZ(vec(Lambda_true),vec(Lambda_mcp_ls));

% Compute the performance metrics
check = [check_zero(vec(Lambda_true),vec(Lambda_scad_g)) check_zero(vec(Lambda_true),vec(Lambda_mcp_g))  ...
    check_zero(vec(Lambda_true),vec(Lambda_scad_ls)) check_zero(vec(Lambda_true),vec(Lambda_mcp_ls)) ];

% check_prop: proportion of true zero entries correctly recovered
check_prop = check./sparsity;

% check_prop: proportion of true non-zero entries correctly recovered
check_prop2 = [Nz_scad_g Nz_mcp_g Nz_scad_ls Nz_mcp_ls]./Nzero;

% mean-square error defined as \|vec(Lambda_true) - vec(Lambda_est)\|^2_2
% with Lambda_est an estimator of Lambda_true
MSE = [ mse(Lambda_true,Lambda_scad_g) mse(Lambda_true,Lambda_mcp_g) ...
    mse(Lambda_true,Lambda_scad_ls) mse(Lambda_true,Lambda_mcp_ls) ];


%% Case (iv) Sparsity pattern in Lambda: General arbitrary sparse structure, diagonal Psi

% general rule: 85% sparsity in Lambda

clear
clc
% Select the sample size:
n = 250;

% Select the dimension and number of factors
p = 60; m = 3;
% p = 120; m = 3 or 4
% p = 180; m = 3;

% total number of parameters and sparsity degree in Lambda
dimension = p*m; sparsity = round(dimension*0.85); threshold = 0.8;
Nzero = dimension - sparsity;

% Simulation of the true sparse loading matrix
Lambda_true = simulate_general_structure(p,m,threshold,sparsity);

% Simulation of the true variance-covariance matrix (diagonal) of the
% idiosyncratic variables
Psi_true = diag(diag((0.5+0.5*rand(p))));

% Construct the true variance-covariance matrix
Sigma_true = Lambda_true*Lambda_true' + Psi_true;

% Generate the observations from a centered normal distribution with
% variance-covariance Sigma_true
X = zeros(n,p);
for ii = 1:n
    X(ii,:) = mvnrnd(zeros(p,1),Sigma_true);
end

% Select the number of folds for the cross-validation to select the optimal
% tuning parameter
fold = 5; grid = (0.1:0.1:4); gamma = grid*sqrt(log(p*m)/n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'Gaussian';

[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_scad_g = transform_factor(Lambda_true,Lambda_scad_g); Lambda_scad_g = lambda_modified(Lambda_true,Lambda_scad_g);
Nz_scad_g = check_NZ(vec(Lambda_true),vec(Lambda_scad_g));

method = 'mcp';
[Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_mcp_g = transform_factor(Lambda_true,Lambda_mcp_g); Lambda_mcp_g = lambda_modified(Lambda_true,Lambda_mcp_g);
Nz_mcp_g = check_NZ(vec(Lambda_true),vec(Lambda_mcp_g));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'LS';

[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_scad_ls = transform_factor(Lambda_true,Lambda_scad_ls); Lambda_scad_ls = lambda_modified(Lambda_true,Lambda_scad_ls);
Nz_scad_ls = check_NZ(vec(Lambda_true),vec(Lambda_scad_ls));

method = 'mcp';
[Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_mcp_ls = transform_factor(Lambda_true,Lambda_mcp_ls); Lambda_mcp_ls = lambda_modified(Lambda_true,Lambda_mcp_ls);
Nz_mcp_ls = check_NZ(vec(Lambda_true),vec(Lambda_mcp_ls));

% Compute the performance metrics
check = [check_zero(vec(Lambda_true),vec(Lambda_scad_g)) check_zero(vec(Lambda_true),vec(Lambda_mcp_g))  ...
    check_zero(vec(Lambda_true),vec(Lambda_scad_ls)) check_zero(vec(Lambda_true),vec(Lambda_mcp_ls)) ];

% check_prop: proportion of true zero entries correctly recovered
check_prop = check./sparsity;

% check_prop: proportion of true non-zero entries correctly recovered
check_prop2 = [Nz_scad_g Nz_mcp_g Nz_scad_ls Nz_mcp_ls]./Nzero;

% mean-square error defined as \|vec(Lambda_true) - vec(Lambda_est)\|^2_2
% with Lambda_est an estimator of Lambda_true
MSE = [ mse(Lambda_true,Lambda_scad_g) mse(Lambda_true,Lambda_mcp_g) ...
    mse(Lambda_true,Lambda_scad_ls) mse(Lambda_true,Lambda_mcp_ls) ];

%% Case (v) Sparsity pattern in Lambda: General arbitrary sparse structure with sparse non-diagonal Psi

% general rule: 85% sparsity in Lambda

clear
clc
% Select the sample size:
n = 250;

% Select the dimension and number of factors
p = 60; m = 3;
% p = 120; m = 3 or 4
% p = 180; m = 3;

% total number of parameters and sparsity degree in Lambda
dimension = p*m; sparsity = round(dimension*0.85); threshold = 0.8;
Nzero = dimension - sparsity;

% total number of distinct parameters and sparsity degree in Psi
dim_psi = p*(p-1)/2; sparsity_psi = dim_psi*0.75;

% Simulation of the true sparse loading matrix
Lambda_true = simulate_general_structure(p,m,threshold,sparsity);

% Simulation of the true variance-covariance matrix (sparse non-diagonal)
% of the idiosyncratic variables
cond = true;
while cond
    iE2=0;
    while (iE2==0)
        Psi_true = full(sprandsym(p,1-sparsity_psi/dim_psi))./10;
        for ll=1:p
            Psi_true(ll,ll) = 0;
        end
        Psi_true = diag(0.5+0.5*rand(p,1)) + Psi_true;
        
        iE2 = min(eig(Psi_true))>0.01;
    end
    K = vech_on(Psi_true,p); K(abs(K)>0)=[];
    cond=(length(K)<sparsity_psi)&&(length(K)>sparsity_psi);
end

%%%%%%%%%%%%%%%%%%%%%% p = 120 %%%%%%%%%%%%%%%%%%%%%
% cond = true;
% while cond
%     iE2=0;
%     while (iE2==0)
%         Psi_true = full(sprandsym(p,1-sparsity_psi/dim_psi))./15;
%         for ll=1:p
%             Psi_true(ll,ll) = 0;
%         end
%         Psi_true = diag(0.5+0.5*rand(p,1)) + Psi_true;
%
%         iE2 = min(eig(Psi_true))>0.01;
%     end
%     K = vech_on(Psi_true,p); K(abs(K)>0)=[];
%     cond=(length(K)<sparsity_psi)&&(length(K)>sparsity_psi);
% end

%%%%%%%%%%%%%%%%%%%%%% p = 180 %%%%%%%%%%%%%%%%%%%%%
% cond = true;
% while cond
%     iE2=0;
%     while (iE2==0)
%         Psi_true = full(sprandsym(p,1-sparsity_psi/dim_psi))./20;
%         for ll=1:p
%             Psi_true(ll,ll) = 0;
%         end
%         Psi_true = diag(0.5+0.5*rand(p,1)) + Psi_true;
%
%         iE2 = min(eig(Psi_true))>0.01;
%     end
%     K = vech_on(Psi_true,p); K(abs(K)>0)=[];
%     cond=(length(K)<sparsity_psi)&&(length(K)>sparsity_psi);
% end


% Construct the true variance-covariance matrix
Sigma_true = Lambda_true*Lambda_true' + Psi_true;

% Generate the observations from a centered normal distribution with
% variance-covariance Sigma_true
X = zeros(n,p);
for ii = 1:n
    X(ii,:) = mvnrnd(zeros(p,1),Sigma_true);
end

% Select the number of folds for the cross-validation to select the optimal
% tuning parameter
fold = 5; grid = (0.1:0.1:4); gamma = grid*sqrt(log(p*m)/n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'Gaussian';

[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_scad_g = transform_factor(Lambda_true,Lambda_scad_g); Lambda_scad_g = lambda_modified(Lambda_true,Lambda_scad_g);
Nz_scad_g = check_NZ(vec(Lambda_true),vec(Lambda_scad_g));

method = 'mcp';
[Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_mcp_g = transform_factor(Lambda_true,Lambda_mcp_g); Lambda_mcp_g = lambda_modified(Lambda_true,Lambda_mcp_g);
Nz_mcp_g = check_NZ(vec(Lambda_true),vec(Lambda_mcp_g));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'LS';

[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_scad_ls = transform_factor(Lambda_true,Lambda_scad_ls); Lambda_scad_ls = lambda_modified(Lambda_true,Lambda_scad_ls);
Nz_scad_ls = check_NZ(vec(Lambda_true),vec(Lambda_scad_ls));

method = 'mcp';
[Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor(X,m,loss,gamma,method,fold,Lambda_first,Psi_first);
Lambda_mcp_ls = transform_factor(Lambda_true,Lambda_mcp_ls); Lambda_mcp_ls = lambda_modified(Lambda_true,Lambda_mcp_ls);
Nz_mcp_ls = check_NZ(vec(Lambda_true),vec(Lambda_mcp_ls));

% Compute the performance metrics
check = [check_zero(vec(Lambda_true),vec(Lambda_scad_g)) check_zero(vec(Lambda_true),vec(Lambda_mcp_g))  ...
    check_zero(vec(Lambda_true),vec(Lambda_scad_ls)) check_zero(vec(Lambda_true),vec(Lambda_mcp_ls)) ];

% check_prop: proportion of true zero entries correctly recovered
check_prop = check./sparsity;

% check_prop: proportion of true non-zero entries correctly recovered
check_prop2 = [Nz_scad_g Nz_mcp_g Nz_scad_ls Nz_mcp_ls]./Nzero;

% mean-square error defined as \|vec(Lambda_true) - vec(Lambda_est)\|^2_2
% with Lambda_est an estimator of Lambda_true
MSE = [ mse(Lambda_true,Lambda_scad_g) mse(Lambda_true,Lambda_mcp_g) ...
    mse(Lambda_true,Lambda_scad_ls) mse(Lambda_true,Lambda_mcp_ls) ];


%% Time-series based experiment
% Lambda and Psi satisfy Case (v)
clear
clc
% Select the sample size:
n = 250;

% Select the dimension and number of factors
p = 60; m = 3;
% p = 120; m = 3 or 4
% p = 180; m = 3;

% total number of parameters and sparsity degree in Lambda
dimension = p*m; sparsity = round(dimension*0.85); threshold = 0.8;
Nzero = dimension - sparsity;

% total number of distinct parameters and sparsity degree in Psi
dim_psi = p*(p-1)/2; sparsity_psi = dim_psi*0.75;

% Simulation of the true sparse loading matrix
Lambda_true = simulate_general_structure(p,m,threshold,sparsity);

% Generation of the factors
% Autoregressive process (diagonal Phi, so each factor follows its own
% AR(1) process)
iE=0;
while (iE==0)
    Phi = diag(0.5+0.35*rand(m,1));
    iE = max(abs(eig(Phi)))<0.99;
end
% Diagonal var-cov for the error term in the autoregressive process for
% the factors
Sigma_u = diag(ones(m,1)-diag(Phi).^2);
% AR dynamic for the factors under Gaussian errors
F = zeros(m,n+1); F(:,1) = (mvnrnd(zeros(m,1),Sigma_u))';
for ii = 2:n+1
    F(:,ii) = Phi*F(:,ii-1) + (mvnrnd(zeros(m,1),Sigma_u))';
end
% remove the initial value of the factors
F = F(:,2:end);

% Generaton of the true sparse var-cov of idiosyncratic errors
cond = true;
while cond
    iE2=0;
    while (iE2==0)
        Psi_true = full(sprandsym(p,1-sparsity_psi/dim_psi))./10;
        for ll=1:p
            Psi_true(ll,ll) = 0;
        end
        Psi_true = diag(0.5+0.5*rand(p,1)) + Psi_true;
        
        iE2 = min(eig(Psi_true))>0.01;
    end
    K = vech_on(Psi_true,p); K(abs(K)>0)=[];
    cond=(length(K)<sparsity_psi)&&(length(K)>sparsity_psi);
end

%%%%%%%%%%%%%%%%%%%%%% p = 120 %%%%%%%%%%%%%%%%%%%%%
% cond = true;
% while cond
%     iE2=0;
%     while (iE2==0)
%         Psi_true = full(sprandsym(p,1-sparsity_psi/dim_psi))./15;
%         for ll=1:p
%             Psi_true(ll,ll) = 0;
%         end
%         Psi_true = diag(0.5+0.5*rand(p,1)) + Psi_true;
%
%         iE2 = min(eig(Psi_true))>0.01;
%     end
%     K = vech_on(Psi_true,p); K(abs(K)>0)=[];
%     cond=(length(K)<sparsity_psi)&&(length(K)>sparsity_psi);
% end

%%%%%%%%%%%%%%%%%%%%%% p = 180 %%%%%%%%%%%%%%%%%%%%%
% cond = true;
% while cond
%     iE2=0;
%     while (iE2==0)
%         Psi_true = full(sprandsym(p,1-sparsity_psi/dim_psi))./20;
%         for ll=1:p
%             Psi_true(ll,ll) = 0;
%         end
%         Psi_true = diag(0.5+0.5*rand(p,1)) + Psi_true;
%
%         iE2 = min(eig(Psi_true))>0.01;
%     end
%     K = vech_on(Psi_true,p); K(abs(K)>0)=[];
%     cond=(length(K)<sparsity_psi)&&(length(K)>sparsity_psi);
% end

% Generation of the vector of observations X
X = zeros(p,n);
for ii = 1:n
    X(:,ii) = Lambda_true*F(:,ii)+(mvnrnd(zeros(p,1),Psi_true))';
end
X = X';

% Select the number of folds for the cross-validation to select the optimal
% tuning parameter
gamma = (0.01:0.1:5)*sqrt(log(p*m)/n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'Gaussian';

[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor_TS(X,m,loss,gamma,method,Lambda_first,Psi_first);
Lambda_scad_g = transform_factor(Lambda_true,Lambda_scad_g); Lambda_scad_g = lambda_modified(Lambda_true,Lambda_scad_g);
Nz_scad_g = check_NZ(vec(Lambda_true),vec(Lambda_scad_g));

method = 'mcp';
[Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor_TS(X,m,loss,gamma,method,Lambda_first,Psi_first);
Lambda_mcp_g = transform_factor(Lambda_true,Lambda_mcp_g); Lambda_mcp_g = lambda_modified(Lambda_true,Lambda_mcp_g);
Nz_mcp_g = check_NZ(vec(Lambda_true),vec(Lambda_mcp_g));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Least squares loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
loss = 'LS';

[Lambda_first,Psi_first] = non_penalized_factor(X,m,loss);

method = 'scad';
[Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor_TS(X,m,loss,gamma,method,Lambda_first,Psi_first);
Lambda_scad_ls = transform_factor(Lambda_true,Lambda_scad_ls); Lambda_scad_ls = lambda_modified(Lambda_true,Lambda_scad_ls);
Nz_scad_ls = check_NZ(vec(Lambda_true),vec(Lambda_scad_ls));

method = 'mcp';
[Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor_TS(X,m,loss,gamma,method,Lambda_first,Psi_first);
Lambda_mcp_ls = transform_factor(Lambda_true,Lambda_mcp_ls); Lambda_mcp_ls = lambda_modified(Lambda_true,Lambda_mcp_ls);
Nz_mcp_ls = check_NZ(vec(Lambda_true),vec(Lambda_mcp_ls));

% Compute the performance metrics
check = [check_zero(vec(Lambda_true),vec(Lambda_scad_g)) check_zero(vec(Lambda_true),vec(Lambda_mcp_g))  ...
    check_zero(vec(Lambda_true),vec(Lambda_scad_ls)) check_zero(vec(Lambda_true),vec(Lambda_mcp_ls)) ];

% check_prop: proportion of true zero entries correctly recovered
check_prop = check./sparsity;

% check_prop: proportion of true non-zero entries correctly recovered
check_prop2 = [Nz_scad_g Nz_mcp_g Nz_scad_ls Nz_mcp_ls]./Nzero;

% mean-square error defined as \|vec(Lambda_true) - vec(Lambda_est)\|^2_2
% with Lambda_est an estimator of Lambda_true
MSE = [ mse(Lambda_true,Lambda_scad_g) mse(Lambda_true,Lambda_mcp_g) ...
    mse(Lambda_true,Lambda_scad_ls) mse(Lambda_true,Lambda_mcp_ls) ];
