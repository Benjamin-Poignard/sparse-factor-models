function [c,ceq] = constr_psi(x)

% Constraints on the parameters of the variance-covariance matrix of the
% idiosyncratic errors (diagonal elements must be positive)

c = -x+0.001; 
ceq = [];
