function [c,ceq] = constr(x,p,m)

% parameter constraint for the first step joint estimation of the factor 
% model parameters (IC5 for Lambda, diagonal assumption for Psi)
% Inputs:
%        - x: full vector of parameters of the factor model
%        - p: dimension of the vector of observations
%        - m: number of factors

% Output: positivity constraint on the diagonal elements of Psi 
c = -x(p*m+1:end)+0.001; 
ceq = [];
