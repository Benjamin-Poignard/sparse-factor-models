function H = proj_defpos(M)

% Projection on the space of positive-definite matrix if required

% Input:
%       - M: square symmetric matrix
% Output:
%       - H: square symmetric and positive-definite matrix obtained by
%       setting the non-negative eigenvalues to 0.01;

if min(eig(M))<eps
    [P,K] = eig(M); K = diag(K);
    K = subplus(K)+0.01; K = diag(K);
    H = P*K*P';
else
    H = M;
end