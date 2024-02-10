function L=likelihoodlambda(S,Lambda,Psi,loss)

Sigma=Lambda*Lambda'+Psi;
p = size(S,2);

switch loss
    case 'Gaussian'
        L = (log(abs(det(Sigma)))+trace(S/Sigma))/p;
    case 'LS'
        L = norm(S-Sigma,'fro')^2/p;
end