% principal components with normalization F'F/T=I

% X is observed

% r is the true number of true factors

% F is T by r matrix of true factors

% Lambda N by r is the true loading matrix

% C=F*Lambda' T by N is the true common component

% chat is the estimated common component

function [ehat,fhat,lambda,ss]=pc_T(y,nfac);

[bigt,bign]=size(y);

yy=y*y';

[Fhat0,eigval,Fhat1]=svd(yy); %for semi-def symmetric matrix, same as eig value decomposition but sorts in descending order

fhat=Fhat0(:,1:nfac)*sqrt(bigt);

lambda=y'*fhat/bigt;

%chi2=fhat*lambda';

%diag(lambda'*lambda)

%diag(fhat'*fhat)                % this should equal the largest eigenvalues

%sum(diag(eigval(1:nfac,1:nfac)))/sum(diag(eigval))

%mean(var(chi2))                 % this should equal decomposition of variance



ehat=y-fhat*lambda';



ve2=sum(ehat'.*ehat')'/bign;

ss=diag(eigval);

