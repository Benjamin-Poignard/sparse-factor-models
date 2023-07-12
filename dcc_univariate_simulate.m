function [simulatedata, H] = dcc_univariate_simulate(parameters,p,q,data)

if isempty(q)
   m=p;
else
   m  =  max(p,q);   
end

[t,k]=size(data);
UncondStd =  sqrt(cov(data));
h=UncondStd.^2*ones(t+m,1);
data=[UncondStd*ones(m,1);data];
RandomNums=randn(t+m,1);
T=size(data,1);

h=garchcore(data,parameters,UncondStd^2,p,q,m,T);
% for t = (m + 1):T
%     h(t) = parameters' * [1 ; data(t-(1:p)).^2;  h(t-(1:q)) ];
% end

simulatedata=data((m+1):T);
H=h((m + 1):T);
