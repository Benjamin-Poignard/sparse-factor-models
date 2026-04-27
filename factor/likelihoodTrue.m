function f=likelihoodTrue(Sy,P,Sigmau,Lambda,lambda)

% f=log|(LL'+Sigmau)|/N+tr(Sy(LL'+Sigmau)^{-1})/N+ penalty

A=Lambda*Lambda'+Sigmau;
f1=log(abs(det(A)));
f2=trace(Sy*inv(A));
f3=0;

B=Sigmau.*P;
for i=1:length(B(1,:))
    f3=f3+norm(B(i,:),1)*lambda;
end;

f=f1+f2+f3;
f=f/length(Sy(1,:));