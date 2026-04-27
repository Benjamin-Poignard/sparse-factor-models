function Lambda = filter_FactorLoadings(Lambda)

[p,m] = size(Lambda);
iI=[];
for i=1:m
    if (Lambda(:,i)==zeros(p,1))
        iI=[iI i];
    end
end
Lambda(:,iI)=[];