function Lambda = simulate_general_structure(p,m,threshold,sparsity)

cond = true; parameter = zeros(p,m);

while cond
    
    for kk=1:p
        for jj=1:m
            cond_dist=true;
            while cond_dist
                param = -2+(2-(-2))*rand(1);
                cond_dist=(param>-0.5 & param<0.5);
            end
            parameter(kk,jj)=param;
        end
    end
    Lambda = (rand(p,m)>threshold).*parameter;
    L = vec(Lambda);
    count = 0;
    for ii = 1:p*m
        if (L(ii)==0)
            count = count+1;
        else
            count = count+0;
        end
    end
    cond = (count>sparsity)||(count<sparsity);
end

