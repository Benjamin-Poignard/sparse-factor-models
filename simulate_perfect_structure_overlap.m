function Lambda = simulate_perfect_structure_overlap(p,m,set,set2,threshold,sparsity)

Lambda_temp = zeros(p,m);
if (m==2)
    Lambda_temp(:,1) = [ones(set,1);zeros(p-set,1)];
    Lambda_temp(:,m) = [zeros(p-set-set2,1);ones(set+set2,1)];
else
    Lambda_temp(:,1) = [ones(set,1);zeros(p-set,1)];
    for kk = 2:m-1
        Lambda_temp(:,kk) = [zeros((kk-1)*set-set2,1);ones(set+set2,1);zeros(p-(kk)*set,1)];
    end
    Lambda_temp(:,m) = [zeros(p-set-set2,1);ones(set+set2,1)];
end


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
    Lambda = (rand(p,m)>threshold).*Lambda_temp.*parameter; L = vec(Lambda);
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