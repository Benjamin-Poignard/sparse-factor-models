function Lambda = simulate_perfect_structure_overlap_full_block(p,m,set,set2)

% Inputs: 
%         - p: dimension
%         - m: number of factors
%         - set: size of the blocks in each column
%         - set2: size of the overlaps
% Output:
%         - Lambda: sparse factors loading matrix with/without overlaps

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
parameter = zeros(p,m);
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

Lambda = parameter.*Lambda_temp;
