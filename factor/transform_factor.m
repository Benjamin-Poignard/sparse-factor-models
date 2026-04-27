function Lambda = transform_factor(Target,L)

% Operation to fix the column identification problem
% Inputs: 
%        - Target: true factor loading matrix
%        - L: estimator of Targer whose columns need to be re-ordered

% Ouptut: 
%        - Lambda: column re-ordered factor loading matrix from L

[p,m] = size(Target); idx = perms(1:size(L,2));

count = 0; mat = zeros(p,m,factorial(m));
for ii = idx'
    count = count+1;
    mat(:,:,count) = L(:,ii);
end
loss = zeros(factorial(m),1);
for kk = 1:factorial(m)
    loss(kk) = sum(abs(vec(abs(mat(:,:,kk))-abs(Target))));
end
[~,id] = min(loss);
Lambda = mat(:,:,id);