function Lambda = lambda_modified(Target,L)

% Operation to fix the sign identification problem
% Inputs: 
%        - Target: true factor loading matrix
%        - L: estimator of Target whose sign entries need to be re-ordered

% Ouptut: 
%        - Lambda: sign re-ordered factor loading matrix from L

m= size(Target,2); Lperm = L; index = 1:m;
for i=1:m
    len = length(index); loss = zeros(len,1); signe = zeros(len,1);
    for j = 1:len
       tmpp = sum((Target(:,i)-L(:,index(j))).^2);
       tmpm = sum((Target(:,i)-(-L(:,index(j)))).^2);
       if (tmpp>tmpm)
           loss(j) = tmpm; signe(j) = -1;
       else 
           loss(j) = tmpp; signe(j) = 1;
       end
    end
    [~,tmpj] = min(loss); Lperm(:,i) = signe(tmpj)*L(:,index(tmpj));
    index = setdiff(index,index(tmpj));
end
Lambda = Lperm;



    