function Mt = vech_on(M,d)

% Inputs: 
% Marix (square) M
% d: size of M

Mt = [];

for i = 1:(d-1)
    
    Mt = [ Mt ; M(i+1:d,i) ];
    
end