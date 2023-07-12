function [c,ceq] = constr_q(x,m)

% Constraints on the Q matrix in the LQ-decomposition
% Q must satisfy Q'xQ = I_m with I_m the m x m identity matrix 

% Output: orthogonality constraint for Q, i.e. Q'*Q = Id, with Id the
% identity matrix

c = []; 
ceq = vec(eye(m)-reshape(x,m,m)'*reshape(x,m,m));