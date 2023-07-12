function C = check_zero(beta0,beta)

p = length(beta);
C=0;
for i = 1:p
   if beta0(i)==0
       if beta(i)==beta0(i)
           C=C+1;
       end
   end
end