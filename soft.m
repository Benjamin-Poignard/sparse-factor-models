function f=soft(A,B)


for i=1:length(A(1,:))
    for j=1:length(A(1,:))
        
        h(i,j)=sign(A(i,j));
        if abs(A(i,j))-B(i,j)>0
            d(i,j)=abs(A(i,j))-B(i,j);
        else d(i,j)=0;
        end;
        f(i,j)=h(i,j)*d(i,j);
    end;
end;

