function y=transx(x,tcode);
n=size(x,1);
small=1e-6;
y=x;
switch(tcode)
    case 1,
        y=x;
    case 2,
        y(1)=0;
        y(2:n)=x(2:n)-x(1:n-1);
    case 3,
        y(1)=0;y(2)=0;
        y(3:n)=x(3:n)-2*x(2:n-1)+x(1:n-2);
    case 4,
        if min(x) < small; y=NaN; else;
            y=log(x);
        end;
    case 5,
        if min(x) < small; y=NaN; else;
            x=log(x);
            y(1)=0;
            y(2:n)=x(2:n)-x(1:n-1);
        end;
    case 6,
        if min(x) < small;  y=NaN; else;
            y(1)=0;y(2)=0;
            x=log(x);
            y(3:n)=x(3:n)-2*x(2:n-1)+x(1:n-2);
        end;
    otherwise,
        y=NaN*ones(n,1);
end;

