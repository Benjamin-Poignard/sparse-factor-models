function error=FEmultiLag(x,xnew,F,Y,h,k,lag)

% model: x_{t+h}= constant+ beta f_t +  y_t+y_{t-1}+y_{t-2}+y_{t-3}...+y_{t-lag}
 % h step ahead forecast

T=length(F(:,1));
r=length(F(1,:));

     %% if with 3 lagged x 
    % W=[ ones(T-h-3,1),  F(4:T-h,:), Y(k,4:T-h)', Y(k,3:T-h-1)',Y(k,2:T-h-2)',Y(k,1:T-h-3)' ];
    % alpha=(W'*W)\W'*x(5:T-h+1); 
    % error=([1, F(T,:),Y(k,T),Y(k,T-1), Y(k,T-2), Y(k,T-3)]*alpha-xnew)^2;
     
     %%
     m=T-h-lag; %lag=0,1,2,3...
     W=ones(m,lag+2+r); W(:,2:r+1)=F(1+lag:T-h,:);
     for t=0:lag
         W(:,r+2+t)=Y(k,1-t+lag:T-h-t)';
     end;
       alpha=(W'*W)\W'*x(lag+2:T-h+1); 
       predictor=ones(1,lag+2+r);
       predictor(2:r+1)=F(T,:);
     for t=0:lag
         predictor(r+2+t)=Y(k,T-t);
     end;
     
        error=(predictor*alpha-xnew)^2;
      
  