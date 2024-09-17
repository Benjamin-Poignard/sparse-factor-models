% The following code replicates the real data experiment based on the
% diffusion index data

% Set h = 12 or 24, with h the forecast horizon
% The code is a direct extension of the code of Bai and Liao (2016):
% https://econweb.rutgers.edu/yl1114/papers/factor3/factor3.html
%% h = 12
clear;
clc
load lndata
series=[];
rawdata=[];
tcode=[];
thr=6;
p=cols(macrodat);
dates=(59+1/12:1/12:108)';
data=trimr(macrodat,12,0);
dates=trimr(dates,12,0);
for i=1:p
    dum=data(:,i);
    m=mean(dum);
    if (isnan(m)==0)
        rawdata=[rawdata dum];
        series=str2mat(series,headertext{1,i});
        tcode=[tcode; vartype(i)];
    else
        disp([i m]);
    end
end
series=trimr(series,1,0);
y=[];
p=cols(rawdata);
for i=1:p
    if (tcode(i)==0)
        tcode(i)=1;
    end
    dum=transx(rawdata(:,i),tcode(i));
    y=[y dum];
end
y=y(49:end,:); % Data 1964.1 - 2007.12
[ehat_T,Fhat_T,lamhat_T,ve2_T]=pc_T(standard(y),8);
y=standard(y)';  % y is N by TT
p=131;
TT=528;
T=round(0.8*TT);
k=6;  % IP:total
kk=114; % CPI: all
%k=2; % % PI less transfers
%k=4;  %  M & T sales
%k=33; % Employ: total

h=12; %12   period ahead forecast; set h = 24 for 24 forecast horizon

for s=1:TT-h+1
    Z(:,s)=mean(y(:,s:s+h-1),2);  % x_{t+h}^h
end

grid = [0.000001:0.001:0.1 0.1:0.05:3];

factors_spec = [5 6 7]; K = length(factors_spec);
IPerror_PCA = zeros(TT-T-h+1,2*K); CPIerror_PCA = zeros(TT-T-h+1,2*K);
IPerror_SAF = zeros(TT-T-h+1,2*K); CPIerror_SAF = zeros(TT-T-h+1,2*K);
IPerror_SGF_scad = zeros(TT-T-h+1,2*K); CPIerror_SGF_scad = zeros(TT-T-h+1,2*K);
IPerror_SGF_mcp = zeros(TT-T-h+1,2*K); CPIerror_SGF_mcp = zeros(TT-T-h+1,2*K);
IPerror_SLSF_scad = zeros(TT-T-h+1,2*K); CPIerror_SLSF_scad = zeros(TT-T-h+1,2*K);
IPerror_SLSF_mcp = zeros(TT-T-h+1,2*K); CPIerror_SLSF_mcp = zeros(TT-T-h+1,2*K);

% Set the lags
Lags = [1 3]; L = length(Lags);

for i=1:TT-T-h+1
    
    % known data
    Y=y(:,i:i+T-1);
    xIP=Z(k,i:i+T-h)';
    xnewIP=Z(k,T+i);
    
    
    xCPI=Z(kk,i:i+T-h)';
    xnewCPI=Z(kk,T+i);
    
    for j = 1:K
        
        r=factors_spec(j);
        
        [FPCA,LPCA]=factor(Y,eye(p),r);
        u=Y-LPCA*FPCA';
        Su=u*u'/(T-4);
        Phi= diag(diag(Su));
        for l = 1:L
            lag = Lags(l);
            IPerror_PCA(i,j+(l-1)*K)=FEmultiLag(xIP,xnewIP, FPCA,Y,h,k,lag);
            CPIerror_PCA(i,j+(l-1)*K)=FEmultiLag(xCPI,xnewCPI, FPCA,Y,h,kk,lag);
        end
        
        gamma = grid*sqrt(log(p*(p+1)/2)/T);
        [Lambda_saf,gamma_opt_saf,Psi_saf] = approx_factor_TS(Y',r,gamma);
        FML_saf = (inv(Lambda_saf'*(Psi_saf\Lambda_saf))*Lambda_saf'*(Psi_saf\Y))';
        for l = 1:L
            lag = Lags(l);
            IPerror_SAF(i,j+(l-1)*K)=FEmultiLag(xIP,xnewIP, FML_saf,Y,h,k,lag);
            CPIerror_SAF(i,j+(l-1)*K)=FEmultiLag(xCPI,xnewCPI, FML_saf,Y,h,kk,lag);
        end
        
        [Lambda_first,Psi_first] = non_penalized_factor(Y',r,'Gaussian');
        loss = 'Gaussian'; method = 'scad';
        gamma = grid*sqrt(log(p*r)/T);
        [Lambda_scad_g,gamma_opt_scad_g,Psi_scad_g] = sparse_factor_TS(Y',r,loss,gamma,method,Lambda_first,Psi_first);
        Lambda_scad_g = filter_FactorLoadings(Lambda_scad_g);
        FML_scad_g = (inv(Lambda_scad_g'*(Psi_scad_g\Lambda_scad_g))*Lambda_scad_g'*(Psi_scad_g\Y))';
        for l = 1:L
            lag = Lags(l);
            IPerror_SGF_scad(i,j+(l-1)*K)=FEmultiLag(xIP,xnewIP,FML_scad_g,Y,h,k,lag);
            CPIerror_SGF_scad(i,j+(l-1)*K)=FEmultiLag(xCPI,xnewCPI,FML_scad_g,Y,h,kk,lag);
        end
        
        [Lambda_first,Psi_first] = non_penalized_factor(Y',r,'Gaussian');
        loss = 'Gaussian'; method = 'mcp';
        gamma = grid*sqrt(log(p*r)/T);
        [Lambda_mcp_g,gamma_opt_mcp_g,Psi_mcp_g] = sparse_factor_TS(Y',r,loss,gamma,method,Lambda_first,Psi_first);
        Lambda_mcp_g = filter_FactorLoadings(Lambda_mcp_g);
        FML_mcp_g = (inv(Lambda_mcp_g'*(Psi_mcp_g\Lambda_mcp_g))*Lambda_mcp_g'*(Psi_mcp_g\Y))';
        for l = 1:L
            lag = Lags(l);
            IPerror_SGF_mcp(i,j+(l-1)*K)=FEmultiLag(xIP,xnewIP,FML_mcp_g,Y,h,k,lag);
            CPIerror_SGF_mcp(i,j+(l-1)*K)=FEmultiLag(xCPI,xnewCPI,FML_mcp_g,Y,h,kk,lag);
        end
        
        [Lambda_first,Psi_first] = non_penalized_factor(Y',r,'LS');
        loss = 'LS'; method = 'scad';
        gamma = grid*sqrt(log(p*r)/T);
        [Lambda_scad_ls,gamma_opt_scad_ls,Psi_scad_ls] = sparse_factor_TS(Y',r,loss,gamma,method,Lambda_first,Psi_first);
        Lambda_scad_ls = filter_FactorLoadings(Lambda_scad_ls);
        FML_scad_ls = (inv(Lambda_scad_ls'*(Psi_scad_ls\Lambda_scad_ls))*Lambda_scad_ls'*(Psi_scad_ls\Y))';
        for l = 1:L
            lag = Lags(l);
            IPerror_SLSF_scad(i,j+(l-1)*K)=FEmultiLag(xIP,xnewIP,FML_scad_ls,Y,h,k,lag);
            CPIerror_SLSF_scad(i,j+(l-1)*K)=FEmultiLag(xCPI,xnewCPI,FML_scad_ls,Y,h,kk,lag);
        end
        
        [Lambda_first,Psi_first] = non_penalized_factor(Y',r,'LS');
        loss = 'LS'; method = 'mcp';
        gamma = grid*sqrt(log(p*r)/T);
        [Lambda_mcp_ls,gamma_opt_mcp_ls,Psi_mcp_ls] = sparse_factor_TS(Y',r,loss,gamma,method,Lambda_first,Psi_first);
        Lambda_mcp_ls = filter_FactorLoadings(Lambda_mcp_ls);
        FML_mcp_ls = (inv(Lambda_mcp_ls'*(Psi_mcp_ls\Lambda_mcp_ls))*Lambda_mcp_ls'*(Psi_mcp_ls\Y))';
        for l = 1:L
            lag = Lags(l);
            IPerror_SLSF_mcp(i,j+(l-1)*K)=FEmultiLag(xIP,xnewIP,FML_mcp_ls,Y,h,k,lag);
            CPIerror_SLSF_mcp(i,j+(l-1)*K)=FEmultiLag(xCPI,xnewCPI,FML_mcp_ls,Y,h,kk,lag);
        end
        
    end
    
end
