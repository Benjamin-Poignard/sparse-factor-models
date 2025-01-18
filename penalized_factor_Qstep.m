function [Lambda_rotated,Q_step] = penalized_factor_Qstep(Lambda_first,m,gamma,method)

% Inputs:
%          - Lambda_first: p x m inital parameter value
%          for the factor loading matrix
%          - m: number of factors (a priori set by the user)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization
% Output:
%          - Lambda_rotated: p x m x length(gamma) inital Lambda candidates
%          for the factor loading matrix; one Lambda_rotated for each gamma
%          - Q_step: estimated Q matrix from the LQ-decomposition
%          satisfying Q'xQ = I_m with I_m the m x m identity matrix; one
%          Q_step for each gamma

% The optimization to get Q_step builds upon the work of  Wen and Yin (2013)
% 'A feasible method for optimization with orthogonality constraints',
% Mathematical Programming 142 (1-2), 397–434.

p = size(Lambda_first,1); sampling_size = 200;
if length(gamma)>1
    Q_step = zeros(m,m,length(gamma));
    parfor i=1:length(gamma)
        loss_eval = zeros(sampling_size,1); Q_sampling = zeros(m,m,sampling_size); pen = gamma(i);
        for kk = 1:sampling_size
            [Q,~] = qr(rand(p,m)'); Q = Q';
            [param_est,out]= OptStiefelGBB(Q,@(x)Q_orthonormal(x,Lambda_first,m,pen,method));
            Q_sampling(:,:,kk) = reshape(param_est,m,m); loss_eval(kk) = out.fval;
        end
        [~,index] = min(loss_eval); Q_step(:,:,i) = Q_sampling(:,:,index);
    end
else
    loss_eval = zeros(sampling_size,1); Q_sampling = zeros(m,m,sampling_size);
    parfor kk = 1:sampling_size
        [Q,~] = qr(rand(p,m)'); Q = Q';
        [param_est,out]= OptStiefelGBB(Q,@(x)Q_orthonormal(x,Lambda_first,m,gamma,method));
        Q_sampling(:,:,kk) = reshape(param_est,m,m); loss_eval(kk) = out.fval;
    end
    [~,index] = min(loss_eval);
    Q_step = Q_sampling(:,:,index);
end
if (size(Q_step,3)>1)
    Lambda_rotated = zeros(p,m,length(gamma));
    for i=1:length(gamma)
        Lambda_rotated(:,:,i) = Lambda_first*Q_step(:,:,i);
    end
else
    Lambda_rotated = Lambda_first*Q_step;
end