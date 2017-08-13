function [maxCorr, Yhat, Yhathat, grad_Yhat, grad_Yhathat, hess_Yhat, hess_Yhathat] = estimateMaxCorr(initX,initXhat,X,Xhat,Y,maxiter,grad_initX,grad_initXhat,grad_X,grad_Xhat,hess_initX,hess_initXhat,hess_X,hess_Xhat)

global zeroApproxi

iX = 1./X;
iX(X >= 0 & X < zeroApproxi) = inf;
iX(X < 0 & X > -zeroApproxi) = -inf;
XiX = [X iX];
iXhat = 1./Xhat;
iXhat(Xhat >= 0 & Xhat < zeroApproxi) = inf;
iXhat(Xhat < 0 & Xhat > -zeroApproxi) = -inf;
XiXhat = [Xhat iXhat];
[n, varnum] = size(X);
Yhat = initX;
Yhathat = initXhat;
R = corrcoef([Yhat Y]);
maxCorr = abs(R(1,2));
if nargout > 3
    [ng, ~] = size(grad_X);
    s = ng/n;
    grad_iX = -grad_X.*repmat(iX.^2,s,1);
    grad_XiX = [grad_X grad_iX];
    grad_Yhat = grad_initX;
    grad_iXhat = -grad_Xhat.*repmat(iXhat.^2,s,1);
    grad_XiXhat = [grad_Xhat grad_iXhat];
    grad_Yhathat = grad_initXhat;
end
if nargout > 5
    hess_iX = (-hess_X.*repmat(kron(X,ones(1,s)),s,1)+2*kron(grad_X,ones(1,s)).*repmat(reshape(grad_X,n,[]),s,1)).*repmat(kron(iX.^3,ones(1,s)),s,1);
    hess_XiX = [hess_X hess_iX];
    hess_Yhat = hess_initX;
    hess_iXhat = (-hess_Xhat.*repmat(kron(Xhat,ones(1,s)),s,1)+2*kron(grad_Xhat,ones(1,s)).*repmat(reshape(grad_Xhat,n,[]),s,1)).*repmat(kron(iXhat.^3,ones(1,s)),s,1);
    hess_XiXhat = [hess_Xhat hess_iXhat];
    hess_Yhathat = hess_initXhat;
end
premaxcorr = -1;
if isnan(maxCorr) 
    maxCorr = premaxcorr+0.1;
end
iter = 0;
disp('----------')
while maxCorr > premaxcorr && iter < maxiter
    iter = iter+1;
    premaxcorr = maxCorr;
    Yhat1 = repmat(Yhat,1,2*varnum).*XiX;
    idx = find(var(Yhat1) < zeroApproxi);
    Yhat1(:,idx) = kron(mean(Yhat1(:,idx)),ones(n,1));
    Yhat2 = repmat(Yhat,1,varnum)+X;
    idx = find(var(Yhat2) < zeroApproxi);
    Yhat2(:,idx) = kron(mean(Yhat2(:,idx)),ones(n,1));
    Yhat3 = repmat(Yhat,1,varnum)-X;
    idx = find(var(Yhat3) < zeroApproxi);
    Yhat3(:,idx) = kron(mean(Yhat3(:,idx)),ones(n,1));
    R1 = corrcoef([Yhat1 Y]);
    R2 = corrcoef([Yhat2 Y]);
    R3 = corrcoef([Yhat3 Y]);
    [maxro1, idx1] = max(abs(R1(end,1:end-1)));
    [maxro2, idx2] = max(abs(R2(end,1:end-1)));
    [maxro3, idx3] = max(abs(R3(end,1:end-1)));
    [maxro, idxall] = max([maxro1 maxro2 maxro3]);
    if maxro > premaxcorr
        maxCorr = maxro;
        if idxall == 1 
            if nargout > 5
                hess_Yhat = hess_Yhat.*repmat(XiX(:,idx1),s,s);
                hess_Yhat = hess_Yhat + repmat(Yhat,s,s).*hess_XiX(:,s*(idx1-1)+1:s*idx1);
                hess_Yhat = hess_Yhat + repmat(grad_Yhat,1,s).*repmat(reshape(grad_XiX(:,idx1),n,[]),s,1);
                hess_Yhat = hess_Yhat + repmat(grad_XiX(:,idx1),1,s).*repmat(reshape(grad_Yhat,n,[]),s,1);
                hess_Yhathat = hess_Yhathat.*repmat(XiXhat(:,idx1),s,s);
                hess_Yhathat = hess_Yhathat + repmat(Yhathat,s,s).*hess_XiXhat(:,s*(idx1-1)+1:s*idx1);
                hess_Yhathat = hess_Yhathat + repmat(grad_Yhathat,1,s).*repmat(reshape(grad_XiXhat(:,idx1),n,[]),s,1);
                hess_Yhathat = hess_Yhathat + repmat(grad_XiXhat(:,idx1),1,s).*repmat(reshape(grad_Yhathat,n,[]),s,1);
            end
            if nargout > 3
                grad_Yhat = grad_Yhat.*repmat(XiX(:,idx1),s,1)+repmat(Yhat,s,1).*grad_XiX(:,idx1);
                grad_Yhathat = grad_Yhathat.*repmat(XiXhat(:,idx1),s,1)+repmat(Yhathat,s,1).*grad_XiXhat(:,idx1);
            end
            Yhat = Yhat1(:,idx1);
            Yhathat = Yhathat.*XiXhat(:,idx1);
            disp(strcat('*',num2str(idx1),', maxCorr=',num2str(maxCorr)))
        elseif idxall == 2
            if nargout > 5
                hess_Yhat = hess_Yhat+hess_X(:,s*(idx2-1)+1:s*idx2);
                hess_Yhathat = hess_Yhathat+hess_Xhat(:,s*(idx2-1)+1:s*idx2);
            end
            if nargout > 3
                grad_Yhat = grad_Yhat+grad_X(:,idx2);
                grad_Yhathat = grad_Yhathat+grad_Xhat(:,idx2);
            end
            Yhat = Yhat2(:,idx2);
            Yhathat = Yhathat+Xhat(:,idx2);
            disp(strcat('+',num2str(idx2),', maxCorr=',num2str(maxCorr)))
        elseif idxall == 3 
            if nargout > 5
                hess_Yhat = hess_Yhat-hess_X(:,s*(idx3-1)+1:s*idx3);
                hess_Yhathat = hess_Yhathat-hess_Xhat(:,s*(idx3-1)+1:s*idx3);
            end
            if nargout > 3
                grad_Yhat = grad_Yhat-grad_X(:,idx3);
                grad_Yhathat = grad_Yhathat-grad_Xhat(:,idx3);
            end
            Yhat = Yhat3(:,idx3);
            Yhathat = Yhathat-Xhat(:,idx3);
            disp(strcat('-',num2str(idx3),', maxCorr=',num2str(maxCorr)))
        end
    end
end
R = corrcoef([Yhat Y]);
maxCorr = abs(R(1,2));
disp('----------')
if maxCorr < 0 || isnan(maxCorr)
    maxCorr = 0;
end
