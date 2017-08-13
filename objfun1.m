function [obj, grad, hess] = objfun1(Z,X,Y)

global maxiter zeroApproxi

disp('objfun1')

[n, varnum] = size(X);
subtasks = length(Z)/n;
ztemp = reshape(Z,n,[]);
zhat = zeros(n,subtasks);
az = zeros(subtasks,1);
bz = zeros(subtasks,1);
grad_az = zeros(n,subtasks);
grad_bz = zeros(n,subtasks);
hess_az = zeros(n,n*subtasks);
hess_bz = zeros(n,n*subtasks);
for i=1:subtasks
    zi = ztemp(:,i);
    [~, zihat] = estimateMaxCorr(ones(n,1),ones(n,1),X,X,zi,maxiter); 
    zhat(:,i) = zihat;
    C = cov([zihat zi]);
    az(i) = C(1,2)/C(1,1);
    bz(i) = mean(zi)-az(i)*mean(zihat);
    grad_zi = ones(n,1);
    grad_zihat = zeros(n,1);
    grad_meanzi = ones(n,1)/n;
    grad_az(:,i) = getGradCov(zi,grad_zi,zihat,grad_zihat)/C(1,1);
    grad_bz(:,i) = grad_meanzi-grad_az(:,i)*mean(zihat);
    hess_zi = zeros(n,1);
    hess_zihat = zeros(n,1);
    hess_az(:,(i-1)*n+1:i*n) = getHessCov(zi,grad_zi,hess_zi,zihat,grad_zihat,hess_zihat)/C(1,1);
    hess_bz(:,(i-1)*n+1:i*n) = -hess_az(:,(i-1)*n+1:i*n)*mean(zihat);
end
yhat = ones(n,1);
yhathat = ones(n,1);
newX = [X ztemp];
newXhat = [X repmat(az',n,1).*zhat+repmat(bz',n,1)];
grad_yhat = zeros(n*subtasks,1);
grad_yhathat = zeros(n*subtasks,1);
grad_Y = zeros(n*subtasks,1);
grad_newX = [zeros(n*subtasks,varnum) kron(eye(subtasks),ones(n,1))];
grad_newXhat = [zeros(n*subtasks,varnum) kron(eye(subtasks),ones(n,1))];
hess_yhat = zeros(n*subtasks,subtasks);
hess_yhathat = zeros(n*subtasks,subtasks);
hess_Y = zeros(n*subtasks,subtasks);
hess_newX = [zeros(n*subtasks,varnum*subtasks) zeros(n*subtasks,subtasks*subtasks)];
hess_newXhat = [zeros(n*subtasks,varnum*subtasks) zeros(n*subtasks,subtasks*subtasks)];
if maxiter < 1
    nonPartIdx = 1:1:subtasks;
    lnonPart = subtasks;
end
for i=1:maxiter
    [~, yhat, yhathat, grad_yhat, grad_yhathat, hess_yhat, hess_yhathat] = ... 
        estimateMaxCorr(yhat,yhathat,newX,newXhat,Y,1,grad_yhat,grad_yhathat,grad_newX,grad_newXhat,hess_yhat,hess_yhathat,hess_newX,hess_newXhat);
    nonPartIdx = [];
    for j=1:subtasks
        if mean(grad_yhat((j-1)*n+1:j*n).^2) > zeroApproxi 
        else
            nonPartIdx = [nonPartIdx j];
        end
    end
    lnonPart = length(nonPartIdx);
    if lnonPart >= maxiter-i
        break;
    end
end
for i=1:lnonPart
    input = ztemp(:,nonPartIdx(i));
    inputhat = az(nonPartIdx(i))*zhat(:,nonPartIdx(i))+bz(nonPartIdx(i));
    grad_input = zeros(n*subtasks,1);
    grad_input(n*(nonPartIdx(i)-1)+1:n*nonPartIdx(i),1) = ones(n,1);
    grad_inputhat = grad_input;
    hess_input = zeros(n*subtasks,subtasks);
    hess_inputhat = zeros(n*subtasks,subtasks);
    [~, yhat, yhathat, grad_yhat, grad_yhathat, hess_yhat, hess_yhathat] = ... 
        estimateMaxCorr(yhat,yhathat,input,inputhat,Y,1,grad_yhat,grad_yhathat,grad_input,grad_inputhat,hess_yhat,hess_yhathat,hess_input,hess_inputhat);
end
for i=1:lnonPart
    if mean(grad_yhat((nonPartIdx(i)-1)*n+1:nonPartIdx(i)*n).^2) > zeroApproxi 
    else
        input = ztemp(:,nonPartIdx(i));
        inputhat = az(nonPartIdx(i))*zhat(:,nonPartIdx(i))+bz(nonPartIdx(i));
        grad_input = zeros(n*subtasks,1);
        grad_input(n*(nonPartIdx(i)-1)+1:n*nonPartIdx(i),1) = ones(n,1);
        grad_inputhat = grad_input;
        hess_input = zeros(n*subtasks,subtasks);
        hess_inputhat = zeros(n*subtasks,subtasks);
        R1 = corrcoef([yhat+input Y]);
        R2 = corrcoef([yhat.*input Y]);
        if abs(R1(1,2)) > abs(R2(1,2))        
            hess_yhat = hess_yhat + hess_input;
            hess_yhathat = hess_yhathat + hess_inputhat;
            grad_yhat = grad_yhat + grad_input;
            grad_yhathat = grad_yhathat + grad_inputhat;
            yhat = yhat+input;
            yhathat = yhathat+inputhat;
            maxcorryzx = abs(R1(1,2));
            disp('------------------------------')
            disp(strcat('+',num2str(nonPartIdx(i)),', maxCorr=',num2str(maxcorryzx)))
            disp('------------------------------')
        else
            hess_yhat = hess_yhat.*repmat(input,subtasks,subtasks);
            hess_yhat = hess_yhat + repmat(yhat,subtasks,subtasks).*hess_input;
            hess_yhat = hess_yhat + repmat(grad_yhat,1,subtasks).*repmat(reshape(grad_input,n,[]),subtasks,1);
            hess_yhat = hess_yhat + repmat(grad_input,1,subtasks).*repmat(reshape(grad_yhat,n,[]),subtasks,1);
            hess_yhathat = hess_yhathat.*repmat(inputhat,subtasks,subtasks);
            hess_yhathat = hess_yhathat + repmat(yhathat,subtasks,subtasks).*hess_inputhat;
            hess_yhathat = hess_yhathat + repmat(grad_yhathat,1,subtasks).*repmat(reshape(grad_inputhat,n,[]),subtasks,1);
            hess_yhathat = hess_yhathat + repmat(grad_inputhat,1,subtasks).*repmat(reshape(grad_yhathat,n,[]),subtasks,1);                
            grad_yhat = grad_yhat.*repmat(input,subtasks,1) + grad_input.*repmat(yhat,subtasks,1);
            grad_yhathat = grad_yhathat.*repmat(inputhat,subtasks,1) + repmat(yhathat,subtasks,1).*grad_inputhat;            
            yhat = yhat.*input;
            yhathat = yhathat.*inputhat;
            maxcorryzx = abs(R2(1,2));
            disp('------------------------------')
            disp(strcat('*',num2str(nonPartIdx(i)),', maxCorr=',num2str(maxcorryzx)))
            disp('------------------------------')
        end
    end
end
R = corrcoef([yhat Y]);
maxcorryzx = abs(R(1,2));
grad_maxcorryzx = sign(R(1,2))*getGradCorr(Y,grad_Y,yhat,grad_yhat);
if nargout > 2
    hess_maxcorryzx = sign(R(1,2))*getHessCorr(Y,grad_Y,hess_Y,yhat,grad_yhat,hess_yhat);
end
C = cov([yhat Y]);
ay = C(1,2)/C(1,1);
by = mean(Y)-ay*mean(yhat);
meanyyhat = mean(Y.*yhat);
meanyhatyhat = mean(yhat.*yhat);
meanyyhathat = mean(Y.*yhathat);
meanyhatyhathat = mean(yhat.*yhathat);
meanyhathatyhathat = mean(yhathat.*yhathat);
meanehat2 = meanyhatyhat + meanyhathatyhathat - 2*meanyhatyhathat;
meaneehat = meanyyhat-meanyyhathat-ay*meanyhatyhat+ay*meanyhatyhathat-by*mean(yhat)+by*mean(yhathat);
obj = var(Y)*(1-maxcorryzx^2) + ay^2*meanehat2 + 2*ay*meaneehat;
obj = obj/var(Y);
grad_numerator = getGradCov(Y,grad_Y,yhat,grad_yhat);
grad_denominator = getGradCov(yhat,grad_yhat,yhat,grad_yhat);
grad_ay = grad_numerator/C(1,1) - grad_denominator*C(1,2)/C(1,1)^2;
grad_meanyhat = grad_yhat/n;
grad_by = -grad_ay*mean(yhat)-ay*grad_meanyhat;
grad_meanyyhat = (repmat(Y,subtasks,1).*grad_yhat)/n;
grad_meanyhatyhat = 2*(repmat(yhat,subtasks,1).*grad_yhat)/n;
temp = grad_yhathat.*reshape(zhat,[],1);
grad_meanyhathat = zeros(n*subtasks,1);
grad_meanyyhathat = zeros(n*subtasks,1);
grad_meanyhatyhathat = zeros(n*subtasks,1);
grad_meanyhathatyhathat = zeros(n*subtasks,1);
for i=1:subtasks
    grad_meanyhathat((i-1)*n+1:i*n) = grad_az(:,i)*mean(temp((i-1)*n+1:i*n))+grad_bz(:,i)*mean(grad_yhathat((i-1)*n+1:i*n));
    grad_meanyyhathat((i-1)*n+1:i*n) = grad_az(:,i)*mean(Y.*temp((i-1)*n+1:i*n))+grad_bz(:,i)*mean(Y.*grad_yhathat((i-1)*n+1:i*n));
    grad_meanyhatyhathat((i-1)*n+1:i*n) = grad_az(:,i)*mean(yhat.*temp((i-1)*n+1:i*n))+grad_bz(:,i)*mean(yhat.*grad_yhathat((i-1)*n+1:i*n));
    grad_meanyhathatyhathat((i-1)*n+1:i*n) = grad_az(:,i)*mean(2*yhathat.*temp((i-1)*n+1:i*n))+grad_bz(:,i)*mean(2*yhathat.*grad_yhathat((i-1)*n+1:i*n));
end
grad_meanyhatyhathat = grad_meanyhatyhathat + (repmat(yhathat,subtasks,1).*grad_yhat)/n;
grad_meanehat2 = grad_meanyhatyhat + grad_meanyhathatyhathat - 2*grad_meanyhatyhathat;
grad_meaneehat = grad_meanyyhat-grad_meanyyhathat-grad_by*mean(yhat)-by*grad_meanyhat+grad_by*mean(yhathat)+by*grad_meanyhathat;
grad_meaneehat = grad_meaneehat -grad_ay*meanyhatyhat-ay*grad_meanyhatyhat+grad_ay*meanyhatyhathat+ay*grad_meanyhatyhathat;
grad = var(Y)*(-2*maxcorryzx*grad_maxcorryzx);
grad = grad + 2*ay*grad_ay*meanehat2 + ay^2*grad_meanehat2 + 2*ay*grad_meaneehat + 2*grad_ay*meaneehat;
grad = grad/var(Y);
if nargout > 2
    hess_numerator = getHessCov(Y,grad_Y,hess_Y,yhat,grad_yhat,hess_yhat);
    hess_denominator = getHessCov(yhat,grad_yhat,hess_yhat,yhat,grad_yhat,hess_yhat);
    hess_ay = (hess_numerator*C(1,1)-hess_denominator*C(1,2))*C(1,1)/C(1,1)^3;
    hess_ay = hess_ay - (grad_numerator*grad_denominator'+grad_denominator*grad_numerator')*C(1,1)/C(1,1)^3;
    hess_ay = hess_ay + 2*(grad_denominator*grad_denominator')*C(1,2)/C(1,1)^3;
    hess_meanyhat = zeros(n*subtasks,n*subtasks);
    hess_meanyyhat = zeros(n*subtasks,n*subtasks);
    hess_meanyhatyhat = zeros(n*subtasks,n*subtasks);
    hess_meanyhathat = zeros(n*subtasks,n*subtasks);
    hess_meanyyhathat = zeros(n*subtasks,n*subtasks);
    hess_meanyhatyhathat = zeros(n*subtasks,n*subtasks);
    hess_meanyhathatyhathat = zeros(n*subtasks,n*subtasks);
    for i=1:subtasks 
        for j=1:subtasks
            hess_meanyhat((i-1)*n+1:i*n,(j-1)*n+1:j*n) = diag(hess_yhat((i-1)*n+1:i*n,j))/n;
            hess_meanyyhat((i-1)*n+1:i*n,(j-1)*n+1:j*n) = diag(Y.*hess_yhat((i-1)*n+1:i*n,j))/n;
            hess_meanyhatyhat((i-1)*n+1:i*n,(j-1)*n+1:j*n) = 2*diag(grad_yhat((i-1)*n+1:i*n).*grad_yhat((j-1)*n+1:j*n))/n+ ... 
                                                             2*diag(yhat.*hess_yhat((i-1)*n+1:i*n,j))/n;
            hess_meanyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) = (grad_az(:,i)*grad_az(:,j)')*mean(zhat(:,i).*zhat(:,j).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                            (grad_az(:,i)*grad_bz(:,j)')*mean(zhat(:,i).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                            (grad_bz(:,i)*grad_az(:,j)')*mean(zhat(:,j).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                            (grad_bz(:,i)*grad_bz(:,j)')*mean(hess_yhathat((i-1)*n+1:i*n,j));
            hess_meanyyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) = (grad_az(:,i)*grad_az(:,j)')*mean(Y.*zhat(:,i).*zhat(:,j).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                             (grad_az(:,i)*grad_bz(:,j)')*mean(Y.*zhat(:,i).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                             (grad_bz(:,i)*grad_az(:,j)')*mean(Y.*zhat(:,j).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                             (grad_bz(:,i)*grad_bz(:,j)')*mean(Y.*hess_yhathat((i-1)*n+1:i*n,j));
            hess_meanyhatyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) = diag(yhathat.*hess_yhat((i-1)*n+1:i*n,j))/n + ... 
                                                                (grad_yhat((i-1)*n+1:i*n).*temp((j-1)*n+1:j*n))*grad_az(:,j)'/n + ... 
                                                                (grad_yhat((i-1)*n+1:i*n).*grad_yhathat((j-1)*n+1:j*n))*grad_bz(:,j)'/n + ... 
                                                                grad_az(:,i)*(grad_yhat((j-1)*n+1:j*n).*temp((i-1)*n+1:i*n))'/n + ... 
                                                                grad_bz(:,i)*(grad_yhat((j-1)*n+1:j*n).*grad_yhathat((i-1)*n+1:i*n))'/n + ...
                                                                (grad_az(:,i)*grad_az(:,j)')*mean(yhat.*zhat(:,i).*zhat(:,j).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                                (grad_az(:,i)*grad_bz(:,j)')*mean(yhat.*zhat(:,i).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                                (grad_bz(:,i)*grad_az(:,j)')*mean(yhat.*zhat(:,j).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                                (grad_bz(:,i)*grad_bz(:,j)')*mean(yhat.*hess_yhathat((i-1)*n+1:i*n,j));
            hess_meanyhathatyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) = 2*(grad_az(:,i)*grad_az(:,j)')*mean(yhathat.*zhat(:,i).*zhat(:,j).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                                   2*(grad_az(:,i)*grad_bz(:,j)')*mean(yhathat.*zhat(:,i).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                                   2*(grad_bz(:,i)*grad_az(:,j)')*mean(yhathat.*zhat(:,j).*hess_yhathat((i-1)*n+1:i*n,j))+ ... 
                                                                   2*(grad_bz(:,i)*grad_bz(:,j)')*mean(yhathat.*hess_yhathat((i-1)*n+1:i*n,j)) + ... 
                                                                   2*(grad_az(:,i)*grad_az(:,j)')*mean(temp((i-1)*n+1:i*n).*temp((j-1)*n+1:j*n))+ ... 
                                                                   2*(grad_bz(:,i)*grad_az(:,j)')*mean(grad_yhathat((i-1)*n+1:i*n).*temp((j-1)*n+1:j*n)) + ... 
                                                                   2*(grad_az(:,i)*grad_bz(:,j)')*mean(temp((i-1)*n+1:i*n).*grad_yhathat((j-1)*n+1:j*n))+ ... 
                                                                   2*(grad_bz(:,i)*grad_bz(:,j)')*mean(grad_yhathat((i-1)*n+1:i*n).*grad_yhathat((j-1)*n+1:j*n));
            if i==j
                hess_meanyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) = hess_meanyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) + ... 
                    hess_az(:,(i-1)*n+1:i*n)*mean(temp((i-1)*n+1:i*n)) + hess_bz(:,(i-1)*n+1:i*n)*mean(grad_yhathat((i-1)*n+1:i*n));
                hess_meanyyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) = hess_meanyyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) + ... 
                    hess_az(:,(i-1)*n+1:i*n)*mean(Y.*temp((i-1)*n+1:i*n)) + hess_bz(:,(i-1)*n+1:i*n)*mean(Y.*grad_yhathat((i-1)*n+1:i*n));
                hess_meanyhatyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) = hess_meanyhatyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) + ... 
                    hess_az(:,(i-1)*n+1:i*n)*mean(yhat.*temp((i-1)*n+1:i*n)) + hess_bz(:,(i-1)*n+1:i*n)*mean(yhat.*grad_yhathat((i-1)*n+1:i*n));
                hess_meanyhathatyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) = hess_meanyhathatyhathat((i-1)*n+1:i*n,(j-1)*n+1:j*n) + ... 
                    2*hess_az(:,(i-1)*n+1:i*n)*mean(yhathat.*temp((i-1)*n+1:i*n)) + 2*hess_bz(:,(i-1)*n+1:i*n)*mean(yhathat.*grad_yhathat((i-1)*n+1:i*n));
            end
        end
    end
    hess_by = -hess_ay*mean(yhat)-ay*hess_meanyhat-(grad_ay*grad_meanyhat'+grad_meanyhat*grad_ay');    
    hess_meanehat2 = hess_meanyhatyhat + hess_meanyhathatyhathat - 2*hess_meanyhatyhathat;
    hess_meaneehat = hess_meanyyhat-hess_meanyyhathat; 
    hess_meaneehat = hess_meaneehat -hess_ay*meanyhatyhat-ay*hess_meanyhatyhat;
    hess_meaneehat = hess_meaneehat -(grad_meanyhatyhat*grad_ay'+grad_ay*grad_meanyhatyhat');
    hess_meaneehat = hess_meaneehat +hess_ay*meanyhatyhathat+ay*hess_meanyhatyhathat;
    hess_meaneehat = hess_meaneehat +grad_ay*grad_meanyhatyhathat'+grad_meanyhatyhathat*grad_ay';    
    hess_meaneehat = hess_meaneehat -hess_by*mean(yhat)-by*hess_meanyhat; 
    hess_meaneehat = hess_meaneehat -(grad_meanyhat*grad_by'+grad_by*grad_meanyhat');    
    hess_meaneehat = hess_meaneehat +hess_by*mean(yhathat)+by*hess_meanyhathat; 
    hess_meaneehat = hess_meaneehat +grad_meanyhathat*grad_by'+grad_by*grad_meanyhathat';
    hess = var(Y)*(-2*maxcorryzx*hess_maxcorryzx-2*(grad_maxcorryzx*grad_maxcorryzx'));
    hess = hess + 2*(grad_ay*grad_ay')*meanehat2 + 2*ay*hess_ay*meanehat2;
    hess = hess + 2*ay*(grad_ay*grad_meanehat2'+grad_meanehat2*grad_ay') + ay^2*hess_meanehat2;
    hess = hess + 2*(grad_ay*grad_meaneehat'+grad_meaneehat*grad_ay') + 2*ay*hess_meaneehat + 2*hess_ay*meaneehat;
    hess = hess/var(Y);
    if sum(sum(abs(hess-hess'))) > zeroApproxi
        disp(strcat('Warning: asymmetric hessian matrix(',num2str(sum(sum(abs(hess-hess')))),')!'))
    end
end
disp(obj)
if obj < 0 
    if obj > -zeroApproxi
        obj = 0;
    else
        disp('Warning: obj is negative!')
        save(strcat('ZNegativeObj.mat'),'Z');
    end
end
