function hess = myhess1(Z,lambda,X,Y)

global maxiter zeroApproxi

disp('myhess1')

[~, ~, hess] = objfun1(Z,X,Y);
if sum(sum(abs(hess-hess'))) > zeroApproxi
    disp(strcat('Warning: asymmetric hessian matrix(',num2str(sum(sum(abs(hess-hess')))),')!'))
end

[n, ~] = size(X);
subtasks = length(Z)/n;
ztemp = reshape(Z,n,[]);
for i=1:subtasks
    zi = ztemp(:,i);
    grad_zi = [zeros((i-1)*n,1); ones(n,1); zeros((subtasks-i)*n,1)];
    grad_zihat = zeros(n*subtasks,1);
    hess_zi = zeros(n*subtasks,subtasks);
    hess_zihat = zeros(n*subtasks,subtasks);
    [~, zihat] = estimateMaxCorr(ones(n,1),ones(n,1),X,X,zi,maxiter);
    R = corrcoef([zihat zi]);
    grad_corrzizihat = getGradCorr(zi,grad_zi,zihat,grad_zihat);
    hess_corrzizihat = getHessCorr(zi,grad_zi,hess_zi,zihat,grad_zihat,hess_zihat);
    hess = hess + lambda.eqnonlin(i)*(-2*R(1,2)*hess_corrzizihat-2*(grad_corrzizihat*grad_corrzizihat'));
    if sum(sum(abs(hess-hess'))) > zeroApproxi
        disp(strcat('Warning: asymmetric hessian matrix(',num2str(sum(sum(abs(hess-hess')))),')!'))
    end
end
