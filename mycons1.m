function [c, ceq, grad_c, grad_ceq] = mycons1(Z,X) 

global maxiter  

disp('mycons1')

[n, ~] = size(X);
subtasks = length(Z)/n;
ztemp = reshape(Z,n,[]);
ceq = zeros(1,subtasks);
grad_ceq = zeros(n*subtasks,subtasks);
for i=1:subtasks
    zi = ztemp(:,i);
    grad_zi = [zeros((i-1)*n,1); ones(n,1); zeros((subtasks-i)*n,1)];
    grad_zihat = zeros(n*subtasks,1);
    [~, zihat] = estimateMaxCorr(ones(n,1),ones(n,1),X,X,zi,maxiter);
    R = corrcoef([zihat zi]);
    grad_corrzizihat = getGradCorr(zi,grad_zi,zihat,grad_zihat);
    ceq(i) = 1-R(1,2)^2;
    grad_ceq(:,i) = -2*R(1,2)*grad_corrzizihat;
end
disp(ceq)
c = [];
grad_c = [];
