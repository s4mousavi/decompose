function [obj, grad, hess] = objfun2(Z)

disp('objfun2')

lz = length(Z);
obj = 0;
grad = zeros(lz,1);
hess = zeros(lz,lz);
disp(obj)
