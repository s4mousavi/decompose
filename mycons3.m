function [c, ceq, grad_c, grad_ceq] = mycons3(Z,X,Y,t)  

disp('mycons3')

[obj, grad] = objfun1(Z,X,Y);
c = obj-t;
grad_c = grad;
disp(c)
ceq = [];
grad_ceq = [];
