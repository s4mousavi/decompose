function hess = myhess3(Z,lambda,X,Y)

global zeroApproxi

disp('myhess3')

[~, ~, hess] = objfun2(Z);
if sum(sum(abs(hess-hess'))) > zeroApproxi
    disp(strcat('Warning: asymmetric hessian matrix(',num2str(sum(sum(abs(hess-hess')))),')!'))
end

[~, ~, hess1] = objfun1(Z,X,Y);
hess = hess + lambda.ineqnonlin(1)*(hess1);
if sum(sum(abs(hess-hess'))) > zeroApproxi
    disp(strcat('Warning: asymmetric hessian matrix(',num2str(sum(sum(abs(hess-hess')))),')!'))
end
