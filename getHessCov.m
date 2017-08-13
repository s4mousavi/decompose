function hess_cov = getHessCov(y,grad_y,hess_y,z,grad_z,hess_z)

global zeroApproxi

n = length(y);
s = length(grad_y)/n;
meanz = mean(z);
meany = mean(y);
hess_cov = hess_z.*repmat(y-meany,s,s)/(n-1);
hess_cov = hess_cov + hess_y.*repmat(z-meanz,s,s)/(n-1);
hess_cov = hess_cov + repmat(grad_y,1,s).*repmat(reshape(grad_z,n,[]),s,1)/(n-1);
hess_cov = hess_cov + repmat(grad_z,1,s).*repmat(reshape(grad_y,n,[]),s,1)/(n-1);
temp = [];
for i=1:s
    rtemp = [];
    for j=1:s
        rtemp = [rtemp diag(hess_cov((i-1)*n+1:i*n,j))];
    end
    temp = [temp; rtemp];
end
if sum(sum(abs(temp-temp'))) > zeroApproxi
    disp(strcat('Warning: asymmetric hessian matrix(',num2str(sum(sum(abs(temp-temp')))),')!'))
end
hess_cov = temp - (grad_y*grad_z' + grad_z*grad_y')/(n*(n-1));
if sum(sum(abs(hess_cov-hess_cov'))) > zeroApproxi
    disp(strcat('Warning: asymmetric hessian matrix(',num2str(sum(sum(abs(hess_cov-hess_cov')))),')!'))
end
