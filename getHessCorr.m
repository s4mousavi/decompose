function hess_corr = getHessCorr(y,grad_y,hess_y,z,grad_z,hess_z)

global zeroApproxi

covyz = cov([y z]);
numerator = covyz(1,2);
denominator = sqrt(covyz(1,1)*covyz(2,2));
grad_covzz = getGradCov(z,grad_z,z,grad_z);
grad_covyy = getGradCov(y,grad_y,y,grad_y);
grad_numerator = getGradCov(y,grad_y,z,grad_z);
temp = grad_covyy*covyz(2,2)+grad_covzz*covyz(1,1);
grad_temp = getHessCov(y,grad_y,hess_y,y,grad_y,hess_y)*covyz(2,2)+getHessCov(z,grad_z,hess_z,z,grad_z,hess_z)*covyz(1,1)+ ...
    grad_covyy*grad_covzz'+grad_covzz*grad_covyy';
grad_denominator = temp/(2*denominator);
hess_numerator = getHessCov(y,grad_y,hess_y,z,grad_z,hess_z);
hess_denominator = (grad_temp*denominator-temp*grad_denominator')/(2*denominator^2);
hess_corr = (hess_numerator*denominator-hess_denominator*numerator)*denominator/(denominator^3);
hess_corr = hess_corr - (grad_numerator*grad_denominator'+grad_denominator*grad_numerator')*denominator/(denominator^3);
hess_corr = hess_corr + 2*(grad_denominator*grad_denominator')*numerator/(denominator^3);
if sum(sum(abs(hess_corr-hess_corr'))) > zeroApproxi
    disp(strcat('Warning: asymmetric hessian matrix(',num2str(sum(sum(abs(hess_corr-hess_corr')))),')!'))
end
