function grad_corr = getGradCorr(y,grad_y,z,grad_z)

covyz = cov([y z]);
numerator = covyz(1,2);
denominator = sqrt(covyz(1,1)*covyz(2,2));
grad_numerator = getGradCov(y,grad_y,z,grad_z);
grad_denominator = (getGradCov(y,grad_y,y,grad_y)*covyz(2,2)+getGradCov(z,grad_z,z,grad_z)*covyz(1,1))/(2*denominator);
grad_corr = (grad_numerator*denominator-numerator*grad_denominator)/(denominator^2);
