function grad_cov = getGradCov(y,grad_y,z,grad_z)

n = length(y);
s = length(grad_y)/n;
meanz = mean(z);
meany = mean(y);
grad_cov = (grad_z.*repmat(y-meany,s,1)+grad_y.*repmat(z-meanz,s,1))/(n-1);
