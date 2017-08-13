function test1()

clear; clc;

global maxiter zeroApproxi 

maxiter = 5; zeroApproxi = 1e-10; 

% thre = [0.14 0.08 0.51 0.72 0.69 0.86];
% trainFile = {'bm1','X05Y1Z15','Concrete','bm4','RatPol2D','SineCosine'};

thre = repmat([0.14 0.08 0.51 0.72 0.69 0.86],10,1);
trainFile = {'bm1','X05Y1Z15','Concrete','bm4','RatPol2D','SineCosine'};

for ll=1:100
    for i=1:length(trainFile)
        load(fullfile('..','data',strcat(trainFile{i},'.mat')));
        X = trainD(:,1:end-1);
        Y = trainD(:,end);
        [n, ~] = size(X);
        for subtasks=1:3
            Z = rand(n,subtasks);
            Z = reshape(Z,[],1);
            clc;
            options = optimset('Algorithm','interior-point','GradObj','on','GradConstr','on','Hessian','on','HessFcn',@(Z,lambda) myhess3(Z,lambda,X,Y),'OutputFcn',@myoutput,'MaxIter',100,'Display','iter','MaxFunEvals',100);
            [Z, ~, ~, ~] = fmincon(@(Z) objfun2(Z),Z,[],[],[],[],[],[],@(Z) mycons3(Z,X,Y,thre(subtasks,i)),options);
            clc;
            options = optimset('Algorithm','interior-point','GradObj','on','GradConstr','on','Hessian','on','HessFcn',@(Z,lambda) myhess2(Z,lambda,X,Y),'OutputFcn',@myoutput,'MaxIter',100,'Display','iter','MaxFunEvals',100);
            [Z, ~, ~, ~] = fmincon(@(Z) objfun1(Z,X,Y),Z,[],[],[],[],[],[],@(Z) mycons2(Z,X,Y,thre(subtasks,i)),options);
            clc;
            options = optimset('Algorithm','interior-point','GradObj','on','GradConstr','on','Hessian','on','HessFcn',@(Z,lambda) myhess1(Z,lambda,X,Y),'OutputFcn',@myoutput,'MaxIter',100,'Display','iter','MaxFunEvals',100);
            [Z, ~, ~, ~] = fmincon(@(Z) objfun1(Z,X,Y),Z,[],[],[],[],[],[],@(Z) mycons1(Z,X),options); 
            for k=1:subtasks
                [~, Zhat] = estimateMaxCorr(ones(n,1),ones(n,1),X,X,Z((k-1)*n+1:k*n),maxiter); 
                C = cov([Zhat Z((k-1)*n+1:k*n)]);
                a = C(1,2)/C(1,1);
                b = mean(Z((k-1)*n+1:k*n))-a*mean(Zhat);
                Z((k-1)*n+1:k*n) = a*Zhat+b;
            end
            fval = objfun1(Z,X,Y);            
            Z = reshape(Z,n,[]);
            sqrtfval = round(sqrt(fval)*10000)/10000;
            fval = round(fval*10000)/10000;
            fileName = strcat(trainFile{i},'-',num2str(fval),'-',num2str(sqrtfval),'-',num2str(subtasks),'.mat');
            save(fullfile(trainFile{i},num2str(subtasks),fileName),'Z');
            thre(subtasks,i) = min(thre(subtasks,i),fval);
        end
    end
end
