function [Xnew,step,iter] = ArmijoStep(X,dX,lossFun,step,params,initLoss)
if nargin < 5
    params = GetArmijoParams();
end
if nargin < 6
    initLoss = lossFun(X);
end

minLoss = initLoss;
normG = norm(dX);
for iter = 1:params.maxiter

    lossNew = lossFun(X - step*dX);
    if lossNew - minLoss < -params.sigma * step * normG,
        Xnew = X - step*dX;
        return;
    end
    %minLoss = min(initLoss,lossNew);

    step = step * params.beta;


end

Xnew = X;