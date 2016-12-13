function [D,M,P] = SynthesisSupDL_rand(X,gtMat,params,D,M,P)
%
% if matlabpool('size') == 0
%     matlabpool open
% end

if ~exist('params','var')
    params = GetParams_AnalysisDL();
end

if ~exist('M','var')
    M = eye(size(D,1));
end
if ~exist('P','var')
    P = eye(size(D,2));
end


params = GetParams_AnalysisDL(params);

isSave = params.saveTempRes;
if isempty(params.run_name)
params.run_name = [params.lossType '_' datestr(now,30)];
end

saveFolder = [pwd '\' params.run_name '\'];
if ~isdir(saveFolder)
    mkdir(saveFolder)
end


% -------------- generate sets
tripletV = GenerateTripletsWithMargin(X, D, M, P, gtMat, params.validSize+params.trainSize, params);
tripletT = tripletV(params.validSize+1:end,:);
tripletV = tripletV(1:params.validSize,:);
nTrain = size(tripletT,1);



[lossV,ratesV] = GetGradients(tripletV,X,M,D,P,params,'rates');
lossV = mean(lossV);
validRate = 'ap';
% validRate = 'eer';
%validRate = 'fpr1';

% -------------- Main loop
tStart = tic;
for iter = 1:params.max_iter
    strValidUpdate = '';
    tStartIter = tic;

    step = params.stepSize*min(1, 0.2*params.max_iter/iter);
%     step = params.stepSize;

    % -------------- SGD: select random subset from training
    nSel  = min(nTrain,params.batchSize);
    isSel = randsample(nTrain,nSel);
    Tsel  = tripletT(isSel,:);


    % -------------- P-step
    if params.isUpdate_P
        [lossP,dP] = GetGradients(Tsel,X,M,D,P,params,'dP');
        lossFunP = @(p)mean(GetGradients(Tsel,X,M,D,p,params));
        [P,stepP,iterP] = ArmijoStep(P,dP,lossFunP,step,params.armijo_params,lossP);

        P = FixMatrix(P,2,params.isDiagonal_P);
    else
        stepP=inf;
        iterP=0;
    end

    % -------------- M-step
    if params.isUpdate_M
        [lossM,dM] = GetGradients(Tsel,X,M,D,P,params,'dM');
        lossFunM = @(m)mean(GetGradients(Tsel,X,m,D,P,params));
        [M,stepM,iterM] = ArmijoStep(M,dM,lossFunM,step,params.armijo_params,lossM);

        M = FixMatrix(M,2,params.isDiagonal_M);
    else
        stepM=inf;
        iterM=0;
    end

    % -------------- D-step
    if params.isUpdate_D
        [lossD,dD] = GetGradients(Tsel,X,M,D,P,params,'dD');
        lossFunD = @(d)mean(GetGradients(Tsel,X,M,d,P,params));
        [D,stepD,iterD] = ArmijoStep(D,dD,lossFunD,step,params.armijo_params,lossD);

        D = FixMatrix(D,1,false,params.lasso_params.pos);
    else
        stepD=inf;
        iterD=0;
    end

    lossT = mean(GetGradients(Tsel,X,M,D,P,params));
    maxArmijo = max([iterD iterM iterP]);
    minStep = min([stepM,stepD,stepP]);

    % -------------- Update validation
    if ~mod(iter,params.validModulusIter)
        % Solve pursuit problem for validation set

        [lossV,ratesV] = GetGradients(tripletV,X,M,D,P,params,'rates');
        lossV = mean(lossV);

        strValidUpdate = '(validation updated)';

        tripletT = GenerateTripletsWithMargin(X, D, M, P, gtMat, params.batchSize, params);
        nTrain = size(tripletT,1);
        if nTrain == 0, break;end
        if isSave, save([saveFolder 'DMP_backup'],'D','M','P','params','iter'); end
%         if iter >= first_valid_check && (1+maxvalidincrease)*lossVprev < lossV
%             fprintf('Train: f = %.4e  Valid: f = %.4e  valid-eer = %.2f%%\nOptimization terminated\n\n', f_, lossV, 100*acc);
%             break
%         end
        %lossVprev = lossV;

%         if maxArmijo == 1
%             step = step / sqrt(params.armijo_params.beta);
%         end
    end

    fprintf('%4d: %.4e [%.4e] valid-%s=%5.2f  step = %.2e (%2d armijos) ,took %.1f ',...
        iter,  lossT, lossV, validRate, 100*ratesV.(validRate), minStep, maxArmijo,toc(tStartIter));
    fprintf('%s \n',strValidUpdate)

    % -------------- update step size
    if ~mod(iter,1)
         save([saveFolder 'DMP_ITER_' num2str(iter)],'D','M','P','params');
    end

end

fprintf('Optimization terminated after %4d iteration\nValidation Score -%.4e\ntook %.1f\n',...
    iter, lossV, toc(tStart));

D = gather(D);
M = gather(M);
P = gather(P);
if isSave, save([saveFolder 'DMP_final'],'D','M','P','params');end

function [lossVal,lossGrad] = GetGradients(Tidxs,X,M,D,P,params,gradName)
lasso_params = params.lasso_params;
D = FixMatrix(D,1,false,lasso_params.pos);
M = FixMatrix(M,2,params.isDiagonal_M);
P = FixMatrix(P,2,params.isDiagonal_P);

descs = {X(Tidxs).desc};
areas = {X(Tidxs).area};
descs = reshape(descs,size(Tidxs));
areas = reshape(areas,size(Tidxs));


Z          = cellfun(@(d)full(mexLasso(M*d',D,lasso_params)),descs,'UniformOutput',0);
[pooledDesc,poolGradFun,dPfun] = cellfun(@(z,a)GetPooledDescriptor(params.poolingMethod, z, a, P),Z,areas,'UniformOutput',0);


[lossValCell,lossGradCell] = cellfun(@(d0,dP,dN)eval_loss_triplet(params.lossType, d0, dP, dN, params.lossAlpha, params.lossMaxNeg),...
    pooledDesc(:,1),...
    pooledDesc(:,2),...
    pooledDesc(:,3),...
    'UniformOutput',0);

lossVal = mean([lossValCell{:}]);
% get gradient w.r.t pooled descriptor
lossGrad = vertcat(lossGradCell{:});

if nargout < 2 || ~exist('gradName','var')
    return
end


if nargout == 2 && exist('gradName','var') && strcmpi(gradName,'rates')
    lossGrad =  GetRates(params.lossType, ...
        cat(3,pooledDesc{:,1}), ...
        cat(3,pooledDesc{:,2}), ...
        cat(3,pooledDesc{:,3}));
    return
end

if strcmp(gradName,'dP')
    % get gradient w.r.t P
    dP = cellfun(@(f,g)f(g),dPfun,lossGrad,'UniformOutput',0);

else
    % get gradient w.r.t Z
    dZ = cellfun(@(f,g)f(g),poolGradFun,lossGrad,'UniformOutput',0);

end


switch gradName
    case 'dD'
        % get gradient w.r.t D
        gradCell = cellfun(@(z,d,dz)lasso_grads(z,M*d',dz, M, D, lasso_params.lambda2,'dD'),...
            Z, descs, dZ,'UniformOutput',0);

    case 'dM'
        % get gradient w.r.t (M*desc)
        gradDesc = cellfun(@(z,d,dz)lasso_grads(z,M*d',dz, M, D, lasso_params.lambda2,'dX'),...
            Z, descs, dZ,'UniformOutput',0);

        % get gradient w.r.t M
        gradCell = cellfun(@mtimes,gradDesc,descs,'UniformOutput',0);
        if params.isDiagonal_M
            gradCell = cellfun(@(p)p.*eye(size(p)),gradCell,'UniformOutput',0);
        end

    case 'dP'
        if params.isDiagonal_P
            dP = cellfun(@(p)p.*eye(size(p)),dP,'UniformOutput',0);
        end
        gradCell = dP;

    otherwise
        error('unknown gradient')
end
% aggregete gradient from all cells
lossGrad = mean(cat(3,gradCell{:}),3);



function rates = GetRates(lossType, desc0, descP, descN)
posDiff = desc0-descP;
negDiff = desc0-descN;


switch lower(lossType)
    case{'l1','l1margin_hinge','l1_lmnn'}
        distFun = @(x)sum(sum(abs(x),1),2);

    case{'l2'}
        distFun = @(x)sum(sum(x.^2,1),2);

    otherwise
        assert(0)
end



[rates.eer,rates.fpr1,rates.fpr01, ~,~,~, ~,roc,prre] = calculate_rates(...
    distFun(posDiff),...
    distFun(negDiff));

isNaN = ~any(isnan(prre),2);
rates.ap = trapz(prre(isNaN,1),prre(isNaN,2));
% plot(prre(isNaN,1),prre(isNaN,2))
isNaN = ~any(isnan(roc),2);
rates.auc = trapz(roc(isNaN,2),roc(isNaN,1));
% % plot(roc(isNaN,2),roc(isNaN,1))


function matOut = FixMatrix(matIn,dim,isDiag,isPos)

matOut = matIn;

if nargin>3 && isPos
    % project to non-negative values
    matOut = max(matOut,0);
end

% isDiag = isequal( matIn~=0 ,logical(eye(size(matIn))));

if isDiag
    diagVals = diag(matOut);
    diagVals = normalize(diagVals,1,1);
    matOut = diag(diagVals) * numel(diagVals);

else
    matOut = normalize(matOut,2,dim);
end



