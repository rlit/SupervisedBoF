% clear all
randn('seed', 0); %#ok<RAND>
rand( 'seed', 0); %#ok<RAND>

startup;

epsilon = 1e-6;
pertubFun = @(x)x + epsilon*randn(size(x));

%% create data
LASSO_PARAMS.pos  = 1;
LASSO_PARAMS.lambda  = .25;
LASSO_PARAMS.lambda2 = .00;
nSamples = 5;
descDim  = 31;
Msize    = 20;
nAtoms   = 48;
LASSO_PARAMS.L = Msize;

H{1}  = rand(nSamples,descDim);
H{2}  = rand(nSamples,descDim);
H{3}  = rand(nSamples,descDim);

H = cellfun(@(x){normalize(x,2,1)},H);


Area = rand(nSamples,1);
D0   = rand(descDim,nAtoms);
M    = rand(Msize,descDim);
D    = M*D0;
P    = rand(nAtoms,nAtoms);
D    = normalize(D,2,1);
M    = normalize(M,2,1);
P    = normalize(P,2,1);


% --------------- dD+dM
D_ = pertubFun(D);
M_ = pertubFun(M);
P_ = pertubFun(P);
diffD = D_ - D;
diffM = M_ - M;
diffP = P_ - P;

% run Lasso
Z   = cellfun(@(d)full(mexLasso(M *d',D ,LASSO_PARAMS)),H,'UniformOutput',0);
Z_D = cellfun(@(d)full(mexLasso(M *d',D_,LASSO_PARAMS)),H,'UniformOutput',0);
Z_M = cellfun(@(d)full(mexLasso(M_*d',D ,LASSO_PARAMS)),H,'UniformOutput',0);

% perform pooling
[pooled,poolGradFun,dPFun]   = cellfun(@(x)GetPooledDescriptor('avg_metric', x, Area,P),Z  ,'UniformOutput' , false);
pooled_D = cellfun(@(x)GetPooledDescriptor('avg_metric', x, Area,P),Z_D,'UniformOutput' , false);
pooled_M = cellfun(@(x)GetPooledDescriptor('avg_metric', x, Area,P),Z_M,'UniformOutput' , false);
pooled_P = cellfun(@(x)GetPooledDescriptor('avg_metric', x, Area,P_),Z,'UniformOutput' , false);


% eval loss
params.lossType   = 'l2_lmnn';
params.lossAlpha  = .3;
params.lossMaxNeg = .01;
[F,dF] = eval_loss_triplet(params.lossType, pooled{1},   pooled{2},   pooled{3},   params.lossAlpha, params.lossMaxNeg);
[F_D]  = eval_loss_triplet(params.lossType, pooled_D{1}, pooled_D{2}, pooled_D{3}, params.lossAlpha, params.lossMaxNeg);
[F_M]  = eval_loss_triplet(params.lossType, pooled_M{1}, pooled_M{2}, pooled_M{3}, params.lossAlpha, params.lossMaxNeg);
[F_P]  = eval_loss_triplet(params.lossType, pooled_P{1}, pooled_P{2}, pooled_P{3}, params.lossAlpha, params.lossMaxNeg);


% calc analytical gradients
dZ = cellfun(@(f,g)f(g),poolGradFun,dF,'UniformOutput',0);
dP = cellfun(@(f,g)f(g),dPFun, dF,'UniformOutput',0);
dD = cellfun(@(z,h,dz)lasso_grads(z,M*h',dz, [], D, LASSO_PARAMS.lambda2,'dD'),Z, H, dZ,'UniformOutput',0);
dX = cellfun(@(z,h,dz)lasso_grads(z,M*h',dz, [], D, LASSO_PARAMS.lambda2,'dX'),Z, H, dZ,'UniformOutput',0);
dM = cellfun(@mtimes,dX, H,'UniformOutput',0);

dD = sum(cat(3,dD{:}),3);
dM = sum(cat(3,dM{:}),3);
dP = sum(cat(3,dP{:}),3);

% validate:
val1 = sum(F_D-F)        / epsilon;
val2 = (diffD(:)'*dD(:)) / epsilon;
fprintf('dD - F`-F=[%10.2e] G*G`=[%10.2e] - diff=%10.3f%%\n',val1,val2,100*(val1-val2)/val2)

val1 = sum(F_M-F)        / epsilon;
val2 = (diffM(:)'*dM(:)) / epsilon;
fprintf('dM - F`-F=[%10.2e] G*G`=[%10.2e] - diff=%10.3f%%\n',val1,val2,100*(val1-val2)/val2)

val1 = sum(F_P-F)        / epsilon;
val2 = (diffP(:)'*dP(:)) / epsilon;
fprintf('dP - F`-F=[%10.2e] G*G`=[%10.2e] - diff=%10.3f%%\n',val1,val2,100*(val1-val2)/val2)

fprintf('\n')
