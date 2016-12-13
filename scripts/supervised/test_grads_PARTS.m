% clear all
randn('seed', 0); %#ok<RAND>
rand( 'seed', 0); %#ok<RAND>

startup;

epsilon = 1e-6;
pertubFun = @(x)x + epsilon*randn(size(x));

%% test grads of Lasso
LASSO_PARAMS.pos  = 1;
LASSO_PARAMS.lambda  = .7;
LASSO_PARAMS.lambda2 = .005;
nSamples = 5;
descDim  = 31;
Msize    = 25;
nAtoms   = 48;
LASSO_PARAMS.L = Msize;

Desc  = rand(nSamples,descDim);
Desc  = normalize(Desc,2,2);

D0   = rand(descDim,nAtoms);
M    = rand(Msize,descDim);
D    = M*D0;
D    = normalize(D,2,1);
M    = normalize(M,2,1);

% ---------------
D_ = pertubFun(D);
M_ = pertubFun(M);
diffD = D_ - D;
diffM = M_ - M;


% run Lasso
Z   = full(mexLasso(M *Desc',D ,LASSO_PARAMS));
Z_D = full(mexLasso(M *Desc',D_,LASSO_PARAMS));
Z_M = full(mexLasso(M_*Desc',D ,LASSO_PARAMS));

% eval loss
F   = .5*sum(Z(:)  .^2) / nSamples;
F_D = .5*sum(Z_D(:).^2) / nSamples;
F_M = .5*sum(Z_M(:).^2) / nSamples;

dZ = Z;

% calc analytical gradients
dD = lasso_grads(Z,M*Desc',dZ, [], D, LASSO_PARAMS.lambda2,'dD');
dX = lasso_grads(Z,M*Desc',dZ, [], D, LASSO_PARAMS.lambda2,'dX');
dM = dX * Desc ;


val1 = sum(F_D-F)        / epsilon;
val2 = (diffD(:)'*dD(:)) / epsilon;
fprintf('dD - F`-F=[%10.2e] G*G`=[%10.2e] - diff=%10.3f%%\n',val1,val2,100*(val1-val2)/val2)

val1 = sum(F_M-F)        / epsilon;
val2 = (diffM(:)'*dM(:)) / epsilon;
fprintf('dM - F`-F=[%10.2e] G*G`=[%10.2e] - diff=%10.3f%%\n',val1,val2,100*(val1-val2)/val2)

fprintf('\n')

%% test grads of GetPooledDescriptor

nSamples = 5;
nAtoms   = 48;

Area  = rand(nSamples,1);
Desc  = rand(nAtoms,nSamples);
P     = randn(nAtoms,nAtoms);

Desc  = normalize(Desc,2,2);
P     = normalize(P,2,2);

Desc_ = pertubFun(Desc);
P_    = pertubFun(P);

diffDesc = Desc_ - Desc;
diffP    = P_ - P;

[pooled, poolGradFun,dPFun] = GetPooledDescriptor('avg_metric', Desc , Area, P);
[pooled_desc]         = GetPooledDescriptor('avg_metric', Desc_, Area, P);
[pooled_P]            = GetPooledDescriptor('avg_metric', Desc , Area, P_);

F      = .5*sum(pooled  .^2);
F_desc = .5*sum(pooled_desc .^2);
F_P    = .5*sum(pooled_P .^2);

G      = pooled;
dDesc  = poolGradFun(G) / nSamples;
dP     = dPFun(G);

val1 = sum(F_desc-F)           / epsilon;
val2 = (diffDesc(:)'*dDesc(:)) / epsilon;
fprintf('dPool - F`-F=[%10.2e] G*G`=[%10.2e] - diff=%10.3f%%\n',val1,val2,100*(val1-val2)/val2)

val1 = sum(F_P-F)        / epsilon;
val2 = (diffP(:)'*dP(:)) / epsilon;
fprintf('dP    - F`-F=[%10.2e] G*G`=[%10.2e] - diff=%10.3f%%\n',val1,val2,100*(val1-val2)/val2)

fprintf('\n')

%% test grads of eval_loss_triplet

H{1} = rand(descDim,1,nSamples);
H{2} = rand(descDim,1,nSamples);
H{3} = rand(descDim,1,nSamples);

H = cellfun(@(x){normalize(x,1,1)},H);

params.lossType   = 'l1_lmnn';
params.lossAlpha  = .3;
params.lossMaxNeg = .0;
[lossValVec,lossGrad] = eval_loss_triplet(params.lossType, H{1}, H{2}, H{3}, params.lossAlpha, params.lossMaxNeg);
lossVal = sum(lossValVec);

for i = 1:3

H_ = H;
H_{i} = pertubFun(H{i});
diffH = H_{i} - H{i};
[lossValVec] = eval_loss_triplet(params.lossType, H_{1}, H_{2}, H_{3}, params.lossAlpha, params.lossMaxNeg);
lossVal_ = sum(lossValVec);

val1 = mean(lossVal_-lossVal)     / epsilon;
val2 = (diffH(:)'*lossGrad{i}(:)) / epsilon;

%remove NaNs
bothZeros = val1==0 & val2==0;
val1(bothZeros) = eps;
val2(bothZeros) = eps;

fprintf('dH{%d} - F`-F=[%10.2e] G*G`=[%10.2e] - diff=%10.3f%%\n',i,val1,val2,100*(val1-val2)/val1)

end
fprintf('\n')
