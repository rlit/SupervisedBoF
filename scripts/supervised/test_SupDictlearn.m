
% clear all
randn('seed', 0); %#ok<RAND>
rand( 'seed', 0); %#ok<RAND>

if matlabpool('size') == 0
%     matlabpool open
end
warning('off','ParCellFun:EmptyPool')

if ~exist('X','var')
    startup

    dictSize = 48;
    LASSO_PARAMS.lambda = .5;
    DICT_DIR_L = [DICT_DIR '_L' sprintf('%.2f',LASSO_PARAMS.lambda)];

    DICTNAME = fullfile(DICT_DIR_L, sprintf('dict%d.mat',dictSize));
    loaded = load(DICTNAME,'D','M');
    D = loaded.D;
    M = loaded.M;
    [~,DICTNAME] = fileparts(DICTNAME);

    GetDescriptorStruct
    X = DESC_STRUCT;

    clear DICT_DIR_L DESC_STRUCT  loaded
end
fprintf('running supervised based on dictionary - %s\n',DICTNAME)


% generate triplets
nTripletsT = 1e3;
nTripletsV = 5e1;

% randBoolVec = @(v,n)randsample(find(v),n,nnz(v)<=n);
gtMat = gt.MASK;
% gtMat(strcmpi(gt.xform,'noise'),:) = 0;
% gtMat(:,strcmpi(gt.xform,'noise')) = 0;
gtMat(0<eye(size(gtMat)))=0;
% hasPos  = any(gtMat>0);
% gtMat = num2cell(gtMat,1);
% tripletIdxs(:,1) = randBoolVec(hasPos,nTripletsT+nTripletsV);
% tripletIdxs(:,2) = arrayfun(@(i)randBoolVec(gtMat{i}== 1,1),tripletIdxs(:,1));%POS
% tripletIdxs(:,3) = arrayfun(@(i)randBoolVec(gtMat{i}==-1,1),tripletIdxs(:,1));%NEG
% % TODO - select only misclassified triplets
% tripletsValid = unique(tripletIdxs(1:nTripletsV     ,:),'rows');
% tripletsTrain = unique(tripletIdxs(nTripletsV+1:end ,:),'rows');
clear randBoolVec maskTmp hasPos tripletIdxs



%%%%%% Parameter settings:
params = GetParams_AnalysisDL();

params.validModulusIter = 5;

params.lasso_params = LASSO_PARAMS;
%params.lasso_params.pos = 0;
% params.D0 = D;

% params.regul_Rec  = 0;
% params.regul_Data = 1;

params.trainSize = nTripletsT;
params.validSize = nTripletsV;
params.batchSize = 50;
params.max_iter = 1e3;
params.stepSize = 1e-3;

params.isUpdate_D = false;
params.isUpdate_M = false;
params.isUpdate_P = true;

params.isDiagonal_M = false;
params.isDiagonal_P = true;


params.armijo_params.maxiter = 6;
params.armijo_params.beta  = 0.2;

% --- new parameters ---------------
params.descNrm   = DESCRIPTOR_NORMALIZATION;

params.lossType   = 'l1_lmnn';
params.lossAlpha  = .5;
params.lossMaxNeg = .05;

params.poolingMethod = 'avg';
params.poolingMethod = 'avg_metric';
% ----------------------------------
P = eye(size(D,2));
params.run_name = [DICTNAME '_DefaultName_' datestr(now,30)];
[D,M,P] = SynthesisSupDL_rand(X,gtMat,params,D,M,P);





