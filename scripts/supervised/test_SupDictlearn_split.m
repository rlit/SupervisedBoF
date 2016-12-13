
% clear all
randn('seed', 0); %#ok<RAND>
rand( 'seed', 0); %#ok<RAND>

if matlabpool('size') == 0
%     matlabpool open
end

if ~exist('X','var')
    startup

    %dictSize = 64;
    dictSize = 48;
    %dictSize = 32;
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

nTripletsT = 5e4;
nTripletsV = 5e1;

% take only training dataset
TRAIN_SET_NAME = 'null+1iso';
isInTraining   = DefineTrainingSet(gt.LABELS,TRAIN_SET_NAME);
gtSel.shapeid  = gt.shapeid(isInTraining);
gtSel.xform    = gt.xform(isInTraining);
gtSel.strength = gt.strength(isInTraining);
gtSel.LABELS   = gt.LABELS(isInTraining);
gtSel.MASK     = gt.MASK(isInTraining,isInTraining);
Xsel = X(isInTraining);


gtMatSel = gtSel.MASK;
gtMatSel(0<eye(size(gtMatSel)))=0;



%%%%%% Parameter settings:
params = GetParams_AnalysisDL();

params.validModulusIter = 5;

LASSO_PARAMS.L = size(X(1).desc,2);
params.lasso_params = LASSO_PARAMS;
params.trainSize = nTripletsT;
params.validSize = nTripletsV;
params.batchSize = 20;
params.max_iter = 1000;
params.stepSize = 3e-2;

params.isUpdate_D = true;
params.isUpdate_M = false;
params.isUpdate_P = false;




params.armijo_params.maxiter = 6;
params.armijo_params.beta  = 0.2;

% --- new parameters ---------------
params.descNrm   = DESCRIPTOR_NORMALIZATION;

params.lossType   = 'l1_lmnn';
params.lossAlpha  = .5;
params.lossMaxNeg = .5;

params.poolingMethod = 'avg_metric';
% ----------------------------------


params.run_name = [DICTNAME ...
    '_L' num2str(LASSO_PARAMS.lambda) ...
    '_margin_' num2str(params.lossMaxNeg) '_' ...
    TRAIN_SET_NAME '_' ...
    datestr(now,30)];
[D] = SynthesisSupDL_rand(Xsel,gtMatSel,params,D);





