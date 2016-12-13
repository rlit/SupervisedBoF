function [PR,RE,roc] = EvalDictFun(DictFileName,shapesToUse)
% Computes sparse codes
%
assert(exist(DictFileName,'file')==2)




startup
paths.EVECS_DIR = EVECS_DIR;
paths.DESC_DIR  = DESC_DIR;
paths.DESCRIPTOR_NORMALIZATION = DESCRIPTOR_NORMALIZATION;


[DictFolder,DictName,tmp] = fileparts(DictFileName);
confusionFolder = fullfile(DictFolder,'confusion');
if ~isdir(confusionFolder)
    mkdir(confusionFolder)
end
confusionTempFile = fullfile(confusionFolder,[DictName,tmp]);

% Load vocabulary / dictoinary
dict = load(DictFileName);
if isfield(dict,'vocab')
    dict.vocab = normalize(dict.vocab, 2, 2);
    dict.D = dict.vocab';
    dict.M = eye(size(dict.D,1));
end
if ~isfield(dict,'P')
    dict.P = eye(size(dict.D,2));
end

poolingMethod = SD_POOLING;
lasso_params = LASSO_PARAMS;
if isfield(dict,'params')
    if isfield(dict.params,'poolingMethod')
        poolingMethod = dict.params.poolingMethod;
    end
    if isfield(dict.params,'lasso_params')
        lasso_params = dict.params.lasso_params;
    end
end


% List of shapes
SHAPES = dir(fullfile(paths.DESC_DIR, FILES_TO_PROCESS));
SHAPES = {SHAPES.name};


if nargin < 2
    shapesToUse = true(size(SHAPES));
end
assert(isequal(size(shapesToUse),size(SHAPES)))



IS_COMPUTE_SSSD=0;
SDs = {};
nShapes = length(SHAPES);

% Statistics
tic;
fprintf(1, 'Computing sparse codes...\n');
nok   = 0;
str = '';



%% calc confusion
if exist(confusionTempFile,'file')
    DIST = load(confusionTempFile);
    DIST = DIST.DIST;
else

D = dict.D;
M = dict.M;
P = dict.P;
parfor s = 1:nShapes

    shapename = SHAPES{s};
%     str = mprintf(str, '%d/%d - %-30s \t ', s,nShapes),shapename);

    shapeData = LoadShapeData(shapename,paths);
    shapeData.desc = normalize(shapeData.desc,paths.DESCRIPTOR_NORMALIZATION,2);
    Z = mexLasso(M*shapeData.desc',D,lasso_params);
    Z = full(Z);
    [SDs{s}] = GetPooledDescriptor(poolingMethod, Z, shapeData.A,P);

    fprintf('%d/%d - %-30s \n', s,nShapes,shapename)
end

% Statistics
fprintf(1, '\nComputation complete\n');
fprintf(1, ' Elapsed time:   %s\n', format_time(toc));
fprintf(1, ' Total Shapes:   %d\n', length(SHAPES));
% fprintf(1, ' Computed:       %d\n', nok);


dist = SD_DISTANCE;
fprintf('Calculating distances...\n');
DIST = bofdist(cat(3,SDs{:}), dist);
save(confusionTempFile,'DIST')

end

DIST = DIST(shapesToUse,shapesToUse);


%%
if ~exist('MASK','var')
    groundtruth_classes
end
M = MASK;
M = M(shapesToUse,shapesToUse);
[idxp,idxn] = posnegidx(M, true);
hasPos  = any(M>0) & shapeid(shapesToUse) <= MAX_POSITIVE_SHAPES;

[eer,fpr1,fpr01, ~,~,~, ~,roc,~] = calculate_rates(DIST(idxp), DIST(idxn));
[cmc,PR,RE] = CalcRatesAtN(DIST,M,hasPos);

fprintf(1, ' %-15s %-8s \t',  DictName, SD_DISTANCE);
fprintf(1, 'EER = %5.2f%% \t FAR@1%% = %5.2f%% \t FAR@0.1%% = %5.2f%%\n', ...
    eer*100, fpr1*100, fpr01*100);

