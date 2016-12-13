function [PR,RE,roc] = EvalVocabFun(VocabFileName,shapesToUse,noiseLevel)
% Computes BoFs
%
assert(exist(VocabFileName,'file')==2)




startup
paths.DESC_DIR  = DESC_DIR;
paths.EVECS_DIR = EVECS_DIR;

%paths.DESCRIPTOR_NORMALIZATION = DESCRIPTOR_NORMALIZATION;


evalNoisedShapes = nargin > 2 && ~isempty(noiseLevel)  && noiseLevel>0;
if evalNoisedShapes
    noisedRoot = [SHAPE_NOISED_DIR '\pct_' sprintf('%4.2f',noiseLevel)];
    assert(isdir(noisedRoot))
    pathsN.SHAPE_DIR  = fullfile(noisedRoot, 'shapes');
    pathsN.EVECS_DIR  = fullfile(noisedRoot, ['evecs.' LB_PARAM]);
    pathsN.DESC_DIR   = fullfile(noisedRoot, ['descriptors.' LB_PARAM]);
    %paths.KERNEL_DIR = fullfile(noisedRoot, ['kernels.' LB_PARAM]);

    BOFsN = {};
else
    pathsN = [];

end

%%
[VocabFolder,VocabName,tmp] = fileparts(VocabFileName);
confusionFolder = fullfile(VocabFolder,'confusion');
if ~isdir(confusionFolder)
    mkdir(confusionFolder)
end
bofFolder = fullfile(VocabFolder,'BoFs');
if ~isdir(bofFolder)
    mkdir(bofFolder)
end
confusionTempFile = fullfile(confusionFolder,[VocabName,tmp]);
bofTempFile = fullfile(bofFolder,[VocabName,tmp]);

%% Load vocabulary
%str = mprintf(str, '%s', dictname);
%load(fullfile(DICT_DIR, dictname));
vocabLoaded = load(VocabFileName);




% List of shapes
SHAPES = dir(fullfile(DESC_DIR, FILES_TO_PROCESS));
SHAPES = {SHAPES.name};


if nargin < 2
    shapesToUse = true(size(SHAPES));
end
assert(isequal(size(shapesToUse),size(SHAPES)))



IS_COMPUTE_SSSD=0;
BOF = {};
nShapes = length(SHAPES);

% Statistics
tic;
fprintf(1, 'Computing BoFs...\n');
nok   = 0;
str = '';



%% calc confusion
if exist(confusionTempFile,'file')
    DIST = load(confusionTempFile);
    DIST = DIST.DIST;
else

desc_nrm = DESCRIPTOR_NORMALIZATION;
sig_scl  = SIGMA_SCALE;
sigma    = vocabLoaded.sigma;
vocab    = vocabLoaded.vocab;
parfor s = 1:nShapes

    shapename = SHAPES{s};
%     str = mprintf(str, '%d/%d - %-30s \t ', s,nShapes),shapename);

    shapeData = LoadShapeData(shapename,paths);
    [BOF{s}] = bof(vocab, sigma*sig_scl, shapeData.desc, desc_nrm, shapeData.A, []);


    if evalNoisedShapes
        shapeData = LoadShapeData(shapename,pathsN);
        [BOFsN{s}] = bof(vocab, sigma*sig_scl, shapeData.desc, desc_nrm, shapeData.A, []);
    end


    fprintf('%d/%d - %-30s \n', s,nShapes,shapename)
end

% Statistics
fprintf(1, '\nComputation complete\n');
fprintf(1, ' Elapsed time:   %s\n', format_time(toc));
fprintf(1, ' Total Shapes:   %d\n', length(SHAPES));
% fprintf(1, ' Computed:       %d\n', nok);


dist = SD_DISTANCE;
fprintf('Calculating distances...\n');
if evalNoisedShapes
    DIST = bofdist2(cat(3,BOF{:}), cat(3,BOFsN{:}), dist);
else
    DIST = bofdist( cat(3,BOF{:}),                  dist);
end
save(bofTempFile,'BOF')
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

fprintf(1, ' %-15s %-8s \t',  VocabName, SD_DISTANCE);
fprintf(1, 'EER = %5.2f%% \t FAR@1%% = %5.2f%% \t FAR@0.1%% = %5.2f%%\n', ...
    eer*100, fpr1*100, fpr01*100);

