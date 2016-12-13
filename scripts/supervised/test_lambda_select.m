startup

% DICT_DIR_ORIG = DICT_DIR;
% DICT_DIR = [DICT_DIR '_pos0'];
% LASSO_PARAMS.pos = false;
% DICT_TRAIN_PARAMS.pos = false;

figH = figure(98656);clf
%%

lambdaVals = [.125:.125:.5 ];
clrs = {'r','g','b','k','m','c','y'};
stls = {'-','--','-.',':'};

for iVal = numel(lambdaVals):-1:1
    curVal = lambdaVals(iVal);
    fprintf('\n\n\n    running for lambda = %.2f \n\n\n\n',curVal)

    DICT_TRAIN_PARAMS.lambda = curVal;
    compute_dict
    %continue
    % restore binary shapesToUse
    shapesToUse  = DefineTrainingSet(LABELS,TRAIN_SET_NAME);% & DefineTrainingSet(LABELS,'SHREC14_allR');
%     shapesToUse  = true(size(LABELS));


    for iDict = 1: numel(DICT_SIZES)
        curDictName = sprintf('dict%d.mat',DICT_SIZES(iDict));
        [PR,RE,roc] = EvalDictFun(fullfile(DICT_DIR_L ,curDictName),shapesToUse);
        h = PlotRateCurves(PR,RE,roc(:,1),1-roc(:,2),figH);
        set(h,'LineWidth', 2,'displayname',sprintf('\\lambda=%.2f - dict%d',curVal,DICT_SIZES(iDict)));
        set(h,'color', clrs{iVal},'linestyle',stls{iDict});
    end
end

set(figH,'color','w')

%% show old results
compute_vocab
shapesToUse  = DefineTrainingSet(LABELS,TRAIN_SET_NAME);
% shapesToUse  = true(size(LABELS));
vocabFileName = fullfile(VOCAB_DIR,'vocab48.mat');

[PR,RE,roc] = EvalVocabFun(vocabFileName,shapesToUse);
h = PlotRateCurves(PR,RE,roc(:,1),1-roc(:,2),figH);
set(h,'LineWidth', 2, 'color', 'k','displayname','softVQ');
set(h,'linestyle','-.','marker','p')
%%
% fRender = myaa(6);
% saveas(fRender,'saved1','png')