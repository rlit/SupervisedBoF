startup;
groundtruth_classes

shapesTest  = 0<DefineTrainingSet(LABELS,'SHREC14_allR');
shapesTrain = 0<DefineTrainingSet(LABELS,TRAIN_SET_NAME);

% shapesToUse  = true(size(LABELS));
% shapesToUse(1:400) = 0;
% shapesTest = shapesTest & shapesToUse;

if matlabpool('size') == 0
    matlabpool open
end


figH = figure(62823);
clf
%% ---------- Original ShapeGoogle resutls

vocabsToUse = {   'vocab48.mat'  };
% vocabsToUse = [vocabsToUse { 'vocab32.mat', 'vocab64.mat'}] ;
vocabFileName = fullfile(VOCAB_DIR,vocabsToUse{1});

[PR,RE,roc] = EvalVocabFun(vocabFileName,shapesTest);
h = PlotRateCurves(PR,RE,roc(:,1),1-roc(:,2),figH);
set(h,'LineWidth', 2, 'color', 'k','displayname','softVQ');
set(h,'linestyle','-.')




%% ---------- plot unsupervised (initial) dictionary
LASSO_PARAMS.lambda = .5;
dictSize = 32;
dictSize = 48;
% dictSize = 64;
refDictPath = sprintf('%s_L%.2f\\dict%d.mat',DICT_DIR,LASSO_PARAMS.lambda,dictSize);

[PR,RE,roc] = EvalDictFun(refDictPath,shapesTest);
h = PlotRateCurves(PR,RE,roc(:,1),1-roc(:,2),figH);
set(h,'LineWidth', 2, 'color', 'r','displayname','UNSUP - initial');
set(h,'linestyle','--')

%% ---------- inspect folder with taining result
runName = '';
runName = 'dict48_justD_margin_0.01_20140116T101356';
% runName = 'dict48_justD_margin_0.05_20140116T102718';
runName = 'dict48_L0.5_justD_margin_0.1_20140129T212016';


dataPath = ['D:\Zsync\Documents\MATLAB\ShapeGoogle\scripts\experiments\' runName '\'];
assert(isdir(dataPath),'"dataPath" does not exist')
figPath = [dataPath 'figs\'];
if ~isdir(figPath)
    mkdir(figPath)
end

files = dir([dataPath '*ITER_*']);
assert(~isempty(files))

iter = cellfun(@(s)sscanf(s,'DMP_ITER_%d'),{files.name});
[iter,perm] = sort(iter);
files = files(perm);

%%
isSave = 1;
legend('off')
PLOT_CLR = 'm';meanNorm=[];
for i = numel(files)%:-1:1
    curFile = files(i).name;

    dict = load([dataPath curFile]);
    assert(dictSize == size(dict.D,2),'file %s has dictSizeof %d, different than %d in the unspuervised one',curFile,size(dict.D,2),dictSize)
%     rowNorm = sum(dict.M.^2,2);
%     meanNorm(i) = mean(rowNorm);
%     meanNorm(i) = sum(abs(dict.r(:)));
%     figure(987565);hold on;plot(diag(dict.P))
%     continue


    saveName = [figPath sprintf('%04d',iter(i))];
    if isSave && exist([saveName '.png'],'file'), continue;end


    [PR,RE,roc] = EvalDictFun([dataPath curFile],shapesTrain);
    h0 = PlotRateCurves(PR,RE,roc(:,1),1-roc(:,2),figH);
    set(h0,'LineWidth', 2, 'color', 'g','displayname',sprintf('train perf - iter %4d',iter(i)));
    set(h0,'linestyle','-.')

    [PR,RE,roc] = EvalDictFun([dataPath curFile],shapesTest);
    h = PlotRateCurves(PR,RE,roc(:,1),1-roc(:,2),figH);
    set(h,'LineWidth', 2, 'color', PLOT_CLR,'displayname',sprintf('test perf - iter %4d',iter(i)));


    legend('show', 'Location', 'southwest');

    if isSave
        
        saveas(figH,saveName,'png')
        saveas(figH,saveName,'fig')

        delete(h)
        delete(h0)
        legend('off');
    end


end

% figure;
% plot(iter,meanNorm)

%%
return
%%
files = dir([figPath '*.png']);

imgs = {};
map  = {};
for i = numel(files):-1:1
    curIM = imread([figPath files(i).name]);
    [imgs{i},map{i}]  = rgb2ind(curIM,50);

    % sort the colormap for consistency
    [map{i},perm] = sortrows(map{i});
    [~,perm] = sort(perm);
    imgs{i} = uint8(perm(imgs{i}+1)-1);
end
% validate colormap consistency
assert(isequal(map{:}))

imgs(end+(1:5)) = imgs(end);
IM = cat(4,imgs{:});

imwrite(IM,map{1},[dataPath 'anim_' runName '.gif'],'gif',...
    'LoopCount',5,...
    'DelayTime',.5)

fprintf('gif done\n')

%%

