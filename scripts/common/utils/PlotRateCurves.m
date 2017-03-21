function [lineHandles] = PlotRateCurves(PR,RE,FP,FN,figHandle)

if exist('figHandle','var') && ishandle(figHandle)
    figure(figHandle)
else
    figure;
end



lineHandles = [];


plotErrorBars = false;

if plotErrorBars
    errPctLo = 10;
    errPctHi = 90;
end



nRecallVals = 11;
maxRecall = 1;%0.9;
recallVals = linspace(0,maxRecall,nRecallVals);

precisionLims = [.2 1];

%%
fHandle = @(x,y)InterpPrRe(x,y,recallVals);
precisionValsCell = cellfun(fHandle,num2cell(RE,1),num2cell(PR,1),'UniformOutput' , false);
precisionValsMat  = vertcat(precisionValsCell{:});

subplot(1,2,1);
hold on
if plotErrorBars
    lineHandles(end+1) = errorbar(recallVals,    mean(precisionValsMat),...
        abs(prctile(precisionValsMat,errPctLo) - mean(precisionValsMat)),...
        abs(prctile(precisionValsMat,errPctHi) - mean(precisionValsMat)));
    SetErrbarWidth(lineHandles(end),1/(nRecallVals-1)/2)
else
    lineHandles(end+1) = plot(recallVals,nanmean(precisionValsMat) );
end

hold off
title('Precision vs. Recall')
ylabel('Precision');
xlabel('Recall');
xlim([0 maxRecall])
ylim(precisionLims)


%%
subplot(1,2,2);
hold on
if plotErrorBars
    lineHandles(end+1) = errorbar(1:size(PR,1), nanmean(PR,2),...
        abs(prctile(PR,errPctLo,2) -            nanmean(PR,2)),...
        abs(prctile(PR,errPctHi,2) -            nanmean(PR,2)));
    SetErrbarWidth(lineHandles(end),1/2)
else
    lineHandles(end+1) = plot(1:size(PR,1), nanmean(PR,2));
end
hold off
title('Precision at N')
ylabel('Precision');
xlabel('# of shapes');
xlim([1 15])
ylim(precisionLims)


    %%
%
% a = subplot(1,3,3);
% hold on
% lineHandles(end+1) = plot(FP, 1-FN);
% hold off
%
% xlabel('FP');
% ylabel('1-FN');
% title('ROC')
% set(a,'Xscale','log')
% xlim([1e-3 1e-1])
% ylim([.9 1])




function prVals = InterpPrRe(RE,PR,reVals)

a = ~isnan(PR);

if nnz(a)==1
    prVals = ones(1,numel(reVals));
    return
end

for i = find(diff(RE(a))==0)'
    RE(i+1) = RE(i) + eps(RE(i));
end
prVals = interp1(RE(a),PR(a),reVals,'linear');

firstNotNan = find(~isnan(prVals),1,'first');
if firstNotNan ~= 1
    prVals(1:firstNotNan-1) = PR(1);
end

%make PR=1 at RE=0
prVals(reVals==0)=1;

function SetErrbarWidth(H,width)
% Acknowledgement to original author: Arnaud Laurent

assert(strcmpi(get(H,'type'),'hggroup'))

hChild = get(H,'children');
xData  = get(hChild(2),'xData');

xData(4:9:end) = xData(1:9:end) - width/2;
xData(5:9:end) = xData(1:9:end) + width/2;

xData(7:9:end) = xData(1:9:end) - width/2;
xData(8:9:end) = xData(1:9:end) + width/2;

set(hChild(2),'xData',xData(:))