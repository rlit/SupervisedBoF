function [cmc,PR,RE] = CalcRatesAtN(distMat,isPosMat,colsToUse)

assert(isequal(size(distMat),size(isPosMat)))

if nargin < 3
colsToUse = true(1,size(distMat,2));
end

%verify zeros on the main diagonal - distance to self is not positive
isPosMat(eye(size(isPosMat))>0) = 0;


[D,perm] = sort(distMat,1);
isPosMatSorted = zeros(size(distMat));
for k=1:size(perm,2),
    isPosMatSorted(:,k) = isPosMat(perm(:,k),k);
end
isPosMatSorted = isPosMatSorted(:,colsToUse);

isPosMatSortedCell = num2cell(isPosMatSorted,1);
[cmc_cell,PR_cell,RE_cell] = cellfun(@CreateSingleCurve,isPosMatSortedCell,'UniformOutput',0);

cmc = [cmc_cell{:}];
RE  = [ RE_cell{:}];
PR  = [ PR_cell{:}];

% % %%
% % cmc = [cmc_cell{:}];
% % % imagesc(([cmc_cell{:}]))
% % % imagesc(diff(Mat))

function [cmc,PR,RE] = CreateSingleCurve(isPosVecSorted)
assert(isvector(isPosVecSorted))

nPos = nnz(isPosVecSorted== 1);
if nPos == 0
    [cmc,PR,RE] = deal([]);
    return
end

isPosVecFiltered = isPosVecSorted(isPosVecSorted~=0)>0;
nFiltered = numel(isPosVecFiltered);

cmc = cumsum(isPosVecFiltered~=0);
RE = cmc  / nPos;
PR = cmc ./ (1:nFiltered)';

% make PR monotonically non-increasing
cumMax = 0;
for ii = nFiltered:-1:1
    if PR(ii) > cumMax
        cumMax = PR(ii);
    else
        PR(ii) = cumMax;
    end
end


vecSize = numel(isPosVecSorted);
if vecSize>nFiltered
    cmc(end+1:vecSize) = cmc(end);
    RE( end+1:vecSize) = RE( end);
    PR( end+1:vecSize) = PR( end);
end


assert(RE(end)==1)
fullRecall = find(RE==1,1,'first');
if fullRecall<vecSize
    PR(fullRecall+1:end)=NaN;
end
