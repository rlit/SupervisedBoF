function triplets = GenerateTripletsWithMargin(X, D, M, P, gtMat,nTripletsMax,params)

descs = {X.desc};
areas = {X.area};

%% calc
Z = cellfun(@(d)full(mexLasso(M*d',D,params.lasso_params)),descs,'UniformOutput',0);
pooledDesc = cellfun(@(z,a)GetPooledDescriptor(params.poolingMethod, z, a, P),Z,areas,'UniformOutput',0);
descMat = cat(3,pooledDesc{:});

DIST = bofdist(descMat, 'l1vec');


%%
maxMarginSize = params.lossMaxNeg;
hasPos  = any(gtMat>0);
tCell = {};
tMargin = {};
for q = find(hasPos)
    [sortedDIST,perm] = sort(DIST(:,q));
    sortedGT = gtMat(perm,q);

    lastPos = find(sortedGT==1,1,'last');
    lastPosDist = sortedDIST(lastPos);

    tooCloseNeg = find(...
        gtMat(:,q) == -1 & ...
        DIST( :,q) < lastPosDist + maxMarginSize);
    if isempty(tooCloseNeg),continue;end


    for curNeg = tooCloseNeg(:)'
        curNegDist = DIST( curNeg,q);
        tooFarPos = find(...
            gtMat(:,q) == 1 & ...
            DIST( :,q) > curNegDist - maxMarginSize);
        if isempty(tooFarPos),continue;end

        V = ones(numel(tooFarPos),1);

        newTriplets = [{q*V} {tooFarPos} {curNeg*V}];
        newMargins  = DIST( tooFarPos,q) - curNegDist + maxMarginSize;
        assert(all(newMargins>0))

        tCell   = [tCell ; newTriplets]; %#ok<AGROW>
        tMargin = [tMargin ; {newMargins}];%#ok<AGROW>

    end

end

triplets = [
    vertcat(tCell{:,1}),...
    vertcat(tCell{:,2}),...
    vertcat(tCell{:,3})];
margins = vertcat(tMargin{:});
nTriplets = size(triplets,1);

nSelect = min(nTripletsMax,nTriplets);
perm = randperm(nTriplets,nSelect);
samp = randsample(nTriplets,nSelect,true,1./margins);
triplets = triplets(perm,:);

%% sanity check - does all triplets have good margin?
% idxPos = sub2ind(size(DIST), triplets(:,1), triplets(:,2));
% idxNeg = sub2ind(size(DIST), triplets(:,1), triplets(:,3));
%
% margin = DIST(idxNeg) - DIST(idxPos);
% [nnz(margin>maxMarginSize) nnz(margin<maxMarginSize)]
%
% hist(margin,100)