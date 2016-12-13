function [loss,lossGrad] = eval_loss_triplet(lossType, desc0, descP, descN, lossAlpha,hingeTh)

if nargin < 6
    hingeTh = 1;
end

if nargin < 5
    lossAlpha = .5;
end

sumMat = @(x)sum(sum(x,2),1);

posDiff = desc0-descP;
negDiff = desc0-descN;

switch lower(lossType)

    case{'l1margin_hinge'}
        margin  = sumMat(abs(negDiff)) - sumMat(abs(posDiff));
        isSubTh = margin < hingeTh;
        loss    = (hingeTh - margin) .* isSubTh;

    case{'l1_lmnn'}
        margin  = sumMat(abs(negDiff)) - sumMat(abs(posDiff));
        isSubTh = margin < hingeTh;
        loss    = lossAlpha  * sumMat(abs(posDiff)) + ...
            (1-lossAlpha) * (hingeTh - margin) .* isSubTh;

    case{'l2_lmnn'}
        margin  = sumMat((negDiff).^2) - sumMat((posDiff).^2);
        isSubTh = margin < hingeTh;
        loss    = lossAlpha  * .5 * sumMat((posDiff).^2) + ...
            (1-   lossAlpha) * .5 * (hingeTh - margin) .* isSubTh;
        
    case{'l1'}
        negSum  = sumMat(abs(negDiff));
        %disp(negSum)
        isSubTh = hingeTh >= negSum;
        loss = lossAlpha  * sumMat(abs(posDiff)) + ...
            (1-lossAlpha) * (hingeTh - negSum) .* isSubTh;

    case{'l2'}
        negSum  = sumMat(negDiff.^2);
        isSubTh = hingeTh >= negSum;
        loss = lossAlpha  * .5 * sumMat(posDiff.^2) + ...
            (1-lossAlpha) * .5 * (hingeTh - negSum) .* isSubTh;

    otherwise
        assert(0)
end



if nargout == 1
    return
end

descDim = size(desc0);
isSubThMat = repmat(isSubTh,descDim(1:2));

switch lower(lossType)
    case{'l1margin_hinge'}
        lossGrad{1} =  isSubThMat .* (sign(posDiff) - sign(negDiff));
        lossGrad{2} = -isSubThMat .*  sign(posDiff);
        lossGrad{3} =  isSubThMat .*  sign(negDiff);

    case{'l1_lmnn'}
        lossGrad{1} =  isSubThMat .* (1-lossAlpha) .* (sign(posDiff) - sign(negDiff)) + lossAlpha  * sign(posDiff);
        lossGrad{2} = -isSubThMat .* (1-lossAlpha) .*  sign(posDiff)                  - lossAlpha  * sign(posDiff);
        lossGrad{3} =  isSubThMat .* (1-lossAlpha) .*  sign(negDiff);

    case{'l2_lmnn'}
        lossGrad{1} =  isSubThMat .* (1-lossAlpha) .* (posDiff - negDiff) + lossAlpha  * posDiff;
        lossGrad{2} = -isSubThMat .* (1-lossAlpha) .*  posDiff            - lossAlpha  * posDiff;
        lossGrad{3} =  isSubThMat .* (1-lossAlpha) .*  negDiff;

    case{'l1'}
        lossGrad{1} =    lossAlpha  * sign(posDiff) - (1-lossAlpha) * sign(negDiff) .*isSubThMat;
        lossGrad{2} =   -lossAlpha  * sign(posDiff);
        lossGrad{3} = (1-lossAlpha) * sign(negDiff) .* isSubThMat;

    case{'l2'}
        lossGrad{1} =    lossAlpha  * posDiff - (1-lossAlpha) * negDiff .*isSubThMat;
        lossGrad{2} =   -lossAlpha  * posDiff;
        lossGrad{3} = (1-lossAlpha) * negDiff .* isSubThMat;

    otherwise
end

