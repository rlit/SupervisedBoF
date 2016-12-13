% Computes distance between an array of BoFs
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.

function D = bofdist(B, method, varargin)


switch lower(method)

    case 'tfidf'               % tf-idf weighted distance
        B = reshape(B, [size(B,1)*size(B,2) 1 size(B,3)]);
        D = zeros(size(B,3));
        idf = varargin{1};
        
        W = zeros(size(idf,1),1,size(B,3));
        W(:,1,:) = repmat(idf,[1 size(B,3)])./squeeze(repmat(sum(B,1),[size(B,1) 1 1]));
        B = W.*B;
        
        D = zeros(size(B,3));
        for i=1:size(B,3),
            d = squeeze(sum(bsxfun(@(x,y)(abs(x-y)), B, B(:,:,i)),1));
            D(:,i) = d;
        end
        
    
    case 'l1vec'               % vector L1 distance
        B = reshape(B, [size(B,1)*size(B,2) 1 size(B,3)]);
        s = repmat(sum(B,1),[size(B,1) 1 1]);
        s(find(s==0)) = 1;
        B = B./s;
        D = zeros(size(B,3));
        for i=1:size(B,3),
            d = squeeze(sum(bsxfun(@(x,y)(abs(x-y)), B, B(:,:,i)),1));
            D(:,i) = d;
        end
        
    case {'l2vec', 'fro'}        
        B = vectorize(B)';
        D = max(squared_dist(B,B),0);
        D = sqrt(D);
    
    case 'l1mat'
        D = zeros(size(B,3));
        for i=1:size(B,3),
            d = squeeze(max(sum(bsxfun(@(x,y)(abs(x-y)), B, B(:,:,i)),1),[],2));
            D(:,i) = d(:);
        end
        
end

function B = vectorize(B)
if size(B,3) > 1,
    B = reshape(B, [size(B,1)*size(B,2) size(B,3)]);
end
