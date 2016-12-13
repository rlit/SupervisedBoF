% Computes bags of features
%
% Usage:  [BOF, SSBOF] = bof(vocab, desc, sigma, A)
%
% Input:  vocab - vocabulary matrix
%         sigma - bandwidth for soft VQ
%         desc  - descriptor matrix
%         nrm   - descriptor normalization (e.g., 'L2')
%         A     - area elements per vertex
%         Kxy   - heat kernel K(x,y,t)
%
% Output: BOF   - spatially-insensitive bag of features
%         SSBOF - spatially-sensitive bags of features
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.

function [BOF, SSBOF,F] = bof(vocab, sigma, desc, nrm, A, Kxy)

% % % hard VQ
% vs    = size(vocab,1);
% desc  = normalize(desc, nrm, 2);
%  
% D = squared_dist(vocab, desc);
% 
% [M,I] = min(D,[],1);
% BOF = hist(I,[1:vs])';
% F = zeros(vs,size(Kxy,1));
% for i = 1:size(Kxy,1)
%    F(I(i),i) = 1; 
% end
% 
% thrdiam = 0.1;
% 
% if nargin > 5 && nargout > 1,
%     SSBOF = {};
%     for t = 1:size(Kxy,3),
%         Ddiff = squared_dist(Kxy,Kxy);
%         diamdiff = max(Ddiff(:));
% 
%         ssbof = zeros(vs,vs);
%         for i = 1:vs
%             for j = 1:vs
%                 d = Ddiff(find(I==i),find(I==j));
%                 ssbof(i,j) = length(find(d<diamdiff*thrdiam));
%             end
%         end
%         
%         SSBOF{t} = ssbof;
%     end
% end




vs    = size(vocab,1);
desc  = normalize(desc, nrm, 2);

D = squared_dist(vocab, desc);
w = exp(-0.5*D/sigma^2);
ws = sum(w,1);             % L1-normalization
%ws = sqrt(sum(w.^2,1));     % L2-normalization
ws(ws <= 0) = 1;
F  = w ./ repmat(ws, [size(w,1) 1]);
%FF = F.*repmat(A(:)',[vs 1]);

%BOF = F*A / sum(A);
BOF = sum(F,2);


if nargin > 5 && nargout > 1,
    SSBOF = {};
    for t = 1:size(Kxy,3),
        SSBOF{t} = F*Kxy(:,:,t)*F';
        %SSBOF{t} = (FF*Kxy(:,:,t)*FF'); % / (A'*Kxy(:,:,t)*A);
    end
end

