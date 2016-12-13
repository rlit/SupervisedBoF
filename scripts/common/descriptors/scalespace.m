% Computes descriptors.
%
% Usage:  [KK0,KK1,KK2] = scalespace(T, L, V)
%
% Input:  T   - vector of times
%         L   - LB eigenvalues
%         V   - corresponding LB eigenvectors
%
% Output: KK0 - Kt(x,x) descriptor (packaged as a VxT matrix with
%               descriptor values per vertex)
%         KK1 - first-order derivative of Kt(x,x) w.r.t. t
%         KK2 - second-order derivative of Kt(x,x) w.r.t. t
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.

function [KK0,KK1,KK2] = scalespace(T, L, V,poT)

TT  = repmat(T(:)', [length(L) 1]);
LL  = repmat(L(:) , [1 length(T)]);
VV  = V.^2;

% use with scale-covariant
if nargin == 4
    KK0 = max(VV*(exp(-LL.*TT).*(TT).^poT), 0);
else
    KK0 = max(VV*(exp(-LL.*TT).*TT), 0);
end

if nargout > 1,
    KK1 = VV*(exp(-LL.*TT).*(1 - LL.*TT));
end
if nargout > 2,
    KK2 = VV*(exp(-LL.*TT).*LL.*(LL.*TT - 2));
end


