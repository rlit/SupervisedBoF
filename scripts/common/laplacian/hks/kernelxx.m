% Computes heat kernel.
%
% Usage:  K = kernelxy(t, eigval, eigvec)
%
% Input:  t   - time, 1 x t
%         eigval   - LB eigenvalues, m x 1
%         eigvec   - corresponding LB eigenvectors, n x m
%
% Output: KK - an n x t matrix with the values of Kt(x,x)
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.

function [Kxx] = kernelxx(T,eigval,eigvec)
TT  = repmat(T(:)', [length(eigval), 1]); % m x t
LL  = repmat(eigval(:) , [1, length(T)]); % m x t
VV  = eigvec.^2; % n x m

Kxx = VV*(exp(-LL.*TT)); % (n x m) x (m x t) 
