% Computes heat kernel.
%
% Usage:  K = kernelxy(t, L, V)
%
% Input:  t   - time
%         L   - LB eigenvalues, n x 1
%         V   - corresponding LB eigenvectors, n x n
%
% Output: KK - an n x n matrix with the values of Kt(x,y)
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.

function [K] = kernelxy(t,L,V)
LL = exp(-t*L);
K  = V*diag(LL)*V';
%Kt = -V*diag(LL.*L)*V';
