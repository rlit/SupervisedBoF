% Computes heat kernel.
%
% Usage:  K = kernelxy(t, L, V)
%
% Input:  t   - times
%         L   - LB eigenvalues
%         V   - corresponding LB eigenvectors
%
% Output: KK - a VxV matrix with the values of Kt(x,y)
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.

function [K] = kernelxy(t,L,V)
LL = exp(-t*L);
K  = V*diag(LL)*V';
%Kt = -V*diag(LL.*L)*V';
