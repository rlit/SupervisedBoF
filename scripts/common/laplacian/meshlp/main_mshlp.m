function [evecs, evals, W, A] = main_mshlp(dtype, shape, max_num_evecs)
% dtype = 'cotangent', 'euclidean', 'geodesic'

assert(strcmp(dtype, 'cotangent') | strcmp(dtype, 'euclidean') | strcmp(dtype, 'geodesic'));

[W,A] = mshlp_matrix(shape, struct('hs',2,'rho',3,'htype','ddr', 'dtype', dtype));
W = -W;
A = sparse([1:length(A)], [1:length(A)], A);
num_evecs = min(size(W, 1) - 1, max_num_evecs);

% Adding A to W and subtracting 1 for numerical stability
rand('seed',0); randn('seed',0);
tic
[evecs, evals] = eigs(W + A, A, num_evecs, 'SM', struct('disp', 0));
tc=toc; fprintf('done eigs(%6.2f sec)\n', tc);
evals = diag(evals) - 1;
assert(all(evals > -1e-6));

% [evecs_, evals_] = eigs(W, A, num_evecs, 0, struct('disp', 0));
% evals_ = diag(evals_);
% max(abs(evals - evals_)) 

[evals, idx] = sort(evals);
evecs = evecs(:,idx);

end % function main_mshlp