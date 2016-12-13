function p = GetParams_AnalysisDL(p)
if nargin == 0
    p = struct;
end

if ~isfield(p,'admm_params')
    p.admm_params = GetAdmmParams(p);
end

if ~isfield(p,'armijo_params')
    p.armijo_params = GetArmijoParams(p);
end

% SGD
if ~isfield(p,'nSamplesSGD'), p.nSamplesSGD = 1e2; end
if ~isfield(p,'batchRatio' ), p.batchRatio = .003; end


if ~isfield(p,'run_name' ), p.run_name = ''; end
if ~isfield(p,'verbose' ), p.verbose = false; end
if ~isfield(p,'verboseIter' ), p.verboseIter = 1; end
if ~isfield(p,'saveTempRes' ), p.saveTempRes = true; end

if ~isfield(p,'stepSize'), p.stepSize = 1e-3; end
if ~isfield(p,'max_iter'), p.max_iter = 1e3; end


if ~isfield(p,'first_valid_check'),  p.first_valid_check = 200; end
if ~isfield(p,'max_valid_increase'), p.max_valid_increase = 0.01; end
if ~isfield(p,'validModulusIter'), p.validModulusIter = 5; end

% gradient regularization
if ~isfield(p,'regul_O'), p.regul_O = 0; end
if ~isfield(p,'regul_M'), p.regul_M = 0; end
if ~isfield(p,'regul_W'), p.regul_W = 0; end

if ~isfield(p,'isDiagonal_M'), p.isDiagonal_M = 0; end
if ~isfield(p,'isDiagonal_P'), p.isDiagonal_P = 0; end

% if ~isfield(p,'isClassifierZ')
%     p.isClassifierZ = true;
% end
% 
% if ~isfield(p,'loss_type')
% % p.loss_type = 'hinge';
% % p.loss_type = 'logit';
% % p.loss_type = 'exp';
% p.loss_type = 'l2';
% end

