function p = GetAdmmParams(p)
if nargin == 0
    p = struct;
end

if ~isfield(p,'lambda1'), p.lambda1 = 5e-1; end
% Disallow lambda2 -- much faster convergence!
if ~isfield(p,'lambda2'), p.lambda2 = 1e-1; end    
if ~isfield(p,'rho'    ), p.rho     = 10;   end
if ~isfield(p,'quiet'  ), p.quiet   = true; end
if ~isfield(p,'alpha'  ), p.alpha   = 1;  end
if ~isfield(p,'maxiter'), p.maxiter = 5e2;  end
if ~isfield(p,'abstol' ), p.abstol  = 1e-6; end
if ~isfield(p,'reltol' ), p.reltol  = 1e-6; end
