function p = GetArmijoParams(p)
if nargin == 0
    p = struct;
end

if ~isfield(p,'beta'   ), p.beta    = 0.2; end
if ~isfield(p,'maxiter'), p.maxiter = 10;  end
if ~isfield(p,'sigma'  ), p.sigma   = 0;   end    

