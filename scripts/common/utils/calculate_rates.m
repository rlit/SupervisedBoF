function [eer,fpr1,fpr01, dee,dfr1,dfr01, d,roc,prre] = calculate_rates(dp, dn, d0)

np = length(dp);
nn = length(dn);

if nargin < 3,
    [d0, idx] = sort([dp(:); dn(:)], 'ascend');

    d0 = [d0(1)-1; d0(:)];
    delta = [d0(2:end)-d0(1:end-1); 0];
    d = d0 + delta;
    
    l = [ones(np,1); -ones(nn,1)];
    p = [0; l(idx)>0];
    n = [0; l(idx)<0];
else
    if isempty(d0),
        d0 = unique([dp(:); dn(:)]);
    end
    d = d0;
    p = histc(dp, d0);
    n = histc(dn, d0);
end

% Precision-recall
% warning off;
pr = cumsum(p)./(cumsum(n)+cumsum(p));
% warning on;
re = cumsum(p)./sum(p);
 
p = p/sum(p);
fn = 1-cumsum(p);
n = n/sum(n);
fp = cumsum(n);

[eer,   dee]   = calculate_eer    (fp, fn, d, d0       );
[fpr1,  dfr1]  = calculate_fpatfn (fp, fn, d, d0, 0.01 );
[fpr01, dfr01] = calculate_fpatfn (fp, fn, d, d0, 0.001);

roc = [fp(:) 1-fn(:)];
prre = [re(:) pr(:)];



% EER
function [eer, dee] = calculate_eer(fp,fn, d,d0)
idx0 = find(fn>fp); 
if isempty(idx0),
    eer = NaN;
    dee = NaN;
else
    idx0 = idx0(end);
    idx1 = idx0 + find(fp(idx0+1:end) > fp(idx0) & fn(idx0+1:end) < fn(idx0));
    if ~isempty(idx1),
        idx1 = idx1(1);
        a    = (fn(idx1)-fn(idx0)) / (fp(idx1)-fp(idx0));
        if a ~= 1,
            eer  = (fn(idx0)-a*fp(idx0)) / (1-a);
            dee  = (eer-fp(idx0))/(fp(idx1)-fp(idx0))*(d0(idx1)-d0(idx0)) + d0(idx0);
        else
            eer  = 0.5*(fp(idx0)+fn(idx0));
            dee  = d(idx0);
        end
    else
        eer  = 0.5*(fp(idx0)+fn(idx0));
        dee  = d(idx0);
    end
end

% FP @ given FN
function [fpr, dpr] = calculate_fpatfn(fp,fn, d,d0, fn_target)
idx0 = find(fn>fn_target); 
if isempty(idx0),
    fpr = NaN;
    dpr = NaN;
else
    idx0 = idx0(end);
    idx1 = idx0 + find(fp(idx0+1:end) > fp(idx0) & fn(idx0+1:end) < fn(idx0));
    if ~isempty(idx1),
        idx1 = idx1(1);
        a    = (fn(idx1)-fn(idx0)) / (fp(idx1)-fp(idx0));
        fpr  = fp(idx0) + (fn_target - fn(idx0))/a;
        dpr  = (fpr-fp(idx0))/(fp(idx1)-fp(idx0))*(d0(idx1)-d0(idx0)) + d0(idx0);
    else
        fpr  = fp(idx0);
        dpr  = d(idx0);
    end
end
