function [idxp,idxn, ip,jp,in,jn] = posnegidx(MASK, ineq);

idxp = find(MASK==1);
idxn = find(MASK==-1);
[ip,jp] = ind2sub(size(MASK), idxp);
[in,jn] = ind2sub(size(MASK), idxn);

if nargin > 1 && ineq,
    idx = jp>ip;
    ip = ip(idx); jp = jp(idx);
    idx = jn>in;
    in = in(idx); jn = jn(idx);
end
