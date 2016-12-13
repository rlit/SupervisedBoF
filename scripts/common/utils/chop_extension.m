function [b, ext] = chop_extension(a)

idx = strfind(a,'.');
if isempty(idx), 
    b = a; 
    ext = '';
else
    b = a(1:idx(end)-1);
    ext = a(idx(end)+1:end);
end



