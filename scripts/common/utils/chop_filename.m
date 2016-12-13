function b = chop_filename(a)

idx1 = strfind(a,'\'); idx1 = [-Inf; idx1(:)];
idx2 = strfind(a,'/'); idx2 = [-Inf; idx2(:)];
idx  = max(idx1(end), idx2(end));
if idx <= 0, 
    b = a; 
else
    b = a(idx+1:end);
end



