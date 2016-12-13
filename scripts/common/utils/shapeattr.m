function s = shapeattr(x)

idx = find(x=='.');
s = [];
s.num      = str2num(x(1:idx(1)-1));
s.xform    = x(idx(1)+1:idx(2)-1);
s.strength = str2num(x(idx(2)+1:idx(3)-1));
