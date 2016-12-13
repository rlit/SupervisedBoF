function str = mprintf(str, varargin)
fprintf(1, '%c', ones(length(str),1)*8);
str = sprintf(varargin{:});
fprintf(1, '%s', str);
