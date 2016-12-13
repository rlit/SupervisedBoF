function str = format_number(num)

th = 0.1;

if num < th*1e3,
    str = sprintf('%d', num);
elseif num < th*1e6,
    str = sprintf('%.2gK', num/1e3);
elseif num < th*1e9,
    str = sprintf('%.2gM', num/1e6);
else
    str = sprintf('%.2gG', num/1e9);
end

