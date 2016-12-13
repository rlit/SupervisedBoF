function [K] = calc_kxx(t, shape, lb_params)

for p=1:length(lb_params)
    K.([lb_params{p} '0']) = kernelxx(t, abs(shape.shape0.(lb_params{p}).evals), shape.shape0.(lb_params{p}).evecs);

    if(isfield(shape, 'shape1'))
        K.([lb_params{p} '1']) = kernelxx(t, abs(shape.shape1.(lb_params{p}).evals), shape.shape1.(lb_params{p}).evecs);
    end
end

end % function calc_kxx()

