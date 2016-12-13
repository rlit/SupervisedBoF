function shapeData = LoadShapeData(shapename,paths)

% Load descriptors and kernels
desc = load(fullfile(paths.DESC_DIR,   shapename), 'desc');
shapeData.desc = desc.desc;

evecsFile = fullfile(paths.EVECS_DIR,  shapename);
if exist(evecsFile,'file')>0
    A = load(evecsFile, 'A');
    A = full(diag(A.A));
else
    warning('LoadShapeData:NoEvecFile','no evecs file found, using equal area')
    A = ones(size(desc.desc,1),1);
end
shapeData.A = A;


if isfield(paths,'KERNEL_DIR')
    ker = load(fullfile(paths.KERNEL_DIR, shapename), 'Kxy', 'Txy');
    shapeData.Kxy = ker.Kxy;
    shapeData.Txy = ker.Txy;
end
