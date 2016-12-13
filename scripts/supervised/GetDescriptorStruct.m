%function s = GetDescriptorStruct(gt)

gt = load(fullfile(GROUNDTRUTH_DIR, 'classes'));

paths.EVECS_DIR = EVECS_DIR;
paths.DESC_DIR = DESC_DIR;

% List of shapes
SHAPES = dir(fullfile(paths.DESC_DIR, FILES_TO_PROCESS));
SHAPES = {SHAPES.name};
nSHAPES = length(SHAPES);

% Decode file names into shape number, xform and strength
s = cellfun(@shapeattr, SHAPES);
shapeid = cellfun(@(x)(x), {s.num}, 'UniformOutput', true);
xform = {s.xform};
strength = cellfun(@(x)(x), {s.strength}, 'UniformOutput', true);

IS_COMPUTE_SSSD = false;
DESC_STRUCT = cell(nSHAPES,1);
for s = 1:nSHAPES


    curIdx = find(...
        strcmp(gt.xform,xform{s}) &...
        gt.shapeid  == shapeid(s) &...
        gt.strength == strength(s),1);

    shapename = SHAPES{curIdx};

    shapeData = LoadShapeData(shapename,paths);

    shapeData.desc = normalize(shapeData.desc,DESCRIPTOR_NORMALIZATION,2);

    shapeData.area = shapeData.A;
    shapeData = rmfield(shapeData,'A');

    DESC_STRUCT{curIdx} = shapeData;


end
DESC_STRUCT = [DESC_STRUCT{:}];
