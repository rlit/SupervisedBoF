% Computes groundtruth classes
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.


if ~isdir(GROUNDTRUTH_DIR)
    mkdir(GROUNDTRUTH_DIR)
end

% List of shapes
SHAPES = dir(fullfile(DESC_DIR, FILES_TO_PROCESS));
SHAPES = {SHAPES.name};

% Decode file names into shape number, xform and strength
s = cellfun(@shapeattr, SHAPES);
shapeid = cellfun(@(x)(x), {s.num}, 'UniformOutput', true);
xform = {s.xform};
strength = cellfun(@(x)(x), {s.strength}, 'UniformOutput', true);

% Assign labels to shape numbers
LABELS = { 'centaur', 'man', 'dog', 'cat', 'man', 'woman', 'horse', ...
           'camel', 'cat', 'elephant', '', 'flamingo', '', ...
           'horse', 'lion', 'other' };
LABELS(16:max(shapeid)) = {'other'};
LABELS = LABELS(shapeid);

% Mask for positives, negatives and don't cares
N = repmat(shapeid(:),[1 length(shapeid)]);
MASK = double(N==N');
MASK(MASK==0) = -1;
MASK(N~=N' & N>15 & N'>15) = 0;
L = repmat(LABELS(:), [1 length(shapeid)]);
M = (N~=N' & strcmpi(L,'man') & (strcmpi(L','man') | strcmpi(L','woman') | strcmpi(L','centaur')));
M = M | (N~=N' & strcmpi(L,'woman') & (strcmpi(L','man') | strcmpi(L','woman') | strcmpi(L','centaur')));
M = M | (N~=N' & strcmpi(L,'horse') & (strcmpi(L','horse') | strcmpi(L','centaur')));
M = M | (N~=N' & strcmpi(L,'cat') & strcmpi(L','cat'));
MASK(M|M'==1) = 0;

% Save
save(fullfile(GROUNDTRUTH_DIR, 'classes'), 'MASK', 'LABELS', 'shapeid', 'xform', 'strength');

