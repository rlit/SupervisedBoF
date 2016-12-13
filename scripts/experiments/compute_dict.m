% Builds synthesis dictionary for descriptors
%
% Written by Roee Litman, Tel Aviv University, 2013


% list of training shapes
groundtruth_classes
shapesToUse = DefineTrainingSet(LABELS,TRAIN_SET_NAME);
shapesToUse = SHAPES(shapesToUse);

% Descriptor dimensionality
tmp = load(fullfile(DESC_DIR, shapesToUse{1}), 'desc');
dimdesc = size(tmp.desc,2);
LASSO_PARAMS.L = dimdesc;

% Allocate space for descriptors
Xp = zeros(8e5, dimdesc, 'single');

% Build training set
fprintf(1, 'Building training set...\n');
count = 0;
nshapes = 0;
str = '';


for n = 1:MAX_POSITIVE_SHAPES,

    SHAPES = dir(fullfile(DESC_DIR, sprintf('%04d.*.mat', n)));
    SHAPES = {SHAPES.name};

    for ii = 1:length(SHAPES)
        if ~ismember(SHAPES{ii},shapesToUse),continue,end

        % Load descriptors
        load(fullfile(DESC_DIR, SHAPES{ii}), 'desc');

        % Add descriptors to training set
        m = size(desc,1);
        Xp(count+[1:m],:) = desc;
        count = count+m;
        nshapes = nshapes+1;

        % Print statistics
        str = mprintf(str, '  Shapes: %-4d \t Descriptors: %s', nshapes, format_number(count));

    end

end

% Print final statistics
mprintf(str, 'Loaded %s descriptors from %d shapes\n', format_number(count), nshapes);

% Select a random subset
idx = randperm(count);
idx = idx(1:min(VOCAB_TRAININGSET_SIZE,count));
Xp = Xp(idx,:);

% Normalize descriptors
Xp = double(Xp);
Xp = normalize(Xp, DESCRIPTOR_NORMALIZATION, 2);

DICT_DIR_L = [DICT_DIR '_L' sprintf('%.2f',DICT_TRAIN_PARAMS.lambda)];
% DICT_DIR_L = [DICT_DIR];
if ~isdir(DICT_DIR_L)
    mkdir(DICT_DIR_L);
end

% Train dictionaries of all sizes
for k=1:length(DICT_SIZES),

    dicsSize = DICT_SIZES(k);

    if SKIP_EXISTING && exist(fullfile(DICT_DIR_L, ['dict' num2str(dicsSize) '.mat']), 'file'),
        fprintf(1, 'Skipping dictionary of size %d, file already exists\n', dicsSize);
        continue;
    end

    fprintf(1, 'Training dictionary of size %d...\n', dicsSize);

    % Train dictionary
    params = DICT_TRAIN_PARAMS;
    params.lasso_params = LASSO_PARAMS;
    params.K = dicsSize;
    D = mexTrainDL_Memory(double(Xp'),params);
    M = eye(size(D,1));


    % Save dictionary
    save(fullfile(DICT_DIR_L, ['dict' num2str(dicsSize) '.mat']), 'D', 'M', 'params','TRAIN_SET_NAME');

end



