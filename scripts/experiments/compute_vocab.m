% Builds vocabularies for descriptors
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.
%
% Altered by Roee Litman, Tel Aviv University, 2013


% list of training shapes
groundtruth_classes
shapesToUse = DefineTrainingSet(LABELS,TRAIN_SET_NAME);
shapesToUse = SHAPES(shapesToUse);

% Descriptor dimensionality
tmp = load(fullfile(DESC_DIR, SHAPES{1}), 'desc');
dimdesc = size(tmp.desc,2);

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

    for ii = 1:length(SHAPES),
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

warning off;
mkdir(VOCAB_DIR);
warning on;    

% Train vocabularies of all sizes
for k=1:length(VOCAB_SIZES),

    nvocab = VOCAB_SIZES(k);
    
    if SKIP_EXISTING && exist(fullfile(VOCAB_DIR, ['vocab' num2str(nvocab) '.mat']), 'file'),
        fprintf(1, 'Skipping vocabulary of size %d, file already exists\n', nvocab);
        continue;
    end
    
    fprintf(1, 'Training vocabulary of size %d...\n', nvocab);
        
    % Train vocabulary
    [idx, vocab] = kmeans(double(Xp), nvocab, ...
                'maxiter', VOCAB_TRAIN_NITER, ...
                'display', 1, ...
                'replicates', VOCAB_TRAIN_REPEATS, ...
                'randstate', 0, ...
                'outlierfrac', VOCAB_TRAIN_OUTLIERS);    
            
    % Compute sigma        
    tree = ann('init', vocab');
    [sig, mind] = ann('search', tree, Xp', 1, 'eps', 1.1);
    sigma = median(mind);
    ann('deinit', tree);
    clear ann;

    % Save vocabulary
    save(fullfile(VOCAB_DIR, ['vocab' num2str(nvocab) '.mat']), 'vocab', 'sigma');
    
end



