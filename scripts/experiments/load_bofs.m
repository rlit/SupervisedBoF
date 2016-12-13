% Load bags of features
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.


% List of vocabularies
VOCABS = dir(fullfile(VOCAB_DIR, 'vocab*.mat'));
VOCABS = {VOCABS.name};

% List of shapes
SHAPES = dir(fullfile(DESC_DIR, FILES_TO_PROCESS));
SHAPES = {SHAPES.name};


% Statistics
tic;
BOFS = {};
for v = 1:length(VOCABS),

    vocabname = VOCABS{v};
    fprintf(1, 'Loading BoFs for vocabulary %s...\t', vocabname);

    str = '';
    B = {};
    for s = 1:length(SHAPES), 

        shapename = SHAPES{s};
        str = mprintf(str, '%s', shapename);    
   
        % Load BOFs
        load(fullfile(fullfile(BOF_DIR, chop_extension(vocabname)), shapename), 'BOF');
        B{s} = [{BOF}, ];
        %load(fullfile(EVECS_DIR, shapename), 'evals'); % bar
        %B{s} = [{BOF}, SSBOF, {evals(1:SHAPE_DNA_SIZE)}];
        
    end    
    str = mprintf(str, '\n');  

    % Reshape into 3D arrays
    for t=1:length(B{1}),
        b          = cellfun(@(x)(x{t}), B, 'UniformOutput', false);
        BOFS{v}{t} = reshape(cell2mat(b), [size(b{1},1) size(b{1},2) length(b)]);
    end
    
end

% Statistics
fprintf(1, '\nComplete\n');
fprintf(1, ' Elapsed time:   %s\n', format_time(toc));
fprintf(1, ' Total shapes:   %d\n', length(SHAPES));
fprintf(1, ' Total vocabs:   %d\n', length(VOCABS));

