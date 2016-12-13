% Computes bags of features
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.


% List of vocabularies
VOCABS = dir(fullfile(VOCAB_DIR, 'vocab*.mat'));
VOCABS = {VOCABS.name};

% List of shapes
SHAPES = dir(fullfile(DESC_DIR, FILES_TO_PROCESS));
SHAPES = {SHAPES.name};

IS_COMPUTE_SSBOFS = false;

% Create directories
for v = 1:length(VOCABS),
    vocabname = VOCABS{v};
    warning off;
    mkdir(fullfile(BOF_DIR, chop_extension(vocabname)));
    warning on;    
end


% Statistics
tic;
fprintf(1, 'Computing bags of features...\n');
nok   = 0;
nskip = 0;
for s = 1:length(SHAPES), 

    shapename = SHAPES{s};
    fprintf(1, '  %-30s \t ', shapename);    
    
    % Skip existing files
    if SKIP_EXISTING,
        skip = true;
        for v = 1:length(VOCABS), 
            vocabname = VOCABS{v};
            if exist(fullfile(fullfile(BOF_DIR, chop_extension(vocabname)), shapename), 'file'),
                continue; 
            end
            skip = false;
            break;
        end
        if skip,
            fprintf(1, 'files already exist, skipping\n');
            nskip = nskip+1;
            continue;
        end
    end
    
    % Load descriptors and kernels
    load(fullfile(DESC_DIR,   shapename), 'desc', 'T');
    if IS_COMPUTE_SSBOFS,
        load(fullfile(KERNEL_DIR, shapename), 'Kxy', 'Txy');
    end
    load(fullfile(EVECS_DIR,  shapename), 'A', 'shape');    
    
    str = '';
    for v = 1:length(VOCABS),
        
        % Skip existing files
        vocabname = VOCABS{v};
        if SKIP_EXISTING && exist(fullfile(fullfile(BOF_DIR, chop_extension(vocabname)), shapename), 'file'),
            str = mprintf(str, '%s skipped', vocabname);
            continue; 
        end
   
        % Load vocabulary
        str = mprintf(str, '%s', vocabname);
        load(fullfile(VOCAB_DIR, vocabname));
        
        % Compute BOFS & save results
        if IS_COMPUTE_SSBOFS
            [BOF, SSBOF] = bof(vocab, sigma*SIGMA_SCALE, desc, DESCRIPTOR_NORMALIZATION, A, Kxy);
            save(fullfile(fullfile(BOF_DIR, chop_extension(vocabname)), shapename), 'BOF', 'SSBOF');

        else
            [BOF] = bof(vocab, sigma*SIGMA_SCALE, desc, DESCRIPTOR_NORMALIZATION, A, []);
            save(fullfile(fullfile(BOF_DIR, chop_extension(vocabname)), shapename), 'BOF');
        end    
        
    end
    
    mprintf(str, '\n');
    nok = nok+1;
    
end

% Statistics
fprintf(1, '\nComputation complete\n');
fprintf(1, ' Elapsed time:   %s\n', format_time(toc));
fprintf(1, ' Total Shapes:   %d\n', length(SHAPES));
fprintf(1, ' Computed:       %d\n', nok);
fprintf(1, ' Skipped:        %d\n', nskip);

