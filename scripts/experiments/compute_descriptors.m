% Computes Kt(x,x) descriptors.
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.

SHAPES = dir(fullfile(EVECS_DIR, FILES_TO_PROCESS));
SHAPES = {SHAPES.name};

% Statistics
tic;
nerr  = 0;
nskip = 0;
nok   = 0;
fprintf(1, 'Computing descriptors...\n');

% Select time ranges
T = timerange(TIME_SAMPLES_PER_OCTAVE, [TIME_START TIME_END]);

warning off;
mkdir(DESC_DIR);
warning on;    


for s = 1:length(SHAPES), 
    
    shapename = SHAPES{s};
    fprintf(1, '  %-30s \t ', shapename);
    
    if SKIP_EXISTING && exist(fullfile(DESC_DIR, shapename), 'file'),
        fprintf(1, 'file already exists, skipping\n');
        nskip = nskip+1;
        continue;
    end

    % Load eigendecomposition
    load(fullfile(EVECS_DIR, shapename), 'evecs', 'evals');
    
    % Compute descriptors
    desc = scalespace(T, abs(evals), evecs);
    
    % Save result
    save(fullfile(DESC_DIR, shapename), 'desc', 'T');
    fprintf(1, 'OK\n');
    nok = nok+1;
     
end

% Statistics
fprintf(1, '\nComputation complete\n');
fprintf(1, ' Elapsed time:   %s\n', format_time(toc));
fprintf(1, ' Total Shapes:   %d\n', length(SHAPES));
fprintf(1, ' Computed:       %d\n', nok);
fprintf(1, ' Skipped:        %d\n', nskip);
fprintf(1, ' Errors:         %d\n', nerr);

