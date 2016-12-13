% Computes LB eigendecomposition.
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.

SHAPES = dir(fullfile(SHAPE_DIR, FILES_TO_PROCESS));
SHAPES = {SHAPES.name};

% Current reference shape
curr_refname = '';

% Statistics
tic;
nerr  = 0;
nskip = 0;
nok   = 0;
fprintf(1, 'Computing LB eigendecomposition...\n');

warning off;
mkdir(EVECS_DIR);
warning on;    


for s = 1:length(SHAPES), 
    
    shapename = SHAPES{s};
    fprintf(1, '  %-30s \t ', shapename);
    
    if SKIP_EXISTING && exist(fullfile(EVECS_DIR, shapename), 'file'),
        fprintf(1, 'file already exists, skipping\n');
        nskip = nskip+1;
        continue;
    end

    % Load reference
    refname = [shapename(1:4) '.null.0.mat'];
    if ~strcmpi(curr_refname, refname),
        curr_refname = refname;    
        load(fullfile(SHAPE_DIR, curr_refname));
        shape_ref = shape;
    end

    % Load shape
    load(fullfile(SHAPE_DIR, shapename));

    % Translation table from shape to shape_ref
    if length(shape.X) == length(shape_ref.X), 
        LUT = [1:length(shape.X)]';
        idx = unique(shape.TRIV(:));
        LUT = LUT(idx);
        ILUT = zeros(length(idx),1);
        ILUT(idx) = 1:length(idx);
        shape.X = shape.X(idx);
        shape.Y = shape.Y(idx);
        shape.Z = shape.Z(idx);
        shape.TRIV = ILUT(shape.TRIV);
    elseif length(shape.X) < length(shape_ref.X), % sub-sample
        LUT = [1:length(shape.X)]';
    else % super-sample
        fprintf(1, 'denser sampling than in reference, skipping\n');
        nskip = nskip+1;
        continue;
    end

    % Laplacian matrices
    try
        max_num_evecs = 200;
        num_vert = length(shape.X);
        num_evecs = min(num_vert - 1, max_num_evecs);
        switch(LB_PARAM)
            case('cot')
                [evecs, evals, W, A] = main_mshlp('cotangent', shape, num_evecs);
            case('euc')
                [evecs, evals, W, A] = main_mshlp('euclidean', shape, num_evecs);
            case('geo')
                [evecs, evals, W, A] = main_mshlp('geodesic', shape, num_evecs);
            case('neu')
                fem_deg = 1;
                [evecs, evals, W, A] = fem(shape, 'neu', num_evecs, fem_deg);
            case('dir')
                fem_deg = 1;
                [evecs, evals, W, A] = fem(shape, 'dir', num_evecs, fem_deg);
            otherwise
                assert(0);
        end
    catch
        fprintf(1, 'error computing eigendecomposition, skipping\n');
        nerr = nerr+1;
        continue;
    end

    % Geodesic distance matrix and canonical form
    %G = fastmarch (shape);
 
    % Save result
    save(fullfile(EVECS_DIR, shapename), ...
         'shape', 'LUT', 'W', 'A', 'evecs', 'evals');
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

