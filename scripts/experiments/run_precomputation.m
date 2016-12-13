% Executes all precomputation steps
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.


startup                     % 0. Setup environment
compute_evecs               % 1. Compute LB eigendecomposition
compute_descriptors         % 2. Compute Kt(x,x) descriptors
compute_kernels             % 3. Compute Kt(x,y) heat kernels
compute_vocab               % 4. Compute vocabularies
compute_bofs                % 5. Compute bags of features

                            % 6. Compute groundtruth
groundtruth_correspondence  % 6a. Correspondences
groundtruth_classes         % 6b. Positives & negatives
