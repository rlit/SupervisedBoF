% run_experiments Executes all experiments steps
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.


% 0. Setup environment
startup                 

% 1. Load bags of features for all vocabs
load_bofs               

% 2. Load groundtruth classes
load(fullfile(GROUNDTRUTH_DIR, 'classes')); % load LABELS, MASK, shapeid, strength, xform  

% 3. Run experiments

close all

% 3a. Error rates of BoFs and SS-BoFs as function of vocabulary size
%experiment_error_rates_vs_vocab_size
 

% 3b. Error rates of SS-BoFs by class
experiment_error_rates_by_class
experiment_mAP

% 3c. Error rates of for different distance functions
%experiment_error_rates_by_distance

% 3d. Error rates of BoFs and SS-BoFs as function of noise level
%experiment_error_rates_noise


