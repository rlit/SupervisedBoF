% Sets up environment for ShapeGoogle experiments and ShapeSIFT benchmarks
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.


%% EIGS Param
LB_PARAM = 'cot'; % 'neu', 'dir', 'cot', 'euc', 'geo'
% 'neu' - 1st order FEM Neumann
% 'dir' - 1st order FEM Dirichlet 
% 'cot' - cotangent weights
% 'euc' - euclidean weights
% 'geo' - geodesic weights

%% Directories

% Root directory for data
%DATA_ROOT_DIR           = fullfile(pwd, '../../data/SHREC');
DATA_ROOT_DIR           = fullfile(pwd, '../../data');

% Relative data directories
SHAPE_DIR               = fullfile(DATA_ROOT_DIR, 'shapes');
GROUNDTRUTH_DIR         = fullfile(DATA_ROOT_DIR, 'groundtruth');
EVECS_DIR               = fullfile(DATA_ROOT_DIR, ['evecs.' LB_PARAM]);
DESC_DIR                = fullfile(DATA_ROOT_DIR, ['descriptors.' LB_PARAM]);
KERNEL_DIR              = fullfile(DATA_ROOT_DIR, ['kernels.' LB_PARAM]);
VOCAB_DIR               = fullfile(DATA_ROOT_DIR, ['vocabs.' LB_PARAM]);
BOF_DIR                 = fullfile(DATA_ROOT_DIR, ['bofs_soft.' LB_PARAM]);
PERF_DIR                = fullfile(DATA_ROOT_DIR, ['perf.' LB_PARAM]);
FIG_DIR                 = fullfile(DATA_ROOT_DIR, 'fig');

% Wildcard for files to process
FILES_TO_PROCESS        = '*.mat';

% Root directory for code
CODE_ROOT_DIR = fullfile(pwd, '../common');


fprintf('USING %s\n',CODE_ROOT_DIR);
%% Code 

% Set path for auxilary and shared code
addpath(CODE_ROOT_DIR);
addpath(fullfile(CODE_ROOT_DIR, 'utils'));          % generic utilities
addpath(genpath(fullfile(CODE_ROOT_DIR, 'laplacian')));      % mesh Laplacian
addpath(fullfile(CODE_ROOT_DIR, 'descriptors'));    % scale space & descriptors
addpath(fullfile(CODE_ROOT_DIR, 'vq'));             % vector quantization
addpath(fullfile(CODE_ROOT_DIR, 'bofs'));           % bags of features
addpath(pwd);


%% Parameters

% Skip existing files in precomputation
SKIP_EXISTING           = true;  
MAX_POSITIVE_SHAPES     = 20; %15;

%% shape dna params
SHAPE_DNA_SIZE = 100;

%% Descriptors

% Descriptors are computed at logarithmic time samples from TIME_START
% to TIME_END with TIME_SAMPLES_PER_OCTAVE samples per scale octave,  
% where scale is sqrt(t).
TIME_START              = 1024; % * 0.0001;             
TIME_END                = 4000; % * 0.0001;
TIME_SAMPLES_PER_OCTAVE = 5;

% Heat kernels Kt(x,y) are computed at these time samples
TIME_KERNEL             = [1024]; % * 0.0001; %[1024 2048 4096];   %[512 1024 2048];


%% Vocabularies

% Vocabularies of these sizes are computed
%VOCAB_SIZES            = [4 8 16 24 32 48 64];
VOCAB_SIZES            = 48;

% Training set size for kmeans 
VOCAB_TRAININGSET_SIZE = 3e5;
VOCAB_TRAIN_NITER      = 250;
VOCAB_TRAIN_REPEATS    = 5;
VOCAB_TRAIN_OUTLIERS   = 0.01;

% How to normalize the descriptor vector 
% Can be 'none', 'L1' or 'L2'
DESCRIPTOR_NORMALIZATION = 'L2';
SIGMA_SCALE            = 2;


%% BoFs

BOF_DISTANCES   = {'l1vec', 'tfidf'};

% Distance between BoFs used for by-class and vocabulary size experiments
BOF_DISTANCE    = 'l1vec';
SSBOF_DISTANCE  = 'l1vec';
SHAPE_DNA_DISTANCE = 'l2vec';

% Vocabulary used for by-class experiements
BOF_VOCAB       = 'vocab48';


%% Misc

% Random generators seeds
rand('seed', 0);
randn('seed', 0);

% Line styles for plots
LINESTYLES = {'-k', '-r', ':g', '-.b', '--c', '-m', ':y', ':k'};

