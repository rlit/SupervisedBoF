% Sets up environment for ShapeGoogle experiments and ShapeSIFT benchmarks
%
% (C) Copyright Alex Bronstein, Michael Bronstein, Maks Ovsjanikov,
% Stanford University, 2009. All Rights Reserved.
%
% Altered by Roee Litman, Tel Aviv University, 2013


%% EIGS Param
LB_PARAM = 'cot'; % 'neu', 'dir', 'cot', 'euc', 'geo'
% 'neu' - 1st order FEM Neumann
% 'dir' - 1st order FEM Dirichlet
% 'cot' - cotangent weights
% 'euc' - euclidean weights
% 'geo' - geodesic weights

%% Directories

% Root directory for data
DATA_ROOT_DIR           = fullfile(pwd, '../../data/SHREC14/All');
DATA_ROOT_DIR           = fullfile(pwd, '../../data/ShapeGoogle');

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
DICT_DIR                = fullfile(DATA_ROOT_DIR, ['dict.' LB_PARAM]);

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
addpath(fullfile(CODE_ROOT_DIR, 'spams'));          % some code from the SPAMS toolbox
addpath(pwd);
addpath(fullfile(pwd, '../supervised'));            % code for supervised dictionary learning


%% Parameters

% Skip existing files in precomputation
SKIP_EXISTING           = true;
MAX_POSITIVE_SHAPES     = 20; %15;
if ~isempty(strfind(DATA_ROOT_DIR,'SHREC14'))
MAX_POSITIVE_SHAPES     = 2000; %15;
end

%% shape dna params
SHAPE_DNA_SIZE = 100;
MAX_MUN_EVECS  = 200;

%% Descriptors
DESC_TYPE = 'HKS';
if ~isempty(strfind(DATA_ROOT_DIR,'SHREC14'))
DESC_TYPE = 'SIHKS'; %'HKS';
end

% Descriptors are computed at logarithmic time samples from TIME_START
% to TIME_END with TIME_SAMPLES_PER_OCTAVE samples per scale octave,
% where scale is sqrt(t).
TIME_START              = 128;
TIME_END                = 8192;
TIME_SAMPLES_PER_OCTAVE = 10;

% Heat kernels Kt(x,y) are computed at these time samples
TIME_KERNEL             = [1024]; % * 0.0001; %[1024 2048 4096];   %[512 1024 2048];

%T = timerange(TIME_SAMPLES_PER_OCTAVE, [TIME_START TIME_END])
%numel(T)

%% Vocabularies

% Vocabularies of these sizes are computed
% VOCAB_SIZES            = [4 8 16 24 32 48 64 128];
VOCAB_SIZES            = 48;

% Training set size for kmeans
VOCAB_TRAININGSET_SIZE = 3e5;
VOCAB_TRAIN_NITER      = 250;
VOCAB_TRAIN_REPEATS    = 5;
VOCAB_TRAIN_OUTLIERS   = 0.01;

TRAIN_SET_NAME = 'null+1iso';

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

%% Dictionary learning
% Dictionary of these sizes are computed
DICT_SIZES            = [32 48 64];%2.^(5:7);

% Training set size for mexTrainDL
DICT_TRAININGSET_SIZE = VOCAB_TRAININGSET_SIZE;

IS_NONNEGATIVE = false;

%DICT_TRAIN_PARAMS.K= ;  % learns a dictionary with ? elements
DICT_TRAIN_PARAMS.lambda=0.5;
DICT_TRAIN_PARAMS.lambda2 = 0.005;
DICT_TRAIN_PARAMS.numThreads = -1; % number of threads
DICT_TRAIN_PARAMS.batchsize = 256;
DICT_TRAIN_PARAMS.pos  = IS_NONNEGATIVE;
DICT_TRAIN_PARAMS.iter = 5000;

%% Sparse decomposition
SD_POOLING = 'avg';

LASSO_PARAMS.mode = 2;
LASSO_PARAMS.lambda     = DICT_TRAIN_PARAMS.lambda;
LASSO_PARAMS.lambda2    = DICT_TRAIN_PARAMS.lambda2;
LASSO_PARAMS.numThreads = DICT_TRAIN_PARAMS.numThreads;
LASSO_PARAMS.pos = IS_NONNEGATIVE; % enforce nonnegativity

LASSO_PARAMS.L = numel(timerange(TIME_SAMPLES_PER_OCTAVE, [TIME_START TIME_END]));


% Distance between BoFs used for by-class and vocabulary size experiments
SD_DISTANCE    = 'l1vec';

%% Misc

% Random generators seeds
% rand('seed', 0);
% randn('seed', 0);

% Line styles for plots
LINESTYLES = {'-k', '-r', ':g', '-.b', '--c', '-m', ':y', ':k'};

