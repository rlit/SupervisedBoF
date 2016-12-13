function isInT = DefineTrainingSet(LABELS,selectionMethod)
% { 'centaur', 'man', 'dog', 'cat', 'man', 'woman', 'horse', ...
%            'camel', 'cat', 'elephant', '', 'flamingo', '', ...
%            'horse', 'lion', 'other' };

if isempty(LABELS)
    groundtruth_classes
end

if ~exist('selectionMethod','var')
    selectionMethod = 'all_classes';
end

% Get generator settings.
warning('off','MATLAB:RandStream:ActivatingLegacyGenerators')
s = rng;
rng(0,'v4')

isInT = false(size(LABELS));

switch selectionMethod
    case 'classes'
        %% select some of the classes
        %trainLabels = {'centaur', 'camel','man','lion', 'flamingo', 'dog'};
        trainLabels = {'woman','lion','dog'};
        for l = 1:numel(trainLabels)
            isInT = isInT | strcmp(LABELS,trainLabels{l});
        end

        nNeg = nnz(~strcmp(LABELS,'other'));
        trainRatio = nnz(isInT) / nNeg;

        otherInT = randsample(find(strcmp(LABELS,'other')),ceil(nNeg*trainRatio));
        isInT(otherInT) = 1;

    case 'all_classes'
        %% select parts of all classes
        minRatio = 0.4;
        unqLabels = unique(LABELS);
        for l = 1:numel(unqLabels)
            isCurLabel = strcmp(LABELS,unqLabels{l});
            nLabel = nnz(isCurLabel);
            isSel = randsample(find(isCurLabel),ceil(minRatio*nLabel));
            isInT(isSel) = 1;
        end

    case 'null+1iso'
        %% select 2 shapes from each class
        isInT([...
            6    16    41    65    85   98,...
            118   133   154   165   188   204,...
            219   231   250   264  282   296   364,...
            428   440   452   502   556   574   588]) = 1;


    otherwise
        error('undefined training set')
end

%% Restore previous generator settings.
rng(s);
warning('on','MATLAB:RandStream:ActivatingLegacyGenerators')


return


%% list number of shapes in every class
% for L = unique(LABELS)
%     fprintf('%s - %d\n',L{1}, nnz(strcmp(LABELS,L{1})))
% end
