function [descPooled, poolGradFun, varargout] = GetPooledDescriptor(poolingMethod, descIn, varargin)

thFactor = 1;
nrmType = 1;

[dSize] = size(descIn,1);

descTH = atan(thFactor*descIn)*2/pi;
% hist([descIn(descIn>0) descTH(descIn>0)],100)
% stem(descIn(:,1:50))
% %%
switch poolingMethod
    case 'avg'
        w = varargin{1};
        w = w/sum(w);
        descPooled = descTH * w;

    case 'avg_metric'
        w = varargin{1};
        w = w/sum(w);
        P = varargin{2};
        descPooled = descTH * w;

    otherwise
        assert(0,'unknown method')
end

if 1
    descPreNorm = descPooled;
    [descPooled,descNorm] = normalize(descPooled,nrmType,1);
end


if strcmp(poolingMethod, 'avg_metric')
    descPreP = descPooled;
    descPooled = P * descPooled;
end


if nargout ==1
    return
end

%% create gradint function
assert(size(descPooled,2)==1,'no support for multiple smaples yet')
switch nrmType
    case 1
        J = (eye(dSize) - sign(descPreNorm)*descPreNorm'/descNorm)    / descNorm;
    case 2
        J = (eye(dSize) -       descPreNorm*descPreNorm'/descNorm.^2) / descNorm;
end


switch poolingMethod
    case 'avg'
        dDescTH = thFactor * numel(w) * 2 /pi ./ (1+(thFactor*descIn).^2);
        poolGradFun = @(g)dDescTH.*(J * g * w');

    case 'avg_metric'
        dDescTH = thFactor * numel(w) * 2 /pi ./ (1+(thFactor*descIn).^2);
        poolGradFun = @(g)dDescTH.*(J * P' * g * w');

        PgradFun = @(g)g * (descPreP)';
        varargout{1} = PgradFun;

    otherwise
        assert(0,'unknown method')
end


