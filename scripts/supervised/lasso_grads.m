function grad = lasso_grads(Y, X, dY, M, D, lambda2,gradName,MtM)

if isempty(M)
    assert(~strcmp(gradName,'dM'))
    M = eye(size(D,1));
end

if ~exist('MtM','var')
    MtM = M'*M;
end

if ~exist('gradName','var')
    gradName = 'dD';
end


if size(Y,2) > 1
    
    % Initialize grads
    switch gradName
        case 'dD'
            grad = zeros(size(D));
        case 'dM'
            grad = zeros(size(M));
        case 'dX'
            grad = zeros(size(X));
        otherwise
            error('unknown gradient')
    end
    
    % compute the gradient for each data vector
    for k=1:size(Y,2)
        if strcmp(gradName,'dX')
            grad(:,k) =   lasso_grads(Y(:,k), X(:,k), dY(:,k), M, D, lambda2,gradName,MtM)/size(Y,2);
        else
            grad = grad + lasso_grads(Y(:,k), X(:,k), dY(:,k), M, D, lambda2,gradName,MtM)/size(Y,2);
        end
    end
    
    return
end

% Find active set
active = (Y~=0);
Nactive = nnz(active);

Dd = D(:,active);

bd = (Dd'*MtM*Dd + lambda2*eye(Nactive))\dY(active);
fit = X-Dd*Y(active);

% Gradient
switch gradName
    case 'dD'
        
        dD = zeros(size(D));
        dD(:,active) = -Dd*bd*Y(active)'+fit*bd';
        dD = MtM*dD;
        grad = dD;
        
    case 'dM'
        
        dM = M*Dd*bd*fit'+M*fit*bd'*Dd';
        grad = dM;
        
    case 'dX'
        dX = MtM * Dd * bd;
        grad = dX;
        
    otherwise
        error('unknown gradient')
end




