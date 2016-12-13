% Very fast version of kmeans clustering.
%
% Cluster the N x p matrix X into k clusters using the kmeans algorithm. It returns the
% cluster memberships for each data point in the N x 1 vector IDX and the K x p matrix of
% cluster means in C. 
%
% Custom implementation of the kmeans algorithm.  In some ways it is less general (for
% example only uses euclidian distance), but it has some options that the matlab version
% does not (for example, it has a notion of outliers and min-cluster size).  It is also
% many times faster than matlab's kmeans.  General kmeans help can be found in help for
% the matlab implementation of kmeans. Note that the although the names and conventions
% for this algorithm are taken from Matlab's implementation, there are slight
% alterations (for example, IDX==-1 is used to indicate outliers).
%
% 
% -------------------------------------------------------------------------
% INPUTS
% 
%  X
% n-by-p data matrix of n p-dimensional vectors.  That is X(i,:) is the ith point in X.
%
%  k
% Integer indicating the maximum nuber of clusters for kmeans to find. Actual number may
% be smaller (for example if clusters shrink and are eliminated).
%
% -------------------------------------------------------------------------
% ADDITIONAL INPUTS
%
% [...] = kmeans2(...,'param1',val1,'param2',val2,...) enables you to
% specify optional parameter name-value pairs to control the iterative
% algorithm used by kmeans. Valid parameters are the following:
%   'replicates'  - Number of times to repeat the clustering, each with a
%                   new set of initial cluster centroid positions. kmeans
%                   returns the solution with the lowest value for sumd.
%   'maxiter'     - Maximum number of iterations. Default is 100.
%   'display'     - Whether or not to display algorithm status (default==0)
%   'randstate'   - seed with which to initialize kmeans.  Useful for
%                   replicability of algoirhtm.
%   'outlierfrac' - maximum fraction of points that can be treated as
%                   outliers   
%   'minCsize'    - minimum size for a cluster (smaller clusters get
%                   eliminated)
%
% -------------------------------------------------------------------------
% OUTPUTS
%
%  IDX
% n-by-1 vector used to indicated cluster membership.  Let X be a set of n points.  Then
% the ID of X - or IDX is a column vector of length n, where each element is an integer
% indicating the cluster membership of the corresponding point in X.  That is IDX(i)=c
% indicates that the ith point in X belongs to cluster c. Cluster labels range from 1 to
% k, and thus k=max(IDX) is typically the number of clusters IDX divides X into.  The
% cluster label "-1" is reserved for outliers.  That is IDX(i)==-1 indicates that the
% given point does not belong to any of the discovered clusters.  Note that matlab's
% version of kmeans does not have outliers.
%
%  C        
% k-by-p matrix of centroid locations.  That is C(j,:) is the cluster centroid of points
% belonging to cluster j.  In kmeans, given X and IDX, a cluster centroid is simply the
% mean of the points belonging to the given cluster, ie: C(j,:) = mean( X(IDX==j,:) ). 
%
%  sumd
% 1-by-k vector of within-cluster sums of point-to-centroid distances. That is sumd(j) is
% the sum of the distances from X(IDX==j,:) to C(j,:). The total sum, sum(sumd), is a
% typical error measure of the quality of a clustering. 
%
% -------------------------------------------------------------------------
%
 
function [IDX,C,sumd] = kmeans (X,k,varargin )

    %%% get input args   (NOT SUPPORTED:  distance, emptyaction, start )
    pnames = {  'replicates' 'maxiter' 'display' 'randstate' 'outlierfrac' 'minCsize'};
    dflts =  {       1        100         0           []          0             1    };
    [errmsg,replicates,maxiter,display,randstate,outlierfrac,minCsize] = ...
                                                    getargs(pnames, dflts, varargin{:});
    error(errmsg);
    if (k<=1) error('k must be greater than 1'); end;
    if(ndims(X)~=2 || any(size(X)==0)) error('Illegal X'); end;
    if (outlierfrac<0 || outlierfrac>=1) 
        error('fraction of outliers must be between 0 and 1'); end;
    noutliers = floor( size(X,1)*outlierfrac );

    % initialize seed if it was not specified by user, otherwise set it.
    if (isempty(randstate)) randstate = rand('state'); else rand('state',randstate); end;

    % run kmeans2_main replicates times
    %msg = ['Running kmeans2 with k=' num2str(k)]; 
    %if (replicates>1) msg=[msg ', ' num2str(replicates) ' times.']; end;
    %if (display) disp(msg); end;

    %fprintf(1, '  k-means    clusters: %d   repetitions: %d\n', k, replicates);
   
    tic;
    bestsumd = inf; 
    for i=1:replicates
        
        [IDX,C,sumd,niters] = kmeans2_main(X,k,noutliers,minCsize,maxiter,display, i, replicates);
        if (sum(sumd)<sum(bestsumd)) bestIDX = IDX; bestC = C; bestsumd = sumd; end
        
    end
    
    IDX = bestIDX; C = bestC; sumd = bestsumd; k = max(IDX);  
    %msg = ['Final number of clusters = ' num2str( k ) ';  sumd=' num2str(sum(sumd))]; 
    %if (display) disp(msg); end;    
       
    % sort IDX to have biggest clusters have lower indicies
    clustercounts = zeros(1,k); for i=1:k clustercounts(i) = sum( IDX==i ); end
    [ids,order] = sort( -clustercounts );  C = C(order,:);  sumd = sumd(order);
    IDX2 = IDX;  for i=1:k IDX2(IDX==order(i))=i; end; IDX = IDX2; 

    cnt = clustercounts; 
    cnt(cnt <= 0) = 1;
    radius = sumd./cnt;

    % Summarize results
    mprintf('', '   Training time  : %s\n', format_time(toc));
    mprintf('', '   Training error : %.4f\n', sum(sumd)/size(X,1));    
    mprintf('', '   Cluster size   : min %-6d \t max %-6d\n', min(clustercounts), max(clustercounts));    
    mprintf('', '   Cluster radius : min %-06.4f \t max %-06.4f \t avg %-06.4f\n', min(radius), max(radius), mean(radius));    
        
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [IDX,C,sumd,niters] = kmeans2_main(X,k,noutliers,minCsize,maxiter,display, repetition, maxrep)    

    % initialize the vectors containing the IDX assignments
    % and set initial cluster centers to be k random X points
    [N p] = size(X);
    IDX = ones(N,1); oldIDX = zeros(N,1);
    index = randperm2(N,k);  C = X(index,:); 
    
    % MAIN LOOP: loop until the cluster assigments do not change
    niters = 0;  
    %ndisdigits = ceil( log10(maxiter-1) );
    %if( display ) fprintf( ['\b' repmat( '0',[1,ndisdigits] )] ); end;
    
    str = '';
    while( sum(abs(oldIDX - IDX)) ~= 0 && niters < maxiter)
        
        fraction = (niters/maxiter + (repetition-1))/maxrep;
        time_elapsed = toc;
        if fraction > 0,
            time_rem = ['remaining: ' format_time(time_elapsed*(1-fraction)/fraction)];
        else
            time_rem = '';
        end
        str = mprintf(str, '  %.1f%% complete   %d.%d / %d.%d   elapsed: %s   %s', ...
            fraction*100, repetition, niters+1, maxrep, maxiter, ...
            format_time(time_elapsed), time_rem);
        
        % Build tree on cluster centers
        tree = ann('init', C');

        % calculate the Euclidean distance between each point and each cluster mean
        % and find the closest cluster mean for each point and assign it to that cluster
        oldIDX = IDX;  
        %D = dist_euclidean( X, C ); 
        %[mind IDX] = min(D,[],2);  
        [IDX,mind] = ann('search', tree, X', 1, 'eps', 1); IDX = double(IDX(:)); mind = mind(:);

        % do not use most distant noutliers elements in computation of cluster centers
        mindsort = sort( mind ); thr = mindsort( end-noutliers );  IDX( mind > thr ) = -1; 

        % discard small clusters [place in outlier set, will get included next time around]
        i=1; while(i<=k) if (sum(IDX==i)<minCsize) IDX(IDX==i)=-1; 
                if(i<k) IDX(IDX==k)=i; end; k=k-1; else i=i+1; end; end
        if( k==0 ) 
            IDX( randint2( 1,1, [1,N] ) ) = 1; 
            k=1; 
        end;
        for i=1:k if ((sum(IDX==i))==0) 
                error('should never happen - empty cluster!'); end; end;        

        % Recalculate the cluster means based on new assignment (loop is compiled - fast!)
        % Actually better then looping over k, because X(IDX==i) is slow. 
        C = zeros(k,p);  counts = zeros(k,1);
        for i=find(IDX>0)' IDx = IDX(i); counts(IDx)=counts(IDx)+1; 
            C(IDx,:) = C(IDx,:)+X(i,:); end
        C = C ./ counts(:,ones(1,p));
        
        ann('deinit', tree);
        
        niters = niters+1;
        %if( display ) 
        %    fprintf( [repmat('\b',[1 ndisdigits]) int2str2(niters,ndisdigits)] ); end;
            %fprintf( [repmat('\b',[1 ndisdigits]) int2str(niters,ndisdigits)] ); end;
            
        %fprintf(1, '%c', ones(strlen,1)*8);
        %str = sprintf('iteration:  %-6d   elapsed time: %.2fh', niters, toc/3600);
        %fprintf(1, '%s', str);
        %strlen = length(str);
    end
    
    str = mprintf(str, '');    

    %fprintf(1, '%c', ones(strlen,1)*8);
    %fprintf(1, 'iterations: %-6d   ', niters);
    
    % record within-cluster sums of point-to-centroid distances 
    sumd = zeros(1,k); for i=1:k sumd(i) = sum( mind(IDX==i) ); end

    
% Utility to process parameter name/value pairs.
%
% Based on code fromt the Matlab Statistics Toolobox "private/statgetargs.m"
%
% [EMSG,A,B,...]=GETARGS(PNAMES,DFLTS,'NAME1',VAL1,'NAME2',VAL2,...) accepts a cell
% array PNAMES of valid parameter names, a cell array DFLTS of default values for the
% parameters named in PNAMES, and additional parameter name/value pairs.  Returns
% parameter values A,B,... in the same order as the names in PNAMES.  Outputs
% corresponding to entries in PNAMES that are not specified in the name/value pairs are
% set to the corresponding value from DFLTS.  If nargout is equal to length(PNAMES)+1,
% then unrecognized name/value pairs are an error.  If nargout is equal to
% length(PNAMES)+2, then all unrecognized name/value pairs are returned in a single cell
% array following any other outputs.
%
% EMSG is empty if the arguments are valid, or the text of an error message if an error
% occurs.  GETARGS does not actually throw any errors, but rather returns an error
% message so that the caller may throw the error. Outputs will be partially processed
% after an error occurs.
%
% EXAMPLE
%   pnames = {'color' 'linestyle', 'linewidth'}
%   dflts  = {    'r'         '_'          '1'}
%   v = {'linew' 2 'nonesuch' [1 2 3] 'linestyle' ':'};
%   [emsg,color,linestyle,linewidth,unrec] = getargs(pnames,dflts,v{:}) % ok
%   [emsg,color,linestyle,linewidth] = getargs(pnames,dflts,v{:})    % error
%
 
function [emsg,varargout]=getargs(pnames,dflts,varargin)
    % We always create (nparams+1) outputs:
    %    one for emsg
    %    nparams varargs for values corresponding to names in pnames
    % If they ask for one more (nargout == nparams+2), it's for unrecognized
    % names/values
    emsg = '';
    nparams = length(pnames);
    varargout = dflts;
    unrecog = {};
    nargs = length(varargin);

    % Must have name/value pairs
    if mod(nargs,2)~=0
        emsg = sprintf('Wrong number of arguments.');
    else
        % Process name/value pairs
        for j=1:2:nargs
            pname = varargin{j};
            if ~ischar(pname)
                emsg = sprintf('Parameter name must be text.');
                break;
            end
            i = strmatch(lower(pname),lower(pnames));
            if isempty(i)
                % if they've asked to get back unrecognized names/values, add this
                % one to the list
                if nargout > nparams+1
                    unrecog((end+1):(end+2)) = {varargin{j} varargin{j+1}};

                    % otherwise, it's an error
                else
                    emsg = sprintf('Invalid parameter name:  %s.',pname);
                    break;
                end
            elseif length(i)>1
                emsg = sprintf('Ambiguous parameter name:  %s.',pname);
                break;
            else
                varargout{i} = varargin{j+1};
            end
        end
    end

    varargout{nparams+1} = unrecog;

    
    
% Returns a random permutation of integers. 
%
% randperm2(n) is a random permutation of the integers from 1 to n.  For example,
% randperm2(6) might be [2 4 5 6 1 3].  randperm2(n,k) is only returns the first k
% elements of the permuation, so for example randperm2(6) might be [2 4].
% 
% This is a faster version of randperm.m if only need first k<<n elements of the random
% permutation.  Also uses less random bits (only k).  Note that this is an implementation
% O(k), versus the matlab implementation which is O(nlogn), however, in practice it is
% often slower for k=n because it uses a loop.
%
% INPUTS
%   n 	- permute 1:n
%   k   - keep only first k outputs
%
% OUTPUTS
%   p 	- k length vector of permutations
%
% EXAMPLE
%   randperm2(10,5)
%
% DATESTAMP
%   29-Sep-2005  2:00pm
 
function p = randperm2(n,k);

    if (nargin<2) k=n; else k = min(k,n); end

    p = 1:n;
    for i=1:k
        r = i + floor( (n-i+1)*rand );     
        t = p(r);  p(r) = p(i);  p(i) = t;
    end
    p = p(1:k);

    
% Calculates the Euclidean distance between vectors [FAST].
%
% Assume X is an m-by-p matrix representing m points in p-dimensional space and Y is an
% n-by-p matrix representing another set of points in the same space. This function
% compute the m-by-n distance matrix D where D(i,j) is the SQUARED Euclidean distance
% between X(i,:) and Y(j,:).  Running time is O(m*n*p).
%
% If x is a single data point, here is a faster, inline version to use:
%   D = sum( (Y - ones(size(Y,1),1)*x).^2, 2 )';
%
% INPUTS
%   X   - m-by-p matrix of m p dimensional vectors 
%   Y   - n-by-p matrix of n p dimensional vectors 
%
% OUTPUTS
%   D   - m-by-n distance matrix
%
% EXAMPLE
%   X=[randn(100,5)]; Y=randn(40,5)+2;
%   D = dist_euclidean( [X; Y], [X; Y] ); im(D)
%
% DATESTAMP
%   29-Sep-2005  2:00pm
%
% See also DIST_CHISQUARED, DIST_EMD

% Piotr's Image&Video Toolbox      Version 1.03   
% Written and maintained by Piotr Dollar    pdollar-at-cs.ucsd.edu 
% Please email me if you find bugs, or have suggestions or questions! 
 
function D = dist_euclidean( X, Y )
    if( ~isa(X,'double') || ~isa(Y,'double'))
        error( 'Inputs must be of type double'); end;
    m = size(X,1); n = size(Y,1);  
    Yt = Y';  
    XX = sum(X.*X,2);        
    YY = sum(Yt.*Yt,1);      
    D = XX(:,ones(1,n)) + YY(ones(1,m),:) - 2*X*Yt;
    
    
    

%%%% code from Charles Elkan with variables renamed
%    m = size(X,1); n = size(Y,1);
%    D = sum(X.^2, 2) * ones(1,n) + ones(m,1) * sum(Y.^2, 2)' - 2.*X*Y';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    
%%%% LOOP METHOD - SLOW
%     [m p] = size(X);  
%     [n p] = size(Y);
%     
%     D = zeros(m,n);
%     ones_m_1 = ones(m,1);
%     for i=1:n
%         y = Y(i,:);
%         d = X - y(ones_m_1,:);
%         D(:,i) = sum( d.*d, 2 );  
%     end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% PARALLEL METHOD THAT IS SUPER SLOW (slower then loop)!
% % Code taken from "MATLAB array manipulation tips and tricks" by Peter J. Acklam
%     Xb = permute(X, [1 3 2]);  
%     Yb = permute(Y, [3 1 2]);
%     D = sum( (Xb(:,ones(1,n),:) - Yb(ones(1,m),:,:)).^2, 3);    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%    


%%%%% USELESS FOR EVEN VERY LARGE ARRAYS X=16000x1000!! and Y=100x1000
%     % call recursively to save memory
%     if( (m+n)*p > 10^5 && (m>1 || n>1))
%         if( m>n )
%             X1 = X(1:floor(end/2),:);
%             X2 = X((floor(end/2)+1):end,:);
%             D1 = dist_euclidean( X1, Y );
%             D2 = dist_euclidean( X2, Y );
%             D = cat( 1, D1, D2 );
%         else
%             Y1 = Y(1:floor(end/2),:);
%             Y2 = Y((floor(end/2)+1):end,:);
%             D1 = dist_euclidean( X, Y1 );
%             D2 = dist_euclidean( X, Y2 );
%             D = cat( 2, D1, D2 );
%         end
%         return;
%     end 
%        
    
    
    
    
