function logp = crp_logprob(T,alpha)
    
    % Log probability of a partition under the CRP.
    %
    % USAGE: logp = crp_logprob(T,alpha)
    %
    % INPUTS:
    %   T - vector of counts for each group in the partition.
    %   alpha - concentration parameter
    
    K = length(T);
    N = sum(T);
    
    logp = K*log(alpha) + sum(gammaln([T alpha])) - gammaln(N+alpha);