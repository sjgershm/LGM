function [P, pz, Z] = LGM(C,v,alpha)
    
    % Latent group model.
    %
    % USAGE: [P, pz, Z] = LGM(C,v,alpha)
    %
    % INPUTS:
    %   C - [M x N] matrix of choices for M agents on N trials. Indicate
    %       missing data with nan.
    %   v - number of choice options on each trial.
    %   alpha (optional) - scalar or vector of concentration parameters (if a
    %                      vector, the model will marginalize over the values). Default: linspace(1e-5,10,6)
    %
    % OUTPUTS:
    %   P - [M x N x V] matrix of choice probabilities
    %   pz - posterior probability of each partition
    %   Z - set of partitions
    %
    % Sam Gershman, May 2017
    
    % initialization
    [M,N] = size(C);
    Z = SetPartition(M);
    g = 1;
    if nargin < 3 || isempty(alpha)
        alpha = linspace(1e-5,10,6);
    end
    A = length(alpha);
    q = zeros(length(Z),M,N,v);
    
    % construct prior
    logp = zeros(length(Z),A);
    
    for j = 1:length(Z)
        h = Z{j};
        K = length(h);
        
        % compute prior
        z = zeros(M,1);
        T = zeros(1,length(K));
        for k = 1:K
            z(h{k}) = k;
            T(k) = length(h{k});
        end
        for a = 1:A
            logp(j,a) = logp(j,a) + crp_logprob(T,alpha(a));
        end
        
        % compute likelihood
        L = zeros(K,max(v),N);
        for n = 1:N
            ix = ~isnan(C(:,n));
            for k = 1:K
                f = double(z(ix)==k);
                logp(j,:) = logp(j,:) + gammaln(v*g) - gammaln(sum(f)+g*v);
                for c = 1:v
                    L(k,c,n) = sum(f.*(C(ix,n)==c));
                    logp(j,:) = logp(j,:) + gammaln(g + L(k,c,n)) - gammaln(g);
                end
            end
        end
        
        % predictive probability
        for n = 1:N
            for m = 1:M
                q(j,m,n,:) = (g + L(z(m),:,n))./(g*v + sum(L(z(m),:,n)));
            end
        end
    end
    
    pz = exp(logp-logsumexp(logp(:)));  % normalize prior
    pz = sum(pz,2);                     % marginalize over alpha
    
    % choice probabilities for missing trials
    if ~isempty(q)
        P = zeros(M,N,v);
        for j = 1:length(Z)
            Q = squeeze(q(j,:,:,:));
            P = P + pz(j)*Q;
        end
    end