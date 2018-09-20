function [probs,beta] = fit_data(choices,P)
    
    % Fit LGM to experimental data.
    %
    % USAGE: [probs,beta] = fit_data(choices)
    %
    % INPUTS:
    %   choices - [N x 1] choices for N trials
    %   P - [N x V] model choice probabilities
    %
    % OUTPUTS:
    %   probs - predicted choice probabilities
    %   beta - fitted choice stochasticity parameter
    %
    % Sam Gershman, May 2017
    
    B = linspace(0.1,10,20);    % range of beta values
    L = -inf;
    
    for i = 1:length(B)
        p = P.^B(i);
        p = bsxfun(@rdivide,p,sum(p,2));
        loglik = 0;
        for n = 1:length(choices)
            loglik = loglik + log(p(n,choices(n)));
        end
        if loglik > L
            probs = p;
            L = loglik;
            beta = B(i);
        end
    end