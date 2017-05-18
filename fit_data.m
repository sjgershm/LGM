function [probs,beta] = fit_data(choices,P)
    
    % Fit LGM to experimental data.
    %
    % USAGE: [probs,beta] = fit_data(choices)
    %
    % INPUTS:
    %   choices - [K x 1] choices for K trials
    %   P - model choice probabilities
    %
    % OUTPUTS:
    %   probs - predicted choice probabilities
    %   beta - fitted choice stochasticity parameter
    %
    % Sam Gershman, May 2017
    
    B = linspace(0.1,10,20);    % range of beta values
    L = -inf;
    
    for i = 1:length(B)
        p = (P.^B(i))./sum(P.^B(i));
        loglik = 0;
        for k = 1:length(choices)
            loglik = loglik + log(p(choices(k)));
        end
        if loglik > L
            probs = p;
            L = loglik;
            beta = B(i);
        end
    end