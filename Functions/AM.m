function[xchain,accept_rate]=AM(x,neglog_p,p_params,prop_flag,Nsamps,C,adapt_int)
% This function implements the Metropolis algorithm for sampling from the
% posterior distribution of the parameters in the x vector.
% 
% INPUTS:
% x         = intial parameter vector.
% neglog_p  = function evaluating the negative-log p(x).
% p_params  = forward model function parameters.
% prop_flag = 0 for normal proposal & 1 for log-normal proposal.
% Nsamps    = the length of the Markov chain.
% C         = covariance of the Gaussian proposal.
% sigsq     = (estimated) sample variance.
% adapt_int = update the covariance estimate every adapt_int samples.
%
% OUTPUTS:
% xchain    = the Markov chain for the unknown parameters x.
% accept_rate = the ratio of # accepted samples to chain length.
%
% Now implement the Metropolis algorithm.
npar        = length(x);
xchain      = zeros(npar,Nsamps);
xold        = x;
xchain(:,1) = xold;
neglog_p_old = feval(neglog_p,xold,p_params); 
R           = chol(C);
naccept     = 0;
hh          = waitbar(0,'Adaptive Metropolis Progress');
for i=1:Nsamps
    hh    = waitbar(i/Nsamps);
    if prop_flag == 0 % normal proposal
        xnew  = xold + R'*randn(npar,1);
    elseif prop_flag == 1 % log-normal proposal
        xnew = exp(log(xold) + R'*randn(npar,1)); 
    end
    % Use the Metropolis ratio to determine probability of the accepting the
    % proposed step. Don't accept the step if it is outside the constraints.
    neglog_p_new = feval(neglog_p,xnew,p_params); 
    if log(rand) < -neglog_p_new+neglog_p_old % log of the MH ratio.
        naccept = naccept + 1;
        xold    = xnew;
        neglog_p_old = neglog_p_new;
    end
    xchain(:,i) = xold;
    if adapt_int > 0 & i > adapt_int;
        if i/adapt_int == fix(i/adapt_int)
            if prop_flag == 0 % normal proposal
                Cadapt = cov(xchain(:,1:i)')+sqrt(eps)*eye(npar);
            elseif prop_flag == 1, % log-normal proposal
                Cadapt = cov(log(xchain(:,1:i)'))+sqrt(eps)*eye(npar);
            end
            R = chol(Cadapt);
        end
    end
end
close(hh)
accept_rate = naccept/Nsamps;