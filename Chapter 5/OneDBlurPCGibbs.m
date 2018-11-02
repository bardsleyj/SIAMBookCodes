%
%  Partially collapsed Gibbs sampler for 1d image deblurring with zero BCs.
%
%  Once the samples are computed, the sample mean is used as an estimator 
%  of the unknown image and empirical quantiles are used to compute 95%
%  credibility intervals for each unknown. 
%
%  The Geweke test is used to determine whether the chain, after burnin, 
%  is in equilibrium, and the integrated auto correlated time and essential 
%  sample size are estimated as described in Chapter 5. 
%
%  written by John Bardsley 2016.
%
%  This code was used to generate figures in Chapter 5.
clear all, close all
path(path,'../Functions');
n       = 512; % No. of grid points
h       = 1/n;
t       = [h/2:h:1-h/2]';
sig     = .05; % Kernel width parameter
kernel  = (1/sqrt(pi)/sig) * exp(-(t-h/2).^2/sig^2);
A       = toeplitz(kernel)*h;

% Set up true solution x_true and data b = A*x_true + error.
x_true  = 50*(.75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4);
Ax      = A*x_true;
err_lev = 2; % Percent error in data
sigma   = err_lev/100 * norm(Ax) / sqrt(n);
eta     =  sigma * randn(n,1);
b       = Ax + eta;
m       = length(b);
Atb     = A'*b;
AtA     = A'*A;
figure(1),
plot(t,x_true,'k',t,b,'ko','LineWidth',1)
legend('true image','blurred, noisy data')

% second derivative precision matrix for prior
L = spdiags([-ones(n,1) 2*ones(n,1) -ones(n,1)],[-1 0 1],n,n);
% Define the structure array and function for evaluating the negative log
% of the marginal density p(del|b,lam).
p_params.AtA = A'*A;
p_params.Atb = A'*b;
p_params.btb = b'*b;
p_params.m   = m;
p_params.n   = n;
p_params.L   = L;
p_params.a0  = 1;
p_params.t0  = .001;
p_params.a1  = 1;
p_params.t1  = .001;
neglog_p     = 'neglog_marginal_1d';
% MCMC sampling
nsamps       = 5000;
xsamp        = zeros(n,nsamps);
lamsamp      = zeros(nsamps,1); lamsamp(1) = 5.2;
delsamp      = zeros(nsamps,1); delsamp(1) = 0.12;
R            = chol(lamsamp(1)*AtA + delsamp(1)*L);
xsamp(:,1)   = R\(R'\(lamsamp(1)*Atb));
nMH          = 5;
sigMH        = 0.1;
h            = waitbar(0,'MCMC samples in progress');
naccept      = 0;
tic
for i = 1:nsamps-1
    h = waitbar(i/(nsamps-1));
    %------------------------------------------------------------------
    % 1. Using conjugacy, sample the noise precision lam=1/sigma^2,
    % conjugate hyperprior: lam~Gamma(a,1/t0)
    lamsamp(i+1) = gamrnd(p_params.a0+m/2,1/(p_params.t0+norm(A*xsamp(:,i)-b)^2/2));
    %------------------------------------------------------------------
    % 2. adaptive Metropolis-within-Gibbs for sampling from p(del|b,lam)
    del_old          = delsamp(i);
    [neglog_p_old,R] = feval(neglog_p,[lamsamp(i+1);del_old],p_params);
    for j=1:nMH
        del_prop = exp(log(del_old)+sigMH*randn); % log-normal proposal
        [neglog_p_prop,R_prop] = feval(neglog_p,[lamsamp(i+1);del_prop],p_params);
        if log(rand) < -neglog_p_prop+neglog_p_old
            naccept      = naccept + 1;
            del_old      = del_prop;
            neglog_p_old = neglog_p_prop;
            R_old        = R_prop;
        end
    end
    delsamp(i+1) = del_old;
    sigMH        = std(log(delsamp(1:i+1)));
    %------------------------------------------------------------------
    % 3. Finally, sample the image from p(x|b,lambda,delta)
    xsamp(:,i+1) = R\(R'\(lamsamp(i+1)*Atb) + randn(n,1));
end
toc
close(h)
% Visualize the MCMC chain
% Plot the sample mean and 95% credibility intervals.
nburnin        = floor(nsamps/10);
xsamp          = xsamp(:,nburnin+1:end);
q              = plims(xsamp(:,:)',[0.025,0.975])';
x_mean         = mean(xsamp(:,:)')';
relative_error = norm(x_true-x_mean)/norm(x_true);
figure(2),
plot(t,x_mean,'k',t,x_true,'-.k',t,q(:,2),'--k',t,q(:,1),'--k')
legend('MCMC sample mean','true image','95% credibility bounds','Location','North')
% Output for (lambda,delta)-chain using sample_plot: ACF, IACT, Geweke test
lamdel_chain = [lamsamp(nburnin+1:end), delsamp(nburnin+1:end)]';
names        = cell(2,1);
names{1}     = '\lambda';
names{2}     = '\delta';
fignum       = 3;
[taux,acfun] = sample_plot(lamdel_chain,names,fignum);
% Plot the autocorrelations functions together.
[~,nacf] = size(acfun);
figure(6)
plot([1:nacf],acfun(1,:),'k',[1:nacf],acfun(2,:),'k--','LineWidth',2)
axis([1,nacf,0,1])
title('ACFs for \lambda and \delta.')
legend(['\tau_{\rm int}(\lambda)=',num2str(taux(1))],['\tau_{\rm int}(\delta)=',num2str(taux(2))])