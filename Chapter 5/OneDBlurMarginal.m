%  
%  Adaptive Metropolis for sampling from the marginal p(lambda,delta|b).
%  1d image deblurring with zero BCs. 
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
% Plot the measurements.
figure(1),
plot(t,x_true,'k',t,b,'ko','LineWidth',1)
legend('true image','blurred, noisy data')

%% Next, sample from the marginal density using adaptive Metropolis.
% Prior precision matrix: negative second derivative with zero BCs.
L = spdiags([-ones(n,1) 2*ones(n,1) -ones(n,1)],[-1 0 1],n,n);
% Define the structure array and function for evaluating the negative log  
% of the marginal density.
p_params.AtA = A'*A;
p_params.Atb = A'*b;
p_params.btb = b'*b;
p_params.m   = m;
p_params.n   = n;
p_params.L   = L;
% hyperpriors: lambda~Gamma(a0,1/t0), delta~Gamma(a1,1/t1)
p_params.a0  = 1;
p_params.t0  = 1e-4;
p_params.a1  = 1;
p_params.t1  = 1e-4;
neglog_p     = 'neglog_marginal_1d';

% MCMC sampling using adaptive Metropolis (AM).
lamdel            = [5.2;0.12];
nsamps            = 10000;
C                 = 0.01*eye(2,2);
adapt_int         = 100;
prop_flag         = 1; % enter 0 for normal & 1 for a lognormal proposal
[lamdel_chain,acc_rate] = AM(lamdel,neglog_p,p_params,prop_flag,nsamps,C,adapt_int);

% Output for (lambda,delta)-chain using sample_plot: ACF, IACT, Geweke test
nburnin      = nsamps/10; 
names        = cell(2,1);
names{1}     = '\lambda';
names{2}     = '\delta';
fignum       = 2;
[taux,acfun] = sample_plot(lamdel_chain(:,nburnin+1:end),names,fignum);
% Plot the autocorrelations functions together.
[~,nacf] = size(acfun);
figure(5)
plot([1:nacf],acfun(1,:),'k',[1:nacf],acfun(2,:),'k--','LineWidth',2)
axis([1,nacf,0,1])
title('ACFs for \lambda and \delta.')
legend(['\tau_{\rm int}(\lambda)=',num2str(taux(1))],['\tau_{\rm int}(\delta)=',num2str(taux(2))])
