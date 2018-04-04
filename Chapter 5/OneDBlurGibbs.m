%  
%  Hierarchical Gibbs sampler for 1d image deblurring with zero BCs.
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
n = 128; % No. of grid points
h = 1/n;
t = [h/2:h:1-h/2]';
sig = .05; % Kernel width.
kernel = (1/sqrt(pi)/sig) * exp(-(t-h/2).^2/sig^2);
A = toeplitz(kernel)*h;

% Set up true solution x_true and data b = A*x_true + error.
x_true = 50*(.75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4);
Ax = A*x_true;
err_lev = 2; % Percent error in data
sigma = err_lev/100 * norm(Ax) / sqrt(n);
eta =  sigma * randn(n,1);
b = Ax + eta;
m = length(b);
Atb = A'*b;
AtA = A'*A;
figure(1),
plot(t,x_true,'k',t,b,'ko','LineWidth',1)
legend('true image','blurred, noisy data')

% second derivative precision matrix, with zero BCs, for prior
L = spdiags([-ones(n,1) 2*ones(n,1) -ones(n,1)],[-1 0 1],n,n);
% MCMC sampling
nsamps  = 10000;
nChol   = 0;
xsamp   = zeros(n,nsamps);
delsamp = zeros(nsamps,1); delsamp(1) = .1;
lamsamp = zeros(nsamps,1); lamsamp(1) = 1;
R       = chol(lamsamp(1)*AtA + delsamp(1)*L);
nChol   = 1;
xsamp(:,1) = R\(R'\(lamsamp(1)*Atb));
% hyperpriors: lambda~Gamma(a,1/t0), delta~Gamma(a1,1/t1)
a0=1; t0=0.0001; a1=1; t1=0.0001;
h = waitbar(0,'MCMC samples in progress');
tic
for i = 1:nsamps-1
    h = waitbar(i/nsamps);
    %------------------------------------------------------------------
    % 1a. Using conjugacy, sample the noise precision lam=1/sigma^2,
    % conjugate prior: lam~Gamma(a0,1/t0)
    lamsamp(i+1) = gamrnd(a0+m/2,1/(t0+norm(A*xsamp(:,i)-b)^2/2));
    %------------------------------------------------------------------
    % 1b. Using conjugacy, sample regularization precisions delta,
    % conjugate prior: delta~Gamma(a1,1/t1);
    delsamp(i+1) = gamrnd(a1+n/2,1./(t1+xsamp(:,i)'*(L*xsamp(:,i))/2));
    %------------------------------------------------------------------
    % 2. Using conjugacy relationships, sample the image.
    R = chol(AtA*lamsamp(i+1) + delsamp(i+1)*L);
    nChol = nChol + 1;
    xsamp(:,i+1) = R \ (R'\(Atb*lamsamp(i+1)) + randn(n,1));
end
toc
close(h)
% Visualize the MCMC chain
% Plot the sample mean and 95% credibility intervals for x.
nburnin        = floor(nsamps/10); 
xsamp          = xsamp(:,nburnin+1:end);
q              = plims(xsamp(:,:)',[0.025,0.975]);
x_mean         = mean(xsamp(:,:)')';
relative_error = norm(x_true-x_mean)/norm(x_true);
figure(2),
plot(t,x_mean,'k',t,x_true,'-.k',t,q(2,:),'--k',t,q(1,:),'--k')
legend('MCMC sample mean','true image','95% credibility bounds','Location','North')
% Output for (lambda,delta)-chain using sample_plot: ACF, IACT, Geweke test
lamdel_chain = [lamsamp(nburnin+1:end), delsamp(nburnin+1:end)]';
names        = ["\lambda","\delta"];
fignum       = 3;
[taux,acfun] = sample_plot(lamdel_chain,names,fignum);
% Plot the autocorrelations functions together.
[~,nacf] = size(acfun);
figure(6)
plot([1:nacf],acfun(1,:),'k',[1:nacf],acfun(2,:),'k--','LineWidth',2)
axis([1,nacf,0,1])
title('ACFs for \lambda and \delta.')
legend(['\lambda: \tau_{\rm int}(\lambda)=',num2str(taux(1))],['\delta: \tau_{\rm int}(\delta)=',num2str(taux(2))])