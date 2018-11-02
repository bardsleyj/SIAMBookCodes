%  
%  RTO for l1-regularized, 1d image deblurring. The prior has the form
%  p(x|del) \propto exp(-del*||Dx||_1),
%  where D is the Haar wavelet transform matrix. A prior transformation is
%  used.
%
%  Once the samples are computed, the sample mean is used as an estimator 
%  of the unknown image and empirical quantiles are used to compute 95%
%  credibility intervals for every unknown. 
%
%  The Geweke test is used to determine whether the chain, after burnin, 
%  is in equilibrium, and the integrated auto correlated time and essential 
%  sample size are estimated as described in Chapter 5. 
% 
%  written by John Bardsley 2017.
%
%  This code was used to generate figures in Chapter 6. 
clear all, close all
path(path,'../Functions');
N = 64; % No. of grid points
h = 1/N;
t = [h/2:h:1-h/2]';
sig = .05; % Kernel width parameter
kernel = (1/sqrt(pi)/sig) * exp(-(t-h/2).^2/sig^2);
A = toeplitz(kernel)*h;

% Set up true solution x_true and data b = A*x_true + error.
x_true = 5*(.75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4);
Ax = A*x_true;
err_lev = 1; % Percent error in data
sigma = err_lev/100 * norm(Ax) / sqrt(N);
eta =  sigma * randn(N,1);
b = Ax + eta;
M = length(b);

%  Haar matrix for regularization function and regularization parameter.
D                = HaarWaveletTransformMatrix(N);
Nresid           = M + length(D(:,1));
% Now compute the MAP estimator using the nonlinear transformation approach
p.A              = A;
p.lam            = 1/sigma^2;
p.del            = 2;
p.b              = b;
p.D              = D;
p.Q              = speye(Nresid);     % this is non-identity when we do RTO.
p.e              = zeros(Nresid,1);   % this is iid Gaussian when we do RTO.
u0               = ones(N,1); % initial guess
[uMAP,rMAP,JMAP] = LevMar(u0,@(u)l1_fun(u,p),0.001,1e-8,100);
mse              = norm(rMAP)^2/(M-N);
xMAP             = D'*g_fn(uMAP,p.del);

% Plot the measurements and model fit.
figure(1), plot(t,b,'ko',t,xMAP,'k',t,x_true,'k--')

%% Now, use RTO-MH to sample from the posterior distribution defined by
%              
%            p(x|b) \propto exp(-lam/2*||A(x)-b||^2-del/2*||D*x||^2).
%
nsamp                = 5000;
[Q,~]                = qr(JMAP,0);
p.Q                  = Q;
p.Nrand              = Nresid;
[zchain,accept_rate] = RTO_MH(uMAP,@(u,p)l1_fun(u,p),p,nsamp);
xchain               = D'*g_fn(zchain,p.del);

% Visualize the MCMC chain
% Plot the sample mean and 95% credibility intervals.
xlims          = plims(xchain',[0.025,0.5,0.975])';
relative_error = norm(x_true-xlims(:,2))/norm(x_true);
figure(3),
plot(t,x_true,'k',t,xlims(:,2),'--k',t,xlims(:,1),'-.k',t,xlims(:,3),'-.k')
legend('true image','MCMC sample median','95% credibility bounds','Location','North')
% Output for individual chains using sample_plot: ACF, IACT, Geweke test.
rng('shuffle')
index        = sort(randsample(N,3));
zchain_index = zchain(index,:);
names        = cell(3,1);
names{1}     = char(['z_{',num2str(index(1)),'}']);
names{2}     = char(['z_{',num2str(index(2)),'}']);
names{3}     = char(['z_{',num2str(index(3)),'}']);
[tau,acf]    = sample_plot(zchain_index,names,4);
[~,nacf]     = size(acf);
% Plot autocorrelation function.
figure(7)
    plot(acf(1,1:nacf),'k'), hold on 
    plot(acf(2,1:nacf),'k--'), 
    plot(acf(3,1:nacf),'k-.'), hold off
axis([1,nacf,0,1])
title(['ACFs for z_{',num2str(index(1)),'}, z_{',num2str(index(2)),'}, and z_{',num2str(index(3)),'}']);
legend(['z_{',num2str(index(1)),'}: \tau_{\rm int}(z_{',num2str(index(1)),'})=',num2str(tau(1))],...
       ['z_{',num2str(index(2)),'}: \tau_{\rm int}(z_{',num2str(index(2)),'})=',num2str(tau(2))],...
       ['z_{',num2str(index(3)),'}: \tau_{\rm int}(z_{',num2str(index(3)),'})=',num2str(tau(3))])


