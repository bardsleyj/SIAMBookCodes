%  
%  RTO for positivity constrainted kernel reconstruction. We use the
%  exponential transformation x=exp(u) and prior
%  p(u|del) \propto exp(-del/2*||Du||_2^2), where D is the 1D difference 
%  matrix, to obtain the posterior density function.
%
%  p(u|b,lam,del)\propto exp(-\lam/2*||A*exp(u)-b||^2-del/2||D*u||_2^2)
%
%  Once the samples are computed, the sample mean is used as an estimator 
%  of the natural log of the unknown image and empirical quantiles are used 
%  to compute 95% credibility intervals for every unknown. 
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
clear all, close all
n      = 80; % No. of grid points
h      = 1/n;
N      = 2*n-1;
A      = tril(ones(N,N))*h;
%
% Next, set up true solution x_true and generate data b = A*x_true + error.
t      = [-1+h:h:1-h]';
sig1   = .1; % Kernel width parameter, left-half
kernelleft = exp(-(t(1:n)).^2/sig1^2);
% Create the right-half of the non-symmetric kernel
sig2   = .2; % Kernel width parameter, right-half
kernelright = exp(-(t(n+1:end)).^2/sig2^2);
% Create the normalized kernel
kernel = [kernelleft;kernelright];
kernel = kernel/sum(kernel)/h; % normalize kernel
x_true = kernel;
% Generate the data b = A*x_true + error.
Ax      = A*x_true;
[M,N]   = size(A);
err_lev = 2; % Percent error in data
sigma   = err_lev/100 * norm(Ax) / sqrt(M);
eta     =  sigma * randn(M,1);
b       = Ax + eta;
figure(1), 
  plot(t,x_true,'k',t,b,'ko')

%  Define the regularization matrix D'*D, where D is the discrete
%  derivative matrix with Neumann boundary conditions.
onesvec          = ones(N,1);
D                = (1/h^2)*spdiags([-onesvec 2*onesvec -onesvec],[-1 0 1],N,N);
D(1,1)           = 1/h^2; % Neumann left BC 
D(N,N)           = 1/h^2; % Neumann right BC
Nresid           = M + length(D(:,1));
% Compute the MAP estimator using the nonlinear transformation approach
p.A              = A;
p.lam            = 1/sigma^2; % true lambda
p.del            = 5e-5;      % hand-tuned delta
p.b              = b;
p.D              = D;
p.Q              = speye(Nresid);     % this is non-identity when we do RTO.
p.e              = zeros(Nresid,1);   % this is iid Gaussian when we do RTO.
u0               = zeros(N,1); % initial guess
[zMAP,rMAP,JMAP] = LevMar(u0,@(z)positivity_fun(z,p),0.001,1e-8,100);

% Plot the measurements and model fit.
figure(2) 
plot(t,exp(zMAP),'k-',t,x_true,'k--')
legend('exp(z_{MAP})','x_{true}')

%% Now, use RTO-MH to sample from the posterior distribution defined by
%              
%            p(x|b) \propto exp(-lam/2*||A(x)-b||^2-del/2*||D*ln(x)||^2).
%
nsamp    = 4000;
[Q,~]    = qr(JMAP,0);
p.Q      = Q;
p.Nrand  = Nresid;
[zchain,accept_rate] = RTO_MH(zMAP,@(z,p)positivity_fun(z,p),p,nsamp);
xchain   = exp(zchain);

% Visualize the MCMC chain
% Plot the sample mean and 95% credibility intervals for x.
xlims          = plims(xchain',[0.025,0.5,0.975])';
relative_error = norm(x_true-xlims(:,2))/norm(x_true)
figure(3),
plot(t,x_true,'k',t,xlims(:,2),'--k',t,xlims(:,1),'-.k',t,xlims(:,3),'-.k')
legend('x_{true}','MCMC sample median','95% credibility bounds')
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


