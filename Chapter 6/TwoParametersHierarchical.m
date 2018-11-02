% This m-file generates data from the small scale nonlinear model appearing
% in Chapter 6. First, we enter the time points, define the nonlinear
% model A(x), and generate the data y.
% First, generate the measurements and solve the nonlinear least squares
% problem using Levenburg Marquardt.
%
%  written by John Bardsley 2016.
%
%  This code was used to generate figures in Chapter 6. 
clear all, close all
path(path,'../Functions');
t        = [1:2:9]';
A        = @(x) x(1)*(1-exp(-x(2)*t));
x_true   = [1;.1]; % true parameter values
sig      = 0.01;
b        = A(x_true)+sig*randn(size(t));
M        = length(b);
N        = length(x_true);

%% Levenburg-Marquardt parameter estimation:
% Compute the maximum likelihood estimator using Levengurg-Marquardt.
p.A              = A;
p.t              = t;
p.lam            = 1;
p.b              = b;
p.Q              = speye(M);     % this is non-identity when we do RTO.
p.e              = zeros(M,1);   % this is iid Gaussian when we do RTO.
x0               = [1;1]; % initial guess
p.Nrand          = M;
[xMAP,rMAP,JMAP] = LevMar(x0,@(x)twoparam_fun(x,p),0.001,1e-8,100);
mse              = norm(rMAP)^2/(M-N);

% Finally we output the results for comparison
figure(1)
plot(t,b,'ko',t,A(x_true),'k--',t,A(xMAP),'k')
legend('data','true model','model fit','Location','NorthWest')

%% Now, use RTO-MH to sample from the posterior distribution defined by
%              
%            p(x_1,x_2|b) \propto exp(-0.5||A(x_1,x_2)-b||^2).
%
% where A(x_1,x_2) = x_1*(1-exp(-x_2*t)) and b are the above measurements.
nsamps  = 5000;
rsamp   = rMAP;
xchain  = zeros(N,nsamps); xchain(:,1) = xMAP;
Axchain = zeros(M,nsamps); Axchain(:,1) = A(xMAP);
lamsamp = zeros(nsamps,1); lamsamp(1)  = 1/mse;
% hyperprior: delta~Gamma(a0,1/t0)
a0 = 1; t0=1e-4; 
h = waitbar(0,'MCMC samples in progress');
tic
for i = 1:nsamps-1
    h = waitbar(i/nsamps);
    %------------------------------------------------------------------
    % 1. Using conjugacy, sample regularization precisions delta,
    % conjugate prior: delta~Gamma(a0,1/t0);
    resid        = b-A(xchain(:,i));
    lamsamp(i+1) = gamrnd(a0+M/2,1./(t0+norm(resid(:))^2/2));
    %------------------------------------------------------------------
    % 2. Using conjugacy relationships, sample the image.
    p.lam  = lamsamp(i+1);
    % Compute the MAP estimator for lamdba and delta and corresponding Q
    p.Q    = speye(M);   % this is non-identity when we do RTO.
    p.e    = zeros(M,1); % this is iid Gaussian when we do RTO.
    [xMAP,rMAP,JMAP] = LevMar(xMAP,@(x)twoparam_fun(x,p),0.001,1e-8,1000);
    [Q,~]  = qr(JMAP,0);
    % Take one step of RTO-MH.
    p.Q            = Q;
    xtemp          = RTO_MH(xMAP,@(x,p)twoparam_fun(x,p),p,1);
    xchain(:,i+1)  = xtemp(:,end);
    Axchain(:,i+1) = A(xtemp(:,end)); 
end
close(h)
% Output for individual chains using sample_plot: ACF, IACT, Geweke test.
nburnin   = nsamps/10;
chain     = [lamsamp(nburnin+1:end)';xchain(:,nburnin+1:end)];
names     = cell(3,1);
names{1}  = '\lambda';
names{2}  = 'x_{1}';
names{3}  = 'x_{2}';
[tau,acf] = sample_plot(chain,names,2);
[~,nacf]  = size(acf);
% Generate pairwise plot for x_1 and x_2.
figure(5)
plot(xchain(1,nburnin+1:end),xchain(2,nburnin+1:end),'k.')
axis([0.5,2,0.04,.18])
% Plot autocorrelation function.
figure(6)
plot(acf(1,1:nacf),'k'), hold on 
plot(acf(2,1:nacf),'k--'), 
plot(acf(3,1:nacf),'k-.'), hold off
axis([1,nacf,0,1])
title(['ACFs for \lambda, x_1, and x_2']);
legend(['\lambda: \tau_{\rm int}(\lambda)=',num2str(tau(1))],...
       ['x_1: \tau_{\rm int}(x_1)=',num2str(tau(2))],...
       ['x_2: \tau_{\rm int}(x_2)=',num2str(tau(3))])
Axlims = plims(Axchain(:,nburnin:end)',[0.025,0.5,0.975]);
figure(7)
plot(t,b,'ko',t,Axlims(2,:),'k-',t,Axlims(1,:),'k--',t,Axlims(3,:),'k--')
legend('measurements','Sample Median','95% credibility bounds','Location','NorthWest')

