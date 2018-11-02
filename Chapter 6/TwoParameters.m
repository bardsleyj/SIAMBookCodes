% This m-file generates data from the small scale nonlinear model appearing
% in Chapter 6. First, we enter the time points, define the nonlinear
% model A(x), and generate the data y.
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
nsamp                = 10000;
p.lam                = 1/mse;
[Q,~]                = qr(JMAP,0);
p.Q                  = Q;
p.Nrand              = M;
[xchain,accept_rate] = RTO_MH(x0,@(x,p)twoparam_fun(x,p),p,nsamp);
% Evaluate the modl at every sample.
Axchain = zeros(M,nsamp);
for i=1:nsamp
    Axchain(:,i)=A(xchain(:,i)); 
end

% Output for individual chains using sample_plot: ACF, IACT, Geweke test.
names            = cell(2,1);
names{1}         = 'x_1';
names{2}         = 'x_2';  
[taux,acf_array] = sample_plot(xchain,names,2);

% Plot the autocorrelations functions together.
[~,nacf] = size(acf_array);
figure(5)
plot([1:nacf],acf_array(1,:),'k',[1:nacf],acf_array(2,:),'k--')
axis([1,nacf,0,1])
title('ACFs for x_1 and x_2.')
legend(['x_1: \tau_{\rm int}(x_1)=',num2str(taux(1))],['x_2: \tau_{\rm int}(x_2)=',num2str(taux(2))])
Axlims = plims(Axchain',[0.025,0.5,0.975]);
figure(6)
plot(t,b,'ko',t,Axlims(2,:),'k-',t,Axlims(1,:),'k--',t,Axlims(3,:),'k--')
legend('measurements','Sample Median','95% credibility bounds','Location','NorthWest')

