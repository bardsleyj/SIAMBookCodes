% This m-file generates data from the small scale nonlinear model appearing
% in Chapter 6. First, we enter the time points, define the nonlinear
% model A(x), and generate the measurements y.
clear all, close all
path(path,'../Functions');
t        = [1:2:9]';
A        = @(x) x(1)*(1-exp(-x(2)*t));
x_true   = [1;.1]; % true parameter values
sig      = 0.01;
b        = A(x_true)+0.01*randn(size(t));
% Plot the measurements and error-free measurements together.
figure(1), 
plot(t,b,'ko',t,A(x_true),'k-')
xlabel('{\bf t}'), ylabel('{\bf b}')

% Adaptive Metropolis (AM):
% Sample from p(x|b)\propto exp(-(1/2/sig^2)*||A(x)-b||^2) using AM. 
p_params.sig         = sig;
p_params.b           = b; 
p_params.A           = A;
neglog_p             = @(x,p) (1/2/p.sig^2)*norm(p.A(x)-p.b)^2;
% Now implement AM for sampling from p(x|b).
C                    = .01*eye(2); % intial proposal covariance.
Nsamps               = 20000; % length of MCMC chain.
nburnin              = floor(Nsamps/2);
adapt_int            = 100;   % proposal covariance update interval. 
prop_flag            = 0; % enter 0 for normal & 1 for a lognormal proposal
[xchain,accept_rate] = AM(x_true,neglog_p,p_params,prop_flag,Nsamps,C,adapt_int);

% Output for x_1- and x_2-chains using sample_plot.
xchain       = xchain(:,nburnin+1:end);
names        = ["x_1","x_2"];
fignum       = 2;
[taux,acfun] = sample_plot(xchain,names,fignum);
% Plot the autocorrelations functions together.
[~,nacf] = size(acfun);
figure(6)
plot([1:nacf],acfun(1,:),'k',[1:nacf],acfun(2,:),'k--','LineWidth',2)
axis([1,nacf,0,1])
title('ACFs for x_1 and x_2.')
legend(['\tau_{\rm int}(x_1)=',num2str(taux(1))],['\tau_{\rm int}(x_2)=',num2str(taux(2))])
