%  
%  Adaptive Metropolis for sampling from the marginal p(lambda,delta|b)
%  for two-dimensional image deblurring with periodic boundary conditions.
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
path(path,'../Functions')
load ../MatFiles/satellite
[nx,ny] = size(x_true); 
N       = nx*ny;
h       = 1/nx;
x       = [-0.5+h/2:h:0.5-h/2]';
[X,Y]   = meshgrid(x);
sig     = 0.02;
kernel  = exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
kernel  = kernel/sum(sum(kernel));
ahat    = fft2(fftshift(kernel));

Ax      = feval('Amult',x_true,ahat);
err_lev = 2; % Percent error in data
sigma   = err_lev/100 * norm(Ax(:)) / sqrt(N);
eta     =  sigma * randn(nx,ny);
b       = Ax + eta;
M       = length(b(:));
btb     = b(:)'*b(:);
bhat    = fft2(b);
figure(1)
  imagesc(x_true), colorbar, colormap(1-gray)
figure(2)
  imagesc(b), colorbar, colormap(1-gray)

% Sample from the marginal density using adaptive Metropolis.
% Construct Fourier representer for the prior precision matrix.
l       = zeros(nx,ny);
l(1, 1) =  4; l(2 ,1) = -1;
l(nx,1) = -1; l(1 ,2) = -1;
l(1,ny) = -1; lhat = abs(fft2(l)).^2;
clear l Ax PSF
% Define the structure array and function for evaluating the negative log  
% of the marginal density.
p_params.ahat = ahat;
p_params.lhat = lhat;
p_params.bhat = bhat;
p_params.btb  = b(:)'*b(:);
p_params.M    = M;
p_params.N    = N;
% hyperpriors: lambda~Gamma(a0,1/t0), delta~Gamma(a1,1/t1)
p_params.a0   = 1;
p_params.t0   = 1e-4;
p_params.a1   = 1;
p_params.t1   = 1e-4;
neglog_p      = 'neglog_marginal_2d';

% MCMC sampling
x             = [10;0.01];
nsamps        = 10000;
C             = 0.1*eye(2,2);
adapt_int     = 100;
prop_flag     = 1;
[lamdel_chain,acc_rate] = AM(x,neglog_p,p_params,prop_flag,nsamps,C,adapt_int);

% Output for (lambda,delta)-chain using sample_plot: ACF, IACT, Geweke test
nburnin      = nsamps/10; 
names        = ["\lambda","\delta"];
fignum       = 2;
[taux,acfun] = sample_plot(lamdel_chain(:,nburnin+1:end),names,fignum);
% Plot the autocorrelations functions together.
[~,nacf] = size(acfun);
figure(6)
plot([1:nacf],acfun(1,:),'k',[1:nacf],acfun(2,:),'k--','LineWidth',2)
axis([1,nacf,0,1])
title('ACFs for \lambda and \delta.')
legend(['\tau_{\rm int}(\lambda)=',num2str(taux(1))],['\tau_{\rm int}(\delta)=',num2str(taux(2))])