%  
%  The gradient scan Gibbs sampler for 2d computed tomography. Biharmonic 
%  prior precision matrix with zero BCs is assumed. Conjugate gradient 
%  without preconditioining was used to compute samples from the posterior.
%
%  Once the samples are computed, the sample mean is used as an estimator 
%  of the unknown image and empirical quantiles are used to compute 95%
%  credibility intervals for every unknown. 
%
%  The Geweke test is used to determine whether the second half of the 
%  (lambda,delta)-chain is in equilibrium, and the IACT, and related 
%  essential sample size, are also estimated for the (lambda,delta)-chain.
% 
%  written by John Bardsley 2016.
%
%  This code was used to generate figures in Chapter 5. 
%% First set up the test case.
clear all, close all
path(path,'../Functions')
load ../MatFiles/SheppLogan
[n,n]     = size(x_true);
ntheta    = 100;
theta     = linspace(-pi/2,pi/2,ntheta);
nz        = 100;
z         = linspace(-0.49,0.49,nz);
[Z,Theta] = meshgrid(z,theta);
A         = Xraymat(Z(:),Theta(:),n);
[M,N]     = size(A);
Ax        = A*x_true(:);
err_lev   = 2;
noise     = err_lev/100 * norm(Ax(:)) / sqrt(ntheta*nz);
b         = reshape(Ax,ntheta,nz) + noise*randn(ntheta,nz);
% Data display
figure(1), imagesc(x_true), colormap(1-gray), colorbar
figure(2), imagesc(b), colormap(1-gray), colorbar

% Sparse matrix representation for discrete negative Laplacian, zero BCs
D         = spdiags([-ones(n,1) ones(n,1)],[-1 0],n+1,n);
I         = speye(n,n); Ds = kron(I,D); Dt = kron(D,I);
Lsq       = Ds'*Ds + Dt'*Dt;
L         = Lsq'*Lsq;        % discrete biharmonic.
Nbar      = N;

% Next choose the regularization parameter and use it to initialize the 
% (lambda,delata,x)-chain. First, store CG iteration information for use 
% within the regularization parameter selection method.
params.max_cg_iter  = 1000;
params.cg_step_tol  = 1e-6;
params.grad_tol     = 1e-6;
params.cg_io_flag   = 0;
params.cg_figure_no = [];
params.precond      = [];
% Store necessary info for matrix vector multiply B*x, where
% B=A'*A+alpha*I.
Bmult               = 'Bmult_TomographyGMRF';
a_params.A          = A;
a_params.L          = L;
params.a_params     = a_params;
% Choose the regularization parameter
Atb                 = A'*b(:);
disp(' *** Computing regularization parameter using GCV *** ')
RegParam_fn         = @(alpha) GCV_Tomography(alpha,b(:),Atb,params,Bmult);
alpha               = fminbnd(RegParam_fn,0,1);
a_params.alpha      = alpha;
params.a_params     = a_params;
[xalpha,iter_hist]  = CG(zeros(N,1),Atb,params,Bmult);
% Estimate lambda and delta using the residuals and alpha.
lambda_est          = 1/norm(b(:)-A*xalpha(:))^2;
delta_est           = alpha*lambda_est;

% ------ Implement the gradient scan Gibbs sampler ------
% Initialization before sampling.  
nsamps     = 5000;
lamsamp    = zeros(nsamps,1); lamsamp(1) = lambda_est;
delsamp    = zeros(nsamps,1); delsamp(1) = delta_est;
xsamp      = zeros(N,nsamps);
xtemp      = xalpha;
xsamp(:,1) = xtemp(:);
% hyperpriors: lambda~Gamma(a,1/t0), delta~Gamma(a1,1/t1)
a0         = 1; 
t0         = 0.0001; 
a1         = 1; 
t1         = 0.0001;
h          = waitbar(0,'MCMC samples in progress');
params.max_cg_iter  = 20; % truncate CG iterations
nCG        = 0;
tic
disp(' *** MCMC samples in progress *** ')
for i=1:nsamps-1
    h = waitbar(i/nsamps);
    %------------------------------------------------------------------
    % 1a. Using conjugacy, sample the noise precision lam=1/sigma^2,
    % conjugate prior: lam~Gamma(a0,1/t0), mean = a0/t0, var = a0/t0^2.
    Axtemp       = A*xtemp;
    lamsamp(i+1) = gamrnd(a0+M/2,1./(t0+norm(Axtemp(:)-b(:))^2/2));
    %------------------------------------------------------------------
    % 1b. Using conjugacy, sample regularization precisions delta,
    % conjugate prior: delta~Gamma(a1,1/t1);
    Lxtemp       = L*xtemp;
    delsamp(i+1) = gamrnd(a1+Nbar/2,1./(t1+xtemp(:)'*Lxtemp(:)/2));
    %------------------------------------------------------------------
    % 2. Use CG to sample x.
    a_params.alpha = delsamp(i+1)/lamsamp(i+1);
    params.a_params = a_params;
    eta = sqrt(lamsamp(i+1))*A'*randn(M,1) + sqrt(delsamp(i+1))*Lsq'*randn(N,1);
    c   = Atb - Bmult_TomographyGMRF(xtemp,a_params) + eta/lamsamp(i+1);
    xtemp = xtemp + CG(zeros(N,1),c,params,Bmult);
    nCG = nCG + params.max_cg_iter;
    xsamp(:,i+1) = xtemp;
end
toc
close(h)     
clear Lxtemp Axtemp xtemp  
% Remove the burn-in samples and visualize the MCMC chain
% Plot the mean and 95% credibility intervals for x
nburnin    = floor(nsamps/10); 
xsamp      = xsamp(:,nburnin+1:end);
q          = plims(xsamp(:,:)',[0.025,0.975])';
x_mean     = mean(xsamp(:,:)')';
relative_error = norm(x_true(:)-x_mean(:))/norm(x_true(:));
figure(3), colormap(1-gray)
  imagesc(reshape(x_mean,n,n)), colorbar
figure(4), colormap(1-gray)
  imagesc(reshape(q(:,2)-q(:,1),n,n)), colorbar
% Output for lambda- and delta-chains using sample_plot.
lamdel_chain = [lamsamp(nburnin+1:end), delsamp(nburnin+1:end)]';
names        = cell(2,1);
names{1}     = '\lambda';
names{2}     = '\delta';
fignum       = 5;
[taux,acfun] = sample_plot(lamdel_chain,names,fignum);
% Plot the autocorrelations functions together.
[~,nacf] = size(acfun);
figure(8)
plot([1:nacf],acfun(1,:),'r',[1:nacf],acfun(2,:),'k--','LineWidth',2)
axis([1,nacf,0,1])
title('ACFs for \lambda and \delta.')
legend(['\tau_{\rm int}(\lambda)=',num2str(taux(1))],['x_2: \tau_{\rm int}(\delta)=',num2str(taux(2))])