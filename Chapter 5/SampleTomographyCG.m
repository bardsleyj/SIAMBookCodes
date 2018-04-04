%  
%  Sampling from a high-dimensional Gaussian arising in the computed 
%  tomography test case. The biharmonic (squared-Laplacian) prior precision 
%  matrix with zero BCs is used. Conjugate gradient iteration without 
%  preconditioning used to compute approximate samples from the posterior.
%
%  Once the samples are computed, the sample mean is used as an estimator 
%  of the unknown image and empirical quantiles are used to compute 95%
%  credibility intervals for every unknown. 
% 
%  written by John Bardsley 2016.
%
%  This code was used to generate figures in Chapter 5. 
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
% True image and data display
figure(1), imagesc(x_true), colormap(1-gray), colorbar
figure(2), imagesc(b), colormap(1-gray), colorbar

% Sparse matrix representation for discrete negative Laplacian, zero BCs
D   = spdiags([-ones(n,1) ones(n,1)],[-1 0],n+1,n);
I   = speye(n,n); Ds = kron(I,D); Dt = kron(D,I);
Lsq = Ds'*Ds + Dt'*Dt;
L   = Lsq'*Lsq;        % discrete biharmonic.
Nbar = N;              % rank of L.

% Choose the regularization parameter and use it to initialize the 
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
lambda_est          = 1/var(b(:)-A*xalpha);
delta_est           = alpha*lambda_est;

% Compute samples using the conjugate gradient iteration. 
nsamps     = 500;
xsamp      = zeros(N,nsamps);
xsamp(:,1) = xalpha;
tic
for i=1:nsamps
    h = waitbar(i/nsamps);
    % Compute approximate sample from p(x|b,lambda,delta) using CG.
    eta = sqrt(lambda_est)*A'*randn(M,1) + sqrt(delta_est)*Lsq'*randn(N,1);
    c   = Atb + eta/lambda_est;
    xsamp(:,i+1) = CG(zeros(N,1),c,params,Bmult);
end
toc
close(h)     
% Compute the sample mean and 95% empirical quantiles.
x_mean = mean(xsamp')';
quants = plims(xsamp',[0.025,0.975])';
% Finally, generate figures.
figure(3), colormap(1-gray)
  imagesc(reshape(xalpha,n,n)), colorbar
figure(4), colormap(1-gray)
  imagesc(reshape(quants(:,2)-quants(:,1),n,n)), colorbar

