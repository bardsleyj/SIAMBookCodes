%  
%  Computed tomography.
%
%  Tikhonov regularization is implemented with the regularization 
%  parameter chosen using GCV with randomized trace estimation. 
%
%  This m-file was used to generate figures in Chapter 3 of the book.
%
%  written by John Bardsley 2016.
%
%  First, create the tomography matrix using Xraymat.m. Then generate the 
%  true image using the Shepp-Logan phantom
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
% Generate noisy data by adding iid Gaussian error to Ax.
err_lev   = 2;
noise     = err_lev/100 * norm(Ax(:)) / sqrt(ntheta*nz);
b         = reshape(Ax,ntheta,nz) + noise*randn(ntheta,nz);
% Plot the tru image and blurred noisy data.
figure(1), imagesc(x_true), colormap(1-gray), colorbar
figure(2), imagesc(b), colormap(1-gray), colorbar

%% Regularization parameter selection.
% Store CG iteration information for use within the regularization
% parameter selection method.
params.max_cg_iter  = 250;
params.cg_step_tol  = 1e-4;
params.grad_tol     = 1e-4;
params.cg_io_flag   = 0;
params.cg_figure_no = [];
params.precond      = [];
% Store necessary info for matrix vector multiply B*x, where
% B=A'*A+alpha*I.
Bmult               = 'Bmult_Tomography';
a_params.A          = A;
params.a_params     = a_params;
% Choose the regularization parameter
Atb                 = A'*b(:);
disp(' *** Computing regularization parameter using GCV *** ')
RegParam_fn         = @(alpha) GCV_Tomography(alpha,b(:),Atb,params,Bmult);
alpha               = fminbnd(RegParam_fn,0,1);

%% With alpha in hand, tighten down on CG tolerances and use CG again
% to solve (A'*A+alpha*I)x=A'*b.
a_params.alpha        = alpha;
params.a_params       = a_params;
params.max_cg_iter    = 500;
params.cg_step_tol    = 1e-6;
params.grad_tol       = 1e-6;
params.precond        = [];
params.precond_params = [];
disp(' *** Computing the regularized solution *** ')
[xalpha,iter_hist]    = CG(zeros(N,1),Atb,params,Bmult);

% Plot the output from CG.
figure(3)
imagesc(reshape(xalpha,n,n),[0,max(x_true(:))]), colorbar, colormap(1-gray)
figure(4),
semilogy(iter_hist(:,2),'k','LineWidth',2), title('CG iteration vs. residual norm')