%  
%  2d image deblurring problem with independent increment (anisotropic)
%  IGMRF prior. Spatially dependent weights are computed iteratively using 
%  Algorithm 4.1. Regularization parameter is computed using GCV. 
%
%  This m-file was used to generate Figures in Chapter 4 of the book.
%
%  written by John Bardsley 2016.
%
%  First, construct the Fourier transform of the circulant-shifted
%  convolution kernel.
clear all, close all
path(path,'../Functions')
load ../MatFiles/satellite
x_true = x_true/max(x_true(:));
[n,n] = size(x_true);
N = n^2;
h = 1/n;
x = [-0.5+h/2:h:0.5-h/2]';
[X,Y]=meshgrid(x);
sig = 0.02;
kernel = h^2*(1/(2*pi)/sig^2)*exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
ahat = fft2(fftshift(kernel));
%
% Next, generate the blurred, noisy data.
Ax = real(ifft2(ahat.*fft2(x_true)));
err_lev = 2; % Percent error in data
sigma = err_lev/100 * norm(Ax(:)) / sqrt(N);
eta =  sigma * randn(n,n);
b = Ax + eta;
bhat = fft2(b);
% Plote the true image and blurred, noisy data.
figure(1) 
  imagesc(x_true), colorbar, colormap(1-gray)
figure(2)
  imagesc(b), colorbar, colormap(1-gray)
%
% GMRF prior precision matrix:
% DFT representation of discrete negative Laplacian with periodic BCs.
% This is used for the CG preconditioner.
lh=zeros(n,n); lh(1,1)=-1; lh(1,2)=1; lhhat=fft2(lh);
lv=zeros(n,n); lv(1,1)=-1; lv(n,1)=1; lvhat=fft2(lv);
lhat = abs(lhhat).^2+abs(lvhat).^2;
% Sparse matrix representation of horizontal and vertical partial
% derivatives, Ds and Dt, with periodic BCs.
D = spdiags([-ones(n,1) ones(n,1)],[0 1],n,n); D(n,1)=1;
I = speye(n,n); Ds = kron(I,D); Dt = kron(D,I);

% Deblur image using conjugate gradient iteration
% Store preliminary CG iteration information
params.max_cg_iter     = 100;
params.cg_step_tol     = 1e-8;
params.grad_tol        = 1e-8;
params.cg_io_flag      = 0;
params.cg_figure_no    = [];
params.precond         = 'Amult';
Bmult_fn               = 'Bmult_IGMRF';
a_params.ahat          = ahat;
a_params.lhat          = lhat;
a_params.Ds            = Ds;
a_params.Dt            = Dt;
c                      = feval('Amult',b,ahat);
beta                   = 0.001; % so the weights are well-defined.

niters = input(' Enter # of iterations of GMRF edge-preserving reconstruction algorithm. ');
for i=1:niters
    if i==1, % negative Laplacian regularization.
        a_params.Lambda_s = ones(n^2,1);
        a_params.Lambda_t = ones(n^2,1);
    else % edge-preserving, anisotropic IGRMF prior.
        a_params.Lambda_s = 1./sqrt((Ds*xalpha(:)).^2+(Dt*xalpha(:)).^2+beta);
        a_params.Lambda_t = 1./sqrt((Ds*xalpha(:)).^2+(Dt*xalpha(:)).^2+beta);
    end
    params.a_params        = a_params;
    % Compute GCV choice of alpha.
    RegParam_fn            = @(alpha) GCV_fn(alpha,b,c,params,Bmult_fn);
    alpha                  = fminbnd(RegParam_fn,0,1);
    % Compute the Tikhonov regularized solution.
    a_params.alpha         = alpha;
    params.a_params        = a_params;
    params.precond_params  = 1./(abs(ahat).^2+alpha*lhat);
    [xalpha,iter_hist]     = CG(zeros(n,n),c,params,Bmult_fn);
    % Plot the Tikhonov solution and CG iteration history.
    figure(3)
      imagesc(xalpha), colorbar, colormap(1-gray)
    figure(4)
      semilogy(iter_hist(:,2),'k*'), title('CG iteration vs. residual norm')
    pause(0.1), drawnow
    fprintf('iteration = %d relative error = %5.3e\n',i,norm(xalpha(:)-x_true(:))/norm(x_true(:)))
end
  