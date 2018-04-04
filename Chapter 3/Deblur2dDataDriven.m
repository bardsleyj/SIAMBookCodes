%  
%  2d image deblurring with data driven boundary conditions. 
%
%  Tikhonov regularization is implemented with the regularization parameter 
%  chosen using GCV with randomized trace estimation. 
%
%  This m-file was used to generate figures in Chapter 3 of the book.
%
%  written by John Bardsley 2016.
%
%  First, construct the Fourier transform of the circulant-shifted
%  convolution kernel and compute the data on the extended domain.
clear all, close all
path(path,'../Functions')
load ../MatFiles/cell
x_true = 100*x_true/max(x_true(:));
[n,n]  = size(x_true);
% Generate data on 256^2 grid w/ periodic BCs, then restrict to 128^2.
h      = 1/n;
x      = [-0.5+h/2:h:0.5-h/2]';
[X,Y]  = meshgrid(x);
sig    = 0.01;
kernel = h^2*(1/(2*pi)/sig^2)*exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
ahat   = fft2(fftshift(kernel));
Ax     = real(ifft2(ahat.*fft2(x_true)));

% Extract a 128x128 subregion from Ax, add iid Gaussian noise, and plot.
n2      = 128; % dimension of subimage is n2xn2
err_lev = 2;   % percent error = 100/SNR.
noisevar= err_lev/100 * norm(Ax(:)) / sqrt(n2^2);
ll      = 101; % index of the lower left corner of the subimage.
index   = ll:ll+n2-1; % indices of the subimge
b       = Ax(index,index) + noisevar*randn(n2,n2);
figure(1), imagesc(x_true), colormap(gray), colorbar,
rectangle('Position',[ll ll n2 n2],'EdgeColor','w','LineWidth',2)
figure(2), imagesc(Ax), colormap(gray), colorbar
rectangle('Position',[ll ll n2 n2],'EdgeColor','w','LineWidth',2)

% Zero pad b and create the mask for computations on the extended domain.
b_pad   = padarray(b,[n/4,n/4]);
bp_hat  = fft2(b_pad);
M       = padarray(ones(size(b)),[n/4,n/4]);

%% Next choose the regularization parameter using GCV.
% Store CG iteration information for use within the regularization
% parameter selection method.
params.max_cg_iter  = 250;
params.cg_step_tol  = 1e-4;
params.grad_tol     = 1e-4;
params.cg_io_flag   = 0;
params.cg_figure_no = [];
% Store information for the preconditioner
params.precond      = 'Amult';
% Store necessary info for matrix vector multiply (A*x) function
Bmult               = 'Bmult_DataDriven';
a_params.ahat       = ahat;
a_params.M          = M;
params.a_params     = a_params;
% Choose the regularization parameter
AtDb                = Amult(b_pad,conj(ahat));
disp(' *** Computing regularization parameter using GCV *** ')
RegParam_fn         = @(alpha) GCV_DataDriven(alpha,b,AtDb,params,Bmult);
alpha               = fminbnd(RegParam_fn,0,1);

%% With alpha in hand, tighten down on CG tolerances and use CG again
% to solve (A'*A+alpha*I)x=A'*b. Then plot the results.
a_params.alpha        = alpha;
params.a_params       = a_params;
params.max_cg_iter    = 500;
params.cg_step_tol    = 1e-6;
params.grad_tol       = 1e-6;
params.precond        = 'Amult';
params.precond_params = 1./(abs(ahat).^2+alpha);
disp(' *** Computing the regularized solution *** ')
[xalpha,iter_hist]    = CG(zeros(n,n),AtDb,params,Bmult);

% Plot of reconstruction.
figure(3)
imagesc(M.*xalpha,[0,max(x_true(:))]), colorbar, colormap(gray)
figure(4),
semilogy(iter_hist(:,2),'k','LineWidth',2), title('CG iteration vs. residual norm')