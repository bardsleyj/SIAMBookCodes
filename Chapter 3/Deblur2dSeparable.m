%  
%  2d image deblurring with separable kernel and zero boundary conditions.
%
%  Tikhonov regularization is implemented with the regularization parameter 
%  chosen using either UPRE or DP. 
%
%  This m-file was used to generate figures in Chapter 3 of the book.
%
%  written by John Bardsley 2016.
%
%  First, create Toeplitz matrices A1 and A2 such that A=kron(A2,A1).
clear all, close all
load ../MatFiles/satellite
[n,n]   = size(x_true);
X_true  = x_true/max(x_true(:));
N       = n^2;
h       = 1/n;
t       = [h/2:h:1-h/2]';
sig     = .02; %%%input(' Kernel width sigma = ');
kernel1 = (1/sqrt(2*pi)/sig) * exp(-(t-h/2).^2/2/sig^2);
kernel2 = kernel1;
A1      = toeplitz(kernel1)*h;
A2      = toeplitz(kernel2)*h;

% Generate the data and compute the least squares solution.
Ax      = (A1*X_true)*A2';
err_lev = 2; % percent error in data
sigma   = err_lev/100 * norm(Ax(:)) / sqrt(N);
eta     =  sigma * randn(n,n);
B       = Ax + eta;
% Plot of the true image, blurred noisy data, and A^{-1}b.
figure(1) 
  imagesc(X_true), colorbar, colormap(1-gray)
figure(2)
  imagesc(B), colorbar, colormap(1-gray)
figure(3)
  imagesc((A1\B)/A2'), colorbar, colormap(1-gray)
  title('A^{-1}b')

% Compute SVDs of A1 and A2 for the computations that follows.
[U1,S1,V1] = svd(A1);
[U2,S2,V2] = svd(A2);
dS1        = diag(S1);
dS2        = diag(S2);
dS1dS2     = dS1*dS2';
Utb        = (U1'*B)*U2;

% Find the UPRE or DP choice of alpha and plot the reconstruction. 
alpha_flag = input(' Enter 1 for UPRE and 2 for DP regularization parameter selection. ');
if alpha_flag == 1 % UPRE choice for the regularization parameter.
    RegParam_fn = @(a) a^2*sum(sum((Utb.^2)./(dS1dS2.^2+a).^2))...
                          +2*sigma^2*sum(sum(dS1dS2.^2./(dS1dS2.^2+a)));
elseif alpha_flag == 2 % DP choice for the regularization parameter.
    RegParam_fn = @(a) abs(a^2*sum(sum((Utb.^2)./(dS1dS2.^2+a).^2))-N*sigma^2);
end
alpha  = fminbnd(RegParam_fn,0,1);
%
% Compute the Tikhonov regularized solution and plot it.
xalpha = V1*((dS1dS2./(dS1dS2.^2+alpha)).*Utb)*V2';
figure(4)
  imagesc(xalpha), colorbar, colormap(1-gray)
