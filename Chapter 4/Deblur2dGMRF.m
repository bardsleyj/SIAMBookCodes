%  
%  2d image deblurring with periodic boundary conditions.
%  Gaussian Markov random field prior with periodic boundary conditions.
%
%  Tikhonov regularization is implemented with the regularization parameter 
%  chosen using GCV.
%
%  This m-file was used to generate figures in Chapter 4 of the book.
%
%  written by John Bardsley 2016.
%
%  First, construct the Fourier transform of the circulant-shifted
%  convolution kernel.
clear all, close all
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
% Construct the Fourier representer for the GMRF prior precision matrix L:
% the discrete negative-Laplacian with periodic boundary conditions.
l = zeros(n,n); l(1, 1) =  4; 
l(2 ,1) = -1; l(n,1) = -1; 
l(1 ,2) = -1; l(1,n) = -1; 
lhat = fft2(l);
%  
% Find the GCV choice for alpha and compute the Tikhonov solution.
G_fn=@(a)(sum(sum((a^2*abs(lhat).^2.*abs(bhat/n).^2)./(abs(ahat).^2+a*(lhat)).^2))) ...
             / (N-sum(sum(abs(ahat).^2./(abs(ahat).^2+a*(lhat)))))^2;
alpha =  fminbnd(G_fn,0,1);
xalpha = real(ifft2((conj(ahat)./(abs(ahat).^2+alpha*(lhat)).*bhat)));
% Plote the Tikhonov regularized solution.
figure(3)
  imagesc(xalpha), colorbar, colormap(1-gray)
  
