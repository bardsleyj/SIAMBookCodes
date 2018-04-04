%  
%  This m-file accompanies Exercise 7 in Chapter 3, which illustrates 
%  how the assumption of periodic boundary conditions yields poor 
%  reconstructions when it is inaccurate.
% 
%  written by John Bardsley 2016.
%
% Generate data and build regularization matrix.
clear all, close all
load ../MatFiles/cell
x_true  = 100*x_true/max(x_true(:));
[n,n]   = size(x_true);
% Generate data on 256^2 grid w/ periodic BCs, then restrict to 128^2.
h       = 1/n;
x       = [-0.5+h/2:h:0.5-h/2]';
[X,Y]   = meshgrid(x);
sig     = 2*h;
kernel  = exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
kernel  = kernel/sum(sum(kernel));
Ax      = real(ifft2(fft2(fftshift(kernel)).*fft2(x_true)));
ll      = 101;
n2      = 128;
indx    = ll:ll+n2-1;
Ax      = Ax(indx,indx);
err_lev = 2;
noise   = err_lev/100 * norm(Ax(:)) / sqrt(n2^2);%input(' The standard deviation of the noise = ');
b       = Ax + noise*randn(n2,n2);
% Plot the true image restricted to the same suregion as the measurements.
figure(1), imagesc(x_true(indx,indx)), colormap(gray), colorbar,
% Plot the measurements.
figure(2), imagesc(b,[0,max(x_true(:))]), colormap(gray), colorbar

% Use 128^2 PSF model with periodic BCs and GCV choice of alpha
% For this, extract the central 128x128 pixels of the integral kernel.
kernel  = kernel(n/4+1:3*n/4,n/4+1:3*n/4);