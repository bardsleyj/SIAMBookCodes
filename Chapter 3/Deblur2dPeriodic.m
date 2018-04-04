%  
%  2d image deblurring with periodic boundary conditions.  
%
%  Tikhonov regularization is implemented with the regularization parameter 
%  chosen using either GCV or L-curve.
%
%  This m-file was used to generate figures in Chapter 3 of the book.
%
%  written by John Bardsley 2016.
%
%  First, construct the Fourier transform of the circulant-shifted,
%  Gaussian convolution kernel.
clear all, close all
load ../MatFiles/satellite
x_true  = x_true/max(x_true(:));
[n,n]   = size(x_true);
N       = n^2;
h       = 1/n;
x       = [-0.5+h/2:h:0.5-h/2]';
[X,Y]   = meshgrid(x);
sig     = 0.02;
kernel  = h^2*(1/(2*pi)/sig^2)*exp(-((X-h/2).^2+(Y-h/2).^2)/2/sig^2);
ahat    = fft2(fftshift(kernel));
% We can now define the function performing multiplication by A and
% generate noisy data by adding iid Gaussian measurement error.
Ax      = real(ifft2(ahat.*fft2(x_true)));
err_lev = 2; %%%input(' Percent error in data = ')
sigma   = err_lev/100 * norm(Ax(:)) / sqrt(N);
eta     = sigma*randn(n,n);
b       = Ax+eta;
bhat    = fft2(b);
% Plot the true image, blurred noisy data, and A^{-1}b.
figure(1) 
  imagesc(x_true), colorbar, colormap(1-gray)
figure(2)
  imagesc(b), colorbar, colormap(1-gray)
figure(3)
  imagesc(real(ifft2(bhat./ahat))), colorbar, colormap(1-gray)
  title('Plot of A^{-1}b')
  
% Now, find the GCV or L-curve choice for alpha and plot the reconstruction
alpha_flag = input(' Enter 1 for GCV and 2 for L-curve regularization parameter selection. ');
if alpha_flag == 1 % GCV
    RegParam_fn  = @(a)(sum(sum((a^2*abs(bhat/n).^2)./(abs(ahat).^2+a).^2)))...
             /(N-sum(sum(abs(ahat).^2./(abs(ahat).^2+a))))^2;
elseif alpha_flag == 2 % L-curve 
    %!!! --- curvatureLcurve.m is at the bottom of this m-file --- !!!
    RegParam_fn  = @(alpha) - curvatureLcurve(alpha,ahat,b,bhat);
end
alpha =  fminbnd(RegParam_fn,0,1);
%
% Compute the regularized solution and plot it.
xalpha    = real(ifft2((conj(ahat)./(abs(ahat).^2+alpha).*bhat)));
figure(4)
  imagesc(xalpha), colorbar, colormap(1-gray)
  
% curvatureLcurve function:
  function[calpha] = curvatureLcurve(alpha,ahat,b,bhat)
% This function evaluates the curvature of the L-curve
% Inputs:
% alpha = regularization parameter
% ahat  = eigenvalues of BCCB matrix A
% b     = right-hand-side/measurement vector.
%
% Outputs:
% calpha = value of the curvature of the L-curve at alpha, see Ch. 3. 
% 
% written by John Bardsley 2016.
%
% Compute minus the curvature of the L-curve
xalpha = real(ifft2((ahat./(abs(ahat).^2+alpha)).*bhat));
ralpha = real(ifft2(ahat.*fft2(xalpha)))-b;
xi     = norm(xalpha)^2;
rho    = norm(ralpha)^2;

% From Vogel 2002. 
N      = length(b(:));
xi_p   = sum(sum(-2*abs(ahat).^2.*abs(bhat/sqrt(N)).^2./(abs(ahat).^2+alpha).^3));
calpha = - ( (rho*xi)*(alpha*rho+alpha^2*xi)+(rho*xi)^2/xi_p ) / ...
           ( rho^2+alpha^2*xi^2)^(3/2);
end