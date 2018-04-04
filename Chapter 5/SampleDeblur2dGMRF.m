%  
%  Sampling from a high-dimensional Gaussian distribution arising in a
%  2d image deblurring inverse problem with periodic boundary conditions
%  and a Gaussian Markov random field prior, also with periodic BCs.
%  The regularization parameter is chosen using GCV. The Gaussian 
%  posterior density function is sampled from using the DFT-based 
%  diagonalization of the covariance described in Chapter 5.  
%
%  Once the samples are computed, the sample mean is used as an estimator 
%  of the unknown image and empirical quantiles are used to compute 95%
%  credibility intervals for every unknown. 
% 
%  written by John Bardsley 2016.
%
%  This m-file was used to generate a plot in Chapter 5 of the book.
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

Ax = real(ifft2(ahat.*fft2(x_true)));
err_lev = 2; % Percent error in data
sigma = err_lev/100 * norm(Ax(:)) / sqrt(N);
rng('default')
eta =  sigma * randn(n,n);
b = Ax + eta;
bhat = fft2(b);
figure(1) 
  imagesc(x_true), colorbar, colormap(1-gray)
figure(2)
  imagesc(b), colorbar, colormap(1-gray)
  
% Fourier representer for the discrete negative-Laplacian precision matrix.
l = zeros(n,n); l(1, 1) =  4; 
l(2 ,1) = -1; l(n,1) = -1; 
l(1 ,2) = -1; l(1,n) = -1; 
lhat = fft2(l);
  
% Find the GCV choice for alpha, then use it to compute xalpha.
G_fn   = @(a)(sum(sum(((10^a)^2*abs(lhat).^4.*abs(bhat).^2)./...
                 (abs(ahat).^2+(10^a)*lhat.^2).^2)))...
             / (N-sum(sum(abs(ahat).^2./(abs(ahat).^2+(10^a)*lhat.^2))))^2;
alpha  =  10^fminbnd(G_fn,-16,0);
xalpha = real(ifft2((conj(ahat)./(abs(ahat).^2+alpha*lhat.^2).*bhat)));

% Now compute samples from the posterior density function.
% First, estimate lambda and delta
resid        = b - real(ifft2(ahat.*fft2(xalpha)));
lambda       = 1/var(resid(:));
% Next, sample from the posterior p(x|b,\lambda,delta)
delta        = alpha*lambda;
nsamps       = 1000;
xsamp        = zeros(N,nsamps);
fourier_filt = lambda*abs(ahat).^2 + delta*abs(lhat).^2;
xMAP         = xalpha;
tic
for i=1:nsamps
    h            = waitbar(i/nsamps);
    E            = randn(n,n)+sqrt(-1)*randn(n,n); 
    xtemp        = xMAP + n*real(ifft2(E./sqrt(fourier_filt)));
    xsamp(:,i+1) = xtemp(:);
end
toc
close(h)   
% Finally, compute the mean and 95% credibility images and plot them.
x_mean = mean(xsamp')';
quants = plims(xsamp',[0.025,0.975])';
figure(3), colormap(1-gray)
  imagesc(reshape(x_mean,n,n)), colorbar
figure(4), colormap(1-gray)
  imagesc(reshape(quants(:,2)-quants(:,1),n,n)), colorbar
