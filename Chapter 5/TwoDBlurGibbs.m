%  
%  Hierarchical Gibbs sampler for 2d image deblurring with periodic BCs.
%
%  Once the samples are computed, the sample mean is used as an estimator 
%  of the unknown image and empirical quantiles are used to compute 95%
%  credibility intervals for each unknown. 
%
%  The Geweke test is used to determine whether the chain, after burnin, 
%  is in equilibrium, and the integrated auto correlated time and essential 
%  sample size are estimated as described in Chapter 5. 
% 
%  written by John Bardsley 2016.
%
%  This code was used to generate figures in Chapter 5. 
clear all, close all
path(path,'../Functions')
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

Ax      = feval('Amult',x_true,ahat);
err_lev = 2; % Percent error in data
sigma   = err_lev/100 * norm(Ax(:)) / sqrt(N);
eta     =  sigma * randn(n,n);
b       = Ax + eta;
bhat    = fft2(b);
figure(1)
  imagesc(x_true), colorbar, colormap(1-gray)
figure(2)
  imagesc(b), colorbar, colormap(1-gray)

% Construct Fourier representer for discrete Laplacian L.
l       = zeros(n,n);
l(1, 1) =  4; l(2 ,1) = -1;
l(n,1)  = -1; l(1 ,2) = -1;
l(1,n)  = -1; lhat = abs(fft2(l)).^2;
clear l Ax PSF

%% MCMC sampling
nsamps  = 10000;
lamsamp = zeros(nsamps,1); lamsamp(1) = 7;
delsamp = zeros(nsamps,1); delsamp(1) = .005;
xsamp   = zeros(N,nsamps);
fourier_filt = lamsamp(1)*abs(ahat).^2 + delsamp(1)*lhat;
xtemp   = real(ifft2(conj(ahat).*(lamsamp(1)*bhat)./fourier_filt + ...
        fft2(randn(n,n))./sqrt(fourier_filt)));
nFFT    = 2;
xsamp(:,1) = xtemp(:);

% hyperprior parameters: lambda~Gamma(a,1/t0), delta~Gamma(a1,1/t1)
a0=1; t0=0.0001; a1=1; t1=0.0001;
h = waitbar(0,'MCMC samples in progress');
tic
for i=1:nsamps-1
    h = waitbar(i/nsamps);
    %------------------------------------------------------------------
    % 1a. Using conjugacy, sample the noise precision lam=1/sigma^2,
    % conjugate prior: lam~Gamma(a0,1/t0), mean = a0/t0, var = a0/t0^2.
    Axtemp       = real(ifft2(ahat.*fft2(xtemp)));
    lamsamp(i+1) = gamrnd(a0+N/2,1./(t0+norm(Axtemp(:)-b(:))^2/2));
    %------------------------------------------------------------------
    % 1b. Using conjugacy, sample regularization precisions delta,
    % conjugate prior: delta~Gamma(a1,1/t1);
    Lxtemp = real(ifft2(lhat.*fft2(xtemp)));
    delsamp(i+1) = gamrnd(a1+(N-1)/2,1./(t1+xtemp(:)'*Lxtemp(:)/2));
    %------------------------------------------------------------------
    % 2. Using conjugacy relationships, sample the image.
    fourier_filt = lamsamp(i+1)*abs(ahat).^2 + delsamp(i+1)*lhat;
    xMAP         = real(ifft2(conj(ahat).*(lamsamp(i+1)*bhat)./fourier_filt));
    E            = randn(n,n)+sqrt(-1)*randn(n,n);
    xtemp        = xMAP + n*real(ifft2(E./sqrt(fourier_filt)));
    nFFT         = nFFT + 2;
    xsamp(:,i+1) = xtemp(:);
end
toc
close(h)     
clear Lxtemp Axtemp xtemp ahat b bhat eta fourier_filt kernel lhat X Y
% Remove the burn-in samples and visualize the MCMC chain
% Plot the mean and 95% credibility intervals for x
nburnin    = floor(nsamps/10); 
xsamp      = xsamp(:,nburnin+1:end);
q          = plims(xsamp(:,:)',[0.025,0.975]);
x_mean     = mean(xsamp(:,:)')';
relative_error = norm(x_true(:)-x_mean(:))/norm(x_true(:));
figure(3), colormap(1-gray)
  imagesc(reshape(x_mean,n,n)), colorbar
figure(4), colormap(1-gray)
  imagesc(reshape(q(2,:)-q(1,:),n,n)), colorbar
% Output for (lambda,delta)-chain using sample_plot: ACF, IACT, Geweke test
lamdel_chain = [lamsamp(nburnin+1:end), delsamp(nburnin+1:end)]';
names        = ["\lambda","\delta"];
fignum       = 5;
[taux,acfun] = sample_plot(lamdel_chain,names,fignum);
% Plot the autocorrelations functions together.
[~,nacf] = size(acfun);
figure(8)
plot([1:nacf],acfun(1,:),'r',[1:nacf],acfun(2,:),'k--','LineWidth',2)
axis([1,nacf,0,1])
title('ACFs for \lambda and \delta.')
legend(['\tau_{\rm int}(\lambda)=',num2str(taux(1))],['x_2: \tau_{\rm int}(\delta)=',num2str(taux(2))])