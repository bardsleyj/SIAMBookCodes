%  
%  1D kernel reconstruction.
%
%  TSVD regularization is implemented and the regularization parameter k 
%  is chosen using one of GCV, or DP.  
%
%  This m-file was used to generate figures in Chapter 2 of the book.
%
%  written by John Bardsley 2016.
%
% First generate the numerical integral matrix 
clear all, close all
n      = 80; %%%input(' No. of grid points = ');
h      = 1/n;
N      = 2*n-1;
A      = tril(ones(N,N))*h;
%
% Next, set up true solution x_true, which will be a non-symmetric kernel. 
% First, we create the left- and right-halves of the kernel, and then we 
% combine them and normalize the resulting kernel to obtain x_true.
t           = [-1+h:h:1-h]';
sig1        = .1; 
kernelleft  = exp(-(t(1:n)).^2/2/sig1^2);
sig2        = .2;
kernelright = exp(-(t(n+1:end)).^2/2/sig2^2);
kernel      = [kernelleft;kernelright];
kernel      = kernel/sum(kernel)/h; % normalize
x_true      = kernel;
% Now, we generate the data b = A*x_true + error.
Ax      = A*x_true;
err_lev = 2; % Percent error in data;
sigma   = err_lev/100 * norm(Ax) / sqrt(N);
eta     =  sigma * randn(N,1);
b       = Ax + eta;
figure(1), 
  plot(t,x_true,'k',t,b,'ko')
%
% Compute the TSVD solution, choosing k using one of UPRE, GCV, or DP.
[U,S,V] = svd(A);
dS      = diag(S); 
Utb     = U'*b;
param_choice = input(' Enter 1 for UPRE, 2 for GCV, or 3 for DP. ');
if param_choice == 1 % UPRE
    RegParam_fn  = @(k) norm(Utb(k+1:N))^2+2*sigma^2*k;
elseif param_choice == 2 % GCV
    RegParam_fn  = @(k) norm(Utb(k+1:N))^2/(N-k)^2;
elseif param_choice == 3 % DP
    RegParam_fn  = @(k) abs(norm(Utb(k+1:N))^2-N*sigma^2);
end
% Find k that minimizes RegParam_fn.
RegFunVals = zeros(N,1);
for i = 1:N, RegFunVals(i)=RegParam_fn(i); end
k = find(RegFunVals == min(RegFunVals));
%
% Now compute the regularized solution and plot it.
phi         = zeros(N,1); phi(1:k)=1; 
idx         = (dS>0);
dSfilt      = zeros(size(dS));
dSfilt(idx) = phi(idx)./dS(idx); 
xfilt       = V*(dSfilt.*(U'*b));
rel_error   = norm(xfilt-x_true)/norm(x_true)
figure(2)
  plot(t,x_true,'b-',t,xfilt,'k-')
