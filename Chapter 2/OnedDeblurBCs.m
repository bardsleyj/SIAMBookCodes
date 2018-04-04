%  
%  1d image deblurring with data-driven boundary conditions.
%
%  Tikhonov regularization is implemented with regularizaton parameter 
%  alpha chosen using the discrepancy principle.
%
%  This m-file was used to generate a figure in Chapter 2 of the book.
%
%  written by John Bardsley 2016.
%
%  Create the Toeplitz matrix A assuming a zero boundary condition on [0,1]
%  then restrict the output (rows of A) to (0.15,0.85).  
clear all, close all
n      = 120; % No. of grid points
h      = 1/n;
t      = [h/2:h:1-h/2]';
sig    = .02; % Kernel width
kernel = (1/sqrt(2*pi)/sig) * exp(-(t-h/2).^2/2/sig^2);
A      = toeplitz(kernel)*h;
% Restrict the output to (0.15,0.85).
ii     = find(t>.15 & t<.85);
A      = A(ii,:);

% Set up true solution x_true and data b = A*x_true + error.
x_true  = .75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4;
x_true  = x_true/norm(x_true);
Ax      = A*x_true;
err_lev = 2; % Percent error in the data.
sigma   = err_lev/100 * norm(Ax) / sqrt(length(Ax));
eta     =  sigma * randn(length(Ax),1);
b       = Ax + eta;

% First, compute the Tikhonov solution using the data driven BCs.
[U,S,V] = svd(A,'econ');
dS      = diag(S); 
dS2     = dS.^2; 
Utb     = U'*b;
% Compute the DP choice of regularization parameter.
RegParam_fn = @(a) (sum((a^2*Utb.^2)./(dS2+a).^2)-n*sigma^2)^2;
alpha       = fminbnd(RegParam_fn,0,1);
% Compute the corresponding regularized solution.
dSfilt = dS./(dS.^2+alpha);
xfilt  = V*(dSfilt.*(U'*b));

% Next, compute the Tikhonov solution using zero BCs on (0.15,0.85), by
% restricting the input (columns of A) also to (0.15,0.85). 
Azero   = A(:,ii);
[U,S,V] = svd(Azero);
dS      = diag(S); 
dS2     = dS.^2; 
Utb     = U'*b;
% Compute the DP choice of regularization parameter for the new test case.
RegParam_fn = @(a) (sum((a^2*Utb.^2)./(dS2+a).^2)-n*sigma^2)^2;
alpha       = fminbnd(RegParam_fn,0,1);
% Compute the corresponding regularized solution.
dSfilt      = dS./(dS.^2+alpha);
xfiltZeroBC = V*(dSfilt.*(U'*b));

% Finally, plot these two solutions on (0.15,0.85) with x_true on (0,1).
figure(1), 
  plot(t,x_true,'k',t(ii),b,'ko'), axis([t(1),t(end), -0.05, 0.35])
  legend('true image','blurred, noisy data','Location','NorthWest')
figure(2)
  plot(t,x_true,'k-',t(ii),xfilt(ii),'k.-',t(ii),xfiltZeroBC,'k+-')
  axis([t(1),t(end), -0.05, 0.35])
  legend('true image','data driven BC reconstruction','zero BC reconstruction','Location','NorthWest')
