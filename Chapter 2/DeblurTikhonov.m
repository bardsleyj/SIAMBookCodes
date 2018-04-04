%  
%  1d image deblurring with zero boundary conditions.
%
%  Tikhonov regularization is implemented with the regularization parameter 
%  alpha chosen using one of UPRE, GCV, DP, or L-curve. 
%
%  This m-file was used to generate Figures in Chapter 2 of the book.
%
%  written by John Bardsley 2016.
%
%  First, create a Toeplitz matrix A using zero BCs and a Gaussian kernel.
clear all, close all
n      = 80;  % No. of grid points
h      = 1/n;
t      = [h/2:h:1-h/2]';
sig    = .03; % kernel width
kernel = (1/sqrt(2*pi)/sig) * exp(-(t-h/2).^2/2/sig^2);
A      = toeplitz(kernel)*h;
%
% Next, set up true solution x_true and data b = A*x_true + error.
x_true  = .75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4;
x_true  = x_true/norm(x_true);
Ax      = A*x_true;
err_lev = 2; % percent error in data.
sigma   = err_lev/100 * norm(Ax) / sqrt(n);
eta     =  sigma * randn(n,1);
b       = Ax + eta;
figure(1), 
  plot(t,x_true,'k',t,b,'ko')
%
% Compute the Tikhonov solution with alpha chosen using one of UPRE, GCV, 
% DP, or L-curve.
[U,S,V] = svd(A);
dS      = diag(S); 
dS2     = dS.^2; 
Utb     = U'*b;
param_choice = input(' Enter 1 for UPRE, 2 for GCV, 3 for DP, or 4 for L-curve. ');
if param_choice == 1 %UPRE
    RegParam_fn = @(a) sum((a^2*Utb.^2)./(dS2+a).^2)+2*sigma^2*sum(dS2./(dS2+a));
elseif param_choice == 2 %GCV
    RegParam_fn = @(a) sum((a^2*Utb.^2)./(dS2+a).^2)/(n-sum(dS2./(dS2+a)))^2;
elseif param_choice == 3 %DP
    RegParam_fn = @(a) abs(sum((a^2*Utb.^2)./(dS2+a).^2)-n*sigma^2);
elseif param_choice == 4 %L-curve: 
    %!!! --- curvatureLcurve.m is at the bottom of this m-file --- !!!
    RegParam_fn = @(alpha) - curvatureLcurve(alpha,A,U,S,V,b);
end
alpha = fminbnd(RegParam_fn,0,1);
%
% Now compute the regularized solution and plot it.
dSfilt    = dS./(dS2+alpha);
xfilt     = V*(dSfilt.*(U'*b));
rel_error = norm(xfilt-x_true)/norm(x_true)
figure(2)
  plot(t,x_true,'b-',t,xfilt,'k-')

% curvatureLcurve function
function[calpha] = curvatureLcurve(alpha,A,U,S,V,b)
% This function evaluates the curvature of the L-curve 
% Inputs:
% alpha = regularization parameter
% A     = model matrix
% USV   = SVD of A
% b     = right-hand-side/measurement vector.
%
% Outputs:
% calpha = value of the curvature of the L-curve at alpha, see Ch. 2. 
% 
% written by John Bardsley 2016.
%
% Compute minus the curvature of the L-curve
dS = diag(S); dS2 = dS.^2;
Utb = U'*b;
xalpha = V*((dS./(dS2+alpha)).*(Utb));
ralpha = A*xalpha-b;
xi = norm(xalpha)^2;
rho = norm(ralpha)^2;
% From Hansen 2010 -- seems to be incorrect.
%zalpha = V*((dS./(dS2+alpha)).*(U'*ralpha));
%xi_p = (4/sqrt(alpha))*xalpha'*zalpha;
%calpha =   2*(xi*rho/xi_p) *...
%           (alpha*xi_p*rho+2*sqrt(alpha)*xi*rho+alpha^2*xi*xi_p) / ...
%           (alpha*xi^2+rho^2)^(3/2);
% From Vogel 2002. 
xi_p = sum(-2*dS2.*Utb.^2./(dS2+alpha).^3);
calpha = - ( (rho*xi)*(alpha*rho+alpha^2*xi)+(rho*xi)^2/xi_p ) / ...
           ( rho^2+alpha^2*xi^2)^(3/2);
end