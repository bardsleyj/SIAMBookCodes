%  
%  1d image deblurring inverse problem with zero boundary conditions.
%  Gaussian Markov random field prior with zero boundary conditions. 
%
%  Tikhonov regularization is implemented with the regularization parameter 
%  chosen by GCV. 
%
%  This m-file was used to generate Figures in Chapter 4 of the book.
%
%  written by John Bardsley 2016.
%
%  First, create the Toeplitz matrix A (a zero boundary condition is
%  assumed) resulting from a symmetric Gaussian kernel. 
clear all, close all
n = 80; % No. of grid points
h = 1/n;
t = [h/2:h:1-h/2]';
sig = .03; % Kernel width
kernel = (1/sqrt(2*pi)/sig) * exp(-(t-h/2).^2/2/sig^2);
A = toeplitz(kernel)*h;
%
% Next, set up true solution x_true and generate data blurred noisy data. 
x_true  = 50*(.75*(.1<t&t<.25) + .25*(.3<t&t<.32) ...
                               + (.5<t&t<1).*sin(2*pi*t).^4);
x_true  = x_true/norm(x_true);
Ax      = A*x_true;
err_lev = 2; % Percent error in data
sigma   = err_lev/100 * norm(Ax) / sqrt(n);
eta     =  sigma * randn(n,1);
b       = Ax + eta;
figure(1),
  plot(t,x_true,'k',t,b,'ko')
%
% GMRF prior precision matrix: discrete second derivative with zero BCs.
L       = spdiags([-ones(n,1) 2*ones(n,1) -ones(n,1)],[-1 0 1],n,n);
%
% Compute GCV choice of alpha a nd compute the Tikhonov solution.
GCV_fn  = @(a) norm(A*((A'*A+a*L)\(A'*b))-b)^2/(n-trace(A*((A'*A+a*L)\A')))^2;
alpha   = fminbnd(GCV_fn,0,1);
xalpha  = (A'*A+alpha*L)\(A'*b);
%
% Plot the Tikhonov solution
figure(2)
  plot(t,x_true,'b',t,xalpha,'k','LineWidth',1)
  legend('true image','blurred, noisy data')
