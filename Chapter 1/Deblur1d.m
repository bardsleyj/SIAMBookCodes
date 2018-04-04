%  
% 1D image deblurring with zero BCs, Example 1.2, Chapter 1. 
% 
% written by John Bardsley 2016.
%
% First, create the Toeplitz matrix A (a zero boundary condition is
% assumed) resulting from a Gaussian kernel. 
clear all, close all
n      = 80; 
h      = 1/n;
t      = [h/2:h:1-h/2]';
gamma  = .03; 
kernel = (1/sqrt(2*pi)/gamma) * exp(-(t-h/2).^2/2/gamma^2);
A      = toeplitz(kernel)*h;
%
% Next, set up true solution x_true and data b = A*x_true + error.
x_true  = 50*(.75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4);
x_true  = x_true/norm(x_true);
Ax = A*x_true;
err_lev = 2; 
sigma   = err_lev/100 * norm(Ax) / sqrt(n);
eta     =  sigma * randn(n,1);
b       = Ax + eta;
figure(1), 
  plot(t,x_true,'k',t,b,'ko')
%
%  Finally, compute the least squares solution as well as the norm of the
%  perturbations of x and b.
figure(2),
  plot(t,A\b,'k')

% Verify ill-posedness inequality (1.21)
delta_x = norm(x_true-A\b)/norm(x_true)
delta_b = norm(b-Ax)/norm(Ax)

% Finally, compute the SVD of A and plot the singular values and vectors.
[U,S,V] = svd(A);
figure(3),
  subplot(221), semilogy(diag(S),'ko'), title('log(\sigma_i)'), axis([1 length(t) 0 1])
  subplot(222), plot(t,V(:,1),'k'), title('v_1')
  subplot(223), plot(t,V(:,5),'k'),title('v_5')
  subplot(224), plot(t,V(:,10),'k'),title('v_{10}')
  
% Verify the ill-posedness inequality (1.27)
ill_posed_lhs = sqrt(sum(sigma^2./(diag(S).^2)))/norm(x_true)
ill_posed_rhs = sqrt(n*sigma^2)/norm(Ax)