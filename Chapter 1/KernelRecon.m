%  
% 1D kernel reconstruction, Example 1.3, Chapter 1.
% 
% written by John Bardsley 2016.
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

%  Finally, compute the least squares solution as well as the norm of the
%  perturbations of x and b.
figure(2),
  plot(t,A\b,'k',t,x_true,'b')

% Verify ill-posedness inequality (1.21)
delta_x = norm(x_true-A\b)/norm(x_true)
delta_b = norm(b-Ax)/norm(Ax)

% Finally, compute the SVD of A and plot the singular values and vectors.
[U,S,V] = svd(A);
figure(3),
  subplot(221), semilogy(diag(S),'ko'), title('log(\sigma_i)'), axis([1 length(t) 0 1])
  subplot(222), plot(t,V(:,1),'k'), title('v_1'), axis([-1 1 -.2 .2])
  subplot(223), plot(t,V(:,5),'k'),title('v_5'), axis([-1 1 -.2 .2])
  subplot(224), plot(t,V(:,10),'k'),title('v_{10}'), axis([-1 1 -.2 .2])
  
% Verify the ill-posedness inequality (1.27)
ill_posed_lhs = sqrt(sum(sigma^2./(diag(S).^2)))/norm(x_true)
ill_posed_rhs = sqrt(n*sigma^2)/norm(Ax)