% 
% This m-file implements sampling from a Gaussian Markov random field in
% both one- and two-dimensions, with precision matrix given by the discrete
% negative-Laplacian with zero boundary conditions. Several plots in Ch 4
% were generated using this code and the methodology is described in Ch 5.
% 
% written by John Bardsley 2016.
%
clear all, close all
% 1D case
% Build the L matrix corresponding to forward or backward difference
%     -x(i-1)-x(i+1)+2*x(i)
%      x(0) = x(n+1) = 0
n       = 128;
one_vec = ones(n,1);
L1D     = spdiags([-one_vec 2*one_vec -one_vec],[-1 0 1],n,n);
R       = chol(L1D);
v       = randn(n,5);
samps   = R\v;
figure(1)
  plot(samps,'k')
  axis([0 128 min(samps(:)) max(samps(:))])

%  2D case
%  Build the L matrix corresponding to the neighborhood relationship
%     -x(i-1,j)-x(i+1,j)-x(i,j-1)-x(i,j+1)+4*x(i) 
%      x(0,j) = x(n+1,j) = x(i,0) = x(i,n+1) = 0
I    = speye(n,n);
L2D  = kron(I,L1D)+kron(L1D,I);
v    = randn(n^2,1);
R    = chol(L2D);
samp = R\v;
figure(2)
  imagesc(reshape(samp,n,n)), colorbar, colormap(gray)

  