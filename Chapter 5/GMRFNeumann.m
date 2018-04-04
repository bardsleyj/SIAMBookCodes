% 
% This m-file implements sampling from a Gaussian Markov random field in
% both one- and two-dimensions with precision matrix given by the discrete
% negative-Laplacian with Neumann boundary conditions. It also implements
% sampling from independent increment IGMRFs, in which edges appear in the  
% samples. Figures in Chapter 4 were generated using this code, and the 
% methodology is described in Ch 5.
% 
% written by John Bardsley 2016.
%
clear all, close all
% 1D iid increment case
% x_{i+1}-x_i ~ N(0,1)
n=128;
one_vec = ones(n,1);
D       = spdiags([-one_vec one_vec],[0 1],n-1,n);
R       = chol(D'*D+sqrt(eps)*speye(n,n));
v       = randn(n-1,5);
samps   = R\(R'\(D'*v));
figure(1)
  plot(samps)
% 1D independent (but not identically distributed) increment case.
% x_{i+1}-x_i ~ N(0,w_i), where w_n/2 = 0.05 and w_i=1 otherwise.
W          = speye(n-1,n-1);
W(n/2,n/2) = 0.05;
WD         = W*D;
R          = chol(WD'*WD+sqrt(eps)*speye(n,n));
samps      = R\(R'\(WD'*v));
figure(2)
  plot(samps,'k')
  axis([0 128 min(samps(:)) max(samps(:))])
% 2D iid increment case
% x_{i+1,j}-x_{i,j} ~ N(0,1) and x_{i,j+1}-x_{i,j} ~ N(0,1)
I      = speye(n,n);
Dh     = kron(I,D);
Dv     = kron(D,I);
D      = [Dh;Dv];
R      = chol(D'*D + sqrt(eps)*speye(n^2,n^2));
v      = randn(max(size(D)),1);
sample = R\(R'\(D'*v));
figure(3)
  imagesc(reshape(sample,n,n)), colormap(gray), colorbar
% 2D independent (but not identically distributed) increment case
% x_{i+1,j}-x_{i,j} ~ N(0,w_ij) and x_{i,j+1}-x_{i,j} ~ N(0,w_ij),
% where w_ij=0.05 on the boundary of a circle and w_ij=1 otherwise.
x = [1/(n+1):1/(n+1):1-1/(n+1)];
[X,Y]=meshgrid(x);
Z = (X-.5).^2+(Y-.5).^2;
circle = (Z(:)<.1); % indicator function on a circle
ngradderiv = sqrt((Dh*circle).^2 + (Dv*circle).^2);
Wdiag = ones(n*(n-1),1);
Wdiag(ngradderiv > 0) = 0.05;
W = spdiags(Wdiag,0,n*(n-1),n*(n-1));
D = [W*Dh;W*Dv];
R = chol(D'*D+sqrt(eps)*speye(n^2,n^2));
sample = R\(R'\(D'*v));
figure(4)
  imagesc(reshape(sample,n,n)), colormap(gray), colorbar
