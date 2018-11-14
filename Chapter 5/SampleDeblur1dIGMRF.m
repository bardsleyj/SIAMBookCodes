%  
%  1d image deblurring with independent increment IGMRF prior: 
%  spatially dependent weights are computed iteratively using 
%  Algorithm 4.1; regularization parameter is computed using GCV. 
%  Then the resulting Gaussian posterior density function is sampled 
%  using the Cholesky factorization of the matrix. 
%
%  Once the samples are computed, the sample mean is used as an estimator 
%  of the unknown image and empirical quantiles are used to compute 95%
%  credibility intervals for every unknown. 
% 
%  written by John Bardsley 2016.
%
%  This m-file was used to generate a plot in Chapter 5 of the book.
clear all, close all
path(path,'../Functions');
n = 80; % No. of grid points 
h = 1/n;
t = [h/2:h:1-h/2]';
sig = .03; % Kernel width sigma
kernel = (1/sqrt(2*pi)/sig) * exp(-(t-h/2).^2/2/sig^2);
A = toeplitz(kernel)*h;

% Next, set up true solution x_true and data b = A*x_true + error.
x_true = 50*(.75*(.1<t&t<.25) + .25*(.3<t&t<.32) + (.5<t&t<1).*sin(2*pi*t).^4);
x_true = x_true/norm(x_true);
Ax = A*x_true;
err_lev = 2; % Percent error in data
sigma = err_lev/100 * norm(Ax) / sqrt(n);
eta =  sigma * randn(n,1);
b = Ax + eta;
figure(1),
plot(t,x_true,'k',t,b,'ko')

% First derivative matrix D for the increments with Neumann BC
D      = spdiags([-ones(n,1) ones(n,1)],[0 1],n-1,n);

% Edge-preserving reconstruction algorithm as defined in Algorithm 4.1.
niters = 10; % number of iterations of Algorithm 4.1
for i=1:niters
    % Compute the weight matrix.
    if i==1, G = speye(n-1,n-1);
    else,    G = diag(1./sqrt((D*xalpha).^2+.001)); end
    % Define the prior precision.
    L = D'*G*D;
    % Compute GCV choice of alpha.
    RegFun = @(a) norm(A*((A'*A+(10^a)*L)\(A'*b))-b)^2/(n-trace(A*((A'*A+(10^a)*L)\A')))^2;
    alpha  = 10^fminbnd(RegFun,-16,0);
    % Compute the Tikhonov regularized solution.
    xalpha = (A'*A+alpha*L)\(A'*b);
    fprintf('iteration = %d\n',i)
end

% Sample from the posterior with prior precision outputed by Algorithm 4.1
% First, estimate lambda and delta and compute xMAP.
lambda  = 1/var(b-A*xalpha);
delta   = alpha*lambda;
R       = chol(lambda*A'*A + delta*L);
xMAP    = R\(R'\(lambda*A'*b));
% Next, sample from the posterior p(x|b,\lambda,delta)
nsamps  = 1000;
xsamp   = repmat(xMAP,1,nsamps) + R\randn(n,nsamps);
% Finally, compute the mean and 95% credibility intervals and plot.
xmean   = mean(xsamp,2);
xquants = plims(xsamp',[0.025,0.975])';
figure(2)
plot(t,x_true,'b-',t,xquants(:,1),'k--',t,xquants(:,2),'k--',t,xmean,'k-')