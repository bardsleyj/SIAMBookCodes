%  OneDPoissonEqnHierarchical.m
%  First set up finite difference discretization of 1-d steady-state 
%  diffusion equation,
%    -d/ds(x(s) du/ds) = f(s),  0 < s < 1,  l = 1,2,
%    u(0) = u_L; u(1) = u_R.
%  We collect data correpsonding to two forcing functions:
%       f(x) = delta(x-1/3)  and f(x) = delta(x-2/3).
%  Then we solve for u and generate data [d_1;d_2] corresponding to the 
%  two forcing functions. 
%
%  After that, we implement a hierarchical Gibbs MCMC method, which makes 
%  use of conjugacy relationships and embedded  RTO-MH samples.
%
%  written by John Bardsley 2016.
%
%  This code was used to generate figures in Chapter 6. 
clear all, close all
path(path,'../Functions');
%  Set up numerical mesh and define the 'true' diffusion coefficient.
N        = 64; % No. of interior FEM nodes
h        = 1 / N;
s        = [h:h:1-h]';     % numerical mesh for u
s_mid    = [h/2:h:1-h/2]'; % numerical mesh for x
x_true   = min(1,1-.5*sin(2*pi*(s_mid-.25))); 

%  Compute right-hand side vectors f_1 and f_2. Incorporate non-homogeneous
%  boundary conditions if necessary.
f        = zeros(N-1,2);
f(floor(N/3),1)   = 1;   % = delta source at left boundary.
f(floor(2*N/3),2) = 1;   % = delta source at right boundary.
f        = 1e3*f;

%  Compute stiffness matrix B(x) and observation matrix C.
onevec   = ones(N,1);
D        = (1/h)*spdiags([-onevec onevec],[-1 0],N,N-1);
B        = @(x) D'*(spdiags(x,0,N,N)*D); % Stiffness matrix.
Bprime   = @(i,v) D(i,:)'*(D(i,:)*v);
C        = speye(N-1); % Observation operator.
A        = @(x) C*(B(x)\f); % Forward Operator.

% Generate measurements.
model    = A(x_true);
M        = length(model(:));
percent_error = 1; % Noise level percentage
sig      = percent_error / 100 * norm(model)  / sqrt(M);
noise    = sig * randn(size(model));
b        = model + noise;

%% Next, estimate x from the measurments using Levenburg-Marquardt
% First define the regularization matrix L.
Dreg     = (1/h)*spdiags([-onevec onevec],[0 1],N-1,N);
Nbar     = N-1; % rank of Dreg'*Dreg
Nresid   = M+length(Dreg(:,1));
% Define parameters that are needed for evaluating the residual and 
% Jacobian function defined below.
p.b      = b;
p.C      = C;
p.B      = B;
p.Bprime = Bprime;
p.f      = f;
p.lam    = 1/sig^2;
p.del    = 0.6;
p.Q      = speye(Nresid);   % this is non-identity when we do RTO.
p.e      = zeros(Nresid,1); % this is iid Gaussian when we do RTO.
p.Lsq    = Dreg;
p.Nrand  = Nresid;
% Compute the MAP estimator using Levenburg-Marquardt optimization.
x0       = ones(size(x_true)); % initial guess
[xMAP,rMAP,JMAP,hist] = LevMar(x0,@(x)diffusion_fun(x,p),0.001,1e-8,1000);
resid    = b-A(xMAP);
var_est  = var(resid(:));
[Q,~]    = qr(JMAP,0);
    
% Plot the measurements and model fit 
figure(1)
plot(s,b,'ko',s,A(x_true),'k-',s,A(xMAP),'k--')
xlabel('s'), title('Plot of data (o), true model (-), and model fit (--)')
figure(2)
plot(s_mid,x_true,'k-',s_mid,xMAP,'k--')
xlabel('s'), ylabel('x'), title('Plot of true (-) and estimated (--) diffusion coefficient')

%% Now, use RTO-MH within Hierarchical Gibbs to sample from 
%              
%  p(x,lam,del|b) \propto exp(-lam/2*||A(x)-b||^2-del/2*||D*x||^2).
%
nsamps   = 5000;
xchain   = zeros(N,nsamps); xchain(:,1) = xMAP;
delsamp  = zeros(nsamps,1); delsamp(1)  = p.del;
lamsamp  = zeros(nsamps,1); lamsamp(1)  = 1/sig^2;
% hyperprior parameters:
a0 = 1; t0=1e-4; 
a1 = 1; t1=1e-4; 
% The MCMC method:
h = waitbar(0,'MCMC samples in progress');
tic
for i = 1:nsamps-1
    h = waitbar(i/nsamps);
    %------------------------------------------------------------------
    % 1a. Using conjugacy, sample regularization precisions delta,
    % conjugate prior: delta~Gamma(a0,1/t0);
    resid        = b-A(xchain(:,i));
    lamsamp(i+1) = gamrnd(a0+M/2,1./(t0+norm(resid(:))^2/2));
    %------------------------------------------------------------------
    % 1b. Using conjugacy, sample regularization precisions delta,
    % conjugate prior: delta~Gamma(a1,1/t1);
    delsamp(i+1) = gamrnd(a1+Nbar/2,1./(t1+norm(Dreg*xchain(:,i))^2/2));
    %------------------------------------------------------------------
    % 2. Sample the diffusion coefficient using RTO.
    p.lam  = lamsamp(i+1);
    p.del  = delsamp(i+1); 
    % Compute the MAP estimator for lamdba and delta and corresponding Q
    p.Q    = speye(Nresid);   % this is non-identity when we do RTO.
    p.e    = zeros(Nresid,1); % this is iid Gaussian when we do RTO.
    [xMAP,rMAP,JMAP] = LevMar(xMAP,@(x)diffusion_fun(x,p),0.001,1e-8,1000);
    [Q,~]  = qr(JMAP,0);
    % Take one step of RTO-MH.
    p.Q    = Q;
    xtemp  = RTO_MH(xMAP,@(x,p)diffusion_fun(x,p),p,1);
    xchain(:,i+1) = xtemp(:,end);
end
close(h)
% Visualize the MCMC chain
% Plot the sample mean and 95% credibility intervals.
nburnin        = nsamps/5;
xchain         = xchain(:,nburnin:end);
delsamp        = delsamp(nburnin:end);
lamsamp        = lamsamp(nburnin:end);
xlims          = plims(xchain',[0.025,0.5,0.975])';
relative_error = norm(x_true-xlims(:,2))/norm(x_true);
figure(3),
plot(s_mid,x_true,'k',s_mid,xlims(:,2),'--k',s_mid,xlims(:,1),'-.k',s_mid,xlims(:,3),'-.k')
legend('x_{true}','Sample Median','95% credibility bounds','Location','North')
% Output for individual chains using sample_plot: ACF, IACT, Geweke test.
names          = cell(3,1);
names{1}       = '\lambda';
names{2}       = '\delta';
index          = randsample(N,1);  % randomly chosen element of x.
xindex         = xchain(index,:);
names{3}       = char(['x_{',num2str(index),'}']);
[tau,acf]      = sample_plot([lamsamp';delsamp';xindex],names,4);
[~,nacf]       = size(acf);
% Plot autocorrelation function.
figure(7)
    plot(acf(1,1:nacf),'k'), hold on 
    plot(acf(2,1:nacf),'k--'), 
    plot(acf(3,1:nacf),'k-.'), hold off
axis([1,nacf,0,1])
title(['ACFs for \lambda, \delta, and x_{',num2str(index),'}']);
legend(['\lambda: \tau_{\rm int}(\lambda)=',num2str(tau(1))],...
       ['\delta: \tau_{\rm int}(\delta)=',num2str(tau(2))],...
       ['x_{',num2str(index),'}: \tau_{\rm int}(x_{',num2str(index),'})=',num2str(tau(3))])
