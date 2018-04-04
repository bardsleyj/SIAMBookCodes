%  First set up finite difference discretization of 1-d steady-state 
%  diffusion equation,
%    -d/ds(x(s) du/ds) = f(s),  0 < s < 1,  l = 1,2,
%    u(0) = u_L; u(1) = u_R.
%  We collect data correpsonding to two forcing functions:
%       f(x) = delta(x-1/3)  and f(x) = delta(x-2/3).
%  Then we solve for u and generate data [b_1;b_2] corresponding to the 
%  two forcing functions. 
%
%  written by John Bardsley 2016.
%
%  This code was used to generate figures in Chapter 6. 
clear all, close all
path(path,'../Functions');
%  Set up numerical mesh and define the 'true' diffusion coefficient.
N        = 64; %%% input(' No. of interior FEM nodes = ');
h        = 1 / N;
s        = [h:h:1-h]';     % numerical mesh for u
s_mid    = [h/2:h:1-h/2]'; % numerical mesh for x
x_true   = min(1,1-.5*sin(2*pi*(s_mid-.25))); 

%  Compute right-hand side vectors f_1 and f_2. Incorporate non-homogeneous
%  boundary conditions if necessary.
f                 = zeros(N-1,2);
f(floor(N/3),1)   = 1;   
f(floor(2*N/3),2) = 1;   
f                 = 1e3*f;

%  Compute stiffness matrix B(x) and observation matrix C.
onevec   = ones(N,1);
D        = (1/h)*spdiags([-onevec onevec],[-1 0],N,N-1);
B        = @(x) D'*(spdiags(x,0,N,N)*D); % Stiffness matrix.
Bprime   = @(i,v) D(i,:)'*(D(i,:)*v);
C        = speye(N-1); % Observation operator.
A        = @(x) C*(B(x)\f); % Forward Operator.

% Generate measurements.
model         = A(x_true);
M             = length(model(:));
percent_error = 1; % Noise level percentage
sig           = percent_error / 100 * norm(model)  / sqrt(M);
noise         = sig * randn(size(model));
b             = model + noise;

%% Next, estimate x from the measurments using Levenburg-Marquardt
% First define the regularization matrix L.
Dreg     = (1/h)*spdiags([-onevec onevec],[0 1],N-1,N);
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
% Compute the MAP estimator using Levenburg-Marquardt optimization.
x0       = ones(size(x_true)); % initial guess
[xMAP,rMAP,JMAP,hist] = LevMar(x0,@(x)diffusion_fun(x,p),0.001,1e-8,1000);

% Plot the measurements and model fit 
figure(1)
plot(s,b,'ko',s,A(x_true),'k-',s,A(xMAP),'k--')
xlabel('s'), title('Plot of data (o), true model (-), and model fit (--)')
figure(2)
plot(s_mid,x_true,'k-',s_mid,xMAP,'k--')
xlabel('s'), ylabel('x'), title('Plot of true (-) and estimated (--) diffusion coefficient')

%% Now, use RTO-MH to sample from the posterior distribution defined by
%              
%            p(x|b) \propto exp(-lam/2*||A(x)-b||^2-del/2*||D*x||^2).
%
nsamp    = 4000;
[Q,~]    = qr(JMAP,0);
p.Q      = Q;
p.Nrand  = Nresid;
[xchain,accept_rate] = RTO_MH(xMAP,@(x,p)diffusion_fun(x,p),p,nsamp);

% Visualize the MCMC chain
% Plot the sample mean and 95% credibility intervals.
xlims          = plims(xchain',[0.025,0.5,0.975]);
relative_error = norm(x_true-xlims(2,:))/norm(x_true);
figure(3),
plot(s_mid,x_true,'k',s_mid,xMAP,'--k',s_mid,xlims(1,:),'-.k',s_mid,xlims(3,:),'-.k')
legend('x_{true}','x_{MAP}','95% credibility bounds','Location','North')
% Output for individual chains using sample_plot: ACF, IACT, Geweke test.
rng('shuffle')
index          = sort(randsample(N,3));
xchain_index   = xchain(index,:);
names(1)       = string(['x_{',num2str(index(1)),'}']);
names(2)       = string(['x_{',num2str(index(2)),'}']);
names(3)       = string(['x_{',num2str(index(3)),'}']);
[tau,acf]   = sample_plot(xchain_index,names',4);
[~,nacf]     = size(acf);
% Plot autocorrelation function.
figure(7)
    plot(acf(1,1:nacf),'k'), hold on 
    plot(acf(2,1:nacf),'k--'), 
    plot(acf(3,1:nacf),'k-.'), hold off
axis([1,nacf,0,1])
title(['ACFs for x_{',num2str(index(1)),'}, x_{',num2str(index(2)),'}, and x_{',num2str(index(3)),'}']);
legend(['x_{',num2str(index(1)),'}: \tau_{\rm int}(x_{',num2str(index(1)),'})=',num2str(tau(1))],...
       ['x_{',num2str(index(2)),'}: \tau_{\rm int}(x_{',num2str(index(2)),'})=',num2str(tau(2))],...
       ['x_{',num2str(index(3)),'}: \tau_{\rm int}(x_{',num2str(index(3)),'})=',num2str(tau(3))])



