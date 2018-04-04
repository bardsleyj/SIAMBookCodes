%
% Simple linear regression, Example 1.1, Chapter 1.
% 
% written by John Bardsley 2016.
%
% Measured weights of lions
b = [420,350,310,280,75]';
% Measured lengths of lions.
ell = [2.4,2.0,2.1,1.8,1.3]';
% Create the design matrix A
A = [ones(size(ell)),ell];
% Finally, compute the least squares solution and plot the best fit line
% together with the data.
xhat = A\b;   % this computes the least squares solution
figure(1)
plot(ell,b,'ok',ell,xhat(1)+xhat(2)*ell,'-k','LineWidth',2)
xlabel('length in meters'), ylabel('weight in pounds')
