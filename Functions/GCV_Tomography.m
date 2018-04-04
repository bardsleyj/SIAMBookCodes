function[GCValpha] = GCV_Tomography(alpha,b,c,params,Bmult_fn)
% This function evaluates the GCV function G(alpha) in the tomography ex.
% Inputs:
% alpha    = regularization parameter
% b        = r.h.s./measurement vector
% c        = \lambda*A'*b
% params   = structure array containing items needed to evaluate A
% Bmult_fn = function for evaluating \lambda*A'*A+\delta I.
%
% Outputs:
% GCValpha   = value of G(alpha).
% 
% written by John Bardsley 2016.
%
% extract a_params from params structure and add alpha for use within CG
a_params               = params.a_params;
a_params.alpha         = alpha;
params.a_params        = a_params;

% compute xalpha using CG, then compute Axalpha
A                      = a_params.A;
[M,N]                  = size(A);
xalpha                 = CG(zeros(N,1),c,params,Bmult_fn);
Axalpha                = A*xalpha;

% Randomized trace estimation
v                      = 2*(rand(M,1)>0.5)-1;
Aalphav                = CG(zeros(N,1),A'*v,params,Bmult_fn);
trAAalpha              = v'*(A*Aalphav);

% Evaluate GCV function.
GCValpha               = norm(Axalpha-b)^2./(M-trAAalpha)^2;