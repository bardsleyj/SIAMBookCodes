function[GCValpha] = GCV_fn(alpha,b,c,params,Bmult_fn)
% This function evaluates the GCV function G(alpha) in the deblurring ex.
% Inputs:
% alpha    = regularization parameter
% b        = r.h.s./measurement vector
% c        = \lambda*A'*b
% params   = structure array containing items needed to evaluate A
% Bmult_fn = function for evaluating \lambda*A'*A+\delta L.
%
% Outputs:
% GCValpha   = value of GCV(alpha).
% 
% written by John Bardsley 2016.
%
% extract a_params from params structure and add alpha for use within CG
[n,n]                  = size(b);
a_params               = params.a_params;
ahat                   = a_params.ahat;
lhat                   = a_params.lhat;
a_params.alpha         = alpha;
params.a_params        = a_params;

% compute xalpha using CG, then compute Axalpha
params.precond_params  = 1./(abs(ahat).^2+alpha*lhat);
xalpha                 = CG(zeros(n,n),c,params,Bmult_fn);
Axalpha                = Amult(xalpha,ahat);

% Randomized trace estimation
v                      = 2*(rand(n,n)>0.5)-1;
Atv                    = Amult(v,conj(ahat));
Aalphav                = CG(zeros(n,n),Atv,params,Bmult_fn);
trAAalpha              = sum(sum(v.*Amult(Aalphav,ahat)));

% Evaluate GCV function.
GCValpha               = norm(Axalpha(:)-b(:))^2./(n^2-trAAalpha)^2;