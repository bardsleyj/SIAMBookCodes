function[GCValpha] = GCV_DataDriven(alpha,b,c,params,Bmult_fn)
% This function evaluates the GCV function G(alpha) in the deblurring case.
% Inputs:
% alpha    = regularization parameter
% b        = r.h.s./measurement vector
% c        = \lambda*A'*b
% params   = structure array containing items needed to evaluate A
% Bmult_fn = function for evaluating \lambda*A'*A+\delta I.
%
% Outputs:
% GCValpha   = value of GCV(alpha).
% 
% written by John Bardsley 2016.
%
% Extract a_params from params structure and add alpha for use within CG.
a_params               = params.a_params;
ahat                   = a_params.ahat;
a_params.alpha         = alpha;
params.a_params        = a_params;
params.precond_params  = 1./(abs(ahat).^2+alpha);

% Compute xalpha using CG, then compute Axalpha.
[nx,ny]                = size(ahat);
xalpha                 = CG(zeros(nx,ny),c,params,Bmult_fn);
Axalpha                = Amult(xalpha,ahat);
Axalpha                = Axalpha(nx/4+1:3*nx/4,nx/4+1:3*nx/4);

% Randomized trace estimation
v                      = 2*(rand(nx/2,ny/2)>0.5)-1;
Dtv                    = padarray(v,[nx/4,ny/4]);
AtDtv                  = Amult(Dtv,conj(ahat));
Aalphav                = CG(zeros(nx,ny),AtDtv,params,Bmult_fn);
AAalphav               = Amult(Aalphav,ahat);
AAalphav               = AAalphav(nx/4+1:3*nx/4,nx/4+1:3*nx/4);
trAAalpha              = sum(sum(v.*AAalphav));

% Evaluate GCV function.
GCValpha               = nx*ny*norm(Axalpha(:)-b(:))^2./(nx*ny-trAAalpha)^2;