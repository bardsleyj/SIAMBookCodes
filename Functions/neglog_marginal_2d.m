function[neglog_p,fourier_filt] = neglog_marginal_1d(x,params)
%
% This function evaluates the negative-log of the marginal density 
% p(lam,del|b) from Chapter 5 of the book. It has the form
% 
% -ln p(lam,del|b) = c(lam,del)+U(lam,del)-(m/2+a0-1)*log(lam)+t0*lam
%                                         -(n/2+a1-1)*log(del)+t1*del;
%
% where c(lam,del) and U(lam,del) are as defined in Chapter 5.
%
% INPUTS:
%           x = (lam,del)
%      params = structure area containing everything required to evaluate 
%               the marginal density (see below).
% OUTPUTS:
%    neglog_p = the value of the negative-log of the marginal x.
%           R = the Cholesky factor corresponding to lam*A'*A+del*L.
%
ahat = params.ahat;
lhat = params.lhat;
bhat = params.bhat;
btb  = params.btb;
M   = params.M;
N   = params.N;
a0  = params.a0;
t0  = params.t0;
a1  = params.a1;
t1  = params.t1;
lam = x(1);
del = x(2);
% evaluate the marginal density at x=[lam;del].
fourier_filt = lam*abs(ahat).^2 + del*lhat;
c_lam_del    = 0.5*sum(log(fourier_filt(:)));
U_lam_del    = 0.5*lam*(btb-lam*sum(abs(ahat(:)).^2.*(abs(bhat(:)).^2/N)./fourier_filt(:)));
neglog_p     = c_lam_del+U_lam_del-(M/2+a0-1)*log(lam)+t0*lam...  
                    -((N-1)/2+a1-1)*log(del)+t1*del;
