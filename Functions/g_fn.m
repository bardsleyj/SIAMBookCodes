function[g,gprime]=g_fn(u,del)
% This function evaluation the transformation function g and its derivative
%
% g(u) = -(1/del)*sign(normcdg(u)-0.5).*log(1-2*abs(normcdf(u)-0.5))
% and
% g'(u) = {normpdf(u)./(del*normcdf(u)), u<0
%         {normpdf(u)./(del*(1-normcdf(u))), u>=0
%
%g = -(1/del)*sign(normcdf(u)-0.5).*log(1-2*abs(normcdf(u)-0.5));
%if nargout==2
%    gprime = (u>=0).*normpdf(u)./(del*(1-normcdf(u))) + ...
%             (u<0).* normpdf(u)./(del*normcdf(u));
%end
g      = -1/del*sign(u).*log(1-2*abs(normcdf(u)-0.5));
gprime =  normpdf(u)./(del*normcdf(-abs(u)));
end