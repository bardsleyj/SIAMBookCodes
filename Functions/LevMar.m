function[x,r,J,iterhist] = LevMar(x,cost_fun,nu0,gradtol,maxiter)
% This m-file implements the Levenburg-Marquardt method described in
% Section 6.2. It is an implementation of Algorithm 3.3.5 in 
% C.T. Kelley, "Iterative Methods for Optimization," SIAM 1999.
%
% b = A(x)+eps  ---->  x_ls = argmin_x {f(x)=||A(x)-b||^2}.
%
% Inputs: x      = initial guess for the optimization.
%         cost_fun = evaluates the residual A(x)-b and its Jacobian J(x).
%         nu0    = default value of nu.
%         gradtol= gradient norm stopping tolerance.
%         maxiter= iteration stopping tolerance.
%
% Outputs: x        = approximate minimizer.
%          r        = residual evaluated at x.
%          J        = Jacobian evaluated at x.
%          iterhist = array containing iteration history information;
[r,J] = feval(cost_fun,x);
g   = J'*r;
ng  = norm(g);
ng0 = ng;
f   = 0.5*r'*r;
nu  = ng;
I   = speye(length(x));
i   = 0;
iterhist = [i ng/ng0 nu];
% parameters required for the Levenburg-Marquardt parameter update
mu0=0; mulow=.25; muhigh=.75; omup=2; omdown=.5;
while ng/ng0 > gradtol & i < maxiter
    i     = i+1;
    s     = (J'*J+nu*speye(length(x)))\g;
    xtemp = x - s;
    [rtemp,Jtemp] = feval(cost_fun,xtemp);
    ftemp = 0.5*rtemp'*rtemp;
    %--------------------------------------------------------------------
    % Levenburg-Marquardt parameter update, Alg. 3.3.4 in Kelley's book. 
    ratio = -2*(f-ftemp)/((xtemp-x)'*g);
    if ratio < mu0
        nu = max(omup*nu,nu0);
    else
        x = xtemp;
        r = rtemp;
        J = Jtemp;
        f = ftemp;
        if ratio < mulow
            nu = max(omup*nu,nu0);
        elseif ratio > muhigh
            nu = omdown*nu;
            if nu < nu0
                nu = 0;
            end
        end
    end
    %--------------------------------------------------------------------
    g   = J'*r;
    ng  = norm(g);
    iterhist = [iterhist; [i ng/ng0 nu]];
    %fprintf('i=%d ||grad||/||grad0||=%2.3e ratio=%2.3e nu=%2.3e\n',i,ng,ratio,nu);
end
    