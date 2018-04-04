function[Qtr,QtJ,r]=l1_fun(u,params)
% This function evaluates the residual and Jacobain of the residual for
% nonlinear least squares functions of the form ||r(u)||^2, where
% r(u) = [(K*D'g(u,del)-d)/sigma;u]
% and the Jacobian of r given by
% J(u) = [K*D'*Jg(u,del)/sigma;I]
A   = params.A;
b   = params.b;
lam = params.lam;
del = params.del;
D   = params.D;
e   = params.e;
Q   = params.Q;
% now evaluate the residual and Jacobian
[g,Jg] = g_fn(u,del);
Dtg    = D'*g;
r      = [sqrt(lam)*(A*Dtg-b);u];
Qtr    = Q'*(r-e);
J      = [sqrt(lam)*A*(D'*diag(Jg));speye(length(u))];
QtJ    = Q'*J;
end