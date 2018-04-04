function[Qtr,QtJ,r]=positivity_fun(u,params)
% This function evaluates the residual and Jacobain of the residual for
% nonlinear least squares functions of the form ||r(u)||^2, where
% r(u) = [sqrt(lam)*A*exp(u);sqrt(del)*D*u]
% and the Jacobian of r given by
% J(u) = [sqrt(lam)*A*diag(exp(u));sqrt(del)*D]
A   = params.A;
b   = params.b;
lam = params.lam;
del = params.del;
D   = params.D;
e   = params.e;
Q   = params.Q;
% now evaluate the residual and Jacobian
expu   = exp(u);
r      = [sqrt(lam)*(A*expu-b);sqrt(del)*D*u];
Qtr    = Q'*(r-e);
J      = [sqrt(lam)*A*diag(expu);sqrt(del)*D];
QtJ    = Q'*J;
end