function[Qtr,QtJ,r]=twoparam_fun(x,params)
% This function evaluates the least squares residual and Jacobian
% r(x) = sqrt(lambda)*Q'*(A(x)-(b+e)) and 
% J(x) = sqrt(lambda)*Q'*Jacobian of A.
A     = params.A;
b     = params.b;
e     = params.e;
Q     = params.Q;
lam   = params.lam;
t     = params.t;
% Now compute the residual r, Jacobian J, Q'*r and Q'*J.
r      = sqrt(lam)*(A(x)-b);
Qtr    = Q'*(r-e);
J      = sqrt(lam)*[1-exp(-x(2)*t), x(1)*t.*exp(-x(2)*t)];
QtJ    = Q'*J;

end % end of function