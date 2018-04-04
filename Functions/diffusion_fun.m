function[Qtr,QtJ,r]=diffusion_fun(x,params)
% This function evaluates the least squares residual and Jacobian
% r(x) = Q'*(A(x)-b) and J(x)=Jacobian of A(x).
C     = params.C;
B     = params.B;
Bprime = params.Bprime;
f     = params.f;
b     = params.b;
e     = params.e;
Q     = params.Q;
lam   = params.lam;
del   = params.del;
Lsq   = params.Lsq;

% Now compute the residual and Jacobian.
[L,U] = lu(B(x));
u     = U\(L\f);
Cu    = C*u;
r     = [sqrt(lam)*(Cu(:)-b(:));sqrt(del)*Lsq*x];
Qtr   = Q'*(r-e);
M     = length(b);
N     = length(x);
J     = zeros(length(r),N);
for i=1:N
    temp       = -sqrt(lam)*C*(U\(L\(Bprime(i,u))));
    J(1:2*M,i) = temp(:);
end
J(2*M+1:end,:) = sqrt(del)*Lsq;
QtJ = Q'*J;

end % end of function