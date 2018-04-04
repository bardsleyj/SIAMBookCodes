  function y_array = Bmult_IGMRF(x_array,params);
%
% Compute array(y), where y = (T'*T + alpha*L)*x. 
% 
% written by John Bardsley 2016.
%

  ahat       = params.ahat;
  alpha      = params.alpha;
  Ds         = params.Ds;
  Dt         = params.Dt;
  Lambda_s   = params.Lambda_s;
  Lambda_t   = params.Lambda_t;

  [n,n]    = size(x_array);
  TtTx_array = Amult(Amult(x_array,ahat),conj(ahat));
  reg_term   = Ds'*(Lambda_s.*(Ds*x_array(:))) + Dt'*(Lambda_t.*(Dt*x_array(:)));
  y_array    = TtTx_array + alpha*reshape(reg_term,n,n);
  
  