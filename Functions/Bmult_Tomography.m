  function y = Bmult_Tomography(x,params);
%
%  Compute array(y), where y = (A'*A+ alpha*I)*x. A is the discrete
%  tomography matrix.
% 
%  written by John Bardsley 2016.
%

  A      = params.A;
  alpha  = params.alpha;
  
  %  Compute A'*(A*x) + alpha*x 
  y      = A'*(A*x) + alpha*x; 
  
  