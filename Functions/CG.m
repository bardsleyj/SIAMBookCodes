function [x,iter_hist,term_code] = CG(x0,b,params,a_mult);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Use conjugate gradient iteration to minimize the quadratic functional
%       J(x) = - b'*x + 0.5*x'*A*x, 
%  where b is an n-vector and A is an n X n symmetric positive
%  definite (SPD) matrix. Equivalently, solve the linear system 
%       A*x = b.
%
%  Input Variables:
%    x0       --- vector containing the initial guess.
%    b        --- vector containing right hand side of system A*x = b.
%    a_mult   --- This is either a matrix containing A, or a text string
%                 containing the name of the MATLAB function which 
%                 implements multiplication by A. In the later case,
%                 to compute y=A*x, call y = feval(a_mult,x,params). 
%                 A must be SPD.
%    precond  --- This is either a matrix containing the preconditioner
%                 M, or a text string containing the name of the MATLAB
%                 function which implements the preconditioner M. In
%                 the latter case, to solve M*x=y, call 
%                 x = feval(precond,y,params). M must be SPD.
%    params   --- MATLAB structure array containing CG parameters
%       and information used by a_mult and m_invert.
%    params.max_cg_iter      Maximimum number of CG iterations.
%    params.cg_step_tol      Stop CG when ||x_k+1 - x_k|| < step_tol.
%    params.cg_io_flag       Output CG info if ioflag = 1.
%    params.cg_figure_no     Figure number for CG output.
%
%  Output Variables:
%    x         --- Approximate solution obtained by CG.
%    iter_hist  --- Array containing iteration history.
%     iter_hist(k,1) = CG iteration number.
%     iter_hist(k,2) = gradient norm at current CG iteration.
%     iter_hist(k,3) = norm of current CG step, ||x_current - x_previous||.
%     iter_hist(k,4) = norm of iterative solution error, ||x_current - x_inf||,
%                      where x_inf is the solution to A*x = -b. This
%                      is computed only if the field params.x_inf exist.
%    term_code --- CG termination code.
%     term_code = 1    Maximum CG iterations reached.
%     term_code = 2    CG step norm stopping tolerance reached.
%     term_code = 3    CG gradient norm stopping tolerance reached.
%     term_code = 4    Negative curvature detected. This suggests that
%                      either A or M is not SPD.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  a_params        = params.a_params;
  precond         = params.precond;
  max_cg_iter     = params.max_cg_iter;
  step_tol        = params.cg_step_tol;
  grad_tol        = params.grad_tol;
  io_flag         = params.cg_io_flag;
  cg_fig          = params.cg_figure_no;
  if isfield(params,'x_inf')
    e_flag = 1;            %  Indicate that params.x_inf exists.
    x_inf = params.x_inf;
  else
    e_flag = 0;            %  Indicate that params.x_inf does not exist.
  end
  sqrt_n = sqrt(length(b(:)));
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  CG initialization.                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if ~isempty(precond)
    precond_params = params.precond_params;
    precond_flag = 1;   %  Use preconditioning.
  else
    precond_flag = 0;   %  No preconditioning.
  end

  x = x0;
  if isnumeric(a_mult)
    Ax = a_mult * x;
  else
    Ax = feval(a_mult,x,a_params);
  end
  g = Ax - b;                       %  Initial gradient g = A*x0 - b.
  ng0 = norm(g(:));
  if precond_flag
    if isnumeric(precond)           %  Apply preconditioner; solve M*z = g.
      z = precond \ g;
    else
      z = feval(precond,g,precond_params);
    end
  else
    z = g;                          %  No preconditioning.
  end
  d = -z;                           %  Initial search direction.
  D = -d(:);
  if isnumeric(a_mult)            %  Compute A*d.
    Ad = a_mult * d;
  else
    Ad = feval(a_mult,d,a_params);
  end
  AD = -Ad(:);
  delta = g(:)'*z(:);               %  delta = g' * M^{-1} * g.
  stepnormvec = [];
  gnorm = ng0;
  term_code = 0;
  cg_iter = 0;
  if e_flag
    enorm = norm(x(:) - x_inf(:)) / norm(x_inf(:));
    iter_hist = [cg_iter gnorm 0 enorm];
  else
    iter_hist = [cg_iter gnorm 0];
  end
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  CG iteration.                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  while term_code == 0
    cg_iter = cg_iter + 1;
    
    dAd = d(:)'*Ad(:);              %  Compute d'*A*d.
    
    if dAd <= 0
      term_code = 1;                %  Negative curvature detected.
      fprintf('***** Negative curvature detected in CG.\n');
      x = [];
      return
    end

    tau = delta / dAd;              %  Line search parameter.
    x = x + tau*d;                  %  Update approximate solution.
    g = g + tau*Ad;                 %  Update gradient g = b + A*x.
    if precond_flag
      if isnumeric(precond);        %  Apply preconditioner; solve M*z = g.
        z = precond \ g;
      else
        z = feval(precond,g,precond_params);
      end
    else
      z = g;                        %  No preconditioning.
    end
    delta_new = g(:)'*z(:);         %  delta = g' * M^{-1} * g.
    my_beta = delta_new / delta;    %  Note: beta is a MATLAB function.
    d = -z + my_beta*d;             %  Update CG search direction.
    if isnumeric(a_mult)            %  Compute A*d.
      Ad = a_mult * d;
    else
      Ad = feval(a_mult,d,a_params);
    end
    
    delta = delta_new;

    %  Compute and store CG convergence information.
    
    snorm = abs(tau)*norm(d(:));
    gnorm = norm(g(:));%sqrt(delta);
    stepnormvec = [stepnormvec; snorm];
    if e_flag
      enorm = norm(x(:)-x_inf(:)) / norm(x_inf(:));
      iter_hist = [iter_hist; [cg_iter gnorm snorm enorm]];
    else
      iter_hist = [iter_hist; [cg_iter gnorm snorm]];
    end
    
    %  Plot CG convergence information.
    
    if ~isempty(cg_fig)
      figure(cg_fig)
      
      %  Plot norm of gradient g = A*x + b vs CG iteration count.
      
      iter_vec = iter_hist(:,1);
      gradnormvec = iter_hist(:,2);
      subplot(221)
      if grad_tol > 0
        gconst = grad_tol * ones(size(iter_vec));
        semilogy(iter_vec,gradnormvec,'o-', iter_vec,gconst,'-')
      else
    	semilogy(iter_vec,gradnormvec,'o-')
      end
      title('CG Gradient Norm')
      xlabel('CG iterate')
      
      %  Plot norm of step x - x_last vs CG iteration count.
      
      subplot(222)
      stepnormvec = iter_hist(:,3);
      ns = max(size(stepnormvec));
      stepnormvec = stepnormvec(2:ns);
      iter_vec = iter_hist(2:ns,1);
      if step_tol > 0
        sconst = step_tol * ones(size(iter_vec));
        semilogy(iter_vec,stepnormvec,'o-', iter_vec,sconst,'-')
      else
        semilogy(iter_vec,stepnormvec,'o-')
      end
      title('Norm of CG Step')
      xlabel('CG iterate')

      if e_flag
	
	%  Plot norm of iterative solution error, x - x_inf, vs CG 
	%  iteration count.

        iter_vec = iter_hist(:,1);
        enormvec = iter_hist(:,4);
        subplot(223)
    	semilogy(iter_vec,enormvec,'o-')
        title('Iterative Soln Error Norm')
        xlabel('CG iterate')
      end
      
    end

    %  Check stopping criteria.
    
    if cg_iter >= max_cg_iter
      term_code = 1;
      %fprintf('***** Max CG iterations exceeded.\n');
      return
    elseif snorm <= step_tol
      term_code = 2;
      %fprintf('***** Step norm stopping tol reached in CG at iter %4.0f.\n',...
      %  cg_iter);
      return
    elseif gnorm/ng0 <= grad_tol
      term_code = 3;
      %fprintf('***** Gradient norm stopping tol reached in CG at iter %4.0f.\n',cg_iter);
      return
    end
    
  end %(while term_code == 0)
