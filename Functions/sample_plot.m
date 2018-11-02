function[taux,acf_array]=sample_plot(x,names,k)
% This m-file plots samples parameters collected in the array b. b_opt is
% the optimal parameter vectors computed using an optimization method.
% Inputs: x = MCMC chains of parameters.
%     names = string of names of the parameters.
%         k = beginning figure number
%
% Outputs: taux = integrated autocorrelation times for parameter subset.
%     acf_array = autocorrelation function for parameter subset.
%
% First, extract the chains for the subset of parameters in index, then be
% sure that the array x has the correct dimensions.
[nparams,nsamp] = size(x);
if nparams>=nsamp
    disp('Array should have dimensionis: # parameters by # samples.')
    return
end
if nparams >= 10
    disp('Choose a smaller subset (<10) of parameters for the plots.')
    return
end

% Compute sample median and 95% credibility intervals for each of the
% parameters. xlims has dimension 3 by nparams.
xlims = plims(x',[0.025,0.5,0.975]);

% Output chain stats and plot autocorrelation functions
acf_array = [];
taux      = zeros(nparams,1);
px        = zeros(nparams,1);
for i = 1:nparams
    [~,px(i)] = geweke(x(i,:)');
    acfxi     = acf(x(i,:)');
    taux(i)   = iact(x(i,:)');
    acf_array = [acf_array; acfxi];
    fprintf('%s chain stats: Geweke p-value = %2.5f, IACT = %2.5f, ESS = %2.5f, 95%% c.i. [%2.5f, %2.5f].\n',names{i},px(i),taux(i),nsamp/taux(i),xlims(1,i),xlims(3,i))
end

% Individual chain plots.
figure(k) 
for i=1:nparams
    subplot(nparams,1,i),plot([1:length(x(i,:))],x(i,:),'k-')
    title([names{i}])%,':  IACT=',num2str(taux(i)),'  p_{Geweke}=',num2str(px(i))])
end

% Individual histograms.
figure(k+1) 
for i=1:nparams
    subplot(nparams,1,i),histogram(x(i,:),20), colormap(1-gray)
    title([names{i}])%,'\tau_{int}=',num2str(taux(i)),' and  p_{Geweke}=',num2str(px(i)))
end

% Pairwise point plots.
figure(k+2)
for i = 1:nparams-1
  for l = (i-1)*nparams+1:i*(nparams-1)
    subplot(nparams-1,nparams-1,l)
    j = l-(i-1)*(nparams-1)+1;
    plot(x(j,:),x(i,:),'k.')
    xlabel([names{j}])
    ylabel([names{i}])
  end
end