% This collection of codes accompanies the SIAM Book:
%
% "Computational Uncertainty Quantification for Inverse Problems"
%
% by Johnathan M. Bardsley
%
% The codes come with no guarantees of any kind.
%
% The figures in the book were generated using these codes,
% though when the codes are run, most results will be slightly 
% different than those in the book, due to different random vector 
% realizations.
%
% The driver codes for specific chapters are found within the 
% respective directories: Chapter 1, Chapter 2, etc. To run a
% driver code, open MATLAB and change directories to the 
% chapter directory of interest, type the name of the file that
% you want to run on the command line and hit return. 
%
% For example, in the Chapter 1 directory, you could type
%
% >> Deblur1d 
%
% and then hit enter. You will several of the figures corresponding
% to the deblurring example in Chapter 1. 
%
% The MatFiles directory contains the true images used in the 
% two-dimensional test cases: satellite.mat, cell.mat, and 
% SheppLogan.mat
%
% The Functions directory contains additional functions needed to 
% run the driver codes, e.g., preconditioned conjugate gradient 
% (CG.m), adaptive Metropolis (AM.m), and the Geweke test (geweke.m),
% to name a few.  