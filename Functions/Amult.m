function[Ax]=Amult(x,ahat)
% This function computes multiplication by a matrix A that has block
% circulant with circulant block structure. It makes use of the DFT and
% IDFT. The inputs are the eigenvalues of A, contained in ahat, and the 
% vector x, while the output is the vector A*x.
% 
% written by John Bardsley 2016.
%
Ax = real(ifft2(ahat.*fft2(x)));
