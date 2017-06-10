function os = itermsg(itermeth,tol,~,i,flag,iter,relres)
%ITERMSG   Displays the final message for iterative methods.
%   ITERMSG(ITERMETH,TOL,MAXIT,I,FLAG,ITER,RELRES)
%
%   See also BICG, BICGSTAB, BICGSTABL, CGS, GMRES, LSQR, MINRES, PCG, QMR,
%   SYMMLQ, TFQMR.

%   Copyright 1984-2010 The MathWorks, Inc. 
%   $Revision: 1.7.4.2 $ $Date: 2010/04/21 21:33:12 $

if flag == 0
    if iter == 0
        if isnan(relres)
            os = sprintf(['The right hand side vector is all zero so %s\n', ...
                'returned an all zero solution without iterating.'], itermeth);
        else
            os = sprintf(['The initial guess has relative residual %0.2g ', ...
                'which is within\nthe desired tolerance %0.2g so %s ', ...
                'returned it without iterating.'], relres, tol, itermeth);
        end
    else
        os = sprintf(['%s converged at %s to a solution with relative ', ...
            'residual %0.2g.'], itermeth, getIterationInfo(iter, true), ...
            relres);
    end
else
    switch flag
        case 1,
            ncnv = sprintf(['%s stopped at %s without converging to the ', ...
                'desired tolerance %0.2g\nbecause the maximum number ', ...
                'of iterations was reached.'], ...
                itermeth, getIterationInfo(i, true), tol);
        case 2,
            ncnv = sprintf(['%s stopped at %s without converging to the ', ...
                'desired tolerance %0.2g\nbecause the system ', ...
                'involving the preconditioner was ill conditioned.'], ...
                itermeth, getIterationInfo(i, true), tol);
        case 3,
            ncnv = sprintf(['%s stopped at %s without converging to the ', ...
                'desired tolerance %0.2g\nbecause the method stagnated.'], ...
                itermeth, getIterationInfo(i, true), tol);
        case 4,
            ncnv = sprintf(['%s stopped at %s without converging to the ', ...
                'desired tolerance %0.2g\nbecause a scalar quantity ', ...
                'became too small or too large to continue computing.'], ...
                itermeth, getIterationInfo(i, true), tol);
        case 5,
            ncnv = sprintf(['%s stopped at %s without converging to the ', ...
                'desired tolerance %0.2g\nbecause the preconditioner is ', ...
                'not symmetric positive definite.'], ...
                itermeth, getIterationInfo(i, true), tol);
    end
    retStr = sprintf('The iterate returned %s has relative residual %0.2g.', ...
        getIterationInfo(iter, false), relres);
    os = sprintf('%s\n%s', ncnv, retStr);
end
disp(os)

function itstr = getIterationInfo(it, verbose)
if length(it) == 2 % gmres
    if verbose
        itstr = sprintf('outer iteration %d (inner iteration %d)', ...
            it(1), it(2));
    else
        itstr = sprintf('(number %d(%d))', it(1), it(2));
    end
elseif fix(it) ~= it % bicgstab
    if verbose
        itstr = sprintf('iteration %.1f', it);
    else
        itstr = sprintf('(number %.1f)', it);
    end
else
    if verbose
        itstr = sprintf('iteration %d', it);
    else
        itstr = sprintf('(number %d)', it);
    end
end

