%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Discriminative Graphical Lasso main function
%%%
%%% Author: Jiun-Yu Kao 
%%% July 22,2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Theta_all W_all bestRatio] = discriGLasso(S_all, rho, maxIt, tol, bestRatio)

%%% Solve for the discriminative graphical lasso for a few categories of signals
%
% For each i in C,
% minimize_{Theta_i > 0} -logdet(Theta_i) + tr(S_i*Theta_i) - 1/C sum_{j~=i}^C tr(S_j*Theta_i) + rho*||Theta||_1

%%% Input:
% S_all : an array for C sample covariance matrices, C is the number of categories 
%     {S_1, ..., S_C}
% rho : regularization parameter
% maxIt : maximum number of iterations
% tol : convergence tolerance level
% bestRatio : parameter r defined in the paper as weight of regularizer 
%             (If not specified, line search will be performed to get the minimum ratio r satisfying initial condition.)
%
%%% Output:
% Theta_all : an array for inverse covariance matrices estimate, {Theta_1, ..., Theta_C}
% W_all : an array for regularized covariance matrices estimate, {W_1, ... W_C}, where W_i = Theta_i^-1


C = size(S_all,2);      % number of categories
p = size(S_all{1,1},1); % number of variables 

if nargin < 4, tol = 1e-6; end
if nargin < 3, maxIt = 1e2; end

% Initialization
W_all = cell(1,C);
Theta_all = cell(1,C);

sumAllCovMat = zeros(p,p);
for cIdx = 1:C
   sumAllCovMat = sumAllCovMat + S_all{1,cIdx};
end

% Line searching for the minimum ratio r that makes (Si-sum{Sj}/r) still positive
% semi-definite
if nargin < 5
   valRatio = 1:0.1:5;
   valRatio = 10.^valRatio;
   for rIdx = 1:length(valRatio)
      cCount = 0;
      for cIdx = 1:C
         [V1,D1] = eig(S_all{1,cIdx} - (sumAllCovMat-S_all{1,cIdx})/valRatio(rIdx));
         if D1(1,1)<((-1)*eps), break; 
         else cCount=cCount+1; end
      end
      if cCount==C
         bestRatio = valRatio(rIdx);
         break;
      end
   end
end
   
for cIdx = 1:C
   S = S_all{1,cIdx};
   % Calculate the critical ratio from the max eigenvalues of S and sumOtherCov 
   W = S + rho * eye(p) - ((sumAllCovMat-S)/bestRatio); % diagonal of W remains unchanged
   [V1,D1] = eig(S - ((sumAllCovMat-S)/bestRatio));
   assert(D1(1,1)>=((-1)*eps), 'S_i-sum{S_j} muse be positive semi-definite.');
   W_old = W;
   i = 0;

% Graphical Lasso loop
while i < maxIt,
    i = i+1;
    for j = p:-1:1,
        jminus = setdiff(1:p,j); % W(jminus,jminus):W11
        [V D] = eig(W(jminus,jminus));
        d = diag(D);
        X = V * diag(sqrt(d)) * V'; % X = W_11^(1/2)
        % compute \sum_{j~=i} s_12^j
        sumOtherCov = zeros(p-1,1);
        for subcIdx = setdiff(1:C,cIdx)
           S_other = S_all{1,subcIdx};
           sumOtherCov = sumOtherCov + S_other(jminus,j);
        end
        Y = V * diag(1./sqrt(d)) * V' * ( S(jminus,j) - (sumOtherCov/bestRatio));    % Y = W_11^(-1/2) * [ s_12_i - 1/r*\sum_{j~=i} s_12^j] 
        b = lassoShooting(X, Y, rho, maxIt, tol);
        W(jminus,j) = W(jminus,jminus) * b; % W_12^i = W_11^i * b
        W(j,jminus) = W(jminus,j)';
    end
    % Stop criterion
    if norm(W-W_old,1) < tol, 
        break; 
    end
    W_old = W;
end

if i == maxIt,
    fprintf('%s\n', 'Maximum number of iteration reached, glasso may not converge.');
end

W_all{1,cIdx} = W;
Theta_all{1,cIdx} = W^-1;

end

end

% Shooting algorithm for Lasso (unstandardized version)
function b = lassoShooting(X, Y, lambda, maxIt, tol)

if nargin < 4, tol = 1e-6; end
if nargin < 3, maxIt = 1e2; end

% Initialization
[n,p] = size(X);
if p > n,
    b = zeros(p,1); % From the null model, if p > n
else
    b = X \ Y;  % From the OLS estimate, if p <= n
end
b_old = b;
i = 0;

% Precompute X'X and X'Y
XTX = X'*X;
XTY = X'*Y;

% Shooting loop
while i < maxIt,
    i = i+1;
    for j = 1:p,
        jminus = setdiff(1:p,j);
        S0 = XTX(j,jminus)*b(jminus) - XTY(j);  % S0 = X(:,j)'*(X(:,jminus)*b(jminus)-Y)
        if S0 > lambda,
            b(j) = (lambda-S0) / norm(X(:,j),2)^2;
        elseif S0 < -lambda,
            b(j) = -(lambda+S0) / norm(X(:,j),2)^2;
        else
            b(j) = 0;
        end
    end
    delta = norm(b-b_old,1);    % Norm change during successive iterations
    if delta < tol, break; end
    b_old = b;
end
if i == maxIt,
    fprintf('%s\n', 'Maximum number of iteration reached, shooting may not converge.');
end

end