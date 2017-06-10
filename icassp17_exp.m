%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% This is the script to compare our proposed discriminative graphical 
%%% lasso algorithm (Disc-GLasso) with other graph learning algorithm (e.g. 
%%% GLasso) on synthetic data.
%%%
%%% Author: Jiun-Yu Kao 
%%% July 22,2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Construct the graph for each class following 8x8 grid patch
addpath('Util/graph_tools/');
Dx = 8; 
Dy = 8;
[edges,N,coords] = lattice3(Dx,Dy,1);

% Specify the edge weights of each direction on each graph
w1x = 0.9; w1y = 0.9;
w2x = 0.9; w2y = 0.1;

W1 = zeros(N,N); % adjacency matrix for graph 1
for m=1:size(edges,1)/2 % edges in the Y-direction
   W1(edges(m,1),edges(m,2)) = w1y;
   W1(edges(m,2),edges(m,1)) = w1y;
end
for m=(size(edges,1)/2)+1 : size(edges,1) % edges in the X-direction
   W1(edges(m,1),edges(m,2)) = w1x;
   W1(edges(m,2),edges(m,1)) = w1x;
end
W2 = zeros(N,N); % adjacency matrix for graph 2
for m=1:size(edges,1)/2 % edges in the Y-direction
   W2(edges(m,1),edges(m,2)) = w2y;
   W2(edges(m,2),edges(m,1)) = w2y;
end
for m=(size(edges,1)/2)+1 : size(edges,1) % edges in the X-direction
   W2(edges(m,1),edges(m,2)) = w2x;
   W2(edges(m,2),edges(m,1)) = w2x;
end

% Compute the Laplacian matrix for each graph
D1 = diag(sum(W1,2));
D2 = diag(sum(W2,2));
L1 = D1-W1;
L2 = D2-W2;
sigma = 1.0;
K1 = inv(sigma*eye(N)+L1); % covariance matrix based on G1
K2 = inv(sigma*eye(N)+L2); % covariance matrix based on G2

% Generate the iid Gaussian signals based on each covariance matrix
addpath('Util/');
nSamples = 2000;
Sig1 = mvnrnd(zeros(1,N), K1, nSamples);
Sig2 = mvnrnd(zeros(1,N), K2, nSamples);

% Calculate the empirical covariance matrix from signal in each class
S1 = cov(Sig1);
S2 = cov(Sig2);
S_all = cell(1,2);
S_all{1,1} = S1;
S_all{1,2} = S2;

rho = 0.05;
% Generate the test signals based on the same distribution as training
% signals
testSig1 = mvnrnd(zeros(1,N), K1, nSamples);
testSig2 = mvnrnd(zeros(1,N), K2, nSamples);
testS1 = cov(testSig1);
testS2 = cov(testSig2);

% Examine the impact of using different weight for regularizer
valRatio = 10:5:100;
result_ratio_glasso = zeros(1,length(valRatio));
result_ratio_disc = zeros(1,length(valRatio));
result_accu_disc = zeros(1,length(valRatio));
result_accu_glasso = zeros(1,length(valRatio));

for ratioIdx=1:length(valRatio)

% Test discriminative graphical lasso algorithm on estimating the precision
% matrices for each class of signals
[Theta_all W_all bestRatio] = discriGLasso(S_all, rho, 1e2, 1e-6, valRatio(ratioIdx));

% Also apply traditional graphical lasso (independently for each category) as comparison
Theta_glasso = cell(1,2);
W_glasso = cell(1,2);
for cIdx=1:2
   [Theta_glasso{1,cIdx} W_glasso{1,cIdx}] = graphicalLasso(S_all{1,cIdx}, rho);
end

% separation measure calculation
result_ratio_disc(ratioIdx) = (trace(testS1*Theta_all{1,2})+trace(testS2*Theta_all{1,1})) / (trace(testS1*Theta_all{1,1})+trace(testS2*Theta_all{1,2}));
result_ratio_glasso(ratioIdx) = (trace(testS1*Theta_glasso{1,2})+trace(testS2*Theta_glasso{1,1})) / (trace(testS1*Theta_glasso{1,1})+trace(testS2*Theta_glasso{1,2}));

% classification
[V1_disc, D] = eig(Theta_all{1,1});
[V2_disc, D] = eig(Theta_all{1,2});
[V1_glasso, D] = eig(Theta_glasso{1,1});
[V2_glasso, D] = eig(Theta_glasso{1,2});
numBasis = floor(size(V1_disc,2)/2);

testSig = [testSig1 ; testSig2];
testLabel = [ones(size(testSig1,1),1) ; ones(size(testSig2,1),1)*2];

predict_label_disc = zeros(length(testLabel),1);
predict_label_glasso = zeros(length(testLabel),1);

for n=1:length(testLabel)
    proj_disc_g1 = testSig(n,:)*V1_disc;
    proj_disc_g2 = testSig(n,:)*V2_disc;
    if (sum(proj_disc_g1(1:numBasis).^2)/sum(proj_disc_g1.^2))  > (sum(proj_disc_g2(1:numBasis).^2)/sum(proj_disc_g2.^2))
        predict_label_disc(n) = 1;
    else
        predict_label_disc(n) = 2;
    end
    proj_glasso_g1 = testSig(n,:)*V1_glasso;
    proj_glasso_g2 = testSig(n,:)*V2_glasso;
    if (sum(proj_glasso_g1(1:numBasis).^2)/sum(proj_glasso_g1.^2))  > (sum(proj_glasso_g2(1:numBasis).^2)/sum(proj_glasso_g2.^2))
        predict_label_glasso(n) = 1;
    else
        predict_label_glasso(n) = 2;
    end
end

test_accuracy_disc = length(find(predict_label_disc==testLabel))/length(testLabel)*100;
disp(sprintf('test accuracy with disc-glasso = %f %', test_accuracy_disc));
test_accuracy_glasso = length(find(predict_label_glasso==testLabel))/length(testLabel)*100;
disp(sprintf('test accuracy with glasso = %f %', test_accuracy_glasso));

result_accu_disc(ratioIdx) = test_accuracy_disc;
result_accu_glasso(ratioIdx) = test_accuracy_glasso;

end

% Plot the results
figure;
plot(valRatio(1:end),result_ratio_disc(1:end),'r-', 'Linewidth', 2.5);
hold on
plot(valRatio(1:end),result_ratio_glasso(1:end),'b--', 'Linewidth', 2.5);
ylim([1.103 1.14]);
xlabel('r');
ylabel('separation measure s');
legend('Disc-GLasso','GLasso');
figure;
plot(valRatio(1:end),result_accu_disc(1:end),'r-', 'Linewidth', 2.5);
hold on
plot(valRatio(1:end),result_accu_glasso(1:end),'b--', 'Linewidth', 2.5);
xlabel('r');
ylabel('classification accuracy');
legend('Disc-GLasso','GLasso');
