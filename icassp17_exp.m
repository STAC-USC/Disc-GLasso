% Construct the graph for each class

addpath('../Util/graph_tools/');

% every signal is an 8x8 patch
Dx = 8; 
Dy = 8;
[edges,N,coords] = lattice3(Dx,Dy,1);
% specify the edge weights of each direction of each graph
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

% compute the Laplacian matrices for each graph
D1 = diag(sum(W1,2));
D2 = diag(sum(W2,2));
L1 = D1-W1;
L2 = D2-W2;
sigma = 1.0;
K1 = inv(sigma*eye(N)+L1); % inverse precision matrix based on G1
K2 = inv(sigma*eye(N)+L2); % inverse precision matrix based on G2

% Generate the iid Gaussian signals based on each precision matrix
addpath('../Util/');
nSamples = 2000;
Sig1 = mvnrnd(zeros(1,N), K1, nSamples);
Sig2 = mvnrnd(zeros(1,N), K2, nSamples);

% Calculate the empirical covariance matrix from signal in each class
S1 = cov(Sig1);
S2 = cov(Sig2);
S_all = cell(1,2);
S_all{1,1} = S1;
S_all{1,2} = S2;
% test discriminative graphical lasso algorithm on estimating the precision
% matrices for each class of signals
rho = 0.05;

% code for examinging the impact of using different ratio 
testSig1 = mvnrnd(zeros(1,N), K1, nSamples);
testSig2 = mvnrnd(zeros(1,N), K2, nSamples);
testS1 = cov(testSig1);
testS2 = cov(testSig2);
valRatio = 5:5:100;
result_ratio_glasso = zeros(1,length(valRatio));
result_ratio_disc = zeros(1,length(valRatio));
result_accu_disc = zeros(1,length(valRatio));
result_accu_glasso = zeros(1,length(valRatio));

for ratioIdx=1:length(valRatio)

[Theta_all W_all bestRatio] = discriGLasso(S_all, rho, 1e2, 1e-6, valRatio(ratioIdx));
% also use traditional graphical lasso (respectively for each category) for comparison
Theta_glasso = cell(1,2);
W_glasso = cell(1,2);
for cIdx=1:2
   [Theta_glasso{1,cIdx} W_glasso{1,cIdx}] = graphicalLasso(S_all{1,cIdx}, rho);
end

% save(sprintf('tempdata/Theta_for_r_%d', ratioIdx), 'Theta_all', 'Theta_glasso', 'W_all', 'W_glasso');

% separation measure calculation
result_ratio_disc(ratioIdx) = (trace(testS1*Theta_all{1,2})+trace(testS2*Theta_all{1,1})) / (trace(testS1*Theta_all{1,1})+trace(testS2*Theta_all{1,2}));
result_ratio_glasso(ratioIdx) = (trace(testS1*Theta_glasso{1,2})+trace(testS2*Theta_glasso{1,1})) / (trace(testS1*Theta_glasso{1,1})+trace(testS2*Theta_glasso{1,2}));

% classification
[V1_disc, D] = eig(Theta_all{1,1});
[V2_disc, D] = eig(Theta_all{1,2});
[V1_glasso, D] = eig(Theta_glasso{1,1});
[V2_glasso, D] = eig(Theta_glasso{1,2});
numBasis = floor(size(V1_disc,2)/2);
% 
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

% save('tempdata/separation_results', 'result_ratio_disc', 'result_ratio_glasso');
% save('tempdata/classification_results', 'result_accu_disc', 'result_accu_glasso');
