% function [W,H] = ICMVNMF_warped(options)
function [W,H,Ht,Hc] = IMCNAS2_warped_options(X,gnd,G,options)
% (alpha,beta,p,max_iter,nn)
nClas = length(unique(gnd));
n_view = length(X);
nSmp_all = length(gnd);
% paras.nSmp_all = nSmp_all;
fea = X;
paras.nSubSpace = options.nSubSpace;

paras.alpha = options.alpha;% can not be set 0. fix this para as 1 !!! tune others
paras.beta = options.beta;% laplacian
% ################################################################
paras.p = 2;
paras.max_iter = options.max_iter;% 15,25 is good 
paras.dim = nClas;
paras.nClas = nClas;
paras.G = G;
paras.nSmp_all = nSmp_all;
paras.nClas = length(unique(gnd));
%% construct affinity matrix
n_neighbors = options.k;% 5,6 is good for 0.1; (2,1)1,2 is good for 0.5
for i = 1:n_view
    sigma = 1;% 1 ecg:2 
    Si = construct_W(fea{i},n_neighbors,sigma);
    paras.S{i} = Si; 
end

tic 
[W,H,Hc,dc_mvnmf_obj] = ICMVNMF_HvHw(fea,paras);
toc
%% 1/3*(H{1}+H{2}+H{3})
Ht = get_Ht(G,H,n_view);


end

function [Ht] = get_Ht(G,H,n_view)
    M = 0;
    for i = 1:n_view
        M = M + sum(G{i}',2);
    end
    Ht = 0;
    for i = 1:n_view
        Ht = Ht + H{i}/(G{i}')*diag(1./M);
    end
end

function W = construct_W(fea,num_knn,sigma)

      opts = [];
      opts.NeighborMode = 'KNN';
      opts.k = num_knn;
      opts.WeightMode = 'Binary';% Binary Cosine HeatKernel
      opts.t = sigma;
      W = constructW(fea,opts);
%       W = (W+W')/2;
end
