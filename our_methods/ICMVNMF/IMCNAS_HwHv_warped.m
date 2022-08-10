function [results] = IMCNAS_HwHv_warped(paras_outer)
Dataname = paras_outer.Dataname;
repeat = paras_outer.repeat;
Data = paras_outer.Data;
Datafold = paras_outer.Datafold;
num_folds = paras_outer.num_folds;

paras_inner.alpha = paras_outer.alpha;
paras_inner.beta = paras_outer.beta;
paras_inner.nSubSpace = paras_outer.nSubSpace;
paras_inner.max_iter = paras_outer.max_iter;
paras_inner.k = paras_outer.k;
% num_folds = 10;% length(folds)    
ACC = [];NMI = [];PUR = [];ARI = [];
for f = 1:num_folds
    if f > 1
        clear folds X truth
    end
    load(Data);
    load(Datafold);

    num_view = length(X);
    numClust = length(unique(truth));
    numInst  = length(truth);

    if strcmp(Dataname,'bbcsport4vbig')
        %{
        ind_folds = folds(f,:,:);
        ind_folds = squeeze(ind_folds);
        disp(['There are ',num2str(size(folds,1)),' folds']);
        %}
        ind_folds = folds{f};
        disp(['There are ',num2str(length(folds)),' folds']);
    else
        ind_folds = folds{f};
        disp(['There are ',num2str(length(folds)),' folds']);
    end
    gnd = truth;
    if strcmp(Dataname,'handwritten')
        gnd = gnd + 1;
    end

    for iv = 1:length(X)
        ind_0 = find(ind_folds(:,iv) == 0);
        %         X0{iv} = X{iv}';
        %         X0{iv}(ind_0,:) = randn(1);
        %         X0{iv} = abs(X{iv}');
        %         X0{iv}(ind_0,:) = rand(1);
        %         X0{iv} = NormalizeFea(X0{iv},1);
        %         X1 = X{iv}';
        %         X{iv} = X_remo_min(X{iv});
        X1 = abs(X{iv}');
        X1 = NormalizeFea(X1,1);
        X1(ind_0,:) = [];% 去掉 缺失样本
        Y{iv} = X1;
        % ------------- 构造缺失视角的索引矩阵 ----------- %
        W1 = eye(numInst);
        W1(ind_0,:) = [];
        G{iv} = W1;
        G{iv} = sparse(G{iv});
    end
    clear X X1 W1 ind_0
    X = Y;
    clear Y

    rng(f*666,'v5normal');

    [Ht,Hc] = ICMVNMF_warped_options(X,gnd,G,paras_inner);
    U = Ht';
%                     U = Hc';

    new_F = U;
    % {
    norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
    %     norm_mat = repmat(sum(new_F,2),1,size(new_F,2));
    %%avoid divide by zero
    for i = 1:size(norm_mat,1)
        if (norm_mat(i,1)==0)
            norm_mat(i,:) = 1;
        end
    end
    new_F = new_F./norm_mat;
    %} 
    for iter_c = 1:repeat% cosine sqEuclidean
        % pre_labels = litekmeans(new_F ,numClust, 'Replicates',20,'Distance','sqEuclidean');
        pre_labels    = kmeans(new_F,numClust,'emptyaction','singleton',...
            'replicates',20,'display','off',...
            'Distance','sqeuclidean');
        result_LatLRR = ClusteringMeasure(gnd, pre_labels);
        % AC(iter_c) = result_LatLRR(1);
        % MIhat(iter_c) = result_LatLRR(2);
        Purity(iter_c) = result_LatLRR(3);
        [AC(iter_c),MI(iter_c),~] = result(pre_labels,gnd);
        [AR(iter_c),~,~,~] = RandIndex(gnd,pre_labels);
    end

    ac = mean(AC);
    nmi = mean(MI);
    pur = mean(Purity);
    ar = mean(AR);
    clear AC MI Purity AR

    ACC = [ACC,ac];
    NMI = [NMI,nmi];
    PUR = [PUR,pur];
    ARI = [ARI,ar];

    disp(strcat('(',num2str(roundn(ac,-3)),',',num2str(roundn(nmi,-3)),...
        ',',num2str(roundn(pur,-3)),',',num2str(roundn(ar,-3)),')'));

end
disp(strcat('Final results: (',...
    num2str(roundn(mean(ACC)*100,-2)),...
    ',',num2str(roundn(mean(NMI)*100,-2)),...
    ',',num2str(roundn(mean(PUR)*100,-2)),...
    ',',num2str(roundn(mean(ARI)*100,-2)),')'));
results{1} = [roundn(mean(ACC)*100,-2); roundn(std(ACC)*100,-2)];
results{2} = [roundn(mean(NMI)*100,-2); roundn(std(NMI)*100,-2)];
results{3} = [roundn(mean(PUR)*100,-2); roundn(std(PUR)*100,-2)];
results{4} = [roundn(mean(ARI)*100,-2); roundn(std(ARI)*100,-2)];
clear ACC NMI PUR ARI

end
