% dbstop if error

clear;
clc

% dataname_list = {'3sources3vbig'};
dataname_list = {'handwritten'};

for idata = 1:length(dataname_list)
    Dataname = dataname_list{idata};
    
    percent_list = {0.3,0.5,0.7,0.9};
%     percent_list = {0.1,0.3,0.5};
    isTwoView = 1;
    for iper = 1:length(percent_list)
        
        percent = percent_list{iper};
        
        [Datafold,Data] = getData(Dataname,percent,isTwoView);
        
        options.k = 3;
        options.nSubSpace = 4;
        alpha_list  = [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3];
        beta_list  = [1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3];
        options.max_iter = 300;
        num_folds = 10; 
        repeat = 5;
        
        for ialpha = 1:length(alpha_list)
            options.alpha = alpha_list(ialpha);
            for ibeta = 1:length(beta_list)
                options.beta = beta_list(ibeta);
                ACC = [];NMI = [];PUR = [];ARI = [];
                Fscore = [];Precision = [];Recall = [];
                for f = 1:num_folds
                    if f > 1
                        clear folds X truth
                    end
                    load(Data);
                    load(Datafold);
                    
                    num_view = length(X);
                    numClust = length(unique(truth));
                    numInst  = length(truth);
                    
                    ind_folds = folds{f};
                    disp(['There are ',num2str(length(folds)),' folds']);
                    gnd = truth;
                    if strcmp(Dataname,'handwritten')
                        gnd = gnd + 1;
                    end
                    
                    for iv = 1:num_view
                        ind_0 = find(ind_folds(:,iv) == 0);
                        X1 = abs(X{iv}');
                        X1 = NormalizeFea(X1,1);
                        X1(ind_0,:) = [];% 去掉 缺失样本
                        Y{iv} = X1;
                        % ------------- 构造缺失视角的索引矩阵 ----------- %
                        W0 = eye(numInst);
                        W0(ind_0,:) = [];
                        G{iv} = W0;
                        G{iv} = sparse(G{iv});
                        ind_1 = find(ind_folds(:,iv) == 1);
                        W1 = eye(numInst);
                        W1(ind_1,:) = [];
                        Gp{iv} = W1;
                        Gp{iv} = sparse(Gp{iv});
                    end
                    clear X X1 W1 ind_0
                    X = Y;
                    clear Y
                    
                    rng(f*666,'v5normal');
                    
                    [W,H,Ht,Hc] = IMCNAS2_warped_options(X,gnd,G,options);
                    
                    U = Ht';
                    
                    new_F = U;
                    % {
                    norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
                    %%avoid divide by zero
                    for i = 1:size(norm_mat,1)
                        if (norm_mat(i,1)==0)
                            norm_mat(i,:) = 1;
                        end
                    end
                    new_F = new_F./norm_mat;
                    %}
                    for iter_c = 1:repeat
                        pre_labels = kmeans(new_F,numClust,'emptyaction','singleton',...
                            'replicates',20,'display','off',...
                            'Distance','sqeuclidean');
                        result_LatLRR = ClusteringMeasure(gnd, pre_labels);
                        % AC(iter_c) = result_LatLRR(1);
                        % MI(iter_c) = result_LatLRR(2);
                        Purity(iter_c)= result_LatLRR(3);
                        [AC(iter_c),MI(iter_c),~] = result(pre_labels,gnd);
                        [AR(iter_c),~,~,~] = RandIndex(gnd,pre_labels);
                        [Fs(iter_c),Pre(iter_c),Rec(iter_c)] = compute_f(gnd,pre_labels);
                    end
                    
                    ac = mean(AC);
                    nmi = mean(MI);
                    pur = mean(Purity);
                    ar = mean(AR);
                    fs = mean(Fs);
                    p = mean(Pre);
                    r = mean(Rec);
                    clear AC MI Purity AR Fs Pre Rec
                    
                    ACC = [ACC,ac];
                    NMI = [NMI,nmi];
                    PUR = [PUR,pur];
                    ARI = [ARI,ar];
                    Fscore = [Fscore, fs];
                    Precision = [Precision, p];
                    Recall = [Recall, r];
                    
                    disp(strcat('(',num2str(roundn(ac,-3)),',',num2str(roundn(nmi,-3)),...
                        ',',num2str(roundn(pur,-3)),',',num2str(roundn(ar,-3)),...
                        ',',num2str(roundn(fs,-3)),',',num2str(roundn(p,-3)),...
                        ',',num2str(roundn(r,-3)),')'));
                    
                end
                disp(['alpha:1e',num2str(log10(options.alpha))]);
                disp(['beta:1e',num2str(log10(options.beta))]);
                disp(strcat('Final results of',[' ',Dataname],':'));
                disp(strcat('(',num2str(roundn(mean(ACC)*100,-2)),...
                    ',',num2str(roundn(mean(NMI)*100,-2)),...
                    ',',num2str(roundn(mean(PUR)*100,-2)),...
                    ',',num2str(roundn(mean(ARI)*100,-2)),...
                    ',',num2str(roundn(mean(Fscore)*100,-2)),...
                    ',',num2str(roundn(mean(Precision)*100,-2)),...
                    ',',num2str(roundn(mean(Recall)*100,-2)),')'));
                para_IMCNAS2{ialpha,ibeta} = [mean(ACC)*100,...
                    mean(NMI)*100,mean(PUR)*100,mean(ARI)*100;
                    std(ACC)*100,std(NMI)*100,std(PUR)*100,...
                    std(ARI)*100];
                clear ACC NMI PUR ARI Fscore Precision Recall
            end
        end
    end
end

