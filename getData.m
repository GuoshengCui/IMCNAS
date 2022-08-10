

function [Datafold,Data] = getData(Dataname,percent,isTwoView)

if ~isTwoView
    percentDel = percent;
    Datafold = ['MV_datasets/',Dataname,'/',Dataname,...
        'RnSp_percentDel_',num2str(percentDel),'.mat'];
    Data = ['MV_datasets/',Dataname,'/',Dataname,'RnSp'];
else
    percent_pair = percent;
    dataroot = ['MV_datasets/',Dataname,'/',Dataname,'_with_G/'];
    Datafold = [dataroot,Dataname,'_Folds_with_G_paired_',...
        num2str(percent_pair),'.mat'];
    Data = [dataroot,Dataname,'_RnSp_with_G'];
end
end