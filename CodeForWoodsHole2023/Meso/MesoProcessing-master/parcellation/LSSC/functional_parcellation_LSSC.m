function [dff_parcels, parcellationmap] = functional_parcellation_LSSC(regsig, cfg)

load('../parcellation/parcells_updated121519.mat','parcells_new');
brain_mask = sum(parcells_new.indicators,3)>0;
parcellationmap=nan(size(brain_mask));

brain_mask(:, 125:131) = false;
brain_mask_left = brain_mask;
brain_mask_right = brain_mask;
brain_mask_left(:,129:end) = false;
brain_mask_right(:,1:128) = false;


allregionspix_left = find(brain_mask_left);
allregionspix_right = find(brain_mask_right);

[R,C] = size(brain_mask);
dFoF_masked_left = regsig(allregionspix_left, :);
dFoF_masked_right = regsig(allregionspix_right, :);

cfg.preProcess=false;
cfg.N_TRIALS=10;
cfg.n_clust = 5000 ;
cfg.makePlots = false;
cfg.thrcluster=0.99;
cfg.NROWS = R;
cfg.NCOLS = C;
cfg.isoverlap = false;


cfg.title_str = 'left';
leftfile = dir(fullfile(cfg.outputfilepath, 'left*.mat'));
if isempty(leftfile)
    runROI_meso_nlm(cfg, dFoF_masked_left, allregionspix_left, brain_mask_left);
    leftfile = dir(fullfile(cfg.outputfilepath, 'left*.mat'));
end
leftparcels = load(fullfile(cfg.outputfilepath,leftfile.name));



cfg.title_str = 'right';
runROI_meso_nlm(cfg, dFoF_masked_right, allregionspix_right, brain_mask_right);
rightfile = dir(fullfile(cfg.outputfilepath, 'right*.mat'));
if isempty(rightfile)
    runROI_meso_nlm(cfg, dFoF_masked_left, allregionspix_left, brain_mask_left);
    rightfile = dir(fullfile(cfg.outputfilepath, 'right*.mat'));
end
rightparcels = load(fullfile(cfg.outputfilepath,rightfile.name));

ROI_list = [leftparcels.ROI_list  rightparcels.ROI_list];
dff_parcels = nan(length(ROI_list), size(regsig,2));
for k = 1:length(ROI_list)
    parcellationmap(ROI_list(k).pixel_list) = k;
    dff_parcels(k, :) = nanmean(regsig(ROI_list(k).pixel_list, :));
end
