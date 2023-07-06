function [dff_blue, mval, sval] = zscore_pixelwise(det_blue, filtLen)
mval = nanmean(det_blue(:, filtLen/2+1:end),2);
sval = nanstd(det_blue(:, filtLen/2+1:end),[],2);


dff_blue = single(bsxfun(@minus, det_blue,mval));
dff_blue = single(bsxfun(@rdivide, dff_blue,sval));
