function [det_sig,basesig] = detrend_pixels(sig_raw, paramsdetrend)

if strcmp(paramsdetrend.method,'FIR')
    filtcoeff = fir1(paramsdetrend.filtLen,paramsdetrend.filtcutoff);
    PixxTime_bl_base = filter(filtcoeff, 1, sig_raw.').';
    PixxTime_bl_base = PixxTime_bl_base(:,paramsdetrend.filtLen/2+1:end); % drop first N/2 frames
    PixxTime_bl_base(:, 1:paramsdetrend.filtLen/2) = nan; % set first N/2 frames equal to nan
    PixxTime_bl_base = cat(2, PixxTime_bl_base, nan(size(PixxTime_bl_base, 1), paramsdetrend.filtLen/2)); % set first N/2 frames equal to nan
elseif strcmp(paramsdetrend.method,'Butter')
    [b,a] = butter(1,paramsdetrend.filtcutoff/(paramsdetrend.samplingFreq/2));
    % have to iterate through each pixel, otherwise matlab tries to do them
    % all at once and we will run out of memory
    PixxTime_bl_base = nan(size(sig_raw),'single');
    nanMask = ~(sum(isnan(sig_raw),1)==size(sig_raw,1));
    for i =1:size(sig_raw,1)
       PixxTime_bl_base(i,nanMask) =  single(filtfilt(b,a,double(sig_raw(i,nanMask))));
    end
end  

det_sig = (bsxfun(@minus, sig_raw,PixxTime_bl_base));
basesig = nanmean(PixxTime_bl_base, 2);
det_sig = (bsxfun(@rdivide, det_sig, basesig));

