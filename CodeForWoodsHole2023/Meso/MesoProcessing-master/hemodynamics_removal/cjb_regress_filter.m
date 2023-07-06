function [sig_noise1, sig_noise2, data_filt, used_patch_size] = cjb_regress_filter(blue_sig, uv_sig, brain_mask, patchsize, inds2process)

% Regression of hempdynamics based on spatial info.
% Input: 
%  blue_sig    - calcium dependent signal - pixels X time (masked)
%  uv_sig      - calcium indipendendt - pixels over time (masked)
%  brain_mask  - matrix of R*C = #pixels of zeors and ones indicating
%               location of brain.
% patchsize    - half size of patch + 1 for spatial info
%inds2process  - which indices to process (usefull for batch processing on farnam)
% Output: 
%   sig_noise1 - estimated standard deviation of noise in blue - pixels X 1
%   sig_noise1 - estimated standard deviation of noise in uv - pixels X 1
%   data_filt  - regressed signal
%     
% Written by:
%    Hadas Benisty, hadas.benisty@gmail.com
%    Boris Landa
%    April 6 2020
%
%------------------------------------------------------------------------


% disp(['Begining of filtering + regression']);


[R,C] = size(brain_mask);
blue_sig = reshape(blue_sig,R,C,[]);
uv_sig = reshape(uv_sig,R,C,[]);

indMap = reshape(cumsum(ones(1,R*C)),R,C);

%braininds = find(brain_mask);

%tic;
[~,~,Nt] = size(blue_sig);
Np = length(inds2process);
sig_noise1 = zeros(Np,1);
sig_noise2 = zeros(Np,1);
mean_blue =  mean(blue_sig,3);
mean_uv =  mean(uv_sig,3);

rmmean_blue = bsxfun(@minus, blue_sig, mean_blue);
rmmean_uv = bsxfun(@minus, uv_sig, mean_uv);

vect_rmmean_blue = reshape(rmmean_blue,R*C,[]);
vect_rmmean_uv = reshape(rmmean_uv,R*C,[]);


data_filt = nan(Np, Nt);
used_patch_size = nan(R,C);
nanlessMap = ~(any(isnan(rmmean_blue),3) | any(isnan(rmmean_uv),3));


for piii=1:length(inds2process)    
    pii = inds2process(piii);
%     if mod(piii,100)==0
%         disp(['Filtering + Regression: processed ' num2str(piii/length(inds2process)) ' pixels']);
%         toc;
%         tic;
%     end
    if ~brain_mask(indMap == pii) | ~nanlessMap(indMap == pii) % cjb changed (was after nn0 definition, but since pii always == nn0, we can save time this way. 
        continue;
    end
    
    [xPix,yPix] = find(indMap == pii);


    [X,Y]=meshgrid(xPix-(patchsize-1)/2:xPix+(patchsize-1)/2,yPix-(patchsize-1)/2:yPix+(patchsize-1)/2);
    keepInds = X(:) > 1 & X(:) < C & Y(:) > 1 & Y(:) < R;% & brain_mask(sub2ind(size(brain_mask),X(:),Y(:))) & nanlessMap(sub2ind(size(brain_mask),X(:),Y(:)));
    
    passMaskInds = brain_mask(sub2ind(size(brain_mask),X(keepInds),Y(keepInds))) & nanlessMap(sub2ind(size(brain_mask),X(keepInds),Y(keepInds)));
    keepInds(keepInds) = passMaskInds;
    
    middlepixel = X(keepInds) ==xPix & Y(keepInds) == yPix;
    
    used_patch_size(pii) = sum(keepInds);
    if used_patch_size(pii) < 1
        continue
    end
    inds = sub2ind(size(brain_mask),X(keepInds),Y(keepInds));
    
    y1 = vect_rmmean_blue(inds,:);
    y2 = vect_rmmean_uv(inds,:);
    
    
    % evaluating noise sigma
    % to do - evalute for more experiments and see what's consistent
%     sig_noise1(pii) = estimateNoiseCov(y1);
%     sig_noise2(pii) = estimateNoiseCov(y2);    
%     sig_noise1=1.9e3;
%     sig_noise2=1.5e3;
    d = size(y1,1);
    y = cat(1, y1, y2);
    sigy = y*y.'/size(y,2);
    sigy1 = sigy(1:d,1:d);
    sigy12 = sigy(1:d,d+1:end);
    sigy2 = sigy(1+d:end,1+d:end);
    sig_noise1(piii) = median(svd(sigy1));
    sig_noise2(piii) = median(svd(sigy2));

    sigx = sigy1-sig_noise1(piii)*eye(d) - sigy12*pinv((sigy2-sig_noise2(piii)*eye(d)))*sigy12';
    % to do - actually we can save time and estimate xhat only for the pixel in the
    % middle of the patch
    % cjb something like this: 
    % [sigx(middlepixel,:) zeros(1,d)]*pinv(sigy)*y% but not completely
    % right
    xhat = [sigx zeros(d)]*pinv(sigy)*y; % cjb changed (since y is equal to what was here)
    data_filt(piii,:) = xhat(middlepixel,:);
    
    
end
