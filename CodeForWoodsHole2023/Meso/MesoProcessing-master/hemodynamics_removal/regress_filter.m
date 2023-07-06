function [sig_noise1, sig_noise2, data_filt] = regress_filter(blue_sig, uv_sig, brain_mask, patch_size, inds2process)

% Regression of hempdynamics based on spatial info.
% Input: 
%  blue_sig    - calcium dependent signal - pixels X time (masked)
%  uv_sig      - calcium indipendendt - pixels over time (masked)
%  brain_mask  - matrix of R*C = #pixels of zeors and ones indicating
%               location of brain.
% patchsize    - (patchsize x patchsize) patch of pixels to use for spatial info
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

%% check input arguments

if mod(patch_size,2) ~= 1 || patch_size < 1 || rem(patch_size,1)~=0
    error('"patch_size" must be an odd whole number greater than 0.')
end

if patch_size^2/2 >= size(blue_sig,2)
    error('The number of frames must be greater than or equal to patch_size^2 / 2.');
end

%%

reach = (patch_size-1)/2; % distance from center pixel of patch to the edge

disp('Begining of filtering + regression');

[R,C] = size(brain_mask);
[Y,X] = meshgrid(1:C,1:R);
X=X(:);Y=Y(:);
braininds = find(brain_mask);
tic;
[~,Nt] = size(blue_sig);
Np = length(inds2process);
sig_noise1 = zeros(Np,1);
sig_noise2 = zeros(Np,1);
mean_blue =  mean(blue_sig,2);
mean_uv =  mean(uv_sig,2);

rmmean_blue = bsxfun(@minus, blue_sig, mean_blue);
rmmean_uv = bsxfun(@minus, uv_sig, mean_uv);

data_filt = nan(Np, Nt);
for piii=1:length(inds2process)    
    pii = inds2process(piii);
    if mod(piii,100)==0
        disp(['Filtering + Regression: processed ' num2str(piii/length(inds2process)) ' pixels']);
        toc;
        tic;
    end
    x = X(pii);y = Y(pii);
    nn = find(abs(X-x) <= reach & abs(Y-y) <= reach);
    
    nn0 = find(abs(X-x) ==0 & abs(Y-y) ==0);
    if ~ismember(nn0 , braininds)
        continue;
    end
    nn = intersect(nn, braininds);
    if isempty(nn)
        continue;
    end
    
    
    nn_inds=[];
    for j=1:length(nn)
        nn_inds(j) = find(braininds == nn(j));
    end
    S1 = sum(rmmean_blue(nn_inds, :),2);
    S2 = sum(rmmean_uv(nn_inds, :),2);
        
    nn_inds=nn_inds(~isnan(S1)&~isnan(S2)); % remove pixels with nans
    nn0_inds = find(braininds == nn0);
   % midpixel = find(nn==nn0);
    midpixel = find(nn_inds==nn0_inds);
    if isempty(midpixel)
        continue;
    end
    y1 = rmmean_blue(nn_inds, :);
    y2 = rmmean_uv(nn_inds, :);

  
    if all(y1(:)==0)&&  all(y2(:)==0)
        continue;
    end
    % evaluating noise sigma
    % to do - evalute for more experiments and see what's consistent
    % sig_noise1(pii) = estimateNoiseCov(y1);
    % sig_noise2(pii) = estimateNoiseCov(y2);    
    % sig_noise1=1.9e3;
    % sig_noise2=1.5e3;
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
    xhat = [sigx zeros(d)]*pinv(sigy)*cat(1, rmmean_blue(nn_inds, :), rmmean_uv(nn_inds, :));
    data_filt(piii,:) = xhat(midpixel,:);
end
