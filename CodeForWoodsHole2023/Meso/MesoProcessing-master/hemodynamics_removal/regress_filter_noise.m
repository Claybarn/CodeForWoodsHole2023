function [sig_noise1, sig_noise2] = regress_filter(blue_sig, uv_sig, brain_mask, patchsize, inds2process)

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
[Y,X]=meshgrid(1:R,1:C);
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


for piii=1:length(inds2process)    
    % get pixel ind to process (pii != piii if don't start at 1 or skip
    % indices to process)
    pii = inds2process(piii);
    
%     if mod(piii,100)==0
%         disp(['Filtering + Regression: processed ' num2str(piii/length(inds2process)) ' pixels']);
%         toc;
%         tic;
%     end

    if ~ismember(pii , braininds) % cjb changed (was after nn0 definition, but since pii always == nn0, we can save time this way. 
        continue;
    end

    x = X(pii);y = Y(pii);
    nn = find(abs(X-x) < patchsize & abs(Y-y) < patchsize); % end up with ((pathsize-1)*2+1,(pathsize-1)*2+1) matrix
    
    % nn0 = find(abs(X-x) == 0 & abs(Y-y) == 0);
    nn0 = pii; % cjb changed
    
    nn = intersect(nn, braininds);
    
    if isempty(nn)
        continue;
    end
    
    nn_inds=zeros(1,length(nn));
    for j=1:length(nn)
        nn_inds(j) = find(braininds == nn(j));
    end
    
    % find if any pixels have nan at some point
    S1 = sum(rmmean_blue(nn_inds, :),2);
    S2 = sum(rmmean_uv(nn_inds, :),2);
    
    nn_inds=nn_inds(~isnan(S1)&~isnan(S2));    % cjb changed
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
%     sig_noise1(pii) = estimateNoiseCov(y1);
%     sig_noise2(pii) = estimateNoiseCov(y2);    
%     sig_noise1=1.9e3;
%     sig_noise2=1.5e3;
    d = size(y1,1);
    y = cat(1, y1, y2);
    sigy = y*y.'/size(y,2);
    sigy1 = sigy(1:d,1:d);
    %sigy12 = sigy(1:d,d+1:end);
    sigy2 = sigy(1+d:end,1+d:end);
    sig_noise1(piii) = median(svd(sigy1));
    sig_noise2(piii) = median(svd(sigy2));
end

