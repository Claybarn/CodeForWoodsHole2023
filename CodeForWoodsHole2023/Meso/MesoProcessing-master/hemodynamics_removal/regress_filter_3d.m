function [sig_noise1, sig_noise2, data_filt] = regress_filter_3d(blue_sig, uv_sig, brain_mask, patch_size, inds_to_process)

[R,C] = size(brain_mask);
sizes = size(blue_sig);
Nt = sizes(end);
Np = length(inds_to_process);
sig_noise1 = zeros(Np,1);
sig_noise2 = zeros(Np,1);
data_filt = nan(Np, Nt);

% new way parameters
reach = (patch_size-1)/2; % how far to reach in order to create desired pixel size
default_poi_loc = patch_size-reach; % where the pixel of interest would be if no caveats

% reshape input to make it 3D, makes it easier to grab patch and has no
% overhead
blue_sig=reshape(blue_sig,R,C,[]);
uv_sig=reshape(uv_sig,R,C,[]);

% take mean of blue
mean_blue =  mean(blue_sig,3);
mean_uv =  mean(uv_sig,3);

% get mask of what pixels have nan values (will show up nan because take
% mean)
nanless_mask = ~isnan(mean_blue) & ~isnan(mean_uv);

% center data
blue_sig = blue_sig - mean_blue;
uv_sig = uv_sig - mean_uv;


for pixel_it=1:length(inds_to_process)    
    pixel_ind = inds_to_process(pixel_it);
    [row,col] = ind2sub([R,C],pixel_ind);
    
    % jump to next pixel if this one is not on brain or has nans
    if ~brain_mask(row,col)||~nanless_mask(row,col)
        continue;
    end
    
    % get patch bounds, accounting for edge of image
    lower_row = max([row-reach,1]);
    upper_row = min([row+reach,R]);
    lower_col = max([col-reach,1]);
    upper_col = min([col+reach,C]);
    % poi = pixel of interest (center pixel of current patch, have to account
    % for ugly things like being near edge of frame and not being on brain
    % mask)

    % create matrix to track our pixel of interest
    poi_tracer = boolean(zeros(patch_size));
    poi_tracer(default_poi_loc,default_poi_loc) = 1;
    % Here we account for being near the edge of the image. Essentially we take the difference
    % between what we would expect the patch size to be and what we can actually
    % use, and adjust accordingly.
    poi_tracer=poi_tracer( 1+lower_row-row+reach  : end-(row+reach-upper_row) , 1+lower_col-col+reach : end-(col+reach-upper_col) );
    % get mask for current patch, accounts for being on brain and not
    % having nans
    patch_mask = brain_mask(lower_row:upper_row,lower_col:upper_col) & nanless_mask(lower_row:upper_row,lower_col:upper_col); % create mask for pixels in this path that are on the brain and have no nan values
    poi_tracer = reshape(poi_tracer,1,[]);
    patch_mask = reshape(patch_mask,1,[]);
    poi_ind = find(poi_tracer(patch_mask));

    if isempty(poi_ind)
        continue;
    end

    % grab patch data and flatten 
    blue_data_seg = reshape(blue_sig(lower_row:upper_row,lower_col:upper_col,:),(upper_row-lower_row+1)*(upper_col-lower_col+1),[]);
    blue_data_seg = blue_data_seg(patch_mask,:);
    uv_data_seg = reshape(uv_sig(lower_row:upper_row,lower_col:upper_col,:),(upper_row-lower_row+1)*(upper_col-lower_col+1),[]);
    uv_data_seg = uv_data_seg(patch_mask,:);


    d = sum(patch_mask);
    y = [blue_data_seg; uv_data_seg];
    sigy = y*y.'/size(y,2);
    sigy1 = sigy(1:d,1:d);
    sigy12 = sigy(1:d,d+1:end);
    sigy2 = sigy(1+d:end,1+d:end);
    sig_noise1(pixel_it) = median(svd(sigy1));
    sig_noise2(pixel_it) = median(svd(sigy2));

    sigx = sigy1-sig_noise1(pixel_it)*eye(d) - sigy12*pinv((sigy2-sig_noise2(pixel_it)*eye(d)))*sigy12';
    % to do - actually we can save time and estimate xhat only for the pixel in the
    % middle of the patch
    xhat = [sigx zeros(d)]*pinv(sigy)*y; % (validpatchsize^2,2xpatchsize^2)*(2xpatchsize^2,2xpatchsize^2)*(2xpatchsize^2,num frames)
    data_filt(pixel_it,:) = xhat(poi_ind,:);
end
end
