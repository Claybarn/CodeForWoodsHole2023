%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run this script one section at a time (highlighted section between double percent
% signs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% make sure local directory is demo_scripts, or below wont work
% adding paths for functions we will need
addpath(genpath('../general_utils/'));
addpath(genpath('../process_tiffs'));
addpath(genpath('../parcellation'));
addpath(genpath('../visualization/'));
addpath(genpath('../hemodynamics_removal/'));
%% inputs
% put path to tiffs here
tiffsPath = '';


%% experimental parameters

% for blue-uv settings
params.signalsExtraction.firstCaFrame = 1; % is the first frame of tiffs blue
params.signalsExtraction.blueuvRatio = 1; % the ratio between blue and uv imaging frames
resizeRatio = .5; % resizeing (1 for not doing anything, 0.5 to get half as many on each axis
%angle2rotate = -180; % angle for rotation of tiffs (depends on where you recorded your sessions!)

% detrending params
params.deterend.filtLen = 150;
params.deterend.filtcutoff = 0.001;
params.deterend.method = 'FIR';
params.deterend.dozscore = true;

%debuging params
batchSize = 500;
batches2load=1;


%% load tiffs
disp(['load tiffs from ' tiffsPath ]);
mov = read_batches_tiffs(batchSize, tiffsPath, batches2load, resizeRatio);

[widthPix,lengthPix,numFrames] = size(mov);

% make sure it worked
subplot(221)
imagesc(mov(:,:,1))
title('Frame 1')

subplot(222)
imagesc(mov(:,:,2))
title('Frame 2')

subplot(223)
imagesc(mov(:,:,3))
title('Frame 3')

subplot(224)
imagesc(mov(:,:,4))
title('Frame 4')

% why do some of the frames look different?


%% demixing channels
% split into calcium and uv frames (were interleved in the mov variable)
disp('Extract signals from tiffs');
[blue_raw, uv_raw, skippedframes, skippedchannels] = demix_2signals(mov, params.signalsExtraction);

%% align to atlas 
% align our images to the allen atlas
% tform file contains data on how to transform our frames to match allen
% atlas
disp('Align to template')
tformfile = fullfile(outputPath, 'tform.mat');
if ~exist(tformfile, 'file') 
    [tform,R,C] = controlPointAlignmentTransform(blue_raw);
    save(tformfile, 'tform','R','C');
else
    load(tformfile, 'tform','R','C');
end
disp('Aligning to template...');

% apply transform to all frames
uv_raw = transform_frames(uv_raw, tform, R, C);
blue_raw = transform_frames(blue_raw, tform, R, C);


% lets look at the mean signal over time
% we sample at 10 Hz, so to create time vector we divide 1:number of frames
% by 10 to convert frame number to seconds. We divide the variable
% numFrames by 2, because it is the total number of frames, ie uv and
% calcium. Half of them should be calcium. Could also call the function
% size on blue_raw to get the number of frames. 

% also note that unlike before, the size of the variable is number of
% pixels by number of frames. In otherwords, it is 2D rather than 3D. We
% collapsed all the pixels into 1 dimension because it is easier to do some
% operations. To get it back to 3D, we can call
% reshape(variableName,256,256,[]) to get a 3D variable. The [] tells the
% funtion to fill in the missing spot itself, and will throw an error if
% the number of elements is changed (because that wouldnt make sense, can
% explain this further)

plot((1:numFrames/2)/10,mean(blue_raw,1))
xlabel('Time (s)');
ylabel('Mean Pixel Intensity');
title('Mean Raw Sig Over Time')
%% detrending
% in this step we do a high pass filter, which means we first transform our
% signal to the frequency domain and then keep the signal that is higher
% than a specificed frequency (this includes 0 Hz signals, ie constant
% offsets). This removes (probably) non-biological slow changes in
% our signal 

blue_raw=reshape(blue_raw, R*C, []);
uv_raw=reshape(uv_raw, R*C, []);
[det_blue,baseline_blue] = detrend_pixels(blue_raw, params.deterend);
[det_uv,baseline_uv] = detrend_pixels(uv_raw, params.deterend);


% lets look at the mean detrended signal over time

plot((1:numFrames/2-params.deterend.filtLen)/10,nanmean(det_blue(:,params.deterend.filtLen/2:end-params.deterend.filtLen/2-1),1))
xlabel('Time (s)');
ylabel('Mean');
title('Mean detrended Sig Over Time')


% If you have enough frames, you should see the signal decay to and
% oscillate around 0


%% zscore pixelwise
% One thing to keep in mind about zscoring (or any form of normalization
% really) is that you lose information on magnitude. Here, we are going to
% zscore each pixel in time. This means if pixel 1 has variance of 3 while
% pixel 2 has a variance of 1000, you could never tell when you go to watch
% the zscored movie. But it seems to help the hemodynamic correction, so we
% do it and then save the mean and std of each pixel


[dff_blue, mval_blue, sval_blue] = zscore_pixelwise(det_blue, params.deterend.filtLen);
[dff_uv, mval_uv, sval_uv] = zscore_pixelwise(det_uv, params.deterend.filtLen);


%% Remove Hemodynamics

% here with both the uv and calcium data we use a path_size by patch_size
% grid around each pixel in the calcium data to predict its value. 
% This seems to do a good job at predicting the hemodynamic artifacts.


patch_size = 9;% set this size according to the amount of time samples you have.

% here we load the allen brain atlas and use it to mask out non brain areas
% of our images (dont want the factor them in this computation since they
% have nothing to do with neural activity). Jess likes to block out the
% midline but it seems like we don't have to do that anymore with our new
% hemodynamic technique. 

load('parcells_updated121519.mat','parcells_new');
brain_mask = parcells_new.CombinedParcells>0;


% need to mask the our data (only take some of the values)so flatten our 2D
% mask to match our the shape of our data
flat_mask = reshape(brain_mask,1,[]);
 

% this function does the hemodynamic correction. Have to mask our data
% before passing it (It is silly but I didnt make this function okay)
[std_blue, std_uv, dff_final]  = regress_filter_3d(...
    dff_blue, ...
    dff_uv, ...
    brain_mask, patch_size, 1:R*C);


% lets see the fruits of our labor
% we will need to adjust the colormap to see anything.
% in the tools drop down menu go to colormap, select parula, check the box
% to specify a range, make the mimum something like -0.01 and maximum
% something like 0.01. Can play with other values. Press apply and close
% the box to view. 
implay(reshape(dff_final,256,256,[]),10);

%% parcellation
% here we parcellate the data by the Allen brain atlas.
dFoF_parcels = parcels_by_Allen_atlas(dff_final);

% the result should be something like 56 by the number of frames. Plot the
% parcel values like we did for the mean value (but dont take mean this
% time!). To see what each parcel is, checkout the parcells_new.description
% field. Plot the ones that have the coolest names.

% Great! Next we can work on aligning it to behavioral data.
