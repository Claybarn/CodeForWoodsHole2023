addpath(genpath('../general_utils/'));
addpath(genpath('../process_tiffs'));
addpath(genpath('../parcellation'));
addpath(genpath('../visualization/'));
addpath(genpath('../hemodynamics_removal/'));
addpath(genpath('../../B-COSFIRE'))
%% inputs


inputPath = '../../CB003_08-20-2021_15-00-00';
tiffsPath = fullfile(inputPath,'meso');
outputPath = '../../dualOutput';
BCOSFIREPath = '../../B-COSFIRE';

spike2matfile = fullfile(outputPath, 'spike2File.mat');

fsspike2 = 5e3; 

%% experimental parameters

% for blue-uv settings
params.signalsExtraction.firstCaFrame = 1; % is the first frame of tiffs blue
params.signalsExtraction.blueuvRatio = 1; % the ratio between blue and uv imaging frames
resizeRatio = .5; % resizeing (1 for not doing anything, 0.5 to get half as many on each axis
%angle2rotate = -180; % angle for rotation of tiffs (depends on where you recorded your sessions!)

% detrending params
params.detrend.filtLen = 1000;
params.detrend.filtcutoff = 0.001;
params.detrend.method = 'FIR';

batchSize = 15000;
batches2load=1; % load all
R = 256;
C = 256;

%% load tiffs
mkdir(outputPath);
disp(['load tiffs from ' tiffsPath ]);
%mov = read_batches_tiffs(batchSize, tiffsPath, batches2load, resizeRatio);
mov = smart_read_tiffs(tiffsPath,true,256,256,0.5);
[widthPix,lengthPix,numFrames] = size(mov);
%% demixing channels
% problem with blue and uv channels, twice as many frames as should be
disp('Extract signals from tiffs');
% sigsMov = demix_3signals(mov, params.signalsExtraction);
%% above didnt work
sigsMov.blue = mov(:,:,1:3:end);
sigsMov.uv = mov(:,:,2:3:end);
sigsMov.green = mov(:,:,3:3:end);
sigsMov.blue = sigsMov.blue(:,:,1:size(sigsMov.green,3));
sigsMov.uv = sigsMov.uv(:,:,1:size(sigsMov.green,3));

%% register 2 cameras
% register images (via blood vessels). Can use bloodVesselInfo and
% tform_register to check registration
[tform_register,bloodVesselInfo] = register_image_tform(mean(sigsMov.blue(:,:,1:100),3),mean(sigsMov.green(:,:,1:100),3),BCOSFIREPath);


tform_register_file = fullfile(outputPath,'tform_register.mat');
save(tform_register_file,'tform_register');

% register green frames to blue
sigsMov.green = transform_frames(sigsMov.green, tform_register, R, C);

subplot(121)
imshowpair(bloodVesselInfo.fixed.respimage,imwarp(bloodVesselInfo.moving.respimage,tform_register,'OutputView',imref2d([256 256]),'Fillvalues',0, 'interp', 'nearest'))
title('Blood Vessel Alignment')
subplot(122)
imshowpair(sigsMov.green(:,:,1),sigsMov.blue(:,:,1))
title('Image Alignment')
%% align to atlas 
disp('Align to template')
tform_allen_file = fullfile(outputPath, 'tform_allen.mat');
if ~exist(tform_allen_file, 'file') 
    [tform_allen,R,C] = controlPointAlignmentTransform(sigsMov.blue);
    save(tform_allen_file, 'tform_allen','R','C');
else
    load(tform_allen_file, 'tform_allen','R','C');
end

disp('Aligning to template...');
sigsMov.uv = transform_frames(sigsMov.uv, tform_allen, R, C);
sigsMov.blue = transform_frames(sigsMov.blue, tform_allen, R, C);
sigsMov.green = transform_frames(sigsMov.green, tform_allen, R, C);


%% detrending
sigsMov.uv = reshape(sigsMov.uv,R*C,[]);
sigsMov.blue = reshape(sigsMov.blue,R*C,[]);
sigsMov.green = reshape(sigsMov.green,R*C,[]);


% TODO compare baseline with butter filter filtfilt function?
[det_uv,~] = detrend_pixels(sigsMov.uv, params.detrend);
det_uv_file = fullfile(outputPath,'det_uv.mat');
save(det_uv_file,'det_uv','-V7.3')
clear det_uv

[det_blue,~] = detrend_pixels(sigsMov.blue, params.detrend);
det_blue_file = fullfile(outputPath,'det_blue.mat');
save(det_blue_file,'det_blue','-V7.3')
clear det_blue

[det_green,~] = detrend_pixels(sigsMov.green, params.detrend);
det_green_file = fullfile(outputPath,'det_green.mat');
save(det_green_file,'det_green','-V7.3')


%% zscore
[zscore_green, green_mean, green_std] = zscore_pixelwise(det_green,1000);
clear det_green
zscore_green_file = fullfile(outputPath,'zscore_green.mat');
save(zscore_green_file,'zscore_green', 'green_mean', 'green_std','-V7.3')
clear zscore_green


load(det_uv_file,'det_uv');
[zscore_uv, uv_mean, uv_std] = zscore_pixelwise(det_uv,1000);
clear det_uv
zscore_uv_file = fullfile(outputPath,'zscore_uv.mat');
save(zscore_uv_file,'zscore_uv', 'uv_mean', 'uv_std','-V7.3')



load(det_blue_file,'det_blue');
[zscore_blue, blue_mean, blue_std] = zscore_pixelwise(det_blue,1000);
clear det_blue
zscore_blue_file = fullfile(outputPath,'zscore_blue.mat');
save(zscore_blue_file,'zscore_blue', 'blue_mean', 'blue_std','-V7.3')


%% Remove Hemodynamics
% select patch size
blue_patch_size = 5;% set this size according to the amount of time samples you have.
green_patch_size = 5;% set this size according to the amount of time samples you have.

load('parcells_updated121519.mat','parcells_new');
brain_mask = parcells_new.CombinedParcells>0;
brain_mask(:,256/2-9:256/2+10) = 0;
flat_mask = reshape(brain_mask,1,[]);

[std_blue, std_uv, noz_blue_final]  = regress_filter(...
    det_blue(flat_mask,:), ...
    det_uv(flat_mask,:), ...
    brain_mask, blue_patch_size, 1:R*C);

blue_final_file = fullfile(outputPath,'noz_blue_final.mat');
save(blue_final_file,'std_blue','std_uv','noz_blue_final','-V7.3')
clear blue_final std_blue std_uv zscore_blue

load(zscore_green_file,'zscore_green')

[std_green, std_uv, noz_green_final]  = regress_filter(...
    det_green(flat_mask,:), ...
    det_uv(flat_mask,:), ...
    brain_mask, green_patch_size, 1:R*C);
green_final_file = fullfile(outputPath,'green_final.mat');
save(green_final_file,'std_green','std_uv','green_final','-V7.3')
clear zscore_green green_final

%% parcellation
% by Allen
noz_blue_parcels = parcels_by_Allen_atlas(noz_blue_final);
green_parcels = parcels_by_Allen_atlas(green_final);

save(fullfile(outputPath, 'allen_parcels.mat'), 'blue_parcels','green_parcels', '-v7.3');


%%
parcel_mask = ~(isnan(blue_parcels(:,1)) | isnan(green_parcels(:,1)));
C = corrcoef((vertcat(blue_parcels(parcel_mask,:),green_parcels(parcel_mask,:)))');
imagesc(C);

%% function parcellation
cfg.outputfilepath = outputPath;
[dff_parcels, parcellationmap] = functional_parcellation_LSSC(green_final(:, params.detrend.filtLen/2+1:end), cfg);

save(fullfile(outputPath, 'final_dFoF_LSSC_parcels.mat'), 'parcellationmap', 'dff_parcels', '-v7.3');

%% smrx
% smrx channels specifications
CHANNELS_NUM = 8;
channels.BLUE = 1;
channels.UV = 2;
channels.FRAMETICKES = 3;
channels.PHOTO_DIODE = 4;
channels.WHEEL = 5;
channels.AIR_PUFF = 6;
channels.PUPIL_CAMERA = 7;

if ~isfile(spike2matfile)
    [timing, channels_data] = process_smrx('../process_spike2/CEDS64ML', outputpath,smrxfile, channels, CHANNELS_NUM);
    save(spike2matfile, 'timing', 'channels_data')
else
    load(spike2matfile, 'timing', 'channels_data')
end
t_spike2 = linspace(0, length(channels_data.startsig)-1,length(channels_data.startsig))/fsspike2; 
% time line for imaging - take every second for 1-1 blue and uv
t_imaging = timing.mesostart(1:2:end)/fsspike2;
L = min(length(t_imaging), size(dFoF_parcels, 2));
dFoF_parcels=dFoF_parcels(:, 1:L);
t_imaging=t_imaging(1:L);
%% sanity check to see that smrx and imaging data are alinged
% plotting v1 and stim traces. Zoom in to see that they match
figure;
h(1) = subplot(3,1,1);plot(t_imaging, dFoF_parcels(1,:));
xlabel('Time [sec]');ylabel('\Delta F/F');
title('V1 Response');
h(2) = subplot(3,1,2);plot(t_spike2, channels_data.startsig);
hold all;plot(t_spike2, channels_data.air_puff);
legend('Stim','Air Puff');
h(3) = subplot(3,1,3);plot(t_spike2, channels_data.wheelspeed);
title('Wheel Speed');
linkaxes(h,'x');


%% Plot mean trial
stim_timestamps = timing.stimstart/fsspike2;
before_win = 40;
after_win = 35;
imagingAl = zeros(size(dFoF_parcels,1), before_win+after_win+1, length(stim_timestamps));

for stim_i = 1:length(stim_timestamps)
inds_imaging = findClosestDouble(t_imaging, stim_timestamps(stim_i));
imagingAl(:, :, stim_i) = dFoF_parcels(:, inds_imaging-before_win:inds_imaging+after_win);
end
fsimaing=10;
t_trials = linspace(-before_win, after_win, after_win+before_win+1)/fsimaing;

figure;subplot(2,1,1);
imagesc(t_trials, 1:size(imagingAl,1), squeeze(imagingAl(2,:,:))')
xlabel('Time [sec]');ylabel('Trials');title('Right V1 Responses Vs Trials');
subplot(2,1,2);plot(t_trials, mean(imagingAl(2,:,:),3))
xlabel('Time [sec]');ylabel('\Delta F/F');
title('Right V1 Averaged Accross Trials');axis tight;


