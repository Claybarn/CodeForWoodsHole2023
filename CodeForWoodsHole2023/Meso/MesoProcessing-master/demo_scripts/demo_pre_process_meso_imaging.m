%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Main script for processing meso imaging - blue-uv
%
% Please note: you should look through all paths and parameters according
% to your experimental setup
%
% Written by Hadas Benisty
% 2/22/2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


addpath(genpath('../general_utils/'));
addpath(genpath('../process_tiffs'));
addpath(genpath('../parcellation'));
addpath(genpath('../visualization/'));
addpath(genpath('../hemodynamics_removal/'));
%% inputs
tiffsPath = 'C:\Users\Clayton\Desktop\HB012\meso';
outputPath = fullfile('test_output_dir');
spike2matfile = fullfile(outputPath, 'spike2File.mat');

fsspike2 = 5e3; 

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
params.deterend.dozscore = 'FIR';

%debuging params
params.verbose = false;
batchSize = 100 ;
batches2load=-1;


%% load tiffs
mkNewDir(outputPath);
disp(['load tiffs from ' tiffsPath ]);
mov = read_batches_tiffs(batchSize, tiffsPath, batches2load, resizeRatio);
%% rotating
% if angle2rotate ~= 0
%     for i=1:size(mov, 3)
%         mov(:,:,i) = imrotate(mov(:,:,i), angle2rotate);
%     end 
% end
[widthPix,lengthPix,numFrames] = size(mov);
%% demixing channels
disp('Extract signals from tiffs');
[blue_raw, uv_raw, skippedframes, skippedchannels] = demix_2signals(mov, params.signalsExtraction);

%% align to atlas 
disp('Align to template')
tformfile = fullfile(outputPath, 'tform.mat');
if ~exist(tformfile, 'file') 
    [tform,R,C] = controlPointAlignmentTransform(blue_raw);
    save(tformfile, 'tform','R','C');
else
    load(tformfile, 'tform','R','C');
end
disp('Aligning to template...');
uv_raw = transform_frames(uv_raw, tform, R, C);
blue_raw = transform_frames(blue_raw, tform, R, C);
%% detrending
blue_raw=reshape(blue_raw, R*C, []);
uv_raw=reshape(uv_raw, R*C, []);
[det_blue,baseline_blue] = detrend_pixels(blue_raw, params.deterend);
[det_uv,baseline_uv] = detrend_pixels(uv_raw, params.deterend);
%% zscore pixelwise
[dff_blue, mval_blue, sval_blue] = zscore_pixelwise(det_blue, params.deterend.filtLen);
[dff_uv, mval_uv, sval_uv] = zscore_pixelwise(det_uv, params.deterend.filtLen);
save(fullfile(outputPath, 'zscored_blue_uv.mat'), 'dff_blue', 'dff_uv', 'R', 'C', ...
    'mval_blue', 'sval_blue', 'mval_uv', 'sval_uv', '-v7.3');
%% Remove Hemodynamics
% select patch size
patch_size = 9;% set this size according to the amount of time samples you have.
assert((patch_size*patch_size)/2<size(dff_blue,2), ... % is this suppose to be patch_size*patch_size)/2 ?
    'please choose smaller patch size or longer session');

load('parcells_updated121519.mat','parcells_new');
brain_mask = parcells_new.CombinedParcells>0;
%brain_mask = zeros(size(parcells_new.CombinedParcells));
%brain_mask((patchsize-1)/2+1:end-(patchsize-1)/2,(patchsize-1)/2+1:end-(patchsize-1)/2) = 1;


flat_mask = reshape(brain_mask,1,[]);
 


[std_blue, std_uv, dff_final,used_patch_size]  = cjb_regress_filter(...
    dff_blue(:,1:500), ...
    dff_uv, ...
    brain_mask, patch_size, 1:R*C);
%dff_final = cat(2, nan(size(dff_final, 1), params.deterend.filtLen/2), dff_final, nan(size(dff_blue, 1), params.deterend.filtLen/2));
save(fullfile(outputPath, 'final_dFoF.mat'), 'std_blue', 'std_uv', 'dff_final', '-v7.3');




%% parcellation
% by Allen
dFoF_parcels = parcels_by_Allen_atlas(dff_final);
save(fullfile(outputPath, 'final_dFoF_allen_parcels.mat'), 'dFoF_parcels', '-v7.3');

%% function parcellation
cfg.outputfilepath = outputPath;
[dff_parcels, parcellationmap] = functional_parcellation_LSSC(dff_final(:, params.deterend.filtLen/2+1:end), cfg);
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



