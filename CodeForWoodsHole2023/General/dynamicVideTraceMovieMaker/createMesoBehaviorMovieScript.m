 
addpath(genpath('C:\Users\Clayton\Desktop\Code\MATLAB\MesoProcessing-master'));
addpath(genpath('C:\Users\Clayton\Desktop\Code\MATLAB\chain_processing'));
addpath(genpath('C:\Users\Clayton\Desktop\Code\MATLAB\Spike2Methods'));
addpath(genpath('C:\Users\Clayton\Desktop\Code\MATLAB\BehaviorAnalysis'));

datasets = dir('S:\JanDualData/CB*');
spike2Path = 'C:\Users\Clayton\Desktop\Code\MATLAB\MesoProcessing-master\process_spike2\CEDS64ML';
load('parcells_updated121519.mat');


fs = 10; % in Hz
stateTransPreTime = 1; % in seconds
stateTransPostTime = 2; % in seconds

boolPlot = true;


%% load data
i=5;
% load meso video data
redData = reconstructCorrectedData(fullfile(datasets(i).folder,datasets(i).name,'hemoCorrectedSig'),'red'); % changed naming scheme
greenData = reconstructCorrectedData(fullfile(datasets(i).folder,datasets(i).name,'hemoCorrectedSig'),'green'); % changed naming scheme


% load parcel data
load(fullfile(datasets(i).folder,datasets(i).name,'allenParcellations.mat'));
greenParcels = allenParcellations.greenData;
redParcels = allenParcellations.redData;


% load spike2 data
load(fullfile(datasets(i).folder,datasets(i).name,'pSpike2Data.mat'),'pSpike2Data');

% load facemap data
facemapFile = dir(fullfile(datasets(i).folder,datasets(i).name,'*proc.mat'));
facemap = load(fullfile(facemapFile(1).folder,facemapFile(1).name));
% account for python and matlab facemamp differences 
if isfield(facemap,'proc')
    temp = facemap.proc.motSVD{end};
    pc = temp(:,1);
else
    pc = facemap.motSVD_1(:,1);
end
% correct for sign of pc (depends on lighting conditions)
if skewness(pc) < 0
    pc = -pc;
end

% load facial video (assumes only 1 avi file in the folder)
facialVideoFile = dir(fullfile(datasets(i).folder,datasets(i).name,'*.avi'));
v = VideoReader(fullfile(facialVideoFile.folder,facialVideoFile.name));

%% align data

% define meso timestamps
mesoTimestamps = pSpike2Data.blueOnTimestamps;

% % align pupil area data to meso data
% alignedPupilArea = alignBehaviorData(mesoTimestamps,pupilData.pupilArea,pSpike2Data.pupilFrameOnTimestamps(1:length(pupilData.pupilArea)));

% align facial motion data to meso data
alignedPC = alignBehaviorData(mesoTimestamps,zscore(pc),pSpike2Data.pupilFrameOnTimestamps(1:length(pc)));

% align locomotion data to meso data
alignedSpeed = alignBehaviorData(mesoTimestamps,pSpike2Data.wheelSpeed,(0:length(pSpike2Data.wheelSpeed)-1)/5000);

%% create parcel outlines

load('parcels_updated12522.mat'); % load allen atlas
parcelEdges = {};
for i =1:size(allen_parcels.indicators,3)
    parcelEdges{end+1} = cell2mat(bwboundaries(allen_parcels.indicators(:,:,i)));
end

bilatDesiredInds = true(56,1);
bilatDesiredInds(21:26) = false;
bilatDesiredInds(53:end) = false;

parcellation_map = nan(256);
desired_parcels = find(bilatDesiredInds);
for i = 1:length(desired_parcels)
    parcellation_map(allen_parcels.indicators(:,:,desired_parcels(i))>0) = i;
end

mask = parcellation_map>0;

mask(:,256/2-(middle_bar_width/2-1):256/2+middle_bar_width/2) = false;


for p1 = 1:256
    for p2 = 1:256
        if ~mask(p1,p2)
            greenData(p1,p2,:) = nan;
            redData(p1,p2,:) = nan;
        end
    end
end

%% define timeframe we want to
% plot

% interactive way
t=tiledlayout(4,1);
ax(1) = nexttile;
plot(redParcels(1,:));
ax(2) = nexttile;
plot(greenParcels(1,:));
ax(3) = nexttile;
plot(alignedSpeed)
ax(4) = nexttile;
plot(alignedPC);

linkaxes(ax,'x');
title(t,'Explore! Find an interesting segment of time.')
input('Ready to select start and end points for the video?');
title(t,'Select start and end points for the video.')

[x,~] = ginput(2);
startInd = round(min(x));
endInd = round(max(x));




%% facial video is a bit different, cant load whole video into memory, so wait till we know what time frame we want, then load those frames

% this gives us the index of meso data that is closest to each respective
% facial video frame [ie alignedDataInds(1200) gives us the index of meso
% data that is closest to the 1200th facial video frame]
alignedDataInds = alignBehaviorDataInds(mesoTimestamps,pSpike2Data.pupilFrameOnTimestamps);

[~,facialStartInd] = min(abs(alignedDataInds-startInd));

frames = read(v,[facialStartInd facialStartInd+(endInd-startInd)]);

%% masking variable, needed for dropped frames
nanGoodData = false(size(mesoTimestamps));
nanGoodData(startInd:endInd) = true;
nanGoodData(isnan(redParcels(1,:)) | isnan(greenParcels(1,:))) = false;
%% define data structures to pass to function that specify plotting data and parameters
alignedMovieData = struct;
alignedTraceData = struct;

alignedMovieData.facialData = struct;
alignedMovieData.redData = struct;
alignedMovieData.greenData = struct;

alignedMovieData.facialData.data = frames(:,:,:,~isnan(redParcels(1,startInd:endInd)) & ~isnan(greenParcels(1,startInd:endInd)));
alignedMovieData.facialData.plotType = 'imshow';


alignedMovieData.redData.data = redData(:,:,nanGoodData);
alignedMovieData.redData.plotType = 'imagesc';
alignedMovieData.redData.colormap = [ones(1,3); parula(256)];
alignedMovieData.redData.title = 'Ca';
alignedMovieData.redData.OutlineParcels = parcelEdges([1,51]);

alignedMovieData.greenData.data = greenData(:,:,nanGoodData);
alignedMovieData.greenData.plotType = 'imagesc';
alignedMovieData.greenData.colormap = [ones(1,3); parula(256)];
alignedMovieData.greenData.title = 'Ne2m';
alignedMovieData.greenData.OutlineParcels = parcelEdges([1,51]);

alignedTraceData.ca = struct;
alignedTraceData.ca.data = [redParcels(1,nanGoodData); redParcels(51,nanGoodData)]';
alignedTraceData.ca.ylim = [-0.02 0.05];
alignedTraceData.ca.ylabel = 'Ca';

alignedTraceData.ne = struct;
alignedTraceData.ne.data = [greenParcels(1,nanGoodData); greenParcels(51,nanGoodData)]';
alignedTraceData.ne.ylim = [-0.02 0.02];
alignedTraceData.ne.ylabel = 'Ne2m';

alignedTraceData.locomotion = struct;
alignedTraceData.locomotion.data = alignedSpeed(nanGoodData);
alignedTraceData.locomotion.ylim = [0 60];
alignedTraceData.locomotion.ylabel = 'Wheel Speed';

alignedTraceData.facialMotion = struct;
alignedTraceData.facialMotion.data = alignedPC(nanGoodData);
alignedTraceData.facialMotion.ylim = [-2 4];
alignedTraceData.facialMotion.ylabel = 'Facial Motion';

createVidTraceMovie(alignedMovieData,alignedTraceData,'test.avi','Position',[100 100 800 900])
