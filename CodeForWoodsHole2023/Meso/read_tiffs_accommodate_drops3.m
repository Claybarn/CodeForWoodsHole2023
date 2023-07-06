function mov = read_tiffs_accommodate_drops3(tiffsPath,fs,varargin)
%%READ_TIFFS_ACCOMMODATE_DROPS load tiff files froma a directory accounting for dropped frames
%
% Required input arguments:
%    tiffsPath: The directory where the tiff files can be found. 
%    fs: The sampling frequency of imaging (overall fs, not per channel)
%
%
% Optional input arguments and default values:
%    ResizeRatio: 1
%    KeepDual: False
%    NumFrames: The Number of detected tiff files
%    Order: [1] (always grab left size of image if not square, otherwise
%    grabs whole image)
%
%
% EXAMPLE USAGE:
% load all images and resize images by factor of 0.5:
% mov = read_tiffs_accommodate_drops(tiffsPath,20,'ResizeRatio',0.5)
%
% load 1000 images and resize:
% mov = read_tiffs_accommodate_drops(tiffsPath,20,'NumFrames',1000,'ResizeRatio',0.5);
% 
% load dual imaging data, keeping both size of the images
% mov = read_tiffs_accommodate_drops(tiffsPath,30,'NumFrames',1000,'ResizeRatio',0.5,'KeepDual',true);
% 
% load dual imaging data, keeping the left (1), left (1) then right (2) side of the image
% mov = read_tiffs_accommodate_drops(tiffsPath,30,'NumFrames',1000,'ResizeRatio',0.5,'Order',[1 1 2];
%
p = inputParser;
addParameter(p,'ResizeRatio',1)
addParameter(p,'KeepDual',false)
addParameter(p,'NumFrames',-1)
addParameter(p,'Order',[1])
parse(p,varargin{:})
% get tiffs to load and sort them 
tiffs=dir(fullfile(tiffsPath, '*.tif'));
names = {tiffs.name};
[~,ndx] = natsort(names);
tiffs=tiffs(ndx);
% see what final size of image will be by loading a sample (sniffing)
im = imread(fullfile(tiffs(1).folder,tiffs(1).name));
im = imresize(im, p.Results.ResizeRatio, 'nearest');
[R,C] = size(im);
dual = false;
if R ~= C && ~p.Results.KeepDual
    R=min(size(im));
    C=R;
    dual = true;
end
if p.Results.NumFrames<0
    n = length(tiffs);
else
    n = p.Results.NumFrames;
end
% get timestamp from first tiff file to find the number of initial frame
% drops and set up our drop detector
text = fileread(fullfile(tiffs(1).folder,tiffs(1).name));
k = strfind(text,'Time_From_Start = ');
timestamp = text(k+18:k+18+12);
prev_timestamp = str2double(timestamp(1:2))*60*60+str2double(timestamp(4:5))*60+str2double(timestamp(7:8))+str2double(timestamp(9:13));
offset = round(prev_timestamp/(1/fs));
% preallocate movie variable space
% num intended frames = sum(frameDiffs)+offset+1
if p.Results.NumFrames<0
    mov = nan(R,C,n+offset+1,'single');
else
    mov = nan(R,C,n,'single');
end
size(mov)
wb = waitbar(0,'Loading tiffs...');
it = 1+offset;
if dual && ~p.Results.KeepDual
    % iterate through each file loading and performing relevant alterations
    % if dual, we take the side of the image that is the brightest (Current
    % way we do dual imaging)
    for i =1:n
        text = fileread(fullfile(tiffs(i).folder,tiffs(i).name));
        k = strfind(text,'Time_From_Start = ');
        timestamp = text(k+18:k+18+12);
        timestamp = str2double(timestamp(1:2))*60*60+str2double(timestamp(4:5))*60+str2double(timestamp(7:8))+str2double(timestamp(9:13));

        im = imread(fullfile(tiffs(i).folder,tiffs(i).name));
        im = imresize(im, p.Results.ResizeRatio, 'nearest');
%         luminance = zeros(2);
%         luminance(1) = sum(im(:,1:C),'all');
%         luminance(2) = sum(im(:,C+1:end),'all');
%         [~,maxLumInd] = max(luminance);
        it = it + round((timestamp-prev_timestamp)/(1/fs));
        maxLumInd = p.Results.Order(mod(it-1,length(p.Results.Order))+1);
        im = im(:,((maxLumInd-1)*C+1):(maxLumInd*C));        
        mov(:,:,it) = im;    
        prev_timestamp = timestamp;
        if mod(i,1000) == 0
            disp(['Read ' num2str(i) ' frames']);
        end
        waitbar(i/n,wb,'Loading tiffs...');
    end
else
    % iterate through each file loading and performing relevant alterations
    for i =1:n
        text = fileread(fullfile(tiffs(i).folder,tiffs(i).name));
        k = strfind(text,'Time_From_Start = ');
        timestamp = text(k+18:k+18+12);
        timestamp = str2double(timestamp(1:2))*60*60+str2double(timestamp(4:5))*60+str2double(timestamp(7:8))+str2double(timestamp(9:13));
        im = imread(fullfile(tiffs(i).folder,tiffs(i).name));
        im = imresize(im, p.Results.ResizeRatio);
        it = it + round((timestamp-prev_timestamp)/(1/fs));
        mov(:,:,it) = im;
        prev_timestamp = timestamp;
        waitbar(i/n,wb,'Loading tiffs...');
    end
end
close(wb);
end