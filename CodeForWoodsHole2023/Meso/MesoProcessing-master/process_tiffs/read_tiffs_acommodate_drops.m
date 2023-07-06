function mov = read_tiffs_acommodate_drops(tiffsPath,offset,frameDiffs,resizeRatio)

% get tiffs to load and sort them 
tiffs=dir(fullfile(tiffsPath, '*.tif'));
names = {tiffs.name};
[~,ndx] = natsort(names);
tiffs=tiffs(ndx);

% see what final size of image will be by loading a sample
im = imread(fullfile(tiffs(1).folder,tiffs(1).name));
im = imresize(im, resizeRatio);
[R,C] = size(im);
dual = false;
if R ~= C
    R=256;
    C=256;
    dual = true;
end

n = length(tiffs);
    
% preallocate movie variable space
% num intended frames = sum(frameDiffs)+offset+1
mov = nan(R,C,sum(frameDiffs)+offset+1);
size(mov)
wb = waitbar(0,'Loading tiffs...');
it = 1+offset;
if dual
    % iterate through each file loading and performing relevant alterations
    % if dual, we take the side of the image that is the brightest (Current
    % way we do dual imaging)
    for i =1:n
        im = imread(fullfile(tiffs(i).folder,tiffs(i).name));
        im = imresize(im, resizeRatio);
        luminance = zeros(2);
        luminance(1) = sum(im(:,1:C),'all');
        luminance(2) = sum(im(:,C+1:end),'all');
        [~,maxLumInd] = max(luminance);
        im = im(:,((maxLumInd-1)*C+1):(maxLumInd*C));        
        mov(:,:,it) = im;
        if i <= length(frameDiffs)
            it = it + frameDiffs(i);
        end
        waitbar(i/n,wb,'Loading tiffs...');
    end
else
    % iterate through each file loading and performing relevant alterations
    for i =1:n
        im = imread(fullfile(tiffs(i).folder,tiffs(i).name));
        %im1 = (im(1:2:end,1:2:end)+im(2:2:end,1:2:end)+im(1:2:end,2:2:end) + im(2:2:end,2:2:end))/4;
        im = imresize(im, resizeRatio);
        mov(:,:,it) = im;
        if i <= length(frameDiffs)
            it = it + frameDiffs(i);
        end
        waitbar(i/n,wb,'Loading tiffs...');
    end
end