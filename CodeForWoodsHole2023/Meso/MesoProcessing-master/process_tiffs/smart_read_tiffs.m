function mov = smart_read_tiffs(tiffsPath,resizeRatio,n)

% get tiffs to load and sort them 
tiffs=dir(fullfile(tiffsPath, '*.tif'));
names = {tiffs.name};
[~,ndx] = natsort(names);
tiffs=tiffs(ndx);

% see what final size of image will be by loading a sample
im = single(imread(fullfile(tiffs(1).folder,tiffs(1).name)));
im = imresize(im, resizeRatio, 'nearest');
[R,C] = size(im);
if R ~= C
    R=256;
    C=256;
    dual = true;
else
    dual = false;
end
if n < 0 || n > length(tiffs)
    n = length(tiffs);
end
    
% preallocate movie variable space
mov = zeros(R,C,n,'single');
size(mov)
wb = waitbar(0,'Loading tiffs...');
if dual
    % iterate through each file loading and performing relevant alterations
    % if dual, we take the side of the image that is the brightest (Current
    % way we do dual imaging)
    for i =1:n
        im = imread(fullfile(tiffs(i).folder,tiffs(i).name));
        im = imresize(im, resizeRatio, 'nearest');
        luminance = zeros(2);
        luminance(1) = sum(im(:,1:C),'all');
        luminance(2) = sum(im(:,C+1:end),'all');
        [~,maxLumInd] = max(luminance);
        im = im(:,((maxLumInd-1)*C+1):(maxLumInd*C));        
        mov(:,:,i) = im;
        waitbar(i/n,wb,'Loading tiffs...');
    end
else
    % iterate through each file loading and performing relevant alterations
    for i =1:n
        im = imread(fullfile(tiffs(i).folder,tiffs(i).name));
        %im1 = (im(1:2:end,1:2:end)+im(2:2:end,1:2:end)+im(1:2:end,2:2:end) + im(2:2:end,2:2:end))/4;
        im = imresize(im, resizeRatio, 'nearest');
        mov(:,:,i) = im;
        waitbar(i/n,wb,'Loading tiffs...');
    end
end