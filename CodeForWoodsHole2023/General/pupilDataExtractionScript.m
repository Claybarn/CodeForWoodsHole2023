
path = 'C:\Users\Clayton\Downloads\pupilVisualFieldMappingTest_mouse2\pupilVisualFieldMappingTest_mouse2\inv_stim';
filename = 'pupil.avi';


medFiltWind = 5;
meanFiltWind = 7;
saccadeSDThresh = 4;
vIn = VideoReader(fullfile(path,filename));

happy = false;
while ~happy
    figure(123);
    happyFrame = read(vIn,sort(randsample(vIn.NumFrames,1)));
    imshow(happyFrame);
    title('Select a good frame for eye cropping')
    x = input('Are you happy with this frame? (y/n): ','s');
    if strcmp(x,'y')
        happy = true;
    end
end

happy = false;
while ~happy
    figure(123);
    imshow(happyFrame);
    title('Select corners to crop the eye')
    [xs,ys]=ginput(4);

    % round so whole numbers for indexing
    xs = round(xs);
    ys = round(ys);

    % get extremes of points picked
    minX = min(xs);
    maxX = max(xs);
    minY = min(ys);
    maxY = max(ys);

    figure(123);
    imshow(happyFrame(minY:maxY,minX:maxX,:));
    title('Cropped eye')
    x = input('Are you happy with this cropping? (y/n): ','s');
    if strcmp(x,'y')
        happy = true;
    end
end



% 
% 
% nameParts = strsplit(filename,'.'); % split file ending (.avi, etc) from file name so we can append an 'eyecropped' tag
% 
% vOut = VideoWriter(fullfile(path,[nameParts{1} '-eyecropped.' nameParts{2}]),'Motion JPEG AVI');
% vOut.Quality = 100;
% vOut.FrameRate = vIn.frameRate;
% open(vOut);
% 
% vIn = VideoReader(fullfile(path,filename));
% while hasFrame(vIn)
%    frame = readFrame(vIn);
%    writeVideo(vOut,histeq(frame(minY:maxY,minX:maxX,1)));
% end



vIn = VideoReader(fullfile(path,filename));
pupilLoc = zeros(vIn.numFrames,2);
pupilArea =  zeros(vIn.numFrames,1);
pupilMetric = zeros(vIn.numFrames,1);
pupilLum = zeros(vIn.numFrames,1);
wb = waitbar(0,'Extracting pupil data...');

[X,Y]=meshgrid(1:maxX-minX+1,1:maxY-minY+1);
for i =1:vIn.numFrames
   frame = readFrame(vIn);
   currFrame = frame(minY:maxY,minX:maxX,1);
    [centers, radii,metric] = imfindcircles(imgaussfilt(histeq(currFrame),2),[6 60],'ObjectPolarity','dark');
    if length(metric) == 1
        pupilArea(i) = radii.^2*pi;
        pupilLoc(i,:) = centers;
        pupilMetric(i) = metric;
        pupilLum(i) = min(currFrame(sqrt((X-centers(1)).^2+(-Y+centers(2)).^2)<=radii),[],'all');
    elseif length(metric) == 0
        pupilArea(i) = nan;
        pupilLoc(i,:) = nan;
        pupilMetric(i) = nan;
    else 
        lum = zeros(size(metric));
        for ii =1:length(metric)
            lum(ii) = min(currFrame(sqrt((X-centers(ii,1)).^2+(-Y+centers(ii,2)).^2)<=radii(ii)),[],'all');
        end
        [pupilLum(i),ind] = min(lum); % optimize for largest circle and circle metric
        
        pupilArea(i) = radii(ind).^2*pi;
        pupilLoc(i,:) = centers(ind,:);
        pupilMetric(i) = metric(ind);
    end
    waitbar(i./vIn.numFrames,wb,'Extracting pupil data...');
end

%% filter pupil data
fPupilArea = filterPupilData(pupilArea,medFiltWind,meanFiltWind);
fPupilLoc(:,1) = filterPupilData(pupilLoc(:,1),meanFiltWind,0);
fPupilLoc(:,2) = filterPupilData(pupilLoc(:,2),meanFiltWind,0);

%% extract saccades (timestamp, magnitude and direction)
pupilMovements = ((fPupilLoc(1:end-1,1)-fPupilLoc(2:end,1)).^2 + (fPupilLoc(1:end-1,2)-fPupilLoc(2:end,2)).^2).^.5;

moveBreakThresh = zscore(pupilMovements)>saccadeSDThresh;
saccadeInds = find(diff(moveBreakThresh)==1)+1;
saccadeMagnitudes = pupilMovements(saccadeInds);

deltaX = diff(fPupilLoc(:,1));
deltaY = flip(diff(flip(fPupilLoc(:,2)))); % since image and y increases from top to bottom
saccadeDirections = atan2(deltaY, deltaX);

%polarhistogram(saccadeDirections(saccadeInds),20)

saccadeFrameInds = saccadeInds + 1;

%% save data
pupilData.pupilArea = fPupilArea;
pupilData.pupilLoc = fPupilLoc;
pupilData.saccadeMagnitudes = saccadeMagnitudes;
pupilData.saccadeDirections = saccadeDirections;
pupilData.saccadeFrameInds = saccadeFrameInds;

save(fullfile(path,'pupilData.mat'),'pupilData');


figure
imshow(frame(minY:maxY,minX:maxX,1))

viscircles(centers(ind,:),radii(ind))






vIn = VideoReader(fullfile(path,filename));

for i =1:vIn.numFrames
   frame = readFrame(vIn);
   [centers, radii,metric] = imfindcircles(imgaussfilt(histeq(frame(minY:maxY,minX:maxX,1)),2),[6 60],'ObjectPolarity','dark');
   imshow(frame(minY:maxY,minX:maxX,1))
    if length(metric) == 1
        viscircles(centers,radii);
    elseif isempty(metric)
        continue
    else 
        [~,ind] = max(radii./max(radii)+metric./max(metric)); % optimize for largest circle and circle metric
        viscircles(centers(ind,:),radii(ind));
   end
pause(0.05)
end

