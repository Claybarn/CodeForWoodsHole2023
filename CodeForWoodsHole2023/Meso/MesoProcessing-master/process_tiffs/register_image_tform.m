function [tform,bloodVesselInfo] = register_image_tform(fixed,moving,BCOSFIREPath)
%%register_image(fixed,moving) regisers two mesoscopic images by vessels
% extracts blood vessels with the B-COSFIRE toolbox and then applies image
% registration.
% fixed and moving must be 2D images of equal dimensions

% normalize intensity of each image such that [0, 1]
fixedMinSub = fixed-min(fixed,[],'all');
fixedIm = ones(1,1,3).*fixedMinSub/max(fixedMinSub,[],'all');

movingMinSub = moving-min(moving,[],'all');
movingIm = ones(1,1,3).*movingMinSub/max(movingMinSub,[],'all');

% find blood vessels in each image
[fixedOutput, ~] = mesoBloodVesselSegmentation(fixedIm,BCOSFIREPath);
[movingOutput, ~] = mesoBloodVesselSegmentation(movingIm,BCOSFIREPath);

fixedResp = fixedOutput.respimage;
movingResp = movingOutput.respimage;
bloodVesselInfo.fixed = fixedOutput;
bloodVesselInfo.moving = movingOutput;

% register image
[optimizer,metric] = imregconfig('multimodal');
optimizer.InitialRadius = optimizer.InitialRadius/5;
optimizer.MaximumIterations = 500;
tform = imregtform(movingResp,fixedResp,'affine',optimizer,metric);
end