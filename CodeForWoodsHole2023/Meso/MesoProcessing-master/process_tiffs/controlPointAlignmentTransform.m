function [tform,R,C] = controlPointAlignmentTransform(inSig)
    % inSig: video data to align to Allen template.
    %does control point matching by selecting points on moving image (insig) a
    % and template(parcellated template) . Need to select at least three points
    load('parcells_updated121519.mat','parcells_new');
    template = parcells_new.CombinedParcells;
    [R,C] = size(template);
    template = template-min(template,[],'all');
    template = template/max(template,[],'all');
    shaped = reshape(inSig,R,C,[]); % if data is not downsampled, this might throw error (i.e. if not 256 by 256)
    maxFrame = max(shaped,[],3);
    maxFrame = (maxFrame-min(maxFrame,[],'all'));
    maxFrame = maxFrame/max(maxFrame,[],'all');
    disp('Please select control points on the brain and close GUI when done');
    rerun = true;
    while rerun
        [movingPoints, fixedPoints]=cpselect(maxFrame,template,parcells_new.movingPoints,parcells_new.fixedPoints,'Wait',true);%call builit in GUI to select points 
        tform = fitgeotrans(movingPoints,fixedPoints,'Similarity');%similatiry based transformation
        Jregistered = imwarp(maxFrame,tform,'OutputView',imref2d(size(template)));
        figure;imshowpair(template,Jregistered,'diff')
        response = input('Are you satisfied with your selection? [y/n]:','s');
        parcells_new.movingPoints = movingPoints;
        if strcmp(response,'y')
            rerun = false;
        end
    end
end




