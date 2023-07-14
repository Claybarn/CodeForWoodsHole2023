function alignmentPointMoveEvent(src,evt)
    tp = ancestor(src,'figure','toplevel');
    points = findobj('Type','images.roi.Point');
    transformedHandle = findobj('UserData',true,'Type','image');
    movingPoints = zeros(3,2);
    fixedPoints = zeros(3,2);
    for p =1:length(points)
        if strcmp(points(p).UserData(1),'m')
            movingPoints(str2num(points(p).UserData(2)),:) = points(p).Position;
        else
            fixedPoints(str2num(points(p).UserData(2)),:) = points(p).Position;
        end
    end

    tform = fitgeotform2d(movingPoints,fixedPoints,"similarity");
    transformed = imwarp(tp.UserData{2},tform,OutputView=imref2d(size(tp.UserData{1})));
    assignin('base', 'tform', tform );
    assignin('base', 'transformed', transformed);
    set(transformedHandle,'Cdata',imfuse(transformed,tp.UserData{1}));
    
end
