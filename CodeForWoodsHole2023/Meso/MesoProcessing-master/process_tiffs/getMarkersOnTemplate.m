function [BW, up, down, left, right] = getMarkersOnTemplate(frame, hl)
BW = zeros(size(frame));
MAXITERS = 10;
for i = 1:MAXITERS
    imagesc(frame);
hold all;

title('Select up');
up = ginput(1);
plot(up(1), up(2),'rx');
title('Select down');
down = ginput(1);
plot(down(1), down(2),'rx');
title('Select left');
left = ginput(1);
plot(left(1), left(2),'rx');

title('Select right');
right = ginput(1);
plot(right(1), right(2),'rx');

plot([up(1) down(1)], [up(2) down(2)],'k');
plot([left(1) right(1)], [left(2) right(2)],'k');
answer = questdlg('Is this OK?', ...
	'', ...
	'Yes','No','Yes');
% Handle response
switch answer
    case 'Yes'
        title('');
        up = round(up);
        down = round(down);
        right = round(right);
        left = round(left);
        
         x=linspace(0,1,100);
        y = round(bsxfun(@times, x(:),up)+ bsxfun(@times, 1-x(:),down));
        inds = sub2ind(size(BW), y(:,2),y(:,1));
        BW(inds) = 1; 
        
        y = round(bsxfun(@times, x(:),right)+ bsxfun(@times, 1-x(:),left));
        inds = sub2ind(size(BW), y(:,2),y(:,1));
        BW(inds) = 1; 
        
        
        
        return;
    case 'No'
        %continue;
end
end
error('Could not get markers correct');