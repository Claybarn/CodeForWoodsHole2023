function mov_trans = transform_frames(mov, tform, R, C)
mov_trans = zeros(size(mov));
if size(mov_trans,3) ~= 1
    sz = size(mov(:,:,1));
    for k=1:size(mov,3)
        
        mov_trans(:,:,k)=imwarp(mov(:,:,k),tform,'OutputView',imref2d(sz),'Fillvalues',nan);
    end
    
else
    if ~exist('R','var') || ~exist('C','var')
        R=256;
        C=256;
    end
    for k=1:size(mov,2)
        x = reshape(mov(:,k),R,C);
        x_trans=imwarp(x,tform,'OutputView',imref2d([R C]),'Fillvalues',nan);
        mov_trans(:,k) = x_trans(:);
    end
    
end
