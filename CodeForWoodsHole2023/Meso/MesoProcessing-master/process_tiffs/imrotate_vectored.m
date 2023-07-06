function outsig = imrotate_vectored(insig,R, C, angle)
if angle == 0
    outsig=insig;
end
for k = 1:size(insig,2)
   x = reshape(insig(:,k), R, C);
   x = imrotate(x, angle);
   outsig(:,k) = x(:);    
end