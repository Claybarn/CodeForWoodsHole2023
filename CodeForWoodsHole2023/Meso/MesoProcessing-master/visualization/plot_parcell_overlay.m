function plot_parcell_overlay(mov,R,C,plotSave,parcells)

meanx=nanmean(mov,2);
mov_trans=reshape(meanx,R,C);

imshow(mat2gray(mov_trans));hold on;
for k=1:size(parcells,3)
    E=parcells(:,:,k);
    [B,~] = bwboundaries(E);hold on;
    for k = 1:length(B)
        boundary = B{k};
        plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 0.1)
    end
end
hold off;

