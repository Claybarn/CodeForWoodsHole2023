function [tform,R,C] = get_alignment_transform_by_mid(maxframe)

load('../parcellation/parcells_updated121519.mat','parcells_new');
Allparcells = parcells_new.CombinedParcells;

[R,C] = size(Allparcells);
allentemplate = zeros(size(Allparcells,1),size(Allparcells,2));
figure;subplot(2,2,1);
imagesc(Allparcells);colormap gray;
hold all;plot([128 128 85 169],[89 226 226 226]-13,'r*');
BWtemplate=zeros(R,C);
x=linspace(0,1,100);
y = round(bsxfun(@times, x(:),[85 226])+ bsxfun(@times, 1-x(:),[169 226]));
inds = sub2ind(size(BWtemplate), y(:,2),y(:,1));
BWtemplate(inds) = 1;
 y = round(bsxfun(@times, x(:),[128 85])+ bsxfun(@times, 1-x(:),[128 226]));
inds = sub2ind(size(BWtemplate), y(:,2),y(:,1));
BWtemplate(inds) = 1;
set(gcf,'Position',[ 1          41        1920         963]);
        
subplot(2,2,2);
[BW2trans, up, down, left, right] = getMarkersOnTemplate(maxframe);

[optimizer, metric] = imregconfig('multimodal');
metric.NumberOfSpatialSamples = 2000;
metric.NumberOfHistogramBins = 10;
metric.UseAllPixels = false;
optimizer.GrowthFactor = 1.02000;
optimizer.Epsilon = 1.50000e-06;
optimizer.InitialRadius = 6.25000e-03;
optimizer.MaximumIterations = 1000;
subplot(2,2,3);
imshowpair(BWtemplate, BW2trans,'Scaling','joint')
[optimizer, metric] = imregconfig('multimodal');
optimizer.InitialRadius = 0.0009;
optimizer.Epsilon = 1.5e-4;
optimizer.GrowthFactor = 1.01;
optimizer.MaximumIterations = 300;
tform = imregtform(BW2trans, BWtemplate, 'affine', optimizer, metric);
movingRegistered = imwarp(BW2trans,tform,'OutputView',imref2d(size(BWtemplate)));
frame_new=imwarp(maxframe,tform,'OutputView',imref2d(size(BW2trans)),'Fillvalues',0);


subplot(2,2,3);
imshowpair(BWtemplate, movingRegistered,'Scaling','joint')


subplot(2,2,4);
plot_parcell_overlay(frame_new(:),R,C,true,parcells_new.indicators);


