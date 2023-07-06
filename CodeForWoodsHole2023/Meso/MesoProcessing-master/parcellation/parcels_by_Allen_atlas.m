function dFoF_parcels = parcels_by_Allen_atlas(dff_blue,parcells_new)
%load('../parcellation/parcells_updated121519.mat','parcells_new');
dff_blue=double(dff_blue);
dFoF_parcels = zeros(size(parcells_new.indicators, 3), size(dff_blue, 2));
for k=1:size(parcells_new.indicators, 3)
    inds = find(parcells_new.indicators(:,:,k) == 1);
   dFoF_parcels(k, :) = nanmean(dff_blue(inds, :),1); %#ok<FNDSB>
    
end
