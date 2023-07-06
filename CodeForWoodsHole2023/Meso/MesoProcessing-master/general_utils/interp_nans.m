function sig = interp_nans(sig)
%% interp_nans(sig) takes 2D data (variables,samples), and linearly interpolates internal nans and sets edge nans equal to nearest nonnan

t = 1:size(sig,2);
for i =1:size(sig,1)
    nanInds = isnan(sig(i,:)); % find nan indices for this pixel
    if sum(nanInds)>0
        sig(i,nanInds) = interp1(t(~nanInds), sig(i,~nanInds), t(nanInds)); % iterpolate nans
        if sum(isnan(sig(i,1)))>0 % if started with a frame drop(s) (cant interpolate) set equal to first not nan
            firstNonNan = find(~isnan(sig(i,:)),1);
            sig(i,1:firstNonNan-1) = sig(i,firstNonNan);
        end
        if sum(isnan(sig(i,end)))>0
            lastNonNan = find(~isnan(sig(i,:)),1,'last');
            sig(i,lastNonNan+1:end) = sig(i,lastNonNan);
        end
    end
end


