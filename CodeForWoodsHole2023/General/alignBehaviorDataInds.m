function alignedDataInds = alignBehaviorDataInds(mesoTimestamps,behaviorTimestamps)
% want to match face camera to meso data 
% find nearest meso timestamp to each face camera timestamp and get the index of the nearest meso timestamp 
alignedDataInds = nan(size(behaviorTimestamps));

for i =1:length(behaviorTimestamps)
    % fast method to find what ind of mesoTimestamps behaviorTimestamps(i)
    % is >=
    ind = findInSorted(mesoTimestamps,behaviorTimestamps(i));
    % check to see if timestamps is closer to found ind or previous
    if ind > 1
        [~,ind2] = min([abs(mesoTimestamps(ind)-behaviorTimestamps(i)),abs(mesoTimestamps(ind-1)-behaviorTimestamps(i))]);
        temp = [ind ind-1];
        ind = temp(ind2);
    end
    alignedDataInds(i) = ind;
end


