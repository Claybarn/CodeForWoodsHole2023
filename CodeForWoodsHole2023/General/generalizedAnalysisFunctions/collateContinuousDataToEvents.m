function collatedMat = collateContinuousDataToEvents(dataTimestamps,data,eventTimestamps,preSamples,postSamples,parametricAxis)
%collateContinuousDataToEvents collates continuous data to events
%
% Arguments:
%
% dataTimestamps: the timestamp for each sample in data
% data: the timeseries data 
% eventTimestamps: the event timestamps 
% preSamples: the number of samples kept before the event
% postSamples: the number of samples kept after the event
% parametricAxis: the axis in data dataTimestamps corresponds to
%   


dataShape = size(data);

dimensionInds = 1:length(dataShape);
dimensionInds(dimensionInds==parametricAxis) = [];


permutedData = permute(data,[parametricAxis dimensionInds]);

dataShapeSansTime = dataShape;
dataShapeSansTime(parametricAxis) = [];
collatedMat = nan(length(eventTimestamps),preSamples+postSamples,prod(dataShapeSansTime(:)),class(data));
for i =1:length(eventTimestamps)
    ind = findInSorted(dataTimestamps,eventTimestamps(i));
    if (ind - preSamples > 0) && (ind + postSamples <= size(data,parametricAxis))
        collatedMat(i,:) = reshape(permutedData(ind-preSamples:ind+postSamples-1,:),1,[]);
    end
end

collatedMat = reshape(collatedMat,[length(eventTimestamps) preSamples+postSamples dataShapeSansTime]);

end