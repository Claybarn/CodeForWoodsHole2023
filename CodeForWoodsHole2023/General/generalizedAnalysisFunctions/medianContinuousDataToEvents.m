function medianMat = medianContinuousDataToEvents(dataTimestamps,data,eventTimestamps,preSamples,postSamples,parametricAxis,varargin)
%medianContinuousDataToEvents finds the median of observations for any data shape.
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



p = inputParser;
addOptional(p,'omitnan',false,@islogical);
parse(p,varargin{:})


dataShape = size(data);

dimensionInds = 1:length(dataShape);
dimensionInds(dimensionInds==parametricAxis) = [];


permutedData = permute(data,[parametricAxis dimensionInds]);

dataShapeSansTime = dataShape;
dataShapeSansTime(parametricAxis) = [];
medianMat = nan(length(eventTimestamps),preSamples+postSamples,prod(dataShapeSansTime(:)),class(data));
for i =1:length(eventTimestamps)
    ind = findInSorted(dataTimestamps,eventTimestamps(i));
    if (ind - preSamples > 0) && (ind + postSamples <= size(data,parametricAxis))
        medianMat(i,:) = reshape(permutedData(ind-preSamples:ind+postSamples-1,:),1,[]);
    end
end
if p.Results.omitnan
    medianMat = reshape(median(medianMat,1,'omitnan'),preSamples+postSamples,prod(dataShapeSansTime(:)));
else
    medianMat = reshape(median(medianMat,1),preSamples+postSamples,prod(dataShapeSansTime(:)));
end

end