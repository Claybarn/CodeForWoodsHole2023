function averagedMat = nanAverageContinuousDataToEvents(dataTimestamps,data,eventTimestamps,preSamples,postSamples,parametricAxis)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
dataShape = size(data);

dimensionInds = 1:length(dataShape);
dimensionInds(dimensionInds==parametricAxis) = [];


permutedData = permute(data,[parametricAxis dimensionInds]);

dataShapeSansTime = dataShape;
dataShapeSansTime(parametricAxis) = [];
averagedMat = zeros([preSamples+postSamples; dataShapeSansTime(:)]',class(data));
added = zeros(size(averagedMat));
for i =1:length(eventTimestamps)
    ind = findInSorted(dataTimestamps,eventTimestamps(i));
    if (ind - preSamples > 0) && (ind + postSamples <= size(data,parametricAxis))
        currData = reshape(permutedData(ind-preSamples:ind+postSamples-1,:),size(averagedMat));
        nanMask = ~isnan(currData);
        averagedMat(nanMask) = averagedMat(nanMask) + currData(nanMask);
        added = added + ~isnan(currData);
    end
end
 averagedMat= averagedMat./added;
 
dimensionInds = 1:length(size(averagedMat));
dimensionInds(dimensionInds==1) = [];

insert = @(a, x, n)cat(2,  x(1:n), a, x(n+1:end));
if parametricAxis > length(dimensionInds)
    dimensionInds(end+1) = 1;
else
    dimensionInds = insert(1,dimensionInds,parametricAxis);
end


 averagedMat = permute(averagedMat,dimensionInds);
