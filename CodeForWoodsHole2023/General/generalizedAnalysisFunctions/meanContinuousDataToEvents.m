function averagedMat = meanContinuousDataToEvents(dataTimestamps,data,eventTimestamps,preSamples,postSamples,parametricAxis,varargin)
%%averageContinuousDataToEvents averge timeseries of any shape to events
%
% Arguments:
%
% dataTimestamps: the timestamp for each sample in data
% data: the timeseries data that will be averaged
% eventTimestamps: the event timestamps data will be averaged to
% preSamples: the number of samples kept before the event
% postSamples: the number of samples kept after the event
% parametricAxis: the axis in data dataTimestamps corresponds to
% 



p = inputParser;
addOptional(p,'omitnan',false,@islogical);
parse(p,varargin{:})

dataShape = size(data);

% # of dimensions for input data
dimensionInds = 1:length(dataShape);
dimensionInds(dimensionInds==parametricAxis) = [];

% reorganize data to make it more ammenable to work with
permutedData = permute(data,[parametricAxis dimensionInds]);

dataShapeSansTime = dataShape;
dataShapeSansTime(parametricAxis) = [];
averagedMat = zeros([preSamples+postSamples; dataShapeSansTime(:)]',class(data));

if p.Results.omitnan
    % calculate mean ignoring nans
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
else
    % calculate mean
    added = 0;
    for i =1:length(eventTimestamps)
        ind = findInSorted(dataTimestamps,eventTimestamps(i));
        if (ind - preSamples > 0) && (ind + postSamples <= size(data,parametricAxis))
            averagedMat = averagedMat + reshape(permutedData(ind-preSamples:ind+postSamples-1,:),size(averagedMat));
            added = added + 1;
        end
    end
    averagedMat= averagedMat/added;
end


%% reformat data to orignal shape
dimensionInds = 1:length(size(averagedMat));
dimensionInds(dimensionInds==1) = [];

insert = @(a, x, n)cat(2,  x(1:n), a, x(n+1:end));
if parametricAxis > length(dimensionInds)
    dimensionInds(end+1) = 1;
else
    dimensionInds = insert(1,dimensionInds,parametricAxis);
end


 averagedMat = permute(averagedMat,dimensionInds);
