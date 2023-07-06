function [xcorrs,lags] = xCorrContinuousData(refData,laggedData,maxLag,parametricAxis,varargin)
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

% reference
refDataShape = size(refData);
refDimensionInds = 1:length(refDataShape);
refDimensionInds(refDimensionInds==parametricAxis) = [];
permutedRefData = permute(refData,[parametricAxis refDimensionInds]);
refDataShapeSansTime = refDataShape;
refDataShapeSansTime(parametricAxis) = [];

% lagged
laggedDataShape = size(laggedData);
laggedDimensionInds = 1:length(laggedDataShape);
laggedDimensionInds(laggedDimensionInds==parametricAxis) = [];
permutedLaggedData = permute(laggedData,[parametricAxis laggedDimensionInds]);
laggedDataShapeSansTime = laggedDataShape;
laggedDataShapeSansTime(parametricAxis) = [];


if refDataShapeSansTime ~= laggedDataShapeSansTime
    error(['Reference and lagged data must have the same shape (not including the time dimension), but have shapes ' num2str(refDataShapeSansTime) ' and ' num2str(laggedDataShapeSansTime) ' respectively.'])
end


lags = -maxLag:maxLag;
xcorrs = nan([length(lags) refDataShapeSansTime],class(refData));

if p.Results.omitnan
    for i = 1:length(lags)
        if lags(i) <= 0
            xcorrs(i,:) = nanCorr(permutedRefData(1:end+lags(i),:),permutedLaggedData(1-lags(i):end,:));
        else
            xcorrs(i,:) = nanCorr(permutedRefData(1+lags(i):end,:),permutedLaggedData(1:end-lags(i),:));
        end
    end
else
    for i = 1:length(lags)
        if lags(i) <= 0
            xcorrs(i,:) = regCorr(permutedRefData(1:end+lags(i),:),permutedLaggedData(1-lags(i):end,:));
        else
            xcorrs(i,:) = regCorr(permutedRefData(1+lags(i):end,:),permutedLaggedData(1:end-lags(i),:));
        end    
    end

end

dimensionInds = 1:length(size(xcorrs));
dimensionInds(dimensionInds==1) = [];

insert = @(a, x, n)cat(2,  x(1:n), a, x(n+1:end));
if parametricAxis > length(dimensionInds)
    dimensionInds(end+1) = 1;
else
    dimensionInds = insert(1,dimensionInds,parametricAxis);
end


xcorrs = permute(xcorrs,dimensionInds);


end


function r = nanCorr(x,y)
    x_sub = bsxfun( @minus, x, mean(x,1,'omitnan') );
    y_sub = bsxfun( @minus, y, mean(y,1,'omitnan') );
    r = bsxfun( @rdivide, sum(bsxfun( @times, x_sub, y_sub),1,'omitnan'), bsxfun( @power, bsxfun( @times, sum(bsxfun(@power, x_sub, 2),1,'omitnan'), sum(bsxfun(@power, y_sub, 2),1,'omitnan')),0.5));
end

function r = regCorr(x,y)
    x_sub = bsxfun( @minus, x, mean(x,1) );
    y_sub = bsxfun( @minus, y, mean(y,1) );
    r = bsxfun( @rdivide, sum(bsxfun( @times, x_sub, y_sub),1), bsxfun( @power, bsxfun( @times, sum(bsxfun(@power, x_sub, 2),1), sum(bsxfun(@power, y_sub, 2),1)),0.5));
end
