function corrs = corrPairwiseContinuousData(data1,data2,parametricAxis,varargin)
%corrPairwiseContinuousData correlates data1 with data2 pairwise
%
% Arguments:
%
% data1: timeseries data
% data2: timeseries data 
% parametricAxis: the axis in data dataTimestamps corresponds to
% omitnan (optional): logical, to ignore nan values in correlation 
%   (default: false)



p = inputParser;
addOptional(p,'omitnan',false,@islogical);
parse(p,varargin{:})

% data1
dataShape1 = size(data1);
dimensionInds1 = 1:length(dataShape1);
dimensionInds1(dimensionInds1==parametricAxis) = [];
permutedData1 = permute(data1,[parametricAxis dimensionInds1]);
data1ShapeSansTime = dataShape1;
data1ShapeSansTime(parametricAxis) = [];
matrixifiedData1 = reshape(permutedData1,size(permutedData1,1),[]);

% data2
dataShape2 = size(data2);
dimensionInds2 = 1:length(dataShape2);
dimensionInds2(dimensionInds2==parametricAxis) = [];
permutedData2 = permute(data2,[parametricAxis dimensionInds2]);
dataShapeSansTime2 = dataShape2;
dataShapeSansTime2(parametricAxis) = [];
matrixifiedData2 = reshape(permutedData2,size(permutedData2,1),[]);


if data1ShapeSansTime ~= dataShapeSansTime2
    error(['data1 and data2 must have the same shape (not including the time dimension), but have shapes ' num2str(data1ShapeSansTime) ' and ' num2str(dataShapeSansTime2) ' respectively.'])
end

corrs = nan(size(matrixifiedData1,2),size(matrixifiedData2,2));

if p.Results.omitnan
    for i = 1:size(corrs,1)
        for ii = 1:size(corrs,2)
            corrs(i,ii) = nanCorr(matrixifiedData1(:,i),matrixifiedData2(:,ii));
        end
    end
else
    for i = 1:size(corrs,1)
        for ii = 1:size(corrs,2)
            corrs(i,ii) = regCorr(matrixifiedData1(:,i),matrixifiedData2(:,ii));
        end
    end
end

dimensionInds = 1:length(size(corrs));
dimensionInds(dimensionInds==1) = [];

insert = @(a, x, n)cat(2,  x(1:n), a, x(n+1:end));
if parametricAxis > length(dimensionInds)
    dimensionInds(end+1) = 1;
else
    dimensionInds = insert(1,dimensionInds,parametricAxis);
end


corrs = permute(corrs,dimensionInds);


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
