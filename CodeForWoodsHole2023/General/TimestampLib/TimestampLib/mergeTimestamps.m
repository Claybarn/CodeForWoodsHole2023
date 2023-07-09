function [onTimestamps,offTimestamps] = mergeTimestamps(onTimestamps,offTimestamps,minInterDurThresh)
interDurs = onTimestamps(2:end)-offTimestamps(1:end-1);
[interDurs,I] = sort(interDurs);
while length(interDurs) > 1 & interDurs(1) < minInterDurThresh
    offTimestamps(I(1)) = offTimestamps(I(1)+1);
    onTimestamps(I(1)+1) = [];
    offTimestamps(I(1)) = [];
    interDurs = onTimestamps(2:end)-offTimestamps(1:end-1);
    [interDurs,I] = sort(interDurs);
end