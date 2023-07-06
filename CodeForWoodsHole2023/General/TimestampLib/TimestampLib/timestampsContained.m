function [timestampsContainedOn,timestampsContainedOff] = timestampsContained(timestamps1On,timestamps1Off,timestamps2On,timestamps2Off)
%TIMESTAMPSCONTAINED returns the timestamps for timestamps1 contained
%within timestamps2
%   check if timestamp1 contained in any timestamp2 (could do more
%   efficiently with any() but keeping for stylistic consistency.

timestampsContainedOn = [];
timestampsContainedOff = [];


for i = 1:length(timestamps1Off)
    for ii = 1:length(timestamps2Off)
        [andOn,andOff] = timestampContained(timestamps1On(i),timestamps1Off(i),timestamps2On(ii),timestamps2Off(ii));
        if ~isempty(andOn)
            timestampsContainedOn(end+1) = andOn;
            timestampsContainedOff(end+1) = andOff;
        end
    end
end


