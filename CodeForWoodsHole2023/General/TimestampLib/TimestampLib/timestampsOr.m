function [timestampsAndOn,timestampsAndOff] = timestampsOr(timestamps1On,timestamp1Off,timestamps2On,timestamps2Off)
%locoFaceConjunctiveStates returns the timestamps corresponding to the
%of (high face and low locomotion), (high or low face and locomotion), and
%(low face and low locomotion)
%   Detailed explanation goes here

timestampsOrOn = [];
timestampsOrOff = [];


for i = 1:length(timestamp1Off)
    for ii = 1:length(timestamps2Off)
        [andOn,andOff] = timestampOr(timestamps1On (i),timestamp1Off(i),timestamps2On(ii),timestamps2Off(ii));
        if ~isempty(andOn)
            timestampsOrOn(end+1) = andOn;
            timestampsOrOff(end+1) = andOff;
        end
    end
end

