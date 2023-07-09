function [timestampsAndOn,timestampsAndOff] = timestampsAnd(timestamps1On ,timestamps1Off,timestamps2On,timestamps2Off)
%locoFaceConjunctiveStates returns the timestamps corresponding to the
%of (high face and low locomotion), (high or low face and locomotion), and
%(low face and low locomotion)
%   Detailed explanation goes here

timestampsAndOn = [];
timestampsAndOff = [];


for i = 1:length(timestamps1Off)
    for ii = 1:length(timestamps2Off)
        [andOn,andOff] = timestampAnd(timestamps1On(i),timestamps1Off(i),timestamps2On(ii),timestamps2Off(ii));
        if ~isempty(andOn)
            timestampsAndOn(end+1) = andOn;
            timestampsAndOff(end+1) = andOff;
        end
    end
end

