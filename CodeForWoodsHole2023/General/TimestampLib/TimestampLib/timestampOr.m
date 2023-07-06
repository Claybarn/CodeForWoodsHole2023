function [returnOn,returnOff] = timestampOr(timestamp1On,timestamp1Off,timestamps2On,timestamp2Off)
%TIMESTAMPAND returns timetamps corresponding to the on and off times of
%any occurence
 % takes the min on time and max off time to find timestamps of any
 % occurence
    returnOn = [];
    returnOff = [];
    if timestampIntersect(timestamp1On,timestamp1Off,timestamps2On,timestamp2Off)
        returnOn = min([timestamp1On timestamp2On]);
        returnOff = max([timestamp1Off timestamp2Off]);
    end
end