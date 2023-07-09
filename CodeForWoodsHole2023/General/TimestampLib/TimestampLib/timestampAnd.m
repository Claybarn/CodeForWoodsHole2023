function [returnOn,returnOff] = timestampAnd(timestamp1On,timestamp1Off,timestamp2On,timestamp2Off)
%TIMESTAMPAND returns timetamps corresponding to the on and off times of overlap
 % takes the max on time and min off time to find timestamps of overlap
    returnOn = [];
    returnOff = [];
    if timestampIntersect(timestamp1On,timestamp1Off,timestamp2On,timestamp2Off)
        returnOn = max([timestamp1On timestamp2On]);
        returnOff = min([timestamp1Off timestamp2Off]);
    end
end