function  verdict = isTimestampContained(timestamp1On,timestamp1Off,timestamp2On,timestamp2Off)
%TIMESTAMPCONTAINED returns timetamps if timestamp1 is contained within
%timestamp2, otherwise empty lists returned. 
 % Keeps format of other timestamp logic tools, so midly cumbersome, but
 % keeping for stylistic purposes.
    verdict = false;
    if timestamp1On>=timestamp2On && timestamp1Off<=timestamp2Off
        verdict = true;
    end
end