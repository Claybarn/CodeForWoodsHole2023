function verdict = areTimestampsContained(timestamps1On,timestamps1Off,timestamps2On,timestamps2Off)
%TIMESTAMPSCONTAINED returns true if the timestamps for timestamps1 are contained
%within timestamps2

verdict = false(size(timestamps1Off));
lastTrue = 1;
for i = 1:length(timestamps1Off)
    for ii = lastTrue:length(timestamps2Off)
        if isTimestampContained(timestamps1On(i),timestamps1Off(i),timestamps2On(ii),timestamps2Off(ii))
            verdict(i) = true;
            lastTrue = ii;
            break;
        end
    end
end