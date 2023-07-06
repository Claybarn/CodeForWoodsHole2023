function boolResult = timestampIntersect(timestamp1On,timestamp1Off,timestamp2On,timestamp2Off)
%TIMESTAMPOVERLAP returns true if on-off timestamps overlap in some period
% of time, false otherwise
%  Less work to take the inverse of the complement of the overlap, i.e.
%  opposite of (first timestamp came after the offset of the
%  second or if the off timestamp came before the onset of the second.
boolResult = ~(timestamp1On>timestamp2Off || timestamp1Off<timestamp2On);
end

