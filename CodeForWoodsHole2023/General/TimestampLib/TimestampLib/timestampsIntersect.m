
function boolVect = timestampsIntersect(timestamps1On,timestamps1Off,timestamps2On,timestamps2Off)
%timestampsIntersect returns boolean vector corresponding to timestamps1. Values are true if
%timestamps1On(i) and timestamps1Off(i) overlaps with any timestamps2On and
%timestamps2Off, false otherwise
%  Asuming timestamps are sorted, quicker to use findInSorted. Returns
%  index of 
boolVect = false(size(timestamps1On));
for i =1:length(timestamps1On)    
        
    ind = findInSorted(timestamps2Off,timestamps1On(i));
    boolVect(i) = timestampIntersect(timestamps1On(i),timestamps1Off(i),timestamps2On(ind),timestamps2Off(ind));
    if ind-1>0
        boolVect(i) =  boolVect(i) | timestampIntersect(timestamps1On(i),timestamps1Off(i),timestamps2On(ind-1),timestamps2Off(ind-1));
    end
end
end
