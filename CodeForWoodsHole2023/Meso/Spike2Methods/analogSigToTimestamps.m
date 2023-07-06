function  [onTimestamps, offTimestamps] = analogSigToTimestamps(inSig,inTime)
%%analogSigToTimestamps transform 2 state analog signal to timestamps 
%   Employs thresholds halfway between min and max values to convert an analog signal into timestamps.
%   inSig: analog signal states are extracted from
%   inTime: time vector used to create timestamps for state transitions

stateVect = inSig>min(inSig,[],'omitnan')+0.5*(max(inSig,[],'omitnan')-min(inSig,[],'omitnan')); % nanmean + thresh to fix weird negative voltage visual triggers

% transform into state changes
stateChange = diff(stateVect); 
% get time of state changes
onTimestamps = inTime(stateChange==1); 
% get time of state changes
offTimestamps = inTime(stateChange==-1);  

if ~isempty(onTimestamps) && ~isempty(offTimestamps)
    % if assigned states "wrong", flip on and off timestamps
    if onTimestamps(1) > offTimestamps(1) 
        tempTimestamps = onTimestamps;
        onTimestamps = offTimestamps;
        offTimestamps = tempTimestamps;
    end
end

% program stopped mid state, so delete last rising edge
if length(onTimestamps)-length(offTimestamps)==1 
    onTimestamps(end)=[];
end

