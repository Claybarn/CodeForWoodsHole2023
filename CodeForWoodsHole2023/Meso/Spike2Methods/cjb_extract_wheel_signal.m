function [wheelVelocity,wheelOn,wheelOff] = cjb_extract_wheel_signal(wheelPosition,fs, WHEEL_DIRECTION, ...
    wind, wheelDiameter, runSpeedThresh, ...
    minRunSpeed, minRunTime)

wheelPosition = double(wheelPosition);
if ~exist('WHEEL_DIRECTION','var')
    WHEEL_DIRECTION = 1;
end
if ~exist('wind','var')
    wind = fs/2;
end
if ~exist('wheelDiameter','var')
    wheelDiameter = 15; % cm
end
if ~exist('runSpeedThresh','var')
    runSpeedThresh = 3; % cm/s
end
if ~exist('minRunSpeed','var')
    minRunSpeed = 20; % cm/s
end
if ~exist('minRunTime','var')
    minRunTime = 0.5; % s
end

% transform wheel postion to running speed
wheelPosition=fillmissing(wheelPosition,'nearest');
if WHEEL_DIRECTION==1
    wheelAngle=(wheelPosition-min(wheelPosition))/(max(wheelPosition)-min(wheelPosition))*2*pi;%change voltage to angle
elseif WHEEL_DIRECTION==-1
    wheelAngle=(1-(wheelPosition-min(wheelPosition))/(max(wheelPosition)-min(wheelPosition)))*2*pi;
end

% make wheel angle cumulative (remove sharp transitions)
wheelAngle = unwrap(wheelAngle);

% get wheel velocity by taking differential of wheel distance with some
% smoothing and filering
wheelVelocity = zeros(size(wheelAngle));
wheelVelocity(2:end) = movmean(medfilt1(diff(wheelAngle),3)*wheelDiameter*fs,wind);

% get wheel speed
wheelSpeed = abs(wheelVelocity);

% get running bouts
stateVect = wheelSpeed > runSpeedThresh;
timestamps = (0:length(wheelSpeed)-1)/fs;
[wheelOn,wheelOff] = analogSigToTimestamps(stateVect,timestamps);
[wheelOn,wheelOff] = mergeTimestamps(wheelOn,wheelOff,minRunTime);

maxRunSpeed = zeros(size(wheelOn));
for i =1:length(wheelOn)
    maxRunSpeed(i) = max(wheelSpeed(round(wheelOn(i)*fs+1):round(wheelOff(i)*fs+1)));
end

toRemove = (wheelOff-wheelOn)<minRunTime | maxRunSpeed < minRunSpeed;
wheelOn(toRemove) = [];
wheelOff(toRemove) = [];


