function [wheel_speed, wheel_on, wheel_off,distance] = extract_wheel_speed_with_correction(wheel_position,fs,wheel_diameter,varargin)
%% EXTRACT_WHEEL_SPEED_WITH_CORRECTION converts voltage position to wheel speed while correction for nonlinearities in reported wheel position
% wheel_position: A vector containing continuous voltage measurements
%                   reflecting wheel position
% fs: The sampling frequency of wheel_position in Hz
% wheel_diameter: The diameter of the wheel in cm

%% Parse Parameters
min_peak_distance = round(fs*0.002);
fast_revolution_time = round(fs/2);
slow_revolution_time =  round(8/5*fs);
num_bins = 1000;
fc = .3;
min_locomotion_speed = 5;
w = gausswin(1250);
w = w/sum(w);



p = inputParser;
addOptional(p,'MinPeakDistance',min_peak_distance);
addOptional(p,'FastRevolutionTime',fast_revolution_time);
addOptional(p,'SlowRevolutionTime',slow_revolution_time);
addOptional(p,'NumBins',num_bins);
addOptional(p,'fc',fc);
addOptional(p,'MinLocomotionSpeed',min_locomotion_speed);
addOptional(p,'SmoothingWindow',w);
parse(p,varargin{:});



%% Interpolate missing values and get an estimate of the wheel angle
wheel_position = fillmissing(wheel_position,'nearest');
wheel_angle = (wheel_position-min(wheel_position))/(max(wheel_position)-min(wheel_position))*2*pi;

%% de-artifact trace 
% by assuming wheel never goes backwards (which I have never seen with trained animals)
% we can store all negatively directed movements into a deficit variable,
% and not count positively directed movements until the deficit is made up.
% This removes artifacts and does some top notch denoising as well.
d = medfilt1(unwrap(wheel_angle),3);
if d(end)<d(1)
    d = -d;
end
deficit = 0;
c = nan(size(d));
c(1) = d(1);
for i =2:length(d)
    change = d(i)-d(i-1);    
    if change < 0
        deficit = deficit + change;
        c(i) = c(i-1);
    else
        c(i) = c(i-1) + reluFun(change+deficit);
        deficit = deficit + change;
        if deficit > 0
            deficit = 0;
        end
    end
end
wheel_angle = wrapTo2Pi(c);


%% Find wheel transition points

[~,highLocs] = findpeaks(wheel_angle,'MinPeakHeight',3/2*pi,'MinPeakProminence',pi,'MinPeakDistance',p.Results.MinPeakDistance);
[~,lowLocs] = findpeaks(-wheel_angle+2*pi,'MinPeakHeight',3/2*pi,'MinPeakProminence',pi,'MinPeakDistance',p.Results.MinPeakDistance);

%% Match high points of wheel signal to following low point
% if there is a low point before a high point, we implicitly ignore it 
% by looking at what follows the first high point only.
% If there is a high point that is not followed by a low point (i.e. at the
% end of the recording), we stop the for loop and ignore that high point
matchedLowLocs = zeros(size(highLocs));
for i = 1:length(highLocs)
    ind = find(lowLocs>highLocs(i),1);
    if isempty(ind) 
        break
    end
    matchedLowLocs(i) = lowLocs(ind);
end

% ignore last reserved high point
highLocs = highLocs(1:i-1);
matchedLowLocs = matchedLowLocs(1:i-1);

%% Test wheel rotation direction, flip high and low points if opposite of expected
if mean( highLocs(2:end) - matchedLowLocs(1:end-1)) > mean(  matchedLowLocs - highLocs)
    temp = highLocs(2:end);
    highLocs = matchedLowLocs(1:end-1);
    matchedLowLocs = temp;
    wheel_angle = -wheel_angle+2*pi;
end


%% Keep revolutions that occur between 0.5 and 1.6 seconds in duration
% assuming 15 cm wheel
toKeep = (matchedLowLocs-highLocs) > p.Results.FastRevolutionTime & (matchedLowLocs-highLocs) < p.Results.SlowRevolutionTime;

finalHighLocs = highLocs(toKeep);
finalLowLocs = matchedLowLocs(toKeep);

if false & length(finalHighLocs) > 30 & length(finalLowLocs) > 30
    %% Calculate average position for points between wheel transition points
    % We assume on average the rate of rotation is constant and invariant of
    % current wheel position. We can then expect wheel position to decrease
    % linearly wheel transition points on average. Here, we divy the wheel
    % position measurements into bins between wheel transition points and take
    % the average for each bin, giving us the average wheel position between transition points. 
    
    
    observedPosition = zeros(p.Results.NumBins,1);
    observedPoints = zeros(p.Results.NumBins,1);
    temporalBinEdges = linspace(0,1,p.Results.NumBins);
    temporalBinEdges(1) = -1;
    temporalBinEdges(end) = 2;
    
    for i =1:length(finalHighLocs)
        t = linspace(0,1,finalLowLocs(i)-finalHighLocs(i));
        currBins = discretize(t,temporalBinEdges);
        for ii =1:length(currBins)
            observedPosition(currBins(ii)) = observedPosition(currBins(ii))+wheel_angle(finalHighLocs(i)+ii-1);
            observedPoints(currBins(ii)) = observedPoints(currBins(ii)) + 1;
        end
    end
    
    % Ensure sorted for future binning. Typically the measurements will already
    % be in descending order, but depending on noise they may not.
    % Qualitatively sorting has negligible effect on the average wheel position
    % curve and is thus a harmless idealization.
    meanObservedPosition = observedPosition./observedPoints;
    desired = linspace(2*pi,0,p.Results.NumBins);
    desired = desired(~isnan(meanObservedPosition));
    meanObservedPosition = sort(meanObservedPosition(~isnan(meanObservedPosition)),'descend');
    
    %% Correct wheel position. 
    % Here we develop bin edges to map all wheel position points to their
    % closest point on our averaged curve to find the corresponding point that
    % would make the rate of rotation constant. 
    
    dataEdges = meanObservedPosition+[3*pi; abs(diff(meanObservedPosition))];
    dataEdges(end) = -pi;
    
    correctedWheelPosition = desired(length(desired)+1-discretize(wheel_angle,flip(dataEdges)));
    
    %% Calculate wheel speed
    %wheel_speed = filtfilt(b,a,-sgolayfilt(diff(unwrap(medfilt1(correctedWheelPosition,3))),3,1001))*fs*0.5*wheel_diameter;
    wheel_speed = -filter(w,1,(diff(medfilt1(unwrap(correctedWheelPosition),3))))*fs*0.5*wheel_diameter;
    
else
    %% didn't have enough fast bouts of locomotion, so lack of correction shouldn't matter anyway
    distance = -medfilt1(unwrap(wheel_angle),3);
    wheel_speed = filter(w,1,(diff(distance)))*fs*0.5*wheel_diameter;
end


%% Calculate locomotion on and off indicies
state_vect = wheel_speed > p.Results.MinLocomotionSpeed;
state_vect(1) = 0;
wheel_on = find(diff(state_vect)==1)+1;
wheel_off = find(diff(state_vect)==-1)+1;

wheel_on = wheel_on(1:length(wheel_off));
[wheel_on,wheel_off] = mergeTimestamps(wheel_on,wheel_off,2500);
adjusted_wheel_on = nan(size(wheel_on));
adjusted_wheel_off = nan(size(wheel_on));

for i = 1:length(wheel_on)
    adjusted_wheel_on(i) = find(wheel_speed(1:wheel_on(i))<=0,1,'last')+1;
    endInd = find(wheel_speed(wheel_off(i):end)<=0,1)+wheel_off(i)-1;
    if ~isempty(endInd)
        adjusted_wheel_off(i) = endInd;
    elseif i==length(wheel_on) %% handle last locomotion bout not terminating before data aquisition
        adjusted_wheel_off(i) = length(wheel_speed);
    end
end



end

function y = reluFun(x)
    y = zeros(size(x));
    y(x>0) = x(x>0);
end
    

