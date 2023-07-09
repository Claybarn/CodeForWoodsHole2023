function states = stateTimestamps(spike2_data,pc,varargin)
%STATETIMESTAMPS identifies states from wheel and facial motion
%   returns timetamps runOn, runOff, sitOn, sitOff, faceOn, and faceOff.
%   TO CHANGE A DEFAULT PARAMETER VALUE:
%   stateTimestamps(spike2_data,pc,'minStateDuration',3)
%   no validation of inputs is performed. 

%% default values
minStateDuration = 5;
minFaceDuration = 5;
mergeTimeLimit = .2;
timeSinceLocOn = 3;%for locomotion state, minimum time since locomotion onset
timeBeforeLocOff = 3;%for locomotion state, minimum time before locomotion offset
timeSinceSitOn = 10;%for quiescence state, minimum time since quiescence onset
timeBeforeSitOff = 10;%for quiescence state, minimum time before quiescence offset
fsSpike2 = 5000;
%% change default values
if any(strcmp('minStateDuration',varargin))
    minStateDuration = varargin{find(strcmp('minStateDuration',varargin))+1};
end
if any(strcmp('timeSinceLocOn',varargin))
	timeSinceLocOn = varargin{find(strcmp('timeSinceLocOn',varargin))+1};
end
if any(strcmp('timeBeforeLocOff',varargin))
    timeBeforeLocOff = varargin{find(strcmp('timeBeforeLocOff',varargin))+1};
end
if any(strcmp('timeSinceSitOn',varargin))
    timeSinceSitOn = varargin{find(strcmp('timeSinceSitOn',varargin))+1};
end
if any(strcmp('timeBeforeSitOff',varargin))
    timeBeforeSitOff = varargin{find(strcmp('timeBeforeSitOff',varargin))+1};
end
if any(strcmp('fsSpike2',varargin))
	fsSpike2 = varargin{find(strcmp('fsSpike2',varargin))+1};
end
if any(strcmp('raw',varargin))
    minStateDuration = 0;
    timeSinceLocOn = 0;
    timeBeforeLocOff = 0;
    timeSinceSitOn = 0;
    timeBeforeSitOff = 0;
end
%% Locomotion

wheelOn = spike2_data.wheelOn(1:length(spike2_data.wheelOff)); % account for locomotion session not ending before recording stopped
wheelOff = spike2_data.wheelOff;
[wheelOn,wheelOff] = mergeTimestamps(wheelOn,wheelOff,mergeTimeLimit);
wheelSpeed = spike2_data.wheelSpeed;
wheelTime = (0:length(wheelSpeed)-1)/fsSpike2;

%  toRemoveInds = false(size(wheelOn));
%  for i =1:length(wheelOn)
%      if nanmax(wheelSpeed(round(wheelOn*fsSpike2):round(wheelOff*fsSpike2))) < 5
%          toRemoveInds(i) = true;
%      end
%  end
%  
%  wheelOn(toRemoveInds) = [];
%  wheelOff(toRemoveInds) = [];

blueMesoTimestamps = spike2_data.blueOnTimestamps;
minRunDur = minStateDuration+timeSinceLocOn+timeBeforeLocOff; %minimum actual locomotion duration including time since locomotion onset, time before locomotion offset and the minimum time period for data analysis


wheelOnFinal = wheelOn(wheelOn > blueMesoTimestamps(1) & wheelOff < blueMesoTimestamps(end) & wheelOff - wheelOn >= minRunDur);
wheelOffFinal = wheelOff(wheelOn > blueMesoTimestamps(1) & wheelOff < blueMesoTimestamps(end) & wheelOff - wheelOn >= minRunDur);


%% Sit (Quiescience)
% define as not running, several seconds from running, and a certain length
sitOn = [0 reshape(wheelOff,1,[])]; %use 0 as the first sit on time;
sitOff = [reshape(wheelOn,1,[]) wheelTime(end)];%use wheelOn times as sit off times;

%find sit on and sit off times during imaging period only
minSitDur = minStateDuration+timeSinceSitOn+timeBeforeSitOff; %actual minimum sit duration accouting for the onset time, offset time and minimum duration of the sustained quiescence epoch  used for analysis

sitOnFinal = sitOn(sitOn > blueMesoTimestamps(1) & sitOff < blueMesoTimestamps(end) & sitOff - sitOn > minSitDur)+timeSinceSitOn;%+TimeSinceSitOn;
sitOffFinal = sitOff(sitOn > blueMesoTimestamps(1) & sitOff < blueMesoTimestamps(end) & sitOff - sitOn > minSitDur)-timeBeforeSitOff;%-TimeBeforeSitOff;

%% Face (facial motion)
% calculate transition points and only keep if they pass our duration
% threshold

%[faceHighOn,faceHighOff] = cjb_changepoints(pc,spike2_data.pupilFrameOnTimestamps,0.6,1);
[faceHighOn,faceHighOff] = cjb_changepoints2(spike2_data.pupilFrameOnTimestamps,pc,1);

% faceHighOnInter = squeeze(faceHighOn(faceHighOff-faceHighOn > minStateDuration));
% faceHighOffInter = squeeze(faceHighOff(faceHighOff-faceHighOn > minStateDuration));
if ~isempty(faceHighOff) & faceHighOff(1) < faceHighOn(1)
    faceHighOff(1) = [];
end
minL = min(length(faceHighOn),length(faceHighOff));
faceHighOn = faceHighOn(1:minL);
faceHighOff = faceHighOff(1:minL);

[faceHighOn,faceHighOff] = mergeTimestamps(faceHighOn,faceHighOff,0.2);
faceHighOnInter = squeeze(faceHighOn(faceHighOff-faceHighOn > minFaceDuration));
faceHighOffInter = squeeze(faceHighOff(faceHighOff-faceHighOn > minFaceDuration));


%[faceLowOn,faceLowOff] = cjb_changepoints(-pc,spike2_data.pupilFrameOnTimestamps,0.4,1);
[faceLowOn,faceLowOff] = cjb_changepoints3(spike2_data.pupilFrameOnTimestamps,pc,1);

minL = min(length(faceLowOn),length(faceLowOff));
faceLowOn = faceLowOn(1:minL);
faceLowOff = faceLowOff(1:minL);

[faceLowOn,faceLowOff] = mergeTimestamps(faceLowOn,faceLowOff,0.2);
faceLowOnInter = faceLowOn(faceLowOff-faceLowOn > minFaceDuration);
faceLowOffInter = faceLowOff(faceLowOff-faceLowOn > minFaceDuration);

%[faceLowOnInter,faceLowOffInter] = timestampsNot(faceHighOnInter,faceHighOffInter,'StartTime',0,'EndTime',spike2_data.pupilFrameOnTimestamps(end));
% [IDX,C,~,~]= kmeans(movmin(movmean(pc,10),10),2,'MaxIter',100000);
% sIDX = IDX;
% [~,I] = sort(C);
% %uIDX = unique(IDX);
% for i=1:length(I)
%     sIDX(IDX==I(i))=i;
% end
% 
% onWhisk = (find(diff(sIDX)==1)+1)/fsFace;
% offWhisk = (find(diff(sIDX)==-1)+1)/fsFace;
% if offWhisk(1)<onWhisk(1)
%     offWhisk(1) = [];
% end
% 
% minLength = min(length(onWhisk),length(offWhisk));
% 
% onWhisk=onWhisk(1:minLength);
% offWhisk=offWhisk(1:minLength);
% 
% [faceHighOnInter,faceHighOffInter] = mergeTimestamps(onWhisk,offWhisk,3);
% 
% 
% faceLowOnInter = faceHighOffInter(1:end-1);
% faceLowOffInter = faceHighOnInter(2:end);


%% states we use
% wheel high and (face high or face low) % just wheel high
% face high contained in wheel low
% sit high and face low

% loco state
states.locoOn = wheelOnFinal + timeSinceLocOn;
states.locoOff = wheelOffFinal - timeBeforeLocOff;

%face high contained in wheel low
% might be better to use sitOnFinal and sitOffFinal here, can get some
% "high face states" where facial motion is actually low because of merge
% and exclusion effects CJB
% original pass comments
%[faceContainOn,faceContainOff] = mergeTimestamps(sitOn,sitOff,minStateDuration); 
%[states.faceOn,states.faceOff] = timestampsAnd(faceHighOnInter,faceHighOffInter,faceContainOn,faceContainOff);
[states.faceOn,states.faceOff] = timestampsAnd(faceHighOnInter,faceHighOffInter,sitOnFinal,sitOffFinal);

% boolMask = timestampsIntersect(faceHighOnInter,faceHighOffInter,sitOn,sitOff);
% 
% states.faceOn = faceHighOnInter(boolMask);
% states.faceOff = faceHighOffInter(boolMask);


%wheel low and face low
[sitOn_inter,sitOff_inter] = timestampsAnd(faceLowOnInter,faceLowOffInter,sitOnFinal,sitOffFinal);

states.sitOn = sitOn_inter(sitOff_inter-sitOn_inter>minStateDuration + timeSinceSitOn + timeBeforeSitOff);
states.sitOff = sitOff_inter(sitOff_inter-sitOn_inter>minStateDuration + timeSinceSitOn + timeBeforeSitOff);

states.sitOn = states.sitOn + timeSinceSitOn;
states.sitOff = states.sitOff - timeBeforeSitOff;
end

