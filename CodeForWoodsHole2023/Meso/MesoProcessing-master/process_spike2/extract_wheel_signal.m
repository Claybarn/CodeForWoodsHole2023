function [wheel, wheelspeed] = extract_wheel_signal(wp, WHEEL_DIRECTION, ...
    MIN_MAX_THRESHOLD, WINDOW, WHEEL_DIAMETER, MAX_WHEEl_SPEED, ...
    SQUARE_DETECT_PARAM, SHORT_BREAK_OR_SHORT_RUN_PARAM, MINIMAL_SPEED_FOR_RUNNING)

if ~exist('WHEEL_DIRECTION','var')
    WHEEL_DIRECTION = 1;
end
if ~exist('MIN_MAX_THRESHOLD','var')
    MIN_MAX_THRESHOLD = 3;
end
if ~exist('VOLTS_RANGE','var')
    VOLTS_RANGE = 5;
end
if ~exist('WINDOW','var')
    WINDOW = 2500;
end
if ~exist('WHEEL_DIAMETER','var')
    WHEEL_DIAMETER = 15; % cm
end
if ~exist('MAX_WHEEl_SPEED','var')
    MAX_WHEEl_SPEED = 60; 
end
if ~exist('SQUARE_DETECT_PARAM','var')
    SQUARE_DETECT_PARAM = 0.05; 
end
if ~exist('SHORT_BREAK_OR_SHORT_RUN_PARAM','var')
    SHORT_BREAK_OR_SHORT_RUN_PARAM = 2500; 
end
if ~exist('MINIMAL_SPEED_FOR_RUNNING','var')
    MINIMAL_SPEED_FOR_RUNNING = 1; 
end

% transform wheel postion to running speed
wp=fillmissing(wp,'nearest');

if WHEEL_DIRECTION==1
    wpangle=(wp-min(wp))/(max(wp)-min(wp))*2*pi;%change voltage to angle
    if max(wp)-min(wp)<MIN_MAX_THRESHOLD
        wpangle=(wp-min(wp))/VOLTS_RANGE*2*pi; % around 5 volt range
    end
elseif WHEEL_DIRECTION==-1
    wpangle=(1-(wp-min(wp))/(max(wp)-min(wp)))*2*pi;
    if max(wp)-min(wp)<MIN_MAX_THRESHOLD
        wpangle=(1-(wp-min(wp))/VOLTS_RANGE)*2*pi; % around 5 volt range
    end
end

wpdist=zeros(length(wpangle),1); %change to distance
for i=2:length(wpangle)
    anglechange=(wpangle(i)-wpangle(i-1));
    if abs(anglechange)>pi
        anglechange=1/2/pi*(anglechange<0);
    end
    wpdist(i)=wpdist(i-1)+anglechange;
end


wpspd=zeros(length(wpdist),1);
i=1;
for i = 1:length(wpdist)
    st=max(1,i-WINDOW/2);
    ed=min(i+WINDOW/2,length(wpdist));
    wpspd(i)=(max(wpdist(st:ed))-min(wpdist(st:ed)))/(ed-st)*5000*0.5*WHEEL_DIAMETER; 
end
wpspd(wpspd>MAX_WHEEl_SPEED)=MAX_WHEEl_SPEED;
wheel=(wpspd>MINIMAL_SPEED_FOR_RUNNING);
wheel(1:2)=0;

[wheelon,wheeloff]=squaredetect(wheel,SQUARE_DETECT_PARAM);
for k=1:length(wheeloff)
    if wheeloff(k)-wheelon(k)<SHORT_BREAK_OR_SHORT_RUN_PARAM
        wheel(wheelon(k):wheeloff(k))=0;
    end
end

[wheelon,wheeloff]=squaredetect(wheel,SQUARE_DETECT_PARAM);
wheelondelete=[];
for k=1:min(length(wheelon),length(wheeloff))-1
    if wheelon(k+1)-wheeloff(k)<SHORT_BREAK_OR_SHORT_RUN_PARAM %short break or shortrun
        wheelondelete=[wheelondelete;k+1];
    end
end
wheelon(wheelondelete)=[];
wheeloff(wheelondelete-1)=[];
wheel=zeros(size(wheel));
for k=1:min(length(wheelon),length(wheeloff))
    wheel(wheelon(k):wheeloff(k))=1;
end


wheelspeed=wpspd;
wheel(1:2)=0;
wheelspeed(1:2)=0;