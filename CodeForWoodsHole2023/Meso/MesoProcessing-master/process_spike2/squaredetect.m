% detect square events, return their on/off location
function [locson,locsoff]=squaredetect(v,threshold)
c1=diff(v);
% plot(c1);
c1max=max(c1);
c1min=min(c1);
%scale to 1/-1
c1=c1/max(c1max,abs(c1min));
[on,locson]=findpeaks(c1,'MinPeakHeight',threshold);
[off,locsoff]=findpeaks(-c1,'MinPeakHeight',threshold);
% hold on;
% plot(locson,on,'.');
% plot(locsoff,off,'.');

% check on=off
% if length(locsoff)~=length(locson)
%     display('on and off detection does not match');
% end
end

