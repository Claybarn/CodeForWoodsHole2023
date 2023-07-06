function [returnOn, returnOff] = timestampsNot(timestampsOn,timestampsOff)
%TIMESTAMPNOT returns timetamps corresponding to the inverse of the input
 % Swaps "(true,true,false,true) to (false,false,true,false).
  % EXAMPLE
 % [returnOn, returnOff] = ...
 % timestampsNot(timestampsOn,timestampsOff,'StartTime',1,'EndTime',100000);
 returnOn = [-inf timestampsOff];
 returnOff = [timestampsOn inf];
 
 
%  StartTimeInd = find(strcmp('StartTime',varargin),1);
%  EndTimeInd = find(strcmp('EndTime',varargin),1);
%  if ~isempty(StartTimeInd)
%     returnOn =[varargin{StartTimeInd+1};  returnOn];
%     if ~ isempty(timestampsOn)
%         returnOff =[timestampsOn(1); returnOff];
%     end
%  end
%  if ~isempty(EndTimeInd)
%      returnOff =[returnOff; varargin{EndTimeInd+1}];
%      if ~ isempty(timestampsOff)
%         returnOn =[returnOn; timestampsOff(end)];
%      end
%  end
% end