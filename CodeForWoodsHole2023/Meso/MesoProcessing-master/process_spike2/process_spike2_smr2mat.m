%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loads spike2 smr file and saves it as a mat file
% input:    
%           data_time_stamp_filename - smr file
%           channels_num             - number of channels
%           analog_rate              - sampling rate
% outputs:
%           data                     - a matrix of all channels from smr
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [data, timestamp] = process_spike2_smr2mat(datapath, outputpath, data_time_stamp_filename, channels_num)



display(strcat('loading in smr: ',data_time_stamp_filename));
if ~isfile(fullfile(datapath, data_time_stamp_filename))
        error('Could not find file');
end
fhand = CEDS64Open(fullfile(datapath, data_time_stamp_filename));
if fhand == -1
    error('Could not open file');
end
ichannum = min(CEDS64MaxChan(fhand),channels_num);
dsec=CEDS64TimeBase(fhand); % sec per time tick
[~, filename] = fileparts(data_time_stamp_filename);

if ~exist(fullfile(outputpath, strcat(filename,'.mat')), 'file')
    maxTimeTicks = CEDS64ChanMaxTime(fhand,1);
    data=nan(maxTimeTicks./20+1,ichannum);
    
    % get waveform data from each channel
    for ichan=1:ichannum
        %file name, channel num, max num of points, start and end time in ticks
        [fRead,fVals,fTime] = CEDS64ReadWaveF(fhand,ichan,maxTimeTicks,0,maxTimeTicks);
        if fRead > 0
        data(fTime+1:fRead+fTime,ichan)=fVals;
        readF(ichan) = fRead;
        end
    end
   
    data=data(1:readF(1)+fTime,:);
    timestamp=(1:20:maxTimeTicks)'*dsec;
    totallength=min(size(data,1),length(timestamp));
    data=data(1:totallength,:);
    timestamp=timestamp(1:totallength);
    save(fullfile(outputpath, strcat(filename,'.mat')),'data','timestamp','-v7.3');
else
    load(fullfile(outputpath, strcat(filename,'.mat')),'data','timestamp');
end
CEDS64CloseAll();

% binarize raw data from smr file
