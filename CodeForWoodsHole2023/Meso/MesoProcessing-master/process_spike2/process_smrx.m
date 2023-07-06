function [timing, channels_data] = process_smrx(cedpath, outputpath,data_smr_time_stamp_filename, channels, CHANNELS_NUM)
%load SON library
CEDS64LoadLib(cedpath);



rf=0;
% parameters for extracting wheel motion
MIN_MAX_THRESHOLD = 3;
VOLTS_RANGE = 5;
WINDOW = 2500;
WHEEL_DIAMETER = 15; % cm
WHEEL_DIRECTION = 1;
MAX_WHEEl_SPEED = 60;
SQUARE_DETECT_PARAM = 0.05;
SHORT_BREAK_OR_SHORT_RUN_PARAM = 2500;
data = process_spike2_smr2mat('', outputpath, data_smr_time_stamp_filename, CHANNELS_NUM);

channels_data = get_channels_data_from_samples(data, channels, rf, ...
    WHEEL_DIRECTION, MIN_MAX_THRESHOLD, WINDOW, WHEEL_DIAMETER, MAX_WHEEl_SPEED, ...
    SQUARE_DETECT_PARAM, SHORT_BREAK_OR_SHORT_RUN_PARAM);
% Other modalities
channels_data.startsig = zeros(size(channels_data.diode));
channels_data.startsig(1:end-500)=channels_data.diode(501:end);
channels_data.sound = zeros(size(channels_data.diode));  % no sound
channels_data.led=zeros(size(channels_data.diode));
channels_data.CaFrameData =zeros(size(channels_data.diode));
% save(fullfile(outputpath,strcat(data_smr_time_stamp_filename,'_binary.mat')),'channels_data');



[timing.stimstart,timing.stimend]=squaredetect(channels_data.diode,.5);
[timing.mesostart,timing.mesoend]=squaredetect(channels_data.mesoframe,.5);
[timing.bluestart,timing.blueend]=squaredetect(channels_data.blue,.5);
[timing.uvstart,timing.uvend]=squaredetect(channels_data.uv,.5);

[timing.airpuffstart,timing.airpuffend]=squaredetect(channels_data.air_puff,.5);



[timing.stimstart,timing.stimend]=squaredetect(channels_data.startsig,.5);
[timing.mesostart,timing.mesoend]=squaredetect(channels_data.mesoframe,.5);
[timing.bluestart,timing.blueend]=squaredetect(channels_data.blue,.5);
[timing.uvstart,timing.uvend]=squaredetect(channels_data.uv,.5);

[timing.airpuffstart,timing.airpuffend]=squaredetect(channels_data.air_puff,.5);
