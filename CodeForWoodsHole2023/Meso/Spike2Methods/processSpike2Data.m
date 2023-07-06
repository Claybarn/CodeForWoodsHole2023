function processedSpike2Data = processSpike2Data(spike2Data)
%%processSpike2 gets relevant info from spike2 files 
% Data is extracted from spike2 files via a dedicated windows specific
% library, then processed according to channel type. The majority of analog
% signals are decomposed into states. Some signals are unaltered, such as
% EEG. The wheel signal is transformed into wheel speed, and periods of
% running are identified with the extract_wheel_signal function.

spike2Fields = fields(spike2Data);
processedSpike2Data = struct;
for i = 1:length(spike2Fields)
        currDataBundle = spike2Data.(spike2Fields{i});
        timestamps = cumsum(ones(size(currDataBundle.data))/currDataBundle.samplingRate)+double(currDataBundle.acquisitionOffset);
        if contains(spike2Fields{i},'BL') || contains(spike2Fields{i},'Blue','IgnoreCase',true)
            [protoBlueOnTimestamps,protoBlueOffTimestamps]=analogSigToTimestamps(spike2Data.(spike2Fields{i}).data,timestamps);
            processedSpike2Data.blueOnTimestamps = protoBlueOnTimestamps; % truncate to match dFoF (calculated elsewhere)
            processedSpike2Data.blueOffTimestamps = protoBlueOffTimestamps;
        elseif contains(spike2Fields{i},'GR') || contains(spike2Fields{i},'Green','IgnoreCase',true)
            [protoGreenOnTimestamps,protoGreenOffTimestamps]=analogSigToTimestamps(spike2Data.(spike2Fields{i}).data,timestamps);
            processedSpike2Data.greenOnTimestamps = protoGreenOnTimestamps; % truncate to match dFoF (calculated elsewhere)
            processedSpike2Data.greenOffTimestamps = protoGreenOffTimestamps;
        elseif contains(spike2Fields{i},'UV','IgnoreCase',true)
            [protoUvOnTimestamps,protoUvOffTimestamps]=analogSigToTimestamps(spike2Data.(spike2Fields{i}).data,timestamps);
             processedSpike2Data.uvOnTimestamps = protoUvOnTimestamps;
             processedSpike2Data.uvOffTimestamps = protoUvOffTimestamps;
        elseif contains(spike2Fields{i},'MesoCam','IgnoreCase',true)
            [protoMesoFrameOnTimestamps,protoMesoFrameOffTimestamps]=analogSigToTimestamps(spike2Data.(spike2Fields{i}).data,timestamps);
            processedSpike2Data.mesoFrameOnTimestamps = protoMesoFrameOnTimestamps;
            processedSpike2Data.mesoFrameOffTimestamps = protoMesoFrameOffTimestamps;
        elseif contains(spike2Fields{i},'Vis','IgnoreCase',true)
            [processedSpike2Data.diodeOnTimestamps,processedSpike2Data.diodeOffTimestamps]=analogSigToTimestamps(spike2Data.(spike2Fields{i}).data,timestamps);
        elseif contains(spike2Fields{i},'wheel','IgnoreCase',true)
            [processedSpike2Data.wheelSpeed,wheel_on,wheel_off] = extract_wheel_speed_with_correction(spike2Data.(spike2Fields{i}).data,5000,15,'MinLocomotionSpeed',1);
             processedSpike2Data.wheelOn = timestamps(wheel_on);
             processedSpike2Data.wheelOff = timestamps(wheel_off);
        elseif contains(spike2Fields{i},'Air','IgnoreCase',true)
            [processedSpike2Data.airPuffOnTimestamps,processedSpike2Data.airPuffOffTimestamps]=analogSigToTimestamps(spike2Data.(spike2Fields{i}).data,timestamps);
        elseif contains(spike2Fields{i},'pupil','IgnoreCase',true)
            [processedSpike2Data.pupilFrameOnTimestamps,processedSpike2Data.pupilFrameOffTimestamps]=analogSigToTimestamps(spike2Data.(spike2Fields{i}).data,timestamps);
        elseif contains(spike2Fields{i},'body','IgnoreCase',true)
            [processedSpike2Data.bodyFrameOnTimestamps,processedSpike2Data.bodyFrameOffTimestamps]=analogSigToTimestamps(spike2Data.(spike2Fields{i}).data,timestamps);
        elseif contains(spike2Fields{i},'LFP') || contains(spike2Fields{i},'ECoG','IgnoreCase',true)
            processedSpike2Data.ECoG = spike2Data.(spike2Fields{i}).data;
        else
            [processedSpike2Data.([spike2Fields{i} 'OnTimestamps']), processedSpike2Data.([spike2Fields{i} 'OffTimestamps'])]=analogSigToTimestamps(spike2Data.(spike2Fields{i}).data,timestamps);
            warning(['Unindetified channel name: ' spike2Fields{i} ', treating as timestamp data']);
        end
end
end
