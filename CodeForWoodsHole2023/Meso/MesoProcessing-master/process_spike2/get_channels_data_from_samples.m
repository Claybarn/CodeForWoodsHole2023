function channels_data = get_channels_data_from_samples(data, channels, rf, ...
    WHEEL_DIRECTION, MIN_MAX_THRESHOLD, WINDOW, WHEEL_DIAMETER, MAX_WHEEl_SPEED, ...
    SQUARE_DETECT_PARAM, SHORT_BREAK_OR_SHORT_RUN_PARAM)

names = fieldnames(channels);
for ni = 1:length(names)
    switch names{ni}
        case 'MOVING_PHOTO_DIODE'
            channels_data.movingdiode=data(:,channels.MOVING_PHOTO_DIODE)-nanmedian(data(:,channels.MOVING_PHOTO_DIODE));
            tmid = round(length(channels_data.movingdiode)/2);
            maxval=nanmean(channels_data.movingdiode(1:1000));
            channels_data.movingdiode(channels_data.movingdiode>maxval)=0;
        case 'BLUE'
            channels_data.blue=(data(:,channels.BLUE)-nanmin(data(:,channels.BLUE))>0.5);
        case 'UV'
            channels_data.uv=(data(:,channels.UV)-nanmin(data(:,channels.UV))>0.5);
        case {'FRAMETICKES' 'RED_MESO'}
            channels_data.mesoframe=(data(:,channels.FRAMETICKES)-nanmin(data(:,channels.FRAMETICKES))>0.5);
        case 'PHOTO_DIODE'
            channels_data.diode = extract_photo_diod_signal(data(:,channels.PHOTO_DIODE), rf);
        case 'WHEEL'
            % Channel5- Wheel
            [channels_data.wheel, channels_data.wheelspeed] = extract_wheel_signal(data(:, channels.WHEEL), WHEEL_DIRECTION, ...
                MIN_MAX_THRESHOLD, WINDOW, WHEEL_DIAMETER, MAX_WHEEl_SPEED, ...
                SQUARE_DETECT_PARAM, SHORT_BREAK_OR_SHORT_RUN_PARAM);
        case 'AIR_PUFF'
            % Channel6- Airpuff
            channels_data.air_puff=(data(:,channels.AIR_PUFF)-nanmean(data(:,channels.AIR_PUFF))>1); %airpuff
        case 'PUPIL_CAMERA'
            % Channel7- get pupil camera
            channels_data.pupil_frame=data(:,channels.PUPIL_CAMERA);
            
            [pupon,pupoff]=squaredetect(channels_data.pupil_frame,0.05*max(channels_data.pupil_frame)); %max value=2.9
            pupon=pupon(1:length(pupoff));
            if pupon(100)-pupon(99)>5000/30*1.7
                maxi=max(channels_data.pupil_frame); %step amp of each frame
                for i=1:length(pupon)-1
                    if round((pupoff(i+1)+pupoff(i))/2)<pupon(end)
                        channels_data.pupil_frame(round((pupon(i+1)+pupon(i))/2):round((pupoff(i+1)+pupoff(i))/2))=maxi;
                    end
                end
            end
            channels_data.pupil_frame=(channels_data.pupil_frame-nanmin(channels_data.pupil_frame)>0.5);
            
        case 'EEG'
            warning('Need to code the EEG');
            
        otherwise
            error('Unindetified channel name');
    end
end



