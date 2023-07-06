function [signal1, signal2, skippedframes, skippedchannels] = demix_2signals(mov, paramsSignalsExtraction)
ratio = paramsSignalsExtraction.blueuvRatio;

firstframe=mov(:,:,1);
secondframe=mov(:,:,2);

[R,C,nframes] = size(mov);

firstframe_linear=double(reshape(firstframe(1/4*R:3/4*R,1/4*C:3/4*C),[],1));
secondframe_linear=double(reshape(secondframe(1/4*R:3/4*R,1/4*C:3/4*C),[],1));

skippedcount=0;
skippedframes=NaN(100,1);skippedchannels=NaN(100,1); % record skipped frames ind and channels
ch=0;% channel0 is uv, channel1 is blue
ind_bl=1; ind_uv=1;
signal1=zeros(R*C,nframes+100,'uint16');
signal2=zeros(R*C,nframes+100,'uint16');

for iframe=1:nframes
    if mod(iframe,2000)==0
        disp(['Checking Dropped Frames, done ',num2str(iframe),' frames']);
    end
    currentframe=mov(:,:,iframe);
    if iframe<10
        uvframe_linear=secondframe_linear;
        blframe_linear=firstframe_linear;
    elseif ind_uv-1>0 && ind_bl-1>0
        %uvframe_linear=imwarp(imgdata_uv(:,:,ind_uv-1),invtform,'OutputView',imref2d(size(template)),'Fillvalues',0);
        %uvframe_linear=double(reshape(uvframe_linear(68:188,68:188),[],1));
        x=reshape(signal1(:,ind_bl-1),R,C);
        blframe_linear=double(reshape(x(1/4*R:3/4*R,1/4*C:3/4*C),[],1));
        x=reshape(signal2(:,ind_uv-1),R,C);
        uvframe_linear=double(reshape(x(1/4*R:3/4*R,1/4*C:3/4*C),[],1));
    end
    if corr(double(reshape(currentframe(1/4*R:3/4*R,1/4*C:3/4*C),[],1)),uvframe_linear)>corr(double(reshape(currentframe(1/4*R:3/4*R,1/4*C:3/4*C),[],1)),blframe_linear) %uv
        ch_current=0;
    else
        ch_current=1;
    end
    if ch_current==ch % frame drop
        skippedcount=skippedcount+1;
        display(['Dropped frame at ',num2str(iframe)]);
        skippedchannels(skippedcount)=~ch;
        skippedframes(skippedcount)=iframe;
        switch ch_current
            case 0 % skipped a blue
                signal1(:,ind_bl)=signal1(:,max(ind_bl-1,1));
                signal2(:,ind_uv)=currentframe(:);
            case 1 % skipped a uv
                signal1(:,ind_bl)=currentframe(:);
                signal2(:,ind_uv)=signal2(:,max(ind_uv-1,2));
        end
        ind_bl=ind_bl+1; ind_uv=ind_uv+1;
        
    else  % if alternating properly
        switch ch_current
            case 0
                signal2(:,ind_uv)=currentframe(:);
                ind_uv=ind_uv+1;
            case 1
                signal1(:,ind_bl)=currentframe(:);
                ind_bl=ind_bl+1;
        end
        ch=~ch;
    end
    
    
end

skippedframes=skippedframes(1:skippedcount);
skippedchannels=skippedchannels(1:skippedcount); % record skipped frames ind and channels
signal1=signal1(:,1:ind_bl-1);
signal2=signal2(:,1:ind_uv-1);




signal1 = single(signal1);
signal2 = single(signal2);

if ratio > 1
    signal1 = interpolate_pixels(signal1, ratio);
    signal2 = interpolate_pixels(signal2, ratio);
end
skippedframes=skippedframes;
skippedchannels=skippedchannels;
