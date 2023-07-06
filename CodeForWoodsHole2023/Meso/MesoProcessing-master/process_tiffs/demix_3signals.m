function sigsMov = demix_3signals(mov, paramsSignalsExtraction)
ratio = paramsSignalsExtraction.blueuvRatio;

        
        [R,C,nframes] = size(mov);
        movLeft = mov(:, 1:C/2,:);
        movRight = mov(:, 1+C/2:end,:);
        clear mov;
        C=C/2;
         firstframe=movRight(:,:,1);
        secondframe=movRight(:,:,2);
        Thirdframe = movLeft(:,:,3);
        
        firstframe_linear=double(reshape(firstframe(68:188,68:188),[],1));
        secondframe_linear=double(reshape(secondframe(68:188,68:188),[],1));
        thirdframe_linear=double(reshape(Thirdframe(68:188,68:188),[],1));
        
        skippedcount=0;
        skippedframes=NaN(100,1);skippedchannels=NaN(100,1); % record skipped frames ind and channels
        next_ch=1;% channel0 is uv, channel1 is blue channel2 is green
        ind_bl=1; ind_uv=1;ind_green=1;
        sigsMov.blue=zeros(R*C,nframes+100,'uint16');
        sigsMov.uv=zeros(R*C,nframes+100,'uint16');
        sigsMov.green=zeros(R*C,nframes+100,'uint16');
        for iframe=1:nframes
            if mod(iframe,2000)==0
                disp(['Checking Dropped Frames, done ',num2str(iframe),' frames']);
            end
            currentframeR=movRight(:,:,iframe);
            currentframeL=movLeft(:,:,iframe);
            if iframe<10
                uvframe_linear=secondframe_linear;
                blframe_linear=firstframe_linear;
                greenframe_linear=thirdframe_linear;
            elseif ind_uv-1>0 && ind_bl-1>0&& ind_green-1>0
                %uvframe_linear=imwarp(imgdata_uv(:,:,ind_uv-1),invtform,'OutputView',imref2d(size(template)),'Fillvalues',0);
                %uvframe_linear=double(reshape(uvframe_linear(68:188,68:188),[],1));
                x=reshape(sigsMov.uv(:,ind_uv-1),R,C);
                uvframe_linear=double(reshape(x(68:188,68:188),[],1));
                x=reshape(sigsMov.blue(:,ind_bl-1),R,C);
                blframe_linear=double(reshape(x(68:188,68:188),[],1));
                x=reshape(sigsMov.green(:,ind_green-1),R,C);
                greenframe_linear=double(reshape(x(68:188,68:188),[],1));
            end
            corrval_curr_uv = corr(double(reshape(currentframeR(68:188,68:188),[],1)),uvframe_linear);
            corrval_curr_blue = corr(double(reshape(currentframeR(68:188,68:188),[],1)),blframe_linear);
            corrval_curr_green = corr(double(reshape(currentframeL(68:188,68:188),[],1)),greenframe_linear);
            [~,mini] = max([corrval_curr_uv, corrval_curr_blue, corrval_curr_green]);
            ch_current = mini-1;
           
            if next_ch~=ch_current % frame drop
             
                skippedcount=skippedcount+1;
                display(['Dropped frame at ',num2str(iframe)]);
                skippedchannels(skippedcount)=next_ch;
                skippedframes(skippedcount)=iframe;
                switch next_ch % fill in for the missed one
                    case 1 % skipped a blue
                        sigsMov.blue(:,ind_bl)=sigsMov.blue(:,max(ind_bl-1,1));     
                        ind_bl=ind_bl+1; 
                    case 0 % skipped a uv                        
                        sigsMov.uv(:,ind_uv)=sigsMov.uv(:,max(ind_uv-1,2));
                        ind_uv=ind_uv+1;
                    case 2 % skipped green
                       sigsMov.green(:,ind_green)=sigsMov.green(:,max(ind_green-1,2)); 
                       ind_green=ind_green+1; 
                end
                switch ch_current % use what we've got
                  case 1 
                        sigsMov.blue(:,ind_bl)=currentframeR(:);
                        ind_bl=ind_bl+1; 
                        next_ch=0;
                    case 0                  
                        sigsMov.uv(:,ind_uv)=currentframeR(:);    
                        ind_uv=ind_uv+1;
                        next_ch=2;
                    case 2 
                       sigsMov.green(:,ind_green)=currentframeL(:);  
                       ind_green=ind_green+1; 
                       next_ch=1;
                end 
                
                
            else  % if alternating properly
                switch ch_current
                    case 0
                        sigsMov.uv(:,ind_uv)=currentframeR(:);
                        ind_uv=ind_uv+1;
                        next_ch = 2;
                    case 1
                        sigsMov.blue(:,ind_bl)=currentframeR(:);
                        ind_bl=ind_bl+1;
                        next_ch = 0;
                    case 2
                        sigsMov.green(:,ind_green)=currentframeL(:);
                        ind_green=ind_green+1;
                        next_ch = 1;
                end
                prevch = ch_current;
            end
            
            
        end
        
        skippedframes=skippedframes(1:skippedcount);
        skippedchannels=skippedchannels(1:skippedcount); % record skipped frames ind and channels
        sigsMov.blue=sigsMov.blue(:,1:ind_bl-1);
        sigsMov.uv=sigsMov.uv(:,1:ind_uv-1);
        sigsMov.green=sigsMov.green(:,1:ind_green-1);
        


fieldNames = fieldnames(sigsMov);
for name_i = 1:length(fieldNames)
    sigsMov.(fieldNames{name_i}) = single(sigsMov.(fieldNames{name_i}));
end
if ratio > 1
    sigsMov.(paramsSignalsExtraction.sig2interpolate) = interpolate_pixels(sigsMov.(paramsSignalsExtraction.sig2interpolate), ratio);
end
sigsMov.skippedframes=skippedframes;
sigsMov.skippedchannels=skippedchannels;
