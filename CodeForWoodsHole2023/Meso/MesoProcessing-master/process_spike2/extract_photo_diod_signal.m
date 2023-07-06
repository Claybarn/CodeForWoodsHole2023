function diode = extract_photo_diod_signal(data, rf)


if rf==0
    diode=tsmovavg(data,'s',50,1); %diode
    diode=diode-nanmean(diode);
    st=find(diode<nanmin(diode)/2,1);
    diode(1:st-10)=0;
    diode=(diode/max(diode)>0.5);
    [vison,visoff]=squaredetect(diode,0.05);
    k=1;
    visondelete=[];
    for k=1:min(length(vison),length(visoff))
        if visoff(k)-vison(k)<1000 || visoff(k)-vison(k)>5000*4 %flickering or long-lasting background
            visondelete=[visondelete;k];
        end
    end
    vison(visondelete)=[];
    visoff(visondelete)=[];
    
    diode=zeros(size(diode));
    for k=1:min(length(vison),length(visoff))
        diode(vison(k):visoff(k))=1;
    end
elseif rf==1
    diode=tsmovavg(data,'s',50,1); %diode
    temp=diff(diode);
    temp=temp>nanmax(temp)/2;
    [on,off]=squaredetect(temp,0.05);
    diode=zeros(size(data));
    for ion=2:length(on)-1 % the first one is psychtoolbox artifacts
        diode(on(ion):on(ion+1)-1)=linspace(0,1,on(ion+1)-on(ion));
    end
end