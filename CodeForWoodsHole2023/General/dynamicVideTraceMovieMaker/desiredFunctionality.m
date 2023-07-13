stateVect = newData.ch2>2.5;
onInds = find(diff(stateVect)==1)+1;

onInds = onInds(1000:end);

ephys_fs = 8000;
frame_rate = 20;
samples_per_frame =  ephys_fs/frame_rate;

fcuts = [0 2 58 59 61 62 100 101];           % Frequencies
mags = [0 1 0 1 0  ];                                    % Passbands & Stopbands
devs = [0.05 0.01 0.05 0.01 0.05 ];                % Tolerances
[n,Wn,beta,ftype] = kaiserord(fcuts,mags,devs,ephys_fs);          % Kaiser Window FIR Specification
n = n + rem(n,2);
hh = fir1(n,Wn,ftype,kaiser(n+1,beta),'noscale');           % Filter Realisation


% filtered_lfp = filtfilt(b,a,double(lfp.data{1}));
% filtered_lfp(:,192) = (filtered_lfp(:,190)+filtered_lfp(:,194))/2;

lfp = filtfilt(hh,1,newData.ch1);

%% specify frames

start_frame = find(onInds>12205500,1);
end_frame = find(onInds>12205500+60*8000,1);

frame_inds = start_frame:end_frame;


%% video 
v = VideoReader('D:\fullTest2/fullTest5_0.avi');
frames = read(v,[start_frame end_frame]);

%%vid 

v = VideoWriter('alignedMobileMesoLFP.mp4');
v.FrameRate = 20;
v.Quality = 100;
open(v);
%%
figure('Position', [10 10 750 1000])
tiledlayout(2,2,'TileSpacing','tight');

ax(1) = nexttile;
mov = imagesc(dfof(:,:,frame_inds(i)-1000),'AlphaData',~isnan(dfof(:,:,1)));
caxis([-2 2])
set(gca, 'visible', 'off')
set(findall(gca, 'type', 'text'), 'visible', 'on')
axis square;

ax(2) = nexttile;
vid = imshow(frames(:,:,1));
set(gca, 'visible', 'off')
set(findall(gca, 'type', 'text'), 'visible', 'on')
axis square;

ax(3) = nexttile([1 2]);
trace = plot(lfp(onInds(i)-ephys_fs+1:onInds(i)));
ylim([-6 6])
set(gca, 'visible', 'off')
set(findall(gca, 'type', 'text'), 'visible', 'on')
writeVideo(v,getframe(gcf));


for i = 2:length(frame_inds)
    mov.CData = dfof(:,:,frame_inds(i)-1000);
    vid.CData = frames(:,:,i);
    trace.YData = lfp(onInds(i)-ephys_fs+1:onInds(i));
    drawnow;
    writeVideo(v,getframe(gcf));
end
