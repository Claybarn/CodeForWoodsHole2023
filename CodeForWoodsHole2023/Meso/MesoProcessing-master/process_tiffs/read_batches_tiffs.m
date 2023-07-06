function outMov = read_batches_tiffs(batchSize, tiffsPath, batches2load, resizeRatio)
tiffs=dir(fullfile(tiffsPath, '*.tif'));
names = {tiffs.name};
[~,ndx] = natsort(names);
tiffs=tiffs(ndx);

% cjb added, pass negative number to read all tiffs
if batches2load < 0
    BatchNum = ceil(length(tiffs)/batchSize);
else
    BatchNum = min(batches2load, ceil(length(tiffs)/batchSize));
end

if BatchNum*batchSize <= length(tiffs) && batches2load > BatchNum
    isleft = true;
else
    isleft = false;
end

st = 1;
outMov=[];
for bi = 1:BatchNum
    disp(['Reading tiffs batch no. ' num2str(bi) ' of ' num2str(BatchNum) ' from ' tiffsPath]);
    mov = readTifStack(tiffs, st,st+batchSize-1);
    if resizeRatio ~= 1
        mov = imresize(mov, resizeRatio);
    end
    if isempty(mov)
        break;
    end
    outMov = cat(3, outMov, mov);
    st = st+batchSize;
end

if isleft
    
    mov = readTifStack(tiffs, st,length(tiffs));
    if resizeRatio ~= 1
        mov = imresize(mov, resizeRatio);
    end
    if isempty(mov)
        return;
    end
    outMov = cat(3, outMov, mov);
end
