function mov = readTifStack(filename, st,en)
l=1;mov=[];
if isstruct(filename)
    en = min(length(filename), en);
    for k = st:en
        %       disp((k-st)/(en-st));
        mov(:,:,l) = imread(fullfile(filename(k).folder,filename(k).name));
        l=l+1;
    end
    
else
    for k = st:en
        mov(:,:,l) = imread(filename,k);
        l=l+1;
    end
end