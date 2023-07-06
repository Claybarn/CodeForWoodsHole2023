function spike2Data = loadSpike2Data(dataSmrxFile,varargin)
%%LOADSPIKE2DATA loads spike2 data into a structure with channel names as
%%fields.
% Data is extracted from spike2 files via a dedicated windows specific
% library.
p = inputParser;
addParameter(p,'cedpath','')
parse(p,varargin{:})

if ~isempty(p.Results.cedpath)
    addpath(genpath(p.Results.cedpath))
    CEDS64LoadLib(p.Results.cedpath) % load in the library to read spike2
end

disp(['loading ' dataSmrxFile]);
 if ~exist(dataSmrxFile,'file') % check if file exists
        error('Could not find file');
 else 
    try
        try
            fhand = CEDS64Open(dataSmrxFile); % open if it exists
        catch
            fhand = CEDS64Open(convertStringsToChars(dataSmrxFile));
        end
    catch
        % if library is not already compiled, will end up here.
        error('Could not load smrx file, try passing the ced path as an optional argument')
    end
    if fhand == -1 % check if it opened correctly
        error('Could not open file');
    end
 end

iChanNum = CEDS64MaxChan(fhand); % get number of channels
dsec=CEDS64TimeBase(fhand); % seconds per sample
% search first 10 channels for a valid channel to find the number of
% samples
for i =1:10
    try
        maxTimeTicks = CEDS64ChanMaxTime(fhand,i);
        if maxTimeTicks<0
            continue
        end
        break
    catch
        continue
    end
end
spike2Data = struct;
for ichan=1:iChanNum
    [ iType ] = CEDS64ChanType( fhand, ichan );
    switch iType
        case 1
            %file name, channel num, max num of points, start and end time in ticks
            [fRead,fVals,i64Time] = CEDS64ReadWaveF(fhand,ichan,maxTimeTicks,0,maxTimeTicks);
            [ ~, sTitleOut ]=CEDS64ChanTitle(fhand,ichan);
            sTitleOut = strrep(sTitleOut,' ','');
            sTitleOut = strrep(sTitleOut,'_','');
            sTitleOut = regexp(sTitleOut,'[A-z]\w+','match');
            dataBundle.data = double(fVals);
            dataBundle.samplingRate = double(fRead/(dsec * maxTimeTicks));
            dataBundle.acquisitionOffset = double(i64Time/(dsec * maxTimeTicks));
            spike2Data.(sTitleOut{:}) = dataBundle;
        case 5
            % should be read markers, but hangs indefinately
            [ ~, cMarkers ] = CEDS64ReadEvents( fhand, ichan, maxTimeTicks, 0, maxTimeTicks);
            [ ~, sTitleOut ]=CEDS64ChanTitle(fhand,ichan);
            sTitleOut = strrep(sTitleOut,' ','');
            sTitleOut = strrep(sTitleOut,'_','');
            sTitleOut = regexp(sTitleOut,'[A-z]\w+','match');
            dataBundle.data = double(cMarkers);
            dataBundle.samplingRate = 1/dsec;
            spike2Data.(sTitleOut{:}) = dataBundle;
    end
end

CEDS64Close(fhand);
