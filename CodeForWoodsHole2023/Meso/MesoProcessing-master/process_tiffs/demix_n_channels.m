function sigsMov = demix_n_channels(mov,n)
    %%DEMIX_N_CHANNELS demix an arbitrary number of alternating channels from movie data 
    % mov: 3D matrix in form (R, C, number of frames)
    % n: number of channels to demix
    
    %% extract constants
    % get shape of movie
    [R,C,numFrames] = size(mov);
    % define portion of frames to compare to check for frame drops
    startR = round(R/4);
    endR = round(3*R/4);
    startC = round(C/4);
    endC = round(3*C/4);
    
    %% set fields for sigsMov variable based on number of channels (n)
    % assumes no frame drops at beginning to facilite checking for frame drops
    for ch = 1:n
        sigsMov.(['ch' num2str(ch)]) = nan(R,C,ceil(numFrames/n));
        sigsMov.(['ch' num2str(ch)])(:,:,1) = mov(:,:,ch);
    end
    
    %% iterate through all remaining frames and match each frame to most
    % similar previous frame from each channel to check for frame drops
    nextCh = 1; % next frame should be first channel, will iterate this as we go
    for f = n+1:numFrames % start with n+1 frame b/c already set first frames above
        linearCurrF = reshape(mov(startR:endR,startC:endC,f),[],1); % flatten to correlate with other frames
        corrVals = zeros(n,1); % preallocate space for corr values
        for ch = 1:n % iterate through each channel, correlating first frame with previous frames from each channel
            corrVals(ch) =  corr(linearCurrF,reshape(sigsMov.(['ch' num2str(ch)])(startR:endR,startC:endC,ceil(f/n-1)),[],1));
        end
        [~,similarCh] = max(corrVals); % get what channel current frame is most correlated with
        if similarCh == nextCh % if most similar frame is the frame we are expecting, iterate normally
            sigsMov.(['ch' num2str(nextCh)])(:,:,ceil(f/n)) = mov(:,:,f); % set channel as
            nextCh = mod(nextCh,n)+1; % iterate expected next channel
        else % frame drop
            disp(['Frame drop at frame: ' num2str(f)])
            % more complicated method that accounts for multiple frame
            % drops in a row
            currCh = nextCh; % make iterator to wrap around channels
            while currCh ~= similarCh % iterate until get to channel frame came from 
                sigsMov.(['ch' num2str(currCh)])(:,:,ceil(f/n)) = nan(R,C); % set dropped frame data to nan
                currCh = mod(currCh,n)+1; % iterate channel 
            end
            sigsMov.(['ch' num2str(similarCh)])(:,:,ceil(f/n)) = mov(:,:,f); % set frame to detected channel
            nextCh = mod(similarCh,n)+1; % iterate channel
        end
    end
    