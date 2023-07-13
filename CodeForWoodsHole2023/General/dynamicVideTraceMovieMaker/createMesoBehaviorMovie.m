function createMesoBehaviorMovie(alignedMovieData,alignedTraceData,movieName,varargin)
%% get plots to plot
movieFields = fields(alignedMovieData);
traceFields = fields(alignedTraceData);

%% get number of plots to plot
numMovies = length(movieFields);
numTraces = length(traceFields);

%% set default parameter values
traceSamples = 100; % default trace samples to show
frameRate = 10; % default frame rate to make video
profile = 'Motion JPEG AVI';
tracerSize = 20; % default scatter point size to indicate value of trace plot

%% check if custom parameters were passed to overwrite defaults
if any(strcmp(varargin,'Position'))
    f = figure('color','white','Position',varargin{find(strcmp(varargin,'Position'))+1});
else
    f = figure('color','white');
end
if any(strcmp(varargin,'TraceSamples'))
    traceSamples = varargin{find(strcmp(varargin,'TraceSamples'))+1};
end
if any(strcmp(varargin,'FrameRate'))
    frameRate = varargin{find(strcmp(varargin,'FrameRate'))+1};
end
if any(strcmp(varargin,'Profile'))
    profile = varargin{find(strcmp(varargin,'Profile'))+1};
end

if any(strcmp(varargin,'TracerSize'))
    tracerSize = varargin{find(strcmp(varargin,'TracerSize'))+1};
end


%% open video writer object
v = VideoWriter(movieName,profile);
v.FrameRate = frameRate;
v.Quality = 100;
open(v);
%% create tiledlayout, will have 1 row for movies and rest of rows for trace
% plots (hence the +1)
tiledlayout(2*numTraces,numMovies,'TileSpacing','tight');

%% get number of frames to create
dims = size(alignedMovieData.(movieFields{1}).data);
numFrames = dims(end);

%% check to make sure all inputs are the same length
for i =2:length(movieFields)
    dims = size(alignedMovieData.(movieFields{i}).data);
    if dims(end) ~= numFrames
        error(['The number of frames in alignedMovieData field ' movieFields{i} ' is not consistent with the number of detected frames ' num2str(numFrames)]);
    end
end

for i =1:length(traceFields)
    if length(alignedTraceData.(traceFields{i}).data) ~= numFrames
        error(['The number of frames in alignedTraceData field ' traceFields{i} ' is not consistent with the number of detected frames ' num2str(numFrames)]);
    end
end

%% structures that will dynamically hold plot data which we will update
moviePlottingData = struct;
tracePlottingData = struct;

% process video data first
for i =1:numMovies
    ax = nexttile([numMovies, 1]);
    % face cam type data
    if strcmp(alignedMovieData.(movieFields{i}).plotType,'imshow')
        moviePlottingData.(movieFields{i}) = imshow(alignedMovieData.(movieFields{i}).data(:,:,:,1));
        % process parameters for this plot
        if isfield(alignedMovieData.(movieFields{i}),'title') 
            title(alignedMovieData.(movieFields{i}).title);
        end
    % meso type data
    elseif strcmp(alignedMovieData.(movieFields{i}).plotType,'imagesc')
        moviePlottingData.(movieFields{i}) = imagesc(alignedMovieData.(movieFields{i}).data(:,:,1));
        axis off
        % process parameters for this plot
        if isfield(alignedMovieData.(movieFields{i}),'title')
            title(alignedMovieData.(movieFields{i}).title);
        end
        if isfield(alignedMovieData.(movieFields{i}),'caxis')
            caxis(alignedMovieData.(movieFields{i}).caxis);
        end
        if isfield(alignedMovieData.(movieFields{i}),'colormap')
            colormap(ax,alignedMovieData.(movieFields{i}).colormap);
        end
        if isfield(alignedMovieData.(movieFields{i}),'OutlineParcels')
            outlineParcels = alignedMovieData.(movieFields{i}).OutlineParcels;
            hold on;
            if isfield(alignedMovieData.(movieFields{i}),'OutlineParcelColors')
                outlineParcelColors = alignedMovieData.(movieFields{i}).OutlineParcelColors;
                for ii = 1:length(outlineParcels)
                    plot(outlineParcels{ii}(:,2),outlineParcels{ii}(:,1),outlineParcelColors(ii));
                end
            else
                for ii = 1:length(outlineParcels)
                    plot(outlineParcels{ii}(:,2),outlineParcels{ii}(:,1));
                end
            end
        end
    end
end
% process traces
for i =1:numTraces
    % take up whole row:
    nexttile([1 numMovies]);
    % nan data at beginning, personally like the way it looks and makes
    % things place nice
    tmp = nan(traceSamples,size(alignedTraceData.(traceFields{i}).data,2));
    tmp(end,:) = alignedTraceData.(traceFields{i}).data(1,:); % set last value of plot to first value of data passed
    tracePlottingData.(traceFields{i}) = plot(tmp);
    hold on;
    % scatter current position (tracer) at time 0
    for ii = 1:length(tracePlottingData.(traceFields{i}))
        tracePlottingData.([traceFields{i} 'Tracer' num2str(ii)]) = scatter(traceSamples,tracePlottingData.(traceFields{i})(ii).YData(end),tracerSize,tracePlottingData.(traceFields{i})(ii).Color);
    end
    % make vertical line to mark time 0
    xline(traceSamples,'--k');
    % process parameters for this plot
    if isfield(alignedTraceData.(traceFields{i}),'title')
        title(alignedTraceData.(traceFields{i}).title);
    end
    if isfield(alignedTraceData.(traceFields{i}),'ylim')
        ylim(alignedTraceData.(traceFields{i}).ylim);
    end
    if isfield(alignedTraceData.(traceFields{i}),'ylabel')
        ylabel(alignedTraceData.(traceFields{i}).ylabel);
    end
    if isfield(alignedTraceData.(traceFields{i}),'xlabel')
        xlabel(alignedTraceData.(traceFields{i}).xlabel);
    end
    if isfield(alignedTraceData.(traceFields{i}),'xticks')
        xticks(alignedTraceData.(traceFields{i}).xticks);
    end
    if isfield(alignedTraceData.(traceFields{i}),'yticks')
        yticks(alignedTraceData.(traceFields{i}).yticks);
    end
    % hacky way to get around matlab quirk
    % want to turn axis off but still have plot attributes like titles and 
    % labels, so have to do the following:
    set(gca, 'visible', 'off')
    set(findall(gca, 'type', 'text'), 'visible', 'on')
end


%% write first frame we created
writeVideo(v,getframe(gcf));

%%
for frameIt =2:numFrames
    for i =1:numMovies
        if strcmp(alignedMovieData.(movieFields{i}).plotType,'imshow')
        	moviePlottingData.(movieFields{i}).CData = alignedMovieData.(movieFields{i}).data(:,:,:,frameIt);
        elseif strcmp(alignedMovieData.(movieFields{i}).plotType,'imagesc')
            moviePlottingData.(movieFields{i}).CData = alignedMovieData.(movieFields{i}).data(:,:,frameIt);
        end
    end
    for i =1:numTraces
        tmp = nan(traceSamples,size(alignedTraceData.(traceFields{i}).data,2));
        tmp(end-min(frameIt,traceSamples)+1:end,:) = alignedTraceData.(traceFields{i}).data(frameIt-min(frameIt,traceSamples)+1:frameIt,:);
        if length(tracePlottingData.(traceFields{i})) > 1
            for ii = 1:length(tracePlottingData.(traceFields{i}))
                tracePlottingData.(traceFields{i})(ii).YData = tmp(:,ii);
                tracePlottingData.([traceFields{i} 'Tracer' num2str(ii)]).YData = tracePlottingData.(traceFields{i})(ii).YData(end);
            end    
        else
            tracePlottingData.(traceFields{i}).YData = tmp;
            tracePlottingData.([traceFields{i} 'Tracer' num2str(i)]).YData = tracePlottingData.(traceFields{i}).YData(end);
        end
    end
    drawnow;
    writeVideo(v,getframe(gcf));
end

close(v);
close(f);
end

