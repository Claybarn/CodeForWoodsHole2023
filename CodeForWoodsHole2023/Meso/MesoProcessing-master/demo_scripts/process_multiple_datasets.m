
addpath(genpath('../general_utils/'));
addpath(genpath('../process_tiffs'));
addpath(genpath('../parcellation'));
addpath(genpath('../visualization/'));
addpath(genpath('../hemodynamics_removal/'));

tiffPaths = {'M:\Imaging\Clayton\RawData\NE_04-27-2021_11-00-00_airpuffs\meso';...
    'M:\Imaging\Hannah\Raw data\0_PavPilot\HB007_GCaMP\Pav9\meso'; ...
    'V:\GRABS_Data\grabAM01\10162019_grabAM01_airpuffs'};



%% experimental parameters

% for blue-uv settings
params.signalsExtraction.firstCaFrame = 1; % is the first frame of tiffs blue
params.signalsExtraction.blueuvRatio = 1; % the ratio between blue and uv imaging frames
resizeRatio = .5; % resizeing (1 for not doing anything, 0.5 to get half as many on each axis

% detrending params
params.deterend.filtLen = 150;
params.deterend.filtcutoff = 0.001;
params.deterend.method = 'FIR';
params.deterend.dozscore = 'FIR';

%debuging params
params.verbose = false;
batchSize = 1;
batches2load=1;



for it = 1:length(tiffPaths)
    %% inputs
    tiffsPath = tiffPaths{it};
    pathParts = strsplit(tiffsPath,'\');
    outputPath = fullfile('S:\CompareMesoProcessings',pathParts{end-1});
    mkNewDir(outputPath);
    disp(['load tiffs from ' tiffsPath ]);
    mov = read_batches_tiffs(batchSize, tiffsPath, batches2load, resizeRatio);
    tformfile = fullfile(outputPath, 'tform.mat');
    if ~exist(tformfile, 'file') 
        [tform,R,C] = controlPointAlignmentTransform(mov(:,:,1));
        save(tformfile, 'tform','R','C');
    else
        load(tformfile, 'tform','R','C');
    end
end

batchSize = 1000;
batches2load = 20;

for it = 1:length(tiffPaths)
    %% inputs
    tiffsPath = tiffPaths{it};
    pathParts = strsplit(tiffsPath,'\');
    outputPath = fullfile('S:\CompareMesoProcessings',pathParts{end-1});

    %% load tiffs
    mkNewDir(outputPath);
    disp(['loading tiffs from ' tiffsPath ]);
    mov = read_batches_tiffs(batchSize, tiffsPath, batches2load, resizeRatio);
    [widthPix,lengthPix,numFrames] = size(mov);

    %% demixing channels
    disp('Extracting signals from tiffs');
    [blue_raw, uv_raw, skippedframes, skippedchannels] = demix_2signals(mov, params.signalsExtraction);
    clear mov;
    
    %% align to atlas 
    disp('Align to template')
    tformfile = fullfile(outputPath, 'tform.mat');
    if ~exist(tformfile, 'file') 
        [tform,R,C] = controlPointAlignmentTransform(blue_raw);
        save(tformfile, 'tform','R','C');
    else
        load(tformfile, 'tform','R','C');
    end
    disp('Aligning to template...');
    uv_raw = transform_frames(uv_raw, tform, R, C);
    blue_raw = transform_frames(blue_raw, tform, R, C);
    %% detrending
    disp('detrending...');
    blue_raw=reshape(blue_raw, R*C, []);
    uv_raw=reshape(uv_raw, R*C, []);
    [det_blue,baseline_blue] = detrend_pixels(blue_raw, params.deterend);
    clear blue_raw;
    [det_uv,baseline_uv] = detrend_pixels(uv_raw, params.deterend);
    clear uv_raw;
    %% zscore pixelwise
    disp('zscoring...');
    [dff_blue, mval_blue, sval_blue] = zscore_pixelwise(det_blue, params.deterend.filtLen);
    clear det_blue;
    [dff_uv, mval_uv, sval_uv] = zscore_pixelwise(det_uv, params.deterend.filtLen);
    clear det_uv;
    save(fullfile(outputPath, 'zscored_blue_uv.mat'), 'dff_blue', 'dff_uv', 'R', 'C', ...
        'mval_blue', 'sval_blue', 'mval_uv', 'sval_uv', '-v7.3');
    %% Remove Hemodynamics
    disp('removing hemodynamics...');
    % select patch size
    patch_size = 13;% set this size according to the amount of time samples you have.
    % must be odd!!!!
    load('parcells_updated121519.mat','parcells_new');
    brain_mask = parcells_new.CombinedParcells>0;
    %brain_mask(:,127-10:127+9) = 0;
    %brain_mask = zeros(size(parcells_new.CombinedParcells));
    %brain_mask((patchsize-1)/2+1:end-(patchsize-1)/2,(patchsize-1)/2+1:end-(patchsize-1)/2) = 1;
    flat_mask = reshape(brain_mask,1,[]);
    [std_blue, std_uv, dff_final]  = regress_filter_3d(...
        dff_blue, ...
        dff_uv, ...
        brain_mask, ...
        patch_size, ...
        1:R*C);
    clear dff_blue;
    clear dff_uv;
    save(fullfile(outputPath, 'final_dFoF.mat'), 'std_blue', 'std_uv', 'dff_final', '-v7.3');
    disp('Done!')
end




