function confidenceBand(t,data,confidence,dimension,varargin)
    ts = tinv(confidence+(1-confidence)/2,sum(~isnan(data),dimension)-1);      % T-Score
    shadedErrorBar(t,squeeze(nanmean(data,dimension)),squeeze(nanstd(data,[],dimension))/squeeze(sqrt(sum(~isnan(data),dimension)))*squeeze(ts),varargin{:})