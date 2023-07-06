function data = filterPupilData(data,medFiltWind,meanFiltWind)
    data = medfilt1(data,medFiltWind);
    if meanFiltWind > 0
        data = movmean(data,meanFiltWind);
    end
    nanx = isnan(data);
    t    = 1:numel(data);
    data(nanx) = interp1(t(~nanx), data(~nanx), t(nanx));
end
