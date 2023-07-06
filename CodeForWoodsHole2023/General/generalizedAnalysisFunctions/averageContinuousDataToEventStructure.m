function averagedStructure = averageContinuousDataToEventStructure(dataTimestamps,data,eventTimestampStructure,preSamples,postSamples,parametricAxis)
%% AVERAGECONTINUOUSDATATOEVENTSTRUCTURE averages continuous data of arbitrary dimensionality to a structure of events
averagedStructure = struct;
names = fields(eventTimestampStructure);
for i =1:length(names)
    averagedStructure.(names{i}) = averageContinuousDataToEventStructure(dataTimestamps,data,eventTimestampStructure.(names{i}),preSamples,postSamples,parametricAxis);
end

end
