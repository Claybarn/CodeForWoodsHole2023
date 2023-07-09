# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 20:13:31 2021

@author: Clayton
"""
from AutoClusterCuration import AutoClusterCuration as acc
import hdf5storage
import numpy as np
import sys

data_path = str(sys.argv[1])
output_path = str(sys.argv[2])
# load dataset
spike_times = np.squeeze(np.load(acc.acc.full_file(data_path,'spike_times.npy')).astype(np.int64))
spike_clusters = np.squeeze(np.load(acc.full_file(data_path,'spike_templates.npy')))
templates = np.load(acc.full_file(data_path,'templates.npy'))
amplitudes = np.squeeze(np.load(acc.full_file(data_path,'amplitudes.npy')))
similar_templates = np.load(acc.full_file(data_path,'similar_templates.npy'))
# neuropixel 1a/b constants
num_channels = 384
fs = 30000
# neuropixel channel map
channel_locations = np.zeros((num_channels,2)) # x and y
# set x positions in um
channel_locations[::4,0] = 0 
channel_locations[1::4,0] = 20
channel_locations[2::4,0] = 40 
channel_locations[3::4,0] = 60 
# set y positions in um
channel_locations[::4,1] = np.arange(num_channels/4) * 40
channel_locations[1::4,1] = np.arange(num_channels/4) * 40
channel_locations[2::4,1] =np.arange(num_channels/4) * 40 + 20
channel_locations[3::4,1] = np.arange(num_channels/4) * 40 + 20
# process  kilosort2 data
correlogram,key = acc.calculate_correlogram(spike_clusters, spike_times, similar_templates, fs=30000, bin_size=0.4, num_bins=45, num_spikes_to_consider=1000)
waveforms = acc.get_waveforms(data_path,spike_times,spike_clusters,num_channels = num_channels)
waveforms[:,191,:] = 0 # Ignore BS reference channel
# merge clusters
new_spike_clusters = acc.merge_clusters(spike_clusters,waveforms,correlogram,key,channel_locations)
# update waveforms
waveforms = acc.append_merged_waveforms(spike_clusters,new_spike_clusters,waveforms)
num_spikes = acc.get_num_spikes(new_spike_clusters)
false_pos = acc.calculate_false_pos(spike_clusters, spike_times.astype(float)/30000)
false_neg = acc.calculate_false_neg(spike_clusters, amplitudes)
cluster_locations = acc.calculate_cluster_locations(waveforms,channel_locations)
main_channels = acc.get_main_channel(waveforms)
waveform_amplitudes = acc.extract_waveform_amplitudes(waveforms)
peak_trough_durations = acc.get_peak_trough_duration(waveforms,fs)
repolarizations = acc.get_repolarization(waveforms,fs)
normalized_waveforms = acc.get_normalized_waveforms(waveforms)
noise_clusters = acc.identify_noise_clusters_post_merge(templates,spike_clusters,new_spike_clusters)
# get what units were automerged
max_spike_templates = np.max(spike_clusters)
max_new_spike_templates = np.max(new_spike_clusters)
was_auto_merged = np.zeros(max_new_spike_templates+1,dtype = bool)
was_auto_merged[max_spike_templates+1:] = True

# get recording start time (kilosort2 output times are relative to start of ephys recording)
ap_timestamps = np.load(acc.full_file(data_path,'timestamps.npy'))
# save data
spikeData = dict()
spikeData['spikeClusters'] = new_spike_clusters
spikeData['spikeTimes'] = spike_times + ap_timestamps[0]
clusterData = dict()
clusterData['clusterIds'] = np.arange(max(spike_clusters),dtype=int)
clusterData['numSpikes'] = num_spikes
clusterData['falsePos'] = false_pos
clusterData['falseNeg'] = false_neg
clusterData['noise'] = noise_clusters
clusterData['autoMerged'] = was_auto_merged 
clusterData['clusterLocations'] = cluster_locations
clusterData['waveformAmplitudes'] = waveform_amplitudes
clusterData['mainChannels'] = main_channels
clusterData['peakTroughDurations'] = peak_trough_durations
clusterData['repolarizations'] = repolarizations
clusterData['normalizedWaveforms'] = normalized_waveforms
clusterData['waveforms'] = waveforms
saveDict = dict()
saveDict['spikeData'] = spikeData
saveDict['clusterData'] = clusterData
hdf5storage.write(saveDict,path='/autoCuratedData/',filename=acc.full_file(output_path,'autoCuratedData.mat'))