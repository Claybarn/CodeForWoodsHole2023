# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 20:34:01 2022

@author: Clayton


This realistic example shows how to override the base Collate Classes for use in real experiments.

"""
  
        
import numpy as np
from scipy import signal
import hdf5storage 
import glob
import neuralynx_io
import os
import platform
from xml.etree import ElementTree
from packaging import version
import sys
from timestamp_lib import timestamps_merge
from collate.CollateClasses import AcquisitionSystem, Experiment



class Neuralynx(AcquisitionSystem):
    def __init__(self,data_dir,system_name,clock_ticks_per_second,experiment,recording, select_channels= None):
        super().__init__(data_dir,system_name,clock_ticks_per_second)
        self.experiment = str(experiment)
        self.recording = str(recording)
        self.select_channels = select_channels
    def load(self):
        # define what channel name we will designate our common clock signal
        alignment_channel_name = 'ArduinoClock'
        # if no select channels, load all nl data in folder
        if self.select_channels is None: 
            # gather all files that end in .ncs
            data_files = glob.glob(os.path.join(self.data_dir,'experiment' + self.experiment, 'recording' + self.recording,'*.ncs'))
        else:
            # gather just the files we asked for
            data_files = [os.path.join(self.data_dir,'experiment' + self.experiment, 'recording' + self.recording,select_channel + '.ncs') for select_channel in self.select_channels]
        for data_file in data_files:
            # load neuralynx data
            data_dict = neuralynx_io.load_ncs(data_file)
            # make channel name the file name
            channel_name = data_file.replace('\\','/').split('/')[-1].split('.')[0]
            # define dictionary to hold meta data 
            meta_data = dict()
            meta_data['fs'] = data_dict['sampling_rate']
            meta_data['data_units'] = data_dict['data_units']
            meta_data['num_samples'] = data_dict['data'].shape[0]
            meta_data['num_channels'] = 1
            meta_data['stream_type'] = 'continuous'
            # add data stream to this acquisition system
            self.add_data_stream(data_dict['time'],data_dict['data'],channel_name,meta_data)
            if channel_name == alignment_channel_name:
                # preprocess alignment datastream and remove it from list of other datastreams
                self.preprocess_alignment_data(self.data_streams.pop()) 
    def preprocess_alignment_data(self,alignment_data_stream):
        # filter and convert our continuous data into event style data.
        # Have to pass as a list element to pass like a pointer 
        alignment_data_stream = med_filt_continuous_to_event(alignment_data_stream,window=7)
        # We just want the onsets as timestamps, not the offsets
        alignment_data_stream.timestamps = alignment_data_stream.timestamps[alignment_data_stream.data == 1]
        alignment_data_stream.data = alignment_data_stream.data[alignment_data_stream.data == 1]
        self.alignment_data_stream = alignment_data_stream
    def preprocess(self):
        for it, data_stream in enumerate(self.data_streams):
            if data_stream.name == 'Camera' or  data_stream.name == 'VisPD':
                self.data_streams[it] = med_filt_continuous_to_event(data_stream,window=7)
            elif data_stream.name == 'wheel':
                # get wheel velocity for this data stream
                data_stream.timestamps,data_stream.data = extract_wheel_velocity(data_stream.timestamps,data_stream.data,data_stream.meta_data['fs'],downsample_factor=40,wheel_diameter = 15,window = int(data_stream.meta_data['fs']/40/2))
                # get locomotion onset and offset times
                on_timestamps, off_timestamps = extract_wheel_states(data_stream.timestamps,data_stream.data,data_stream.meta_data['fs'])
                # create conglomerate timestamp vector
                timestamps = np.hstack((on_timestamps,off_timestamps))
                # create on-off indicators for data field
                data = np.hstack((np.ones(on_timestamps.shape),-np.ones(off_timestamps.shape)))
                # get sorting indices
                I = np.argsort(timestamps) 
                # sort timestamps
                timestamps = timestamps[I]
                # sort to match sorted timestamps
                data = data[I]
                # bit a of faux pa to add an element to this list you are iterating through, but oh well
                # notice we are keeping both the wheel speed and locomotion start and end times
                # by not converting the wheel velocity data stream into events
                self.add_data_stream(timestamps,data,'Locomotion') 
            else:
                pass


class OpenEphys(AcquisitionSystem):  
    def __init__(self,data_dir,system_name,clock_ticks_per_second,experiment,recording):
        super().__init__(data_dir,system_name,clock_ticks_per_second)
        self.experiment = str(experiment)
        self.recording = str(recording)
    
    def load(self):
        # first loading the meta data so we know the OE version (important for later steps)
        meta_data = self._load_xml_info()
        # downsample lfp data, taking into account how the files are saved by OE version
        # here we 'cheat' a bit and preprocess in our load function, but this way
        # we can keep the memory load low
        downsampled_data_timestamps,downsampled_data = self._load_and_preprocess_lfp(oe_version=meta_data['OE_Version'])
        # add downsampled data to data stream
        self.add_data_stream(downsampled_data_timestamps,downsampled_data,'lfp',meta_data)
        
        # load spike data
        timestamps,data,ks2_meta_data = self._load_autocurate_data()
        # add spike data as data stream
        self.add_data_stream(timestamps,data,'spikes',ks2_meta_data)
        
        # get the common clock data for OE
        timestamps,data = self._load_and_preprocess_alignment_data(oe_version= meta_data['OE_Version'])
        # add clock data as a data stream
        self.add_data_stream(timestamps,data,'ClockSignal',dict())
        # treat as alignment datastream
        self.preprocess_alignment_data(self.data_streams.pop())
    def preprocess_alignment_data(self,alignment_data_stream):
        self.alignment_data_stream = alignment_data_stream
    
    def preprocess(self):
        pass
    
    def _load_and_preprocess_lfp(self,downsample_factor=5,num_channels=384,oe_version='0.6.1'):
        """ load LFP data, downsampling by factor of 5 by default"""
        # load raw lfp data based on OE version
        if version.parse(oe_version)>=version.parse('0.6.1'): # should check the exact version where sample number numpy file name changed
            raw_data = np.fromfile(os.path.join(self.data_dir,'experiment' + self.experiment,'recording' + self.recording, 'continuous','Neuropix-PXI-100.ProbeA-LFP','continuous.dat'),dtype=np.int16)
        else:
            raw_data = np.fromfile(os.path.join(self.data_dir,'experiment' + self.experiment,'recording' + self.recording, 'continuous','Neuropix-PXI-100.1','continuous.dat'),dtype=np.int16)
        # get number of samples. Should be (total number of elements)/(number of channels)
        num_samples = int(raw_data.shape[0]/num_channels)
        # reshape data to be samples by channels
        data = raw_data.reshape((num_samples,num_channels))
        # clear data
        del raw_data
        if version.parse(oe_version)>=version.parse('0.6.1'): # should check the exact version where sample number numpy file name changed
            timestamps = np.load(os.path.join(self.data_dir,'experiment' + self.experiment,'recording' + self.recording,'continuous','Neuropix-PXI-100.ProbeA-LFP','sample_numbers.npy'))*12
        else:
            timestamps = np.load(os.path.join(self.data_dir,'experiment' + self.experiment,'recording' + self.recording,'continuous','Neuropix-PXI-100.1','timestamps.npy'))*12
        # downsample data by binned averaging
        downsampled_timestamps,downsampled_data = downsample_data(timestamps,data,downsample_factor)
        return downsampled_timestamps,downsampled_data 
    
    def _load_xml_info(self,lfp=True):
        # function to load settings.xml file associated with OE experiments
        all_info = XML2Dict(os.path.join(self.data_dir,'settings.xml'))
        try:
            # if this succeeds without error, then new stype of saving settings.xml files
            if lfp:
                meta_data = all_info['SIGNALCHAIN']['PROCESSOR']['Neuropix-PXI']['STREAM']['ProbeA-LFP']
            else:
                meta_data = all_info['SIGNALCHAIN']['PROCESSOR']['Neuropix-PXI']['STREAM']['ProbeA-AP']
            meta_data['OE_Version'] = all_info['INFO']['VERSION']
            # should work for newer version of OE (tested 0.6.1)
            meta_data['num_channels'] = all_info['SIGNALCHAIN']['PROCESSOR']['Neuropix-PXI']['STREAM']['ProbeA-AP']['channel_count']
            # get channel positions
            x_pos = []
            for i in list(all_info['SIGNALCHAIN']['PROCESSOR']['Neuropix-PXI']['EDITOR']['NP_PROBE']['ELECTRODE_XPOS'].keys()):
                x_pos.append(float(all_info['SIGNALCHAIN']['PROCESSOR']['Neuropix-PXI']['EDITOR']['NP_PROBE']['ELECTRODE_XPOS'][i]))
            y_pos = []
            for i in list(all_info['SIGNALCHAIN']['PROCESSOR']['Neuropix-PXI']['EDITOR']['NP_PROBE']['ELECTRODE_YPOS'].keys()):
                y_pos.append(float(all_info['SIGNALCHAIN']['PROCESSOR']['Neuropix-PXI']['EDITOR']['NP_PROBE']['ELECTRODE_YPOS'][i]))
            meta_data['channel_map'] = np.vstack((np.array(x_pos),np.array(y_pos))).T
        except:
            # process old style settings.xml file
            meta_data = dict()
            meta_data['OE_Version'] = all_info['INFO']['VERSION']
            # should work for older version of OE (tested 0.4.5)
            meta_data['num_channel'] = int(len(list(all_info['SIGNALCHAIN']['PROCESSOR']['Sources/Neuropix-PXI']['CHANNEL_INFO']['CHANNEL'].keys()))/2)
            meta_data['channel_map'] = None
        return meta_data
    
    def _load_autocurate_data(self):
        # have to backstep here to grab the kilsort data, data_dir points to the OE folder    
        #autocurated_data = loadmat_v7(os.path.join(self.data_dir,'../Kilosort2','experiment' + self.experiment,'recording' + self.recording,'autoCuratedData.mat'))
        autocurated_data = hdf5storage.loadmat(os.path.join(self.data_dir,'../Kilosort2','experiment' + self.experiment,'recording' + self.recording,'autoCuratedData.mat'))
        # get timestamps data from dictionary
        timestamps = autocurated_data['autoCuratedData']['spikeData']['spikeTimes']
        # get clusters from dictionary
        data = autocurated_data['autoCuratedData']['spikeData']['spikeClusters']
        # set cluster metrics as meta data
        meta_data = autocurated_data['autoCuratedData']['clusterData']
        return timestamps,data,meta_data
    
    def _load_and_preprocess_alignment_data(self,oe_version='0.6.1'):
        # newer verions of OE changes timestamps.npy to be in units of seconds when they were previously sample numbers (now called sample_numbers.npy)
        # we are going to load the data corresponding to sample number and convert to seconds later for consistency with kilosort2 data which only outputs spikes in sample number
        if version.parse(oe_version)>=version.parse('0.6.1'):
            timestamps = np.load(os.path.join(self.data_dir,'experiment' + self.experiment,'recording' + self.recording,'events','Neuropix-PXI-100.ProbeA-AP','sample_numbers.npy'))
            channel_states = np.load(os.path.join(self.data_dir,'experiment' + self.experiment,'recording' + self.recording,'events','Neuropix-PXI-100.ProbeA-AP','channel_states.npy'))
        else:
            timestamps = np.load(os.path.join(self.data_dir,'experiment' + self.experiment,'recording' + self.recording, 'events','Neuropix-PXI-100.0','TTL_1','timestamps.npy'))
            channel_states = np.load(os.path.join(self.data_dir,'experiment' + self.experiment,'recording' + self.recording, 'events','Neuropix-PXI-100.0','TTL_1','channel_states.npy'))
        if channel_states[0] == 1:
            timestamps  = timestamps[::2]
        else:
            timestamps  = timestamps[1::2]
        return timestamps,np.ones_like(timestamps)


def med_filt_continuous_to_event(data_stream,window=3):
    """ 
    
    Performs median filtering and converts data stream to an event-stype data stream
            
    """
    data_stream.data = np.abs(signal.medfilt(data_stream.data,7))
    threshold = (np.min(data_stream.data)+np.max(data_stream.data))/2
    data_stream.continuous_to_event(threshold=threshold)
    return data_stream

def downsample_data(timestamps,data,downsample_factor,circular=False):
    """ performs downsampling by taking average of temporal num_bins of size downsample_factor """
    # if vector, make shape (num_samples,1)
    if len(data.shape) == 1:
        data = np.expand_dims(data,1)
    if len(data.shape) != 2:
        raise ValueError("data must be 1 or 2 dimensional.")
        # Although, it is fun to think about generalizing this to an arbitrary dimensionality...
    # implicitly round down to get number of bins
    num_bins = int(data.shape[0]/downsample_factor)
    # number of stragglers that don't fit in to a bin
    remainder = np.mod(data.shape[0],downsample_factor)
    if remainder:
        # if stragglers, make space to take their average
        downsampled_data = np.empty((num_bins+1,data.shape[1]),dtype=data.dtype)
        if circular:
            # take circular mean of bins
            downsampled_data[:-1,:] = circular_mean(data[:num_bins*downsample_factor,:].reshape((num_bins,downsample_factor,data.shape[1])),1)
            # deal with stragglers
            downsampled_data[-1,:] = circular_mean(data[-remainder:,:],0)
        else:
            # take mean of bins
            downsampled_data[:-1,:] = np.mean(data[:num_bins*downsample_factor,:].reshape((num_bins,downsample_factor,data.shape[1])),1)
            # deal with stragglers
            downsampled_data[-1,:] = np.mean(data[-remainder:,:],0)
    else:
         if circular:
              # take circular mean of bins
             downsampled_data = circular_mean(data[:num_bins*downsample_factor,:].reshape((num_bins,downsample_factor,data.shape[1])),1)
         else:
              # take mean of bins
             downsampled_data = np.mean(data[:num_bins*downsample_factor,:].reshape((num_bins,downsample_factor,data.shape[1])),1)
    downsampled_timestamps = timestamps[int(downsample_factor/2)::downsample_factor]
    # account for having mod(data.shape[0],downsample_factor) < int(downsample_factor/2) by extrapolating timestamps
    if downsampled_timestamps.shape[0]+1 == downsampled_data.shape[0]:
        downsampled_timestamps = np.append(downsampled_timestamps,downsampled_timestamps[-1] + np.mean(np.diff(downsampled_timestamps)))
    return downsampled_timestamps,downsampled_data


def circular_mean(angles,axis=0):
    """ calculate circular mean """
    mean_sin = np.mean(np.sin(angles),axis)
    mean_cos = np.mean(np.cos(angles),axis)
    mean_angle = np.arctan2(mean_sin,mean_cos)
    mean_angle = np.mod(mean_angle,np.pi*2)
    return mean_angle


def extract_wheel_velocity(timestamps,data,sampling_frequency,downsample_factor=1,wheel_diameter = 15,window = 1):
    # calculate what sampling frequency will be after downsampling
    new_sampling_frequency = sampling_frequency/downsample_factor
    # find min and max wheel position values to convert to wheel voltage to phase
    min_wheel_position = np.min(data)
    max_wheel_position = np.max(data-min_wheel_position)
    # calculate wheel phase
    wheel_phase = (data-min_wheel_position)/max_wheel_position*2*np.pi
    # downsample the wheel phase
    downsampled_timestamps,downsampled_phase = downsample_data(timestamps,wheel_phase,downsample_factor,circular=True)
    # preallocate space for the change in phase, allows us to handle loss of one sample from diff() elegantly
    phase_diff = np.zeros_like(downsampled_phase)
    phase_diff[1:] = np.diff(downsampled_phase,axis=0)
    # unwrap phase
    phase_diff[phase_diff>np.pi] =  np.pi*2-phase_diff[phase_diff>np.pi]
    phase_diff[phase_diff<-np.pi] = np.pi*2+ phase_diff[phase_diff<-np.pi]
    cumulative_phase = signal.medfilt(np.cumsum(phase_diff),3) # do some light noise removal
    # make smoothing window odd 
    if window % 2 == 0:
        window += 1
    # smooth unwrapped phase
    smooth_cumulative_phase = np.convolve(cumulative_phase,np.ones(window)/window,'valid')
    # preallocate space to accomodate samples lost from 'valid' convolution and diff()nelegantly
    wheel_speed = np.zeros_like(cumulative_phase)
    wheel_speed[int(window/2)+1:-int(window/2)] = np.diff(smooth_cumulative_phase)*wheel_diameter/2*new_sampling_frequency
    return downsampled_timestamps,wheel_speed


def extract_wheel_states(wheel_time,wheel_speed,timebase_sampling_frequency,run_speed_min=1,merge_thresh_seconds = 3,run_duration_seconds=3):
    # find samples above threshold
    state_vect = wheel_speed>run_speed_min
    # find on and off times for threshold crossings
    change_pts = np.diff(state_vect.astype(int)) # calculate change in state
    raw_wheel_on = wheel_time[np.squeeze(np.nonzero(change_pts==1))+1] # convert indices to time
    raw_wheel_off = wheel_time[np.squeeze(np.nonzero(change_pts==-1))+1] # convert indices to time
    # merge 
    m_wheel_on,m_wheel_off = timestamps_merge(raw_wheel_on,raw_wheel_off,timebase_sampling_frequency*merge_thresh_seconds)
    m_wheel_on = np.array(m_wheel_on)
    m_wheel_off = np.array(m_wheel_off)
    wheel_on = m_wheel_on[(m_wheel_off-m_wheel_on)>timebase_sampling_frequency*run_duration_seconds]
    wheel_off = m_wheel_off[(m_wheel_off-m_wheel_on)>timebase_sampling_frequency*run_duration_seconds]
    return wheel_on, wheel_off


"""
on windows, hdf5storage returns numpy array of objects which is difficult to work with,
so convert to dictionary.

 
 """
def loadmat_v7(filename):
    data = hdf5storage.loadmat(filename)
    if platform.system() == 'Windows':
        data_dict = dict()
        keys = list(data.keys())
        for key in keys:
            data_dict[key] = rec_fun(data[key])
        return data_dict
    else:
        return data

def rec_fun(arg):
    """ recursive functions for loading matlab V7.3 files into a dictionary """
    data_dict = dict()
    try:
        keys = arg[0,0].dtype.names
    except:
        keys = arg.dtype.names
    if keys is None:
        return arg[0,0]
    else:
        for key in keys:
            try:
                data_dict[key] = rec_fun(arg[key])
            except:
                data_dict[key] = rec_fun(arg[0,0][key])
        return data_dict


""" XML reading code (see below) is from open ephys analysis tools, and thus subject to the following license """

"""
Allen Institute Software License - This software license is the 2-clause BSD
license plus a third clause that prohibits redistribution for commercial
purposes without further permission.
 
Copyright 2018. Allen Institute. All rights reserved.
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
 
3. Redistributions for commercial purposes are not permitted without the
Allen Institute's written permission.
For purposes of this license, commercial purposes is the incorporation of the
Allen Institute's software into anything for which you will charge fees or
other compensation. Contact terms@alleninstitute.org for commercial licensing
opportunities.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

def Root2Dict(El):
    Dict = {}
    if El.getchildren():
        for SubEl in El:
            if SubEl.keys():
                if SubEl.get('name'):
                    if SubEl.tag not in Dict: Dict[SubEl.tag] = {}
                    Dict[SubEl.tag][SubEl.get('name')] = Root2Dict(SubEl)
                    Dict[SubEl.tag][SubEl.get('name')].update(
                        {K: SubEl.get(K) for K in SubEl.keys() if K != 'name'}
                    )
                else:
                    Dict[SubEl.tag] = Root2Dict(SubEl)
                    Dict[SubEl.tag].update(
                        {K: SubEl.get(K) for K in SubEl.keys() if K != 'name'}
                    )
            else:
                if SubEl.tag not in Dict: Dict[SubEl.tag] = Root2Dict(SubEl)
                else:
                    No = len([k for k in Dict if SubEl.tag in k])
                    Dict[SubEl.tag+'_'+str(No+1)] = Root2Dict(SubEl)
        return(Dict)
    else:
        if El.items(): return(dict(El.items()))
        else: return(El.text)


def XML2Dict(File):
    Tree = ElementTree.parse(File); Root = Tree.getroot()
    Info = Root2Dict(Root)
    return(Info)




if __name__ == "__main__":
    path = str(sys.argv[1])
    experiment_num = str(1)
    recording_num = str(1)
    oe = OpenEphys(os.path.join(path,'OpenEphys'),'OpenEphys',30000,experiment_num,recording_num)
    nl = Neuralynx(os.path.join(path,'Neuralynx'),'Neuralynx',40000,experiment_num,recording_num,select_channels=['ArduinoClock','VisPD','wheel'])
    
    experiment = Experiment([oe,nl])
    experiment.process()
    experiment.save_to_file(os.path.join(path,'collated_exp'+experiment_num+'_rec'+recording_num))



