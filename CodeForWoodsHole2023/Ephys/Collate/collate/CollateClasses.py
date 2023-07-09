
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 13:25:14 2021

Architecture goal:

argmax(a * generality + b * simplicity)

where a and b are scalar coefficients and generality and simplicity are ill
defined concepts.


@author: Clayton Barnes
"""

__author__ = "Clayton Barnes"
__version__ = 0.1

import numpy as np

from scipy import signal
import warnings
import hdf5storage
import matplotlib.pyplot as plt


class SystemAlignmentError(Exception):
    def __init__(self,expression):
        self.expression = expression
 

class DataStream:
    """
    Generic class intended to hold any type of data that would be collected during an experiment.
    The main fields are data and timestamps. data is hopefully self explanatory.
    timestamps is a vector containing a timestamp for each sample in data.
    All other meta data (fs, number of channels, channel name, channel type, etc) 
    is stored in the meta_data field. Finally, the aligned element indicates if
    the data has been aligned to a common reference time. 
    """
    def __init__(self,timestamps,data,name,meta_data=dict(),time_axis=0):
        self.data = data # shape should go into meta data
        self.timestamps = timestamps
        self.name = name
        self.meta_data = meta_data
        self.time_axis = time_axis
        self.aligned = False
    
    def continuous_to_event(self,threshold=0.5,discretization_function=None):
        # diff loses 1 sample, so adjust accordingly
        diff_time = self.timestamps[1:] 
        # check if custom discretization function was passed
        if callable(discretization_function):
            # discretize continuous signal
            state_vector = discretization_function(self.data)
            # get timestamps for rise times
            on_timestamps = diff_time[np.squeeze(np.argwhere(state_vector>0))] 
            # get timestamps for fall times
            off_timestamps = diff_time[np.squeeze(np.argwhere(state_vector<0))] 
        else:
            state_vector = np.diff((self.data>threshold).astype(int))
            # get timestamps for rise times
            on_timestamps = diff_time[np.squeeze(np.argwhere(state_vector>0))] 
            # get timestamps for fall times
            off_timestamps = diff_time[np.squeeze(np.argwhere(state_vector<0))] 
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
        # override datastream values
        self.timestamps = timestamps
        self.data = data
    
    def plot(self,ax=None):
        if ax is None:
            plt.plot(self.timestamps,self.data,label=self.name)
        else:
            ax.plot(self.timestamps,self.data,label=self.name)


class AcquisitionSystem:
    """
    Generic class intended to hold 'DataStream's. If more than 1 acquisition 
    system is used in an experiment, each system should receive a common clock
    signal. This clock DataStream is kept seperated to allow for alignment across
    systems. User generated classes that inherit these fields should define a load,
    preprocess_alignment_data (which is called during the load function) and 
    preprocess functions that load data based on data directory,convert analog 
    clock data to timestamps (if need be), and dynamically preprocess
    DataStreams (see examples), respectively.
    """
    def __init__(self,data_dir,name,clock_ticks_per_second):
        self.data_dir = str(data_dir)
        self.name = str(name)
        self.clock_ticks_per_second = float(clock_ticks_per_second)
        self.data_streams = []
        self.alignment_data_stream = None
    
    def __eq__(self,other):
        try:
            return self.name == other.name
        except:
            return self.name == other
    
    def add_data_stream(self,timestamps,data,name,meta_data=dict()):
        self.data_streams.append(DataStream(timestamps,data,name,meta_data))
    
    def add_alignment_data_stream(self,timestamps,data,name,meta_data=dict()):
        self.alignment_data_stream = DataStream(timestamps,data,name,meta_data)


class Experiment:
    """
    Generic class intended to hold and perform functions for 'AcquisitionSystem's,
    and subsequently 'DataStream's. Process function triggers the loading of data,
    alignment to common timebase, and user defined preprocessing methods (see examples)
    """
    def __init__(self,acquisition_systems=None,data_file=None,experiment_info=None):
        self.acquisition_systems = acquisition_systems
        self.experiment_info = experiment_info
        self.reference_system = None
        self.data_dict = None
        if data_file is None:
            if type(self.acquisition_systems) is not list:
                self.acquisition_systems = [self.acquisition_systems]
            if self.acquisition_systems is None or len(self.acquisition_systems) < 1:
                raise SystemAlignmentError("At least 1 acquisition system is required.")
            else:
                # make first acquisition system the reference system
                self.reference_system = self.acquisition_systems[0]
        else:
            self.load_from_file(data_file)
    
    #def __str__(self):
    #    # TODO make this recursive with 
    #    curr_str = []
    #    curr_str += 'Experiment Info\n'
    #    curr_str += 'Number of acquisition systems: ' + str(len(self.acquisition_systems)) +'\n'
    #    curr_str += 'Acquisition System(s) Info'+'\n'
    #    for acquisition_system in self.acquisition_systems:
    #        curr_str += '    ' + acquisition_system.name + ', number of data streams: ' + str(len(acquisition_system.data_streams)) +'\n'
    #        for data_stream in acquisition_system.data_streams:
    #           curr_str + '        ' + data_stream.name +', number of samples: ' + str(len(data_stream.timestamps)) + ', data shape: ' + str(data_stream.data.shape), ', alignment status: ' + str(data_stream.data.aligned) +'\n'
    #    return curr_str
    
    def _load_data(self):
       for acquisition_system in self.acquisition_systems:
           acquisition_system.load()
    
    def _preprocess_data(self):
       for acquisition_system in self.acquisition_systems:
           acquisition_system.preprocess()
    
    def _align_acquisition_systems(self):
        self._verify_alignment_streams()
        # convert reference system timestamps to seconds first
        for data_stream in self.reference_system.data_streams:
            data_stream.timestamps = data_stream.timestamps/self.reference_system.clock_ticks_per_second 
            data_stream.aligned = True
        self.reference_system.alignment_data_stream.timestamps = self.reference_system.alignment_data_stream.timestamps/self.reference_system.clock_ticks_per_second 
        # covert all other timestamps into aligned seconds
        for acquisition_system in self.acquisition_systems:
            if acquisition_system is not self.reference_system:
                poly1d_fn = self.timebase_transform(self.reference_system.alignment_data_stream.timestamps, acquisition_system.alignment_data_stream.timestamps)
                # store for record keeping
                acquisition_system.alignment_function = poly1d_fn 
                for data_stream in acquisition_system.data_streams:
                    data_stream.timestamps = poly1d_fn(data_stream.timestamps)
                    data_stream.aligned = True
                # transform clock signal for sanity check
                acquisition_system.alignment_data_stream.timestamps = poly1d_fn(acquisition_system.alignment_data_stream.timestamps)
        
    def _verify_alignment_streams(self):
        for acquisition_system in self.acquisition_systems:
            if  (acquisition_system.alignment_data_stream is None) or not hasattr( self.reference_system.alignment_data_stream,'timestamps'):
                raise SystemAlignmentError(acquisition_system.name + ': alignment stream not properly defined.')
    
    def _create_data_dict(self):
        self.data_dict = dict()
        for acquisition_system in self.acquisition_systems:
            for data_stream in acquisition_system.data_streams:
                if data_stream.name not in self.data_dict.keys():
                    self.data_dict[data_stream.name] = data_stream
                else:
                    warnings.warn('Duplicate channel names, data stream overwritten.')
    
    def timebase_transform(self,ref_align_timestamps,to_align_timestamps):
        """
        Takes clock signal timestamps and returns a linear model to transform
        timestamps in the same timebase as to_align_timestamps and convert them
        to the ref_align_timestamps timebase
        """
        lags = signal.correlation_lags(len(ref_align_timestamps)-1,len(to_align_timestamps)-1,mode='full')
        corr = signal.correlate(np.diff(ref_align_timestamps),np.diff(to_align_timestamps),mode='full')
        lag = lags[np.argmax(corr)]
        if lag > 0:
            ref_align_timestamps=ref_align_timestamps[lag:]
        else:
            to_align_timestamps=to_align_timestamps[-lag:]
        min_len = min(len(ref_align_timestamps),len(to_align_timestamps))
        ref_align_timestamps = ref_align_timestamps[:min_len]
        to_align_timestamps = to_align_timestamps[:min_len]
        coef = np.polyfit(to_align_timestamps,ref_align_timestamps,1)
        poly1d_fn = np.poly1d(coef)
        return poly1d_fn
    
    def process(self):
        self._load_data()
        self._preprocess_data()
        self._align_acquisition_systems()
        self._create_data_dict()
    
    def save_to_file(self,file):
        to_save = dict()
        for data_stream_name in self.data_dict:
            to_save[data_stream_name] = {'timestamps': [ self.data_dict[data_stream_name].timestamps],
                                         'data': [self.data_dict[data_stream_name].data],
                                         'meta_data': self.data_dict[data_stream_name].meta_data}
        hdf5storage.savemat(file, to_save, format='7.3', compress=False)
    
    def load_from_file(self,file):
        self.data_dict = dict()
        loaded_data = hdf5storage.loadmat(file)
        for data_stream_name in loaded_data:
            self.data_dict[data_stream_name] = DataStream(loaded_data[data_stream_name].timestamps,loaded_data[data_stream_name].data,data_stream_name,loaded_data[data_stream_name].meta_data)
    
    def plot(self,data_stream_names,ax=None):
        if type(data_stream_names) is list:
            if ax is None:
                for data_stream_name in data_stream_names:
                    self.data_dict[data_stream_name].plot()
            elif type(ax) is list:
                if len(ax) == 1:
                    self.data_dict[data_stream_name].plot(ax=ax[0])
                elif len(ax) == len(data_stream_names):
                    for it,data_stream_name in enumerate(data_stream_names):
                        self.data_dict[data_stream_name].plot(ax=ax[it])
                else:
                    raise ValueError('The length of "ax" is greater than 1 and not equal to the length of "data_stream_names".')

