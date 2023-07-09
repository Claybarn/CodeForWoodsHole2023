# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:41:27 2021

@author: Clayton

This library contains methods for timestamp data. Functions that begin with
'timestamp' (singular) perform functions on pairs of on-off timestamps.
Functions that begin with 'timestamps' (plural) perform functions on arrays of
on-off timestamps. 
"""
import numpy as np

def timestamp_and(timestamp1_on,timestamp1_off,timestamp2_on,timestamp2_off):
    #_check_valid_timestamps(timestamp1_on,timestamp1_off,timestamp2_on,timestamp2_off)
    timestamp_and_on = None
    timestamp_and_off = None
    if timestamp_intersect(timestamp1_on,timestamp1_off,timestamp2_on,timestamp2_off):
        timestamp_and_on = max([timestamp1_on, timestamp2_on])
        timestamp_and_off = min([timestamp1_off, timestamp2_off])
    return timestamp_and_on, timestamp_and_off


def timestamp_or(timestamp1_on,timestamp1_off,timestamp2_on,timestamp2_off):
    #_check_valid_timestamps(timestamp1_on,timestamp1_off,timestamp2_on,timestamp2_off)
    timestamp_or_on = None
    timestamp_or_off = None
    if timestamp_intersect(timestamp1_on,timestamp1_off,timestamp2_on,timestamp2_off):
        timestamp_or_on = min([timestamp1_on, timestamp2_on])
        timestamp_or_off = max([timestamp1_off, timestamp2_off])
    return timestamp_or_on, timestamp_or_off


def timestamp_contained(timestamp1_on,timestamp1_off,timestamp2_on,timestamp2_off):
    #_check_valid_timestamps(timestamp1_on,timestamp1_off,timestamp2_on,timestamp2_off)
    timestamp_contained_on = None
    timestamp_contained_off = None
    if timestamp1_on>=timestamp2_on and timestamp1_off<=timestamp2_off:
        timestamp_contained_on = timestamp1_on
        timestamp_contained_off = timestamp1_off
    return timestamp_contained_on, timestamp_contained_off


def timestamp_intersect(timestamp1_on,timestamp1_off,timestamp2_on,timestamp2_off):
    return not (timestamp1_on>timestamp2_off or timestamp1_off<timestamp2_on)

def timestamps_not(timestamps1_on,timestamps1_off,start_timestamp=None,end_timestamp=None):
    _check_valid_timestamps(timestamps1_on,timestamps1_off)
    timestamps_not_on = timestamps1_off[:-1]
    timestamps_not_off = timestamps1_on[1:]
    if start_timestamp is not None:
        timestamps_not_on = np.insert(timestamps_not_on,0,start_timestamp) 
        if len(timestamps1_on)>0:
           timestamps_not_off = np.insert(timestamps_not_off,0,timestamps1_on[0]) 
    if end_timestamp is not None:
        timestamps_not_off = np.append(timestamps_not_off,end_timestamp)
        if len(timestamps1_off)>0:
            timestamps_not_on = np.append(timestamps_not_on, timestamps1_off[-1])
    return timestamps_not_on,timestamps_not_off


def timestamps_and(timestamps1_on,timestamps1_off,timestamps2_on,timestamps2_off):
    _check_valid_timestamps(timestamps1_on,timestamps1_off,timestamps2_on,timestamps2_off)
    timestamps_and_on = []
    timestamps_and_off = []
    for i in range(len(timestamps1_off)):
        for ii in range(len(timestamps2_off)):
            and_on,and_off = timestamp_and(timestamps1_on[i],timestamps1_off[i],timestamps2_on[ii],timestamps2_off[ii])
            if and_on is not None:
                timestamps_and_on.append(and_on)
                timestamps_and_off.append(and_off)
    return timestamps_and_on,timestamps_and_off


def timestamps_or(timestamps1_on,timestamps1_off,timestamps2_on,timestamps2_off):
    _check_valid_timestamps(timestamps1_on,timestamps1_off,timestamps2_on,timestamps2_off)
    timestamps_or_on = []
    timestamps_or_off = []
    for i in range(len(timestamps1_off)):
        for ii in range(len(timestamps2_off)):
            and_on,and_off = timestamp_or(timestamps1_on[i],timestamps1_off[i],timestamps2_on[ii],timestamps2_off[ii])
            if and_on is not None:
                timestamps_or_on.append(and_on)
                timestamps_or_off.append(and_off)
    return timestamps_or_on,timestamps_or_off


def timestamps_contained(timestamps1_on,timestamps1_off,timestamps2_on,timestamps2_off):
    _check_valid_timestamps(timestamps1_on,timestamps1_off,timestamps2_on,timestamps2_off)
    timestamps_contained_on = []
    timestamps_contained_off = []
    for i in range(len(timestamps1_off)):
        for ii in range(len(timestamps2_off)):
            and_on,and_off = timestamp_contained(timestamps1_on[i],timestamps1_off[i],timestamps2_on[ii],timestamps2_off[ii])
            if and_on is not None:
                timestamps_contained_on.append(and_on)
                timestamps_contained_off.append(and_off)
    return timestamps_contained_on, timestamps_contained_off


def timestamps_intersect(timestamps1_on,timestamps1_off,timestamps2_on,timestamps2_off):
    _check_valid_timestamps(timestamps1_on,timestamps1_off,timestamps2_on,timestamps2_off)
    timestamps_intersect_bool = np.zeros(len(timestamps1_on),dtype=bool)
    for i in range(len(timestamps1_on)):    
        ind = np.searchsorted(timestamps2_off,timestamps1_on[i])
        if (ind>0) and (ind<len(timestamps2_on)):
            timestamps_intersect_bool[i] = timestamp_intersect(timestamps1_on[i],timestamps1_off[i],timestamps2_on[ind-1],timestamps2_off[ind-1]) or timestamp_intersect(timestamps1_on[i],timestamps1_off[i],timestamps2_on[ind],timestamps2_off[ind])
    return timestamps_intersect_bool


def timestamps_merge(on_timestamps,off_timestamps,min_dur_thresh):
    _check_valid_timestamps(on_timestamps,off_timestamps)
    # inter_durs = np.array(on_timestamps[1:])-np.array(off_timestamps[:-1])
    # on_timestamps = list(on_timestamps)
    # off_timestamps = list(off_timestamps)
    # inter_durs = list(inter_durs)
    # while (len(inter_durs) > 1) and (np.min(inter_durs) < min_dur_thresh):
    #     ind = np.argmin(inter_durs)
    #     off_timestamps[ind] = off_timestamps[ind+1]
    #     del off_timestamps[ind+1]
    #     del on_timestamps[ind+1]
    #     del inter_durs[ind]
    # return on_timestamps,off_timestamps   
    merge_vect = (on_timestamps[1:]-off_timestamps[:-1])<min_dur_thresh
    new_len = len(merge_vect)-np.sum(merge_vect)+1
    new_on = np.empty(new_len)
    new_on[0] = on_timestamps[0]
    new_off = np.empty(new_len)
    new_off[0] = off_timestamps[0]
    it = 0
    for i in range(len(merge_vect)):
      if merge_vect[i]:
          new_off[it] = off_timestamps[i]
      else:
          it += 1
          new_on[it] = on_timestamps[i+1]
          new_off[it] = off_timestamps[i+1]
    return new_on,new_off



def _check_valid_timestamps(*argv):
    if len(argv[0]) != len(argv[1]):
        if len(argv)>2:
            raise ValueError('First pair of on and off timestamps have different lengths: ' + str(len(argv[0])) + ' and ' + str(len(argv[1])))
        else:
            raise ValueError('On and off timestamps have different lengths: ' + str(len(argv[0])) + ' and ' + str(len(argv[1])))
    if len(argv)>2:
        if len(argv[2]) != len(argv[3]):
            raise ValueError('Second pair of on and off timestamps have different lengths: ' + str(len(argv[2])) + ' and ' + str(len(argv[3])))

