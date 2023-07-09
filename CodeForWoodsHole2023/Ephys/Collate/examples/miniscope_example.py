# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:46:49 2022

@author: Clayton
"""

import glob
import cv2
import numpy as np
import json
from scipy.signal import butter
from scipy.signal import sosfiltfilt
import keyboard
from natsort import natsorted
import os
from matplotlib import pyplot as plt
from tqdm import tqdm 
from collate.CollateClasses import AcquisitionSystem, Experiment
from scipy.interpolate import interp1d
import eventlogic
from scipy.io import loadmat
from scipy import signal


spatial_denoise('D:/fullTest2/My_V4_Miniscope')

for it,df in enumerate(data_files):
    meso_data[:,:,it*frames_per_file:(it+1)*frames_per_file] = np.load(df)
            
            
            
            
dff = ((meso_data[::2,::2,:]+meso_data[1::2,::2,:]+meso_data[::2,1::2,:]+meso_data[1::2,1::2,:])/4).astype(np.float32) # down sample and convert to float
dff = ((dff[:2*int(dff.shape[0]/2):2,::2,:]+dff[1:2*int(dff.shape[0]/2):2,::2,:]+dff[:2*int(dff.shape[0]/2):2,1::2,:]+dff[1:2*int(dff.shape[0]/2):2,1::2,:])/4) # down sample and convert to float
sos = butter(1, 0.001, 'lp', fs=20, output='sos')
                
for i in range(dff.shape[0]):
     for ii in range(dff.shape[1]):
         curr_lp_data = sosfiltfilt(sos,dff[i,ii,:])
         dff[i,ii,:] -= curr_lp_data
         dff[i,ii,:] /= curr_lp_data 

def spatial_denoise(dataDir,dataFilePrefix=''):
    # Values users can modify:
    #dataDir = "C:/Users/Clayton/Desktop/2022_07_29/16_43_17/My_V4_Miniscope/"
    # Modify FFT using a circle mask around center
    """ parameters for mobile moso denoising that work well """
    # Values users can modify:
    goodRadius = 150 # 500
    notchHalfWidth = 1
    centerHalfHeightToLeave = 20
    # Makes sure path ends with '/'
    if (dataDir[-1] != "/"):
        dataDir = dataDir + "/"
    """loading video metadata """
    with open(os.path.join(dataDir,'metaData.json')) as f:
        meta_data = json.load(f)
    #frame_rate = meta_data['frameRate']
    roi = meta_data['ROI']
    frames_per_file = meta_data['framesPerFile']
    startingFileNum = 0
    framesPerFile = frames_per_file
    rows = roi['height']
    cols = roi['width']
    crow,ccol = int(rows/2) , int(cols/2)
    """ create fft mask """
    #maskFFT = np.zeros((rows,cols,2), np.float32)
    maskFFT = np.zeros((rows,cols,2), np.float32)
    #cv2.circle(maskFFT,(crow,ccol),goodRadius,1,thickness=-1)
    maskFFT = cv2.circle(maskFFT,(ccol,crow,),goodRadius,1,thickness=-1)
    maskFFT[(crow+centerHalfHeightToLeave):,(ccol-notchHalfWidth):(ccol+notchHalfWidth),0] = 0
    maskFFT[:(crow-centerHalfHeightToLeave),(ccol-notchHalfWidth):(ccol+notchHalfWidth),0] = 0
    maskFFT[(crow-notchHalfWidth):(crow+notchHalfWidth),ccol+centerHalfHeightToLeave:,0] = 0
    maskFFT[(crow-notchHalfWidth):(crow+notchHalfWidth),:ccol-centerHalfHeightToLeave,0] = 0
    maskFFT[:,:,1] = maskFFT[:,:,0]
    frameStep = 1 #1 # Should be set to 1 for saving
    # Select one below -
    #compressionCodec = "FFV1"
    # --------------------
    
    fileNum = startingFileNum
    #sumFFT = None
    running = True
    if not os.path.exists(dataDir + "Denoised"):
        os.mkdir(dataDir + "Denoised")
    while (os.path.exists(dataDir + dataFilePrefix + "{:.0f}.avi".format(fileNum)) and running is True):
        cap = cv2.VideoCapture(dataDir + dataFilePrefix + "{:.0f}.avi".format(fileNum))
        #writeFile = cv2.VideoWriter(dataDir + "Denoised/" + dataFilePrefix + "denoised{:.0f}.avi".format(fileNum),  
        #                    codec, 60, (cols,rows), isColor=False)
        writeFile = os.path.join(dataDir,"Denoised",dataFilePrefix + "denoised{:.0f}.npy".format(fileNum))
        curr_data = np.zeros((rows,cols,framesPerFile),dtype=np.uint8)
        fileNum = fileNum + 1
        for frameNum in tqdm(range(0,framesPerFile, frameStep), total = framesPerFile/frameStep, desc ="Running file {:.0f}.avi".format(fileNum - 1)):
            _ = cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
            ret, frame = cap.read()
            # frameNum = frameNum + frameStep 
            # print(frameCount)
            if (ret is False):
                break
            else:
                frame = frame[:,:,1]
                dft = cv2.dft(np.float32(frame),flags = cv2.DFT_COMPLEX_OUTPUT|cv2.DFT_SCALE)
                dft_shift = np.fft.fftshift(dft)
                fshift = dft_shift * maskFFT
                f_ishift = np.fft.ifftshift(fshift)
                img_back = cv2.idft(f_ishift)
                img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
                #meanF = img_back.mean()
                #img_back = img_back * (1 + (meanFiltered[frameCount] - meanF)/meanF)
                img_back[img_back >255] = 255
                img_back = np.uint8(img_back)
                curr_data[:,:,frameNum] = img_back
        curr_data = curr_data[:,:,:frameNum]
        np.save(writeFile,curr_data)


class Miniscope(AcquisitionSystem):
    def __init__(self,data_dir,system_name):
        with open(os.path.join(data_dir,'metaData.json')) as f:
            meta_data = json.load(f)
        clock_ticks_per_second = meta_data['frameRate']
        super().__init__(data_dir,system_name,clock_ticks_per_second)
    def load(self):
        #spatial_denoise(self.data_dir) 
        data_files = natsorted(glob.glob(os.path.join(self.data_dir,'Denoised','*.npy')))
        with open(os.path.join(self.data_dir,'metaData.json')) as f:
            meta_data = json.load(f)
        roi = meta_data['ROI']
        frames_per_file = meta_data['framesPerFile']
        meta_data=meta_data
        meso_data = np.zeros((roi['height'],roi['width'],frames_per_file*len(data_files)),dtype=np.uint8)
        for it,df in enumerate(data_files):
            meso_data[:,:,it*frames_per_file:(it+1)*frames_per_file] = np.load(df)
        # add data stream to this acquisition system
        self.add_data_stream(np.arange(meso_data.shape[-1]),meso_data,'Miniscope',meta_data)
        self.add_alignment_data_stream(np.arange(meso_data.shape[-1]),np.ones(meso_data.shape[-1]),'FrameCounts')
    def preprocess(self):
        for it, data_stream in enumerate(self.data_streams):
            if data_stream.name == 'Miniscope':
                global_diff_vect = np.sum(np.diff(data_stream.data,axis=-1),axis=(0,1))
                artifact_vect = np.insert(np.abs((global_diff_vect-np.mean(global_diff_vect))/np.std(global_diff_vect)) >= 3,-1,0)
                good_vect = np.invert(artifact_vect)
                interp_inds = np.squeeze(np.argwhere(artifact_vect))
                t = np.arange(data_stream.data.shape[-1])
                for x in range(data_stream.meta_data['ROI']['height']):
                    for y in range(data_stream.meta_data['ROI']['width']):
                        f = interp1d(t[good_vect], data_stream.data[x,y,good_vect])
                        data_stream.data[x,y,artifact_vect] = f(interp_inds)
                dff = ((data_stream.data[::2,::2,:]+data_stream.data[1::2,::2,:]+data_stream.data[::2,1::2,:]+data_stream.data[1::2,1::2,:])/4).astype(float) # down sample and convert to float
                dff = ((dff[:2*int(dff.shape[0]/2):2,::2,:]+dff[1:2*int(dff.shape[0]/2):2,::2,:]+dff[:2*int(dff.shape[0]/2):2,1::2,:]+dff[1:2*int(dff.shape[0]/2):2,1::2,:])/4) # down sample and convert to float
                sos = butter(1, 0.001, 'lp', fs=data_stream.meta_data['frameRate'], output='sos')
                for i in range(dff.shape[0]):
                    for ii in range(dff.shape[1]):
                        curr_lp_data = sosfiltfilt(sos,dff[i,ii,:])
                        dff[i,ii,:] -= curr_lp_data
                        dff[i,ii,:] /= curr_lp_data
                self.data_streams[it].data = dff
            else:
                pass

class Ephys(AcquisitionSystem):
    def __init__(self,data_dir,system_name,clock_ticks_per_second):
            super().__init__(data_dir,system_name,clock_ticks_per_second)
    def load(self):
        data = loadmat(os.path.join(self.data_dir,'data.mat'))
        self.add_data_stream(np.arange(data['data']['ch1'][0,0].shape[0]),data['data']['ch1'][0,0],'LFP')
        self.add_data_stream(np.arange(data['data']['ch3'][0,0].shape[0]),data['data']['ch3'][0,0],'EMG')
        state_vect = np.squeeze(data['data']['ch2'][0,0] > 2.5)
        on_inds = np.argwhere(np.diff(state_vect.astype(int))==1)+1
        off_inds = np.argwhere(np.diff(state_vect.astype(int))==-1)+1
        events = eventlogic.events_from_vectors(np.squeeze(on_inds),np.squeeze(off_inds))
        events.merge(100)
        ons,offs = events._unravel_events()
        self.add_alignment_data_stream(ons,np.ones_like(ons),'SyncChannel')
    def preprocess(self):
        lfp_sos = signal.butter(2, .5, 'hp', fs=8000, output='sos')
        emg_sos = signal.butter(2, 10, 'hp', fs=8000, output='sos')
        for it, data_stream in enumerate(self.data_streams):
            if data_stream.name == 'LFP':
                self.data_streams[it].data = signal.sosfiltfilt(lfp_sos, self.data_streams[it].data)
            elif data_stream.name == 'EMG':
                self.data_streams[it].data = signal.sosfiltfilt(emg_sos, self.data_streams[it].data)
        pass


ephys = Ephys('S:/fullTest1','Ephys',8000)

ms = Miniscope('S:/fullTest1/2022_10_13/17_12_25/My_V4_Miniscope','Miniscope')

experiment = Experiment([ephys,ms])
experiment.process()
experiment.save_to_file(os.path.join(path,'collated_exp'+experiment_num+'_rec'+recording_num))
