import numpy as np
from sonpy import lib as sp
import sys
# install instructions (requires python=3.7)
# conda create --name spike2 python=3.7
# conda activate spike2 / source activate spike2 (if on windows)
# conda install pip
# pip install sonpy 
# conda install scipy

# usage:
# python extractSpike2.py path/to/file.smrx path/to/output

def extract_spike2(filename):
    file = sp.SonFile(filename,True)  
    if file.GetOpenError() != 0:
        print('Error opening ' + filename + ', error: ',sp.GetErrorString(file.GetOpenError()))
    # Data storage and function finder
    DataReadFunctions = {
        sp.DataType.Adc:        sp.SonFile.ReadInts,
        sp.DataType.EventFall:  sp.SonFile.ReadEvents,
        sp.DataType.EventRise:  sp.SonFile.ReadEvents,
        sp.DataType.EventBoth:  sp.SonFile.ReadEvents,
        sp.DataType.Marker:     sp.SonFile.ReadMarkers,
        sp.DataType.AdcMark:    sp.SonFile.ReadWaveMarks,
        sp.DataType.RealMark:   sp.SonFile.ReadRealMarks,
        sp.DataType.TextMark:   sp.SonFile.ReadTextMarks,
        sp.DataType.RealWave:   sp.SonFile.ReadFloats
    }
    DataReadTypes = {
        sp.DataType.Adc:        'Adc',
        sp.DataType.EventFall:  'EventFall',
        sp.DataType.EventRise:  'EventRise',
        sp.DataType.EventBoth:  'EventBoth',
        sp.DataType.Marker:     'Marker',
        sp.DataType.AdcMark:    'AdcMark',
        sp.DataType.RealMark:   'RealMark',
        sp.DataType.TextMark:   'TextMark',
        sp.DataType.RealWave:   'RealWave'
    }
    # spike 2 starts timestamps at 0, so can just record fs for each channel. 
    MyData = []
    channel_num = []
    channel_fs = []
    channel_type = []
    # Loop through channels retrieving data
    for i in range(file.MaxChannels()):
        if file.ChannelType(i) != sp.DataType.Off:
            try:
                MyData.append(np.array(DataReadFunctions[file.ChannelType(i)](file, i, int(file.ChannelMaxTime(i)/file.ChannelDivide(i)), 0, file.ChannelMaxTime(i))))
                channel_num.append(i)
                channel_fs.append(1/(file.ChannelDivide(i)*file.GetTimeBase()))
                channel_type.append(DataReadTypes[file.ChannelType(i)])	
            except:
                pass
    spike2_data = dict()
    for it, chan in enumerate(channel_num):
        data_dict = dict()
        data_dict['data'] = MyData[it]
        data_dict['fs'] = channel_fs[it]
        data_dict['type'] = channel_type[it]
        spike2_data[file.GetChannelTitle(chan)] = data_dict
    return spike2_data


