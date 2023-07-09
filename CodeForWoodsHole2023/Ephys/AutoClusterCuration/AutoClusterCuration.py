# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:58:00 2021

@author: Clayton

This library contains methods for the autocurration of kilosort2 data.
To compensate for the "over-splitting" of kilosort2, correlograms between pairs
of clusters are examined for statisitcally significant differences in the number
of spikes (lognormal) for a given proportion of spikes that come before the center
bin of a correlogram. Coupled with waveform amplitude and cluster distance 
requirements, this method is tailored for merging bursting cells that might be 
split up due to their change in waveform character with successive spikes.
Complementary to merging, this library also contains methods for the 
identification of noise templates, and waveform characterization metrics.

You can also call this function as a script to process kilosort2 data from
neuropixel1(a/b?) recordings via:
    
    python AutoClusterCuration.py data_path output_path


TODO: Fully document, make it a more authentic library, speed up correlogram 
calculation by method phy2 uses, and more vetting.
"""


import numpy as np
import numba
from numba import prange
import glob 
import scipy.optimize
import sys

center_vect = np.array([0.26861492, 0.59709601, 0.44760537, 0.30068877, 0.25509305,
       0.22015293, 0.19948817, 0.15799062, 0.12345578, 0.10529966,
       0.08856479, 0.07202871, 0.05744233, 0.0513385 , 0.04072589,
       0.03523523, 0.02803807, 0.02654449, 0.02333782, 0.0183325 ,
       0.01518445, 0.01586018, 0.01275769, 0.010446  , 0.00951412,
       0.00988068, 0.00960744, 0.00596982, 0.00601363, 0.00712424,
       0.00875602, 0.00472361, 0.00476887, 0.00651582, 0.00633348,
       0.00308042, 0.0044427 , 0.00616653, 0.00524355, 0.00182876,
       0.00445183, 0.00629399])


loading_vect = np.array([ 0.69513072,  0.20185457, -0.33440671, -0.27818196, -0.28608765,
       -0.24017402, -0.19275232, -0.16316315, -0.13964936, -0.10712628,
       -0.06767548, -0.05585796, -0.03820023, -0.0211211 , -0.00128932,
        0.00301544,  0.00823093,  0.01835685,  0.025683  ,  0.02644433,
        0.02928565,  0.03450431,  0.03915692,  0.03868921,  0.04050215,
        0.04278534,  0.04402669,  0.0432333 ,  0.04322418,  0.045442  ,
        0.04814661,  0.0449823 ,  0.04421252,  0.04712965,  0.04538452,
        0.04365522,  0.04433125,  0.04751047,  0.04544363,  0.04326772,
        0.04508066,  0.04697539])


def full_file(part1,part2):
    """
    

    Parameters
    ----------
    part1 : TYPE
        DESCRIPTION.
    part2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if part1[-1] != '/':
        return part1 + '/' + part2
    else:
        return part1 + part2

def dist_2d(pt1,pt2):
    """
    

    Parameters
    ----------
    pt1 : TYPE
        DESCRIPTION.
    pt2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)

def identify_noise_clusters(templates):
    """
    

    Parameters
    ----------
    templates : TYPE
        DESCRIPTION.

    Returns
    -------
    cluster_classification : TYPE
        DESCRIPTION.

    """
    
    num_clusters = templates.shape[0]
    cluster_classification = np.zeros(num_clusters,dtype=bool)
    for it in range(num_clusters):
        main_chan = np.argmax(np.sum(abs(templates[it,:,:]),0))
        norm_fft_template = np.abs(np.fft.rfft(templates[it,:,main_chan]/np.sum(abs(templates[it,:,main_chan]))))
        norm_fft_template -= center_vect
        oned_embedding = np.sum(loading_vect*norm_fft_template) # take dot product
        if oned_embedding > 0:
            cluster_classification[it] = True
    return cluster_classification




## define function for generating error rate
@numba.jit(nopython=True,parallel=True)
def calculate_false_pos(spike_clusters, spike_times):
    """
     Calculates false positive with a firing rate model free method. The incidence
     of 2 spikes in a given time bin of the absolute refractory period of (1.5 ms)
     is used with an estimate of the units firing rate to p

    Parameters
    ----------
    spike_clusters : TYPE
        DESCRIPTION.
    spike_times : FLOAT
        To play nice with numba, this variable must be converted to the float
        data type AND BE CONVERTED TO SECONDS before being passed to this function.
        TODO: Fix this less than elegant solution
        

    Returns
    -------
    false_pos_rate : TYPE
        DESCRIPTION.

    """
    num_clusters = np.max(spike_clusters)+1
    num_bins = int(spike_times[-1]*1000/1.5+0.5)# round up number of bins to span recording time
    false_pos_rate = np.zeros((num_clusters))
    print('Calculating false positive rates...')
    for i in range(num_clusters):
        curr_data=spike_times[spike_clusters==i]
        if curr_data.shape[0] == 0: # account for unused cluster ids
            continue
        bin_inds = (curr_data*1000/1.5).astype(np.int32) # get bin indices for each spike time
        binned = np.zeros(num_bins).astype(np.int32)
        for ii in range(bin_inds.shape[0]): # bin
            binned[bin_inds[ii]] += 1
        c1 = np.sum(binned == 2)/num_bins # proportion double spikes
        c2 = np.sum(binned) # num spikes
        p2 = c1*num_bins/c2
        p1 = c2/num_bins-c1/c2*num_bins
        false_pos_rate[i] = p2*num_bins/((p1+p2)*num_bins)
    print('Done!')
    return false_pos_rate 


@numba.jit(nopython=True,parallel=True)
def calculate_false_neg(spike_clusters, amplitudes, num_histogram_bins = 500, histogram_smoothing_value = 3.0,truncate=4.0):
    # inspired from: https://github.com/AllenInstitute/ecephys_spike_sorting and scipy gaussian filter 1d
    # calculate weights for smoothing
    # remember to squeeze spike_clusters
    sd = float(histogram_smoothing_value)
    num_histogram_bins = int(num_histogram_bins)
    sigma = sd
    total_units = np.max(spike_clusters)+1
    false_neg_rate = np.zeros(total_units)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    radius = lw
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    weights = phi_x[::-1]
    print('Calculating false negative rates...')
    for i in prange(total_units):
        curr_data = amplitudes[spike_clusters == i] # get amplitudes of current clusterID
        h,b = np.histogram(curr_data, num_histogram_bins) # bin data
        h2 = np.zeros(h.shape[0]+2*lw) # pad data for smoothing. Overkill for typical purposes, but could be relevant if care about contamination of highly contaminated units
        h2[int(lw):int(lw)+h.shape[0]] = h
        h2[0:int(lw)] = h[int(lw):0:-1]
        h2[-int(lw):] = h[:-int(lw)-1:-1]
        pdf = np.correlate(h2, weights) # perform smoothing on padded data
        peak_index = np.argmax(pdf) # find maximum point in distribution
        if peak_index < int(num_histogram_bins/2):
            smooth_buff = np.zeros(2*(num_histogram_bins-peak_index-1)+1) # second buffer to more adequately smooth the data near the cutoff point
            mid_point = int((len(smooth_buff)-1)/2) # where new mid point is
            smooth_buff[mid_point:] = h[peak_index:] # assign right side of the distribution
            smooth_buff[:mid_point] = h[:peak_index:-1] # assign left side of the distribution by reversing right
            smooth_buff[mid_point-peak_index:mid_point]=h[:peak_index] # fill in left side of the distribution with the values we do have, overwrites some values in previous step
            pdf = np.correlate(smooth_buff, weights) # smooth the data 
            support = b[:-1] # reverse order for diff
            bin_size = np.mean(np.diff(support)) # get bin size
            norm_fact = 1/(np.sum(pdf)*bin_size) # get factor so we can convert the variable 'pdf' into a true pdf
            pdf = pdf*norm_fact # convert to true pdf
            mid_point = int((len(pdf)-1)/2) # new mid point since lose some edge values when smoothing. Don't worry about since very negligibly small
            fraction_missing = np.sum(pdf[:mid_point-peak_index])*bin_size # integrate cutoff values of pdf
        else: # peak index is in second half of bins, causes problems and shouldn't have any amplitude cutoff
            fraction_missing = 0
        if fraction_missing < 0.5:
            false_neg_rate[i] = fraction_missing
        else:
            false_neg_rate[i] = 0.5
    print('Done!')
    return false_neg_rate


@numba.jit(nopython=True,parallel=True)
def get_num_spikes(spike_clusters):
    """
    

    Parameters
    ----------
    spike_clusters : TYPE
        DESCRIPTION.

    Returns
    -------
    num_spikes : TYPE
        DESCRIPTION.

    """
    num_clusters = np.max(spike_clusters)+1
    num_spikes = np.zeros(num_clusters)
    for i in prange(num_clusters):
        num_spikes[i] = np.sum(spike_clusters==i)
    return num_spikes


def calculate_template_amplitudes(spike_clusters,amplitudes):
    """
    

    Parameters
    ----------
    spike_clusters : TYPE
        DESCRIPTION.
    amplitudes : TYPE
        DESCRIPTION.

    Returns
    -------
    template_amplitudes : TYPE
        DESCRIPTION.

    """
    num_clusters = np.max(spike_clusters)+1
    template_amplitudes = np.zeros(num_clusters)
    for i in range(num_clusters):
        template_amplitudes[i] = np.mean(amplitudes[spike_clusters==i])
    return template_amplitudes


def calculate_correlogram(spike_clusters, spike_times, fs=30000, bin_size=0.4, num_bins=45, num_spikes_to_consider=1000): 
    """
    

    Parameters
    ----------
    spike_clusters : TYPE
        DESCRIPTION.
    spike_times : TYPE
        DESCRIPTION.
    similar_templates : TYPE
        DESCRIPTION.
    fs : TYPE, optional
        DESCRIPTION. The default is 30000.
    bin_size : TYPE, optional
        DESCRIPTION. The default is 0.4.
    num_bins : TYPE, optional
        DESCRIPTION. The default is 45.
    num_spikes_to_consider : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    cluster_correlogram : 3D ARRAY, FLOAT
        3D array in format (bins, descending order of similiar clusters, clusters)
    arg_sorted_sim_mat : TYPE
        DESCRIPTION.

    """
    """
    fs: sampling frequency (Hz)
    bin_size: size in ms of each bin (win num_bins determines time around spike to consider)
    num_bins: number of bins to use for correlogram (with bin_size determines time around spike to consider)
    num_spikes_to_consider: maximum reference spikes to consider
    """
    num_clusters = np.max(spike_clusters)+1
    arg_sorted_sim_mat = np.zeros((num_clusters,num_clusters)) 
    # get indices that would sort matrix
    for curr_cluster in range(num_clusters):
        arg_sorted_sim_mat[curr_cluster,:] = np.arange(num_clusters) 
    cluster_correlogram = np.zeros((num_bins,num_clusters,num_clusters)) # data matrix to hold correlogram
    bin_size_samples = int(bin_size*fs/1000) 
    range_ = int((num_bins-1)/2*bin_size_samples+bin_size_samples/2) # edges of correlogram
    print('Calculating correlograms...')
    print('')
    for curr_cluster in prange(num_clusters): # iterate through each cluster
       curr_spikes = spike_times[spike_clusters==curr_cluster] # get spikes for this neruon
       np.random.shuffle(curr_spikes)
       for sim_template in range(32): # iterate through 32 most similar clusters
           sim_temp_spikes = spike_times[spike_clusters==arg_sorted_sim_mat[curr_cluster,sim_template]].astype(np.int64)
           for curr_spike in range(min((curr_spikes.shape[0],num_spikes_to_consider))):
               cluster_correlogram[:,sim_template,curr_cluster] += np.histogram(sim_temp_spikes-curr_spikes[curr_spike],bins=num_bins,range=(-range_,range_))[0]
    return cluster_correlogram,arg_sorted_sim_mat


def get_waveforms(data_path,spike_times,spike_clusters,samples_per_waveform=82,pre_samples=20,spikes_per_waveform=1000,num_channels=384,bit_volts = 0.19499999284744262695):
    """
    

    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.
    spike_times : TYPE
        DESCRIPTION.
    spike_clusters : TYPE
        DESCRIPTION.
    samples_per_waveform : TYPE, optional
        DESCRIPTION. The default is 82.
    pre_samples : TYPE, optional
        DESCRIPTION. The default is 20.
    spikes_per_waveform : TYPE, optional
        DESCRIPTION. The default is 1000.
    num_channels : TYPE, optional
        DESCRIPTION. The default is 384.
    bit_volts : TYPE, optional
        DESCRIPTION. The default is 0.19499999284744262695.

    Returns
    -------
    waveforms : TYPE
        DESCRIPTION.

    """
    num_clusters = np.max(spike_clusters)+1
    binary_file = glob.glob(full_file(data_path,'*.dat'))
    if len(binary_file) < 1: # TODO: add else
        binary_file = glob.glob(full_file(data_path,'*.bin'))
    raw_data = np.memmap(binary_file[0], dtype='int16')
    recorded_samples = int(raw_data.size/num_channels)
    data = np.reshape(raw_data, (recorded_samples,num_channels))
    post_samples = samples_per_waveform-pre_samples
    waveforms = np.zeros((samples_per_waveform,num_channels,num_clusters))
    print('Getting waveforms...')
    print('')
    for it  in prange(num_clusters):
        mean_waveform = np.zeros((samples_per_waveform,num_channels),dtype=float)
        curr_spike_times = np.copy(spike_times[spike_clusters==it])
        np.random.shuffle(curr_spike_times)
        added = np.int(0)
        spike_it = np.int(0)
        while added < spikes_per_waveform and spike_it < curr_spike_times.shape[0]:
            if curr_spike_times[spike_it]+post_samples < recorded_samples and curr_spike_times[spike_it]-pre_samples >=0:
                mean_waveform+=np.squeeze(data[int(curr_spike_times[spike_it]-pre_samples):int(curr_spike_times[spike_it]+post_samples),:].astype(float))
                added +=1
            spike_it+=1
        mean_waveform/=added
        mean_waveform-=mean_waveform[0,:] # normalize channels
        waveforms[:,:,it] = mean_waveform*bit_volts
    print('Done!')
    return waveforms


def extract_waveform_amplitudes(waveforms):
    """
    

    Parameters
    ----------
    waveforms : TYPE
        DESCRIPTION.

    Returns
    -------
    waveform_amplitudes : TYPE
        DESCRIPTION.

    """
    num_clusters = waveforms.shape[2]
    waveform_amplitudes = np.zeros(num_clusters)
    for it  in range(num_clusters):
        mean_waveform = np.squeeze(waveforms[:,:,it])
        arg_max = np.argmax(np.abs(mean_waveform))
        [x,y] = np.unravel_index(arg_max,mean_waveform.shape)
        waveform_amplitudes[it] = mean_waveform[x,y]
    return waveform_amplitudes


def get_candidate_good_clusters(total_error,num_spikes,template_amplitudes,nonnoise_clusters,total_error_rate_thresh = 0.02, num_spikes_thresh = 100, template_amplitude_thresh = 30):
    """
    

    Parameters
    ----------
    total_error : TYPE
        DESCRIPTION.
    num_spikes : TYPE
        DESCRIPTION.
    template_amplitudes : TYPE
        DESCRIPTION.
    nonnoise_clusters : TYPE
        DESCRIPTION.
    total_error_rate_thresh : TYPE, optional
        DESCRIPTION. The default is 0.02.
    num_spikes_thresh : TYPE, optional
        DESCRIPTION. The default is 100.
    template_amplitude_thresh : TYPE, optional
        DESCRIPTION. The default is 30.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    candidate_good_clusters = (total_error<total_error_rate_thresh)*(num_spikes>num_spikes_thresh)*(template_amplitudes>template_amplitude_thresh)*(nonnoise_clusters)
    return np.squeeze(np.argwhere(candidate_good_clusters))


def norm_pdf(x,mu,delta,s):
    """
    
    returns normal distribution of x given mu and dleta with vertical shift s

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    delta : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return 1/(delta*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu)/delta)**2)+s



def get_cliques(edges):
    """
    Simple clique finder.
    
    Parameters
    ----------
    edges : list of tuples with integers to denote graph nodes

    Returns
    -------
    cliques : returns list of lists with each sub list indicating the nodes of a clique

    """
    cliques = []
    for tup in edges:
        added = False
        for it, c in enumerate(cliques):
            if tup[0] in c or tup[1] in c:
                c.append(tup[0])
                c.append(tup[1])
                cliques[it] =  list(np.unique(c))
                added = True
        if not added:
            cliques.append(list(tup))
    return cliques


def calculate_correlogram_proportions(correlogram):
    """
    

    Parameters
    ----------
    correlogram : TYPE
        DESCRIPTION.

    Returns
    -------
    proportions : TYPE
        DESCRIPTION.
    correlogram_num_spikes : TYPE
        DESCRIPTION.

    """
    proportions = np.zeros(correlogram.shape[1]*correlogram.shape[2])
    correlogram_num_spikes = np.zeros(correlogram.shape[1]*correlogram.shape[2])
    # really should not have flipped dimmensionality here, but it works
    for i in range(correlogram.shape[2]):
        for ii in range(correlogram.shape[1]):
            correlogram_num_spikes[i*correlogram.shape[1]+ii] = (np.sum(correlogram[:int(correlogram.shape[0]/2),ii,i])+np.sum(correlogram[int(correlogram.shape[0]/2)+2:,ii,i]))
            proportions[i*correlogram.shape[1]+ii] = np.sum(correlogram[:int(correlogram.shape[0]/2),ii,i])/correlogram_num_spikes[i*correlogram.shape[1]+ii]
    return proportions, correlogram_num_spikes


def calculate_correlogram_means_stds(proportions,correlogram_num_spikes,num_bins = 1000):
    """
    Uses moving window to get lognormal mean and std for each proportion value by fitting a log normal distribution

    Parameters
    ----------
    proportions : TYPE
        DESCRIPTION.
    correlogram_num_spikes : TYPE
        DESCRIPTION.
    num_bins : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    correlogram_means : TYPE
        DESCRIPTION.
    correlogram_stds : TYPE
        DESCRIPTION.

    """
    bin_step=.8/num_bins
    stds = np.zeros(num_bins)
    means = np.zeros(num_bins)
    for i in range(num_bins):
        curr_data = correlogram_num_spikes[(proportions>0+i*bin_step)*(proportions<+0.2+i*bin_step)]
        means[i] = np.mean(np.log(curr_data[curr_data>3]))
        stds[i] = np.std(np.log(curr_data[curr_data>3]))
    # x values to match with the 'ys's computing in the moving window above
    bin_mids = [(0.1+i*bin_step) for i in range(num_bins)]
    # fit normal distribution curves to mean log number of spikes and std log number of spikes
    mean_popt, mean_pcov = scipy.optimize.curve_fit(norm_pdf, bin_mids, means)
    std_popt, std_pcov = scipy.optimize.curve_fit(norm_pdf, bin_mids, stds)
    # calculate expected mean number of spikes and std of spikes for each correlogram 
    correlogram_means = norm_pdf(proportions,*mean_popt)
    correlogram_stds = norm_pdf(proportions,*std_popt)
    return correlogram_means, correlogram_stds


def calculate_cluster_locations(waveforms,channel_locations):
    """
    Get cluster locations by weighting each channels' location by the clusters' waveforms

    Parameters
    ----------
    waveforms : TYPE
        DESCRIPTION.
    channel_locations : TYPE
        DESCRIPTION.

    Returns
    -------
    cluster_locations : TYPE
        DESCRIPTION.

    """
    max_abs_waves = np.max(np.abs(waveforms),axis=0)
    weightings = max_abs_waves/np.sum(max_abs_waves,axis=0)
    cluster_locations = np.zeros((waveforms.shape[2],2))
    cluster_locations[:,0] = np.dot(weightings.T,channel_locations[:,0])
    cluster_locations[:,1] = np.dot(weightings.T,channel_locations[:,1])
    return cluster_locations

def generate_candidate_merge_graph(candidate_merges,merge_proportions,key):
    """
    

    Parameters
    ----------
    candidate_merges : 2D ARRAY, BOOL
        DESCRIPTION.
    2D boolean array that maps onto merge_proportions and key. Indicates what
    clusters to consider for merges.
        DESCRIPTION.
    merge_proportions : 2D ARRAY, FLOAT64
        Encodes proportion of spikes in first half of correlogram. 
        merge_proportions[10,20] encodes the proportion of the 20th most similar
        cluster to 10's spikes that came in the first half of the corerlogram
    key : 2D ARRAY, INT
        Key mapping cluster (first dimension index) to 32 similar clusters

    Returns
    -------
    candidate_merge_graph : TYPE
        DESCRIPTION.

    """
    inter = np.argwhere(np.sum(candidate_merges,1))
    if inter.shape[0] > 1: 
        clusters_with_candidate_merges = np.squeeze(inter)
    elif inter.shape[0] == 1:
        clusters_with_candidate_merges = inter[0]
    else:
        return dict()
    candidate_merge_graph = dict()
    for i in range(clusters_with_candidate_merges.shape[0]):
        edges = list(key[clusters_with_candidate_merges[i],candidate_merges[clusters_with_candidate_merges[i],:]].astype(int))
        weights = list(merge_proportions[clusters_with_candidate_merges[i],candidate_merges[clusters_with_candidate_merges[i],:]])
        candidate_merge_graph[clusters_with_candidate_merges[i]] = [(edges[i],weights[i]) for i in range(len(edges))]
    return candidate_merge_graph


def filter_merges(candidate_merge_graph,waveform_amplitudes,cluster_locations,distance_thresh):
    """
    

    Parameters
    ----------
    candidate_merge_graph : DICTIONARY
        Dictionary graph structure where the key is the cluster id and the data
        a tuple with the first element being the cluster id of a potential 
        cluster to merge with and the second element being the proportion of 
        spikes that come before the center bin.
    waveform_amplitudes : 1D ARRAY, FLOAT64 
        Array of waveform amplitudes calculated from binary data.
    cluster_locations : 2D ARRAY, FLOAT64
        coordinates for each channel location. First dimension should be the 
        number of channels and the second dimension should have length 2 for
        2d probe layouts (other layouts not supported).
    distance_thresh : FLOAT64
        Threshold of minimum physical distance to consider merging 2 clusters.

    Returns
    -------
    merge_cliques : LIST
        A list of lists of clusters to merge. The clusters in merge_clusters[0] 
        should be merged.

    """
    to_merge = []
    for k in candidate_merge_graph.keys():
        for i in range(len(candidate_merge_graph[k])):
            if (waveform_amplitudes[k]*waveform_amplitudes[candidate_merge_graph[k][i][0]] >0) and dist_2d(cluster_locations[k,:],cluster_locations[candidate_merge_graph[k][i][0],:]) < distance_thresh: # check to make sure waveforms are the same sign (merging somatic and axonal spike is not desired behavior)
                waveform_amp_difference = abs(waveform_amplitudes[k])-abs(waveform_amplitudes[candidate_merge_graph[k][i][0]]) # subtract waveform amplitude of waveform cluster to potentially merge with
                curr_proportion = candidate_merge_graph[k][i][1]
                if(curr_proportion < 0.5 and waveform_amp_difference > 0) or (curr_proportion > 0.5 and waveform_amp_difference < 0):
                    to_merge.append((k,candidate_merge_graph[k][i][0]))
    merge_cliques = get_cliques(to_merge)
    return merge_cliques


def _merge_clusters(spike_clusters,merge_cliques):
    """
    

    Parameters
    ----------
    spike_clusters : TYPE
        DESCRIPTION.
    merge_cliques : TYPE
        DESCRIPTION.

    Returns
    -------
    new_spike_clusters : TYPE
        DESCRIPTION.

    """
    last_cluster_id = np.max(spike_clusters)
    new_spike_clusters = np.copy(spike_clusters)
    for i in range(len(merge_cliques)):
        bool_indexing_array = np.zeros(spike_clusters.shape[0],dtype=bool)
        for ii in range(len(merge_cliques[i])): # will put 1 where cluster id should be changed
            bool_indexing_array += spike_clusters==merge_cliques[i][ii]
        new_spike_clusters[bool_indexing_array] = last_cluster_id + i + 1 # give new cluster id
    return new_spike_clusters


def merge_clusters(spike_clusters,waveforms,correlogram,key,channel_locations,nonnoise_clusters = [-1],correlogram_std_thresh=3,distance_thresh=100):
    if nonnoise_clusters[0] == -1:
        nonnoise_clusters = np.arange(correlogram.shape[2])
    
    # get only putative neural correlograms
    clean_correlogram = correlogram[:,:,nonnoise_clusters]
    
    # calculate proportion of correlogram filling spikes that come before the reference spikes and the total number of spikes
    proportions, correlogram_num_spikes = calculate_correlogram_proportions(clean_correlogram)
    
    # get fit correlogram mean and std log num spikes
    correlogram_means, correlogram_stds = calculate_correlogram_means_stds(proportions,correlogram_num_spikes)
    
    # get boolean vector for which correlograms pass our threshold, which we take to be candidates for merges
    clean_candidate_merges = np.zeros(clean_correlogram.shape[1]*clean_correlogram.shape[2],dtype = bool)
    clean_candidate_merges[correlogram_num_spikes>0] = np.log(correlogram_num_spikes[correlogram_num_spikes>0]) > correlogram_means[correlogram_num_spikes>0] + correlogram_std_thresh * correlogram_stds[correlogram_num_spikes>0]
    
    # convert back to where first dimension (0 index) is cluster id
    candidate_merges = np.zeros((correlogram.shape[2],correlogram.shape[1]),dtype = bool)
    candidate_merges[nonnoise_clusters,:] = clean_candidate_merges.reshape((clean_correlogram.shape[2],clean_correlogram.shape[1]))
    
    # convert proportions to same indexing regime
    merge_proportions = np.zeros((correlogram.shape[2],correlogram.shape[1]))
    merge_proportions[nonnoise_clusters,:] = proportions.reshape((clean_correlogram.shape[2],clean_correlogram.shape[1]))
    
    # create graph data structure to encode potential merges between clusters and the proportion of spikes coming before the center bin
    candidate_merge_graph = generate_candidate_merge_graph(candidate_merges,merge_proportions,key)
    
    # get waveform amplitudes
    waveform_amplitudes = extract_waveform_amplitudes(waveforms)
    
    # get cluster locations
    cluster_locations = calculate_cluster_locations(waveforms,channel_locations)
    
    # generate list of tuples of clusters to merge
    merge_cliques = filter_merges(candidate_merge_graph,waveform_amplitudes,cluster_locations,distance_thresh)    
    
    # merge
    new_spike_clusters = _merge_clusters(spike_clusters,merge_cliques)
    
    return new_spike_clusters


def get_main_channel(waveforms):
    """
    
    Parameters
    ----------
    waveforms : TYPE
        DESCRIPTION.
    
    Returns
    -------
    waveform_amplitudes : TYPE
        DESCRIPTION.

    """
    num_clusters = waveforms.shape[2]
    main_channel = np.zeros(num_clusters,dtype = int)
    for it  in range(num_clusters):
        mean_waveform = np.squeeze(waveforms[:,:,it])
        arg_max = np.argmax(np.abs(mean_waveform))
        [x,y] = np.unravel_index(arg_max,mean_waveform.shape)
        main_channel[it] = y
    return main_channel


def append_merged_waveforms(old_spike_clusters,new_spike_clusters,waveforms):
    """
    

    Parameters
    ----------
    old_spike_clusters : TYPE
        DESCRIPTION.
    new_spike_clusters : TYPE
        DESCRIPTION.
    waveforms : TYPE
        DESCRIPTION.

    Returns
    -------
    new_waveforms : TYPE
        DESCRIPTION.

    """
    new_waveforms = np.zeros((waveforms.shape[0],waveforms.shape[1],np.max(new_spike_clusters)+1))
    new_waveforms[:,:,:waveforms.shape[2]] = waveforms
    new_cluster_ids = np.arange(np.max(old_spike_clusters)+1,np.max(new_spike_clusters)+1)
    old_num_spikes = get_num_spikes(old_spike_clusters)
    for cluster in  new_cluster_ids:
        old_clusters = np.unique(old_spike_clusters[new_spike_clusters==cluster])
        total_spikes = 0
        for old_cluster in old_clusters:
            new_waveforms[:,:,cluster] += waveforms[:,:,old_cluster] * old_num_spikes[old_cluster]
            total_spikes += old_num_spikes[old_cluster]
        new_waveforms[:,:,cluster] /= total_spikes
    return new_waveforms


def identify_noise_clusters_post_merge(templates,old_spike_clusters,new_spike_clusters):
    """
    Returns boolean vector indicating clusters identified as noise. Handles
    merged clustrs by setting the new cluster's corresponding value as true if
    at least one of the previous clusters was identified as noise.

    Parameters
    ----------
    templates : TYPE
        DESCRIPTION.
    old_spike_clusters : TYPE
        DESCRIPTION.
    new_spike_clusters : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    noise_clusters = np.zeros(np.max(new_spike_clusters)+1,dtype = bool)
    noise_clusters[:np.max(old_spike_clusters)+1] = identify_noise_clusters(templates)
    new_cluster_ids = np.arange(np.max(old_spike_clusters)+1,np.max(new_spike_clusters)+1)
    for cluster in new_cluster_ids:
        old_clusters = np.unique(old_spike_clusters[new_spike_clusters==cluster])
        for old_cluster in old_clusters:
            noise_clusters[cluster] |= noise_clusters[old_cluster]  
    return noise_clusters
 

def normalize_waveform(waveform):
    """
    

    Parameters
    ----------
    waveform : TYPE
        DESCRIPTION.

    Returns
    -------
    norm_waveform : TYPE
        DESCRIPTION.

    """
    min_val = np.min(waveform)
    min_ind = np.argmin(waveform)
    max_val = np.max(waveform[min_ind:])
    norm_waveform = np.copy(waveform)
    norm_waveform[waveform<0] = waveform[waveform<0]/np.abs(min_val)
    norm_waveform[waveform>0] = waveform[waveform>0]/np.abs(max_val)
    return norm_waveform


def get_normalized_waveforms(waveforms):
    """
    

    Parameters
    ----------
    waveforms : TYPE
        DESCRIPTION.

    Returns
    -------
    normalized_waveforms : TYPE
        DESCRIPTION.

    """
    normalized_waveforms = np.zeros((waveforms.shape[0],waveforms.shape[2]))
    num_clusters = waveforms.shape[2]
    main_channels = get_main_channel(waveforms)
    for i in range(num_clusters):
        normalized_waveforms[:,i] = normalize_waveform(waveforms[:,main_channels[i],i])
    return normalized_waveforms


def get_peak_trough_duration(waveforms,fs):
    """
    

    Parameters
    ----------
    waveforms : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.

    Returns
    -------
    pt_durations : TYPE
        DESCRIPTION.

    """
    num_clusters = waveforms.shape[2]
    pt_durations = np.zeros(num_clusters)
    main_channels = get_main_channel(waveforms)
    for i in range(num_clusters):
        waveform = waveforms[:,main_channels[i],i]
        min_ind = np.argmin(waveform)
        max_ind = np.argmax(waveform[min_ind:])
        pt_durations[i] = max_ind / fs * 1000 # convert to ms 
    return pt_durations


def get_repolarization(waveforms,fs,repolarization_time = 0.666):
    """
    

    Parameters
    ----------
    waveforms : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    repolarization_time : TYPE, optional
        DESCRIPTION. The default is 0.666 ms.

    Returns
    -------
    repolarizations : TYPE
        DESCRIPTION.

    """
    num_clusters = waveforms.shape[2]
    repolarizations = np.ones(num_clusters)
    main_channels = get_main_channel(waveforms)
    repolarization_samples = int(np.round(repolarization_time*fs/1000))
    for i in range(num_clusters):
        waveform = waveforms[:,main_channels[i],i];
        norm_waveform = normalize_waveform(waveform)
        min_ind = np.argmin(norm_waveform) # TODO change to max_ind after min_ind??
        max_ind = np.argmax(norm_waveform[min_ind:])
        if min_ind+max_ind+repolarization_samples < norm_waveform.shape[0]:
             repolarizations[i]  = 1 - norm_waveform[min_ind+max_ind+repolarization_samples]
        else:
             repolarizations[i] = np.nan
    return repolarizations


def calculate_local_variation(spike_times,spike_clusters):
    num_clusters = np.max(spike_clusters)+1
    local_variation = np.zeros(num_clusters)
    for i in range(num_clusters):
        curr_isi = np.diff(spike_times[spike_clusters == i].astype(float),axis=0)
        local_variation[i] = np.mean(3*(curr_isi[:-1]/curr_isi[1:]-1)**2/(curr_isi[:-1]/curr_isi[1:]+1)**2)
    return local_variation        


if __name__ == "__main__":
    import hdf5storage
    data_path = str(sys.argv[1])
    output_path = str(sys.argv[2])
    # load dataset
    spike_times = np.squeeze(np.load(full_file(data_path,'spike_times.npy')).astype(np.int64))
    spike_clusters = np.squeeze(np.load(full_file(data_path,'spike_templates.npy')))
    templates = np.load(full_file(data_path,'templates.npy'))
    amplitudes = np.squeeze(np.load(full_file(data_path,'amplitudes.npy')))
    similar_templates = np.load(full_file(data_path,'similar_templates.npy'))
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
    correlogram,key = calculate_correlogram(spike_clusters, spike_times, similar_templates, fs=fs, bin_size=0.4, num_bins=45, num_spikes_to_consider=1000)
    waveforms = get_waveforms(data_path,spike_times,spike_clusters,num_channels = num_channels)
    waveforms[:,191,:] = 0
    # merge clusters
    new_spike_clusters = merge_clusters(spike_clusters,waveforms,correlogram,key,channel_locations)
    # update waveforms
    waveforms = append_merged_waveforms(spike_clusters,new_spike_clusters,waveforms)
    num_spikes = get_num_spikes(new_spike_clusters)
    false_pos = calculate_false_pos(new_spike_clusters, spike_times.astype(float)/fs)
    false_neg = calculate_false_neg(new_spike_clusters, amplitudes)
    cluster_locations = calculate_cluster_locations(waveforms,channel_locations)
    main_channels = get_main_channel(waveforms)
    waveform_amplitudes = extract_waveform_amplitudes(waveforms)
    peak_trough_durations = get_peak_trough_duration(waveforms,fs)
    repolarizations = get_repolarization(waveforms,fs)
    normalized_waveforms = get_normalized_waveforms(waveforms)
    noise_clusters = identify_noise_clusters_post_merge(templates,spike_clusters,new_spike_clusters)
    # save data
    spikeData = dict()
    spikeData['spikeClusters'] = new_spike_clusters
    spikeData['spikeTimes'] = spike_times
    clusterData = dict()
    clusterData['clusterIds'] = np.arange(max(new_spike_clusters)+1,dtype=int)
    clusterData['numSpikes'] = num_spikes
    clusterData['falsePos'] = false_pos
    clusterData['falseNeg'] = false_neg
    clusterData['noise'] = noise_clusters
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
    hdf5storage.write(saveDict,path='/autoCuratedData/',filename=full_file(output_path,'autoCuratedData.mat'))




