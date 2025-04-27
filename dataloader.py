import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wfdb
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from wfdb import processing

from scipy.io import loadmat
from tqdm import tqdm_notebook
from scipy.signal import decimate, resample
from biosppy.signals import ecg
from biosppy.signals.tools import filter_signal

import utils
from utils import *

class ECGWindowAlignedDataset(Dataset):
    def __init__(self, df, window, nb_windows, src_path):
        ''' Return window length segments from ecg signal startig from random qrs peaks
            df: trn_df, val_df or tst_df
            window: ecg window length e.g 2500 (5 seconds)
            nb_windows: number of windows to sample from record
        '''
        self.df = df
        self.window = window
        self.nb_windows = nb_windows
        self.src_path = src_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get data
        row = self.df.iloc[idx]
        filename = str(self.src_path/(row.Patient + '.hea'))
        data, hdr = load_challenge_data(filename)
        seq_len = data.shape[-1] # get the length of the ecg sequence
        #change!!
        # # Get top (normalized) features (excludes signal duration and demo feats)
        # top_feats = utils.all_feats[all_feats.filename == row.Patient][feature_names[:nb_feats]].values
        # # First, convert any infs to nans
        # top_feats[np.isinf(top_feats)] = np.nan
        # # Replace NaNs with feature means
        # top_feats[np.isnan(top_feats)] = feat_means[None][np.isnan(top_feats)]
        # # Normalize wide features
        # feats_normalized = (top_feats - feat_means) / feat_stds
        # # Use zeros (normalized mean) if cannot find patient features
        # if not len(feats_normalized):
        #     feats_normalized = np.zeros(nb_feats)[None]
        
        # Apply band pass filter
        if filter_bandwidth is not None:
            data = apply_filter(data, filter_bandwidth)
        
        # Polarity check, per selected channel
        for ch_idx in polarity_check:
            try:
                # Get BioSPPy ECG object, using specified channel
                ecg_object = ecg.ecg(signal=data[ch_idx], sampling_rate=fs, show=False)

                # Get rpeaks and beat templates
                rpeaks = ecg_object['rpeaks']
                templates, rpeaks = extract_templates(data[ch_idx], rpeaks)

                # Polarity check (based on extremes of median templates)
                templates_min = np.min(np.median(templates, axis=1)) 
                templates_max = np.max(np.median(templates, axis=1))

                if np.abs(templates_min) > np.abs(templates_max):
                    # Flip polarity
                    data[ch_idx] *= -1
                    templates *= -1
            except:
                continue

        # Detect qrs complexes (use lead II)
        xqrs = processing.XQRS(data[1,:], fs=500.)
        xqrs.detect(verbose=False)
        
        data = normalize(data)
        lbl = row[classes].values.astype(int)
        
        qrs = xqrs.qrs_inds[xqrs.qrs_inds < seq_len - self.window] # keep qrs complexes that allow a full window
        
        # Window too large, adjust sequence with padding
        if not len(qrs):
            # Add just enough padding to allow qrs find
            pad = np.abs(np.min(seq_len - window, 0)) + xqrs.qrs_inds[0] + 1
            data = np.pad(data, ((0,0),(0,pad)))
            seq_len = data.shape[-1] # get the length of the ecg sequence
            qrs = xqrs.qrs_inds[xqrs.qrs_inds < seq_len - self.window] # keep qrs complexes that allow a full window
    
        starts = np.random.randint(len(qrs), size=self.nb_windows) # get start indices of ecg segment (from qrs complex)
        starts = qrs[starts]
        ecg_segs = np.array([data[:,start:start+self.window] for start in starts])
       #change!!
        return ecg_segs, lbl

class ECGWindowPaddingDataset(Dataset):
    def __init__(self, df, window, nb_windows, src_path):
        ''' Return randome window length segments from ecg signal, pad if window is too large
            df: trn_df, val_df or tst_df
            window: ecg window length e.g 2500 (5 seconds)
            nb_windows: number of windows to sample from record
        '''
        self.df = df
        self.window = window
        self.nb_windows = nb_windows
        self.src_path = src_path

    def __len__(self):
        return len(self.df)
    #when i call a class , all the function will run?
    def __getitem__(self, idx):
        # Get data
        row = self.df.iloc[idx]
        filename = str(self.src_path/(row.Patient + '.hea'))
        data, hdr = load_challenge_data(filename)
        seq_len = data.shape[-1] # get the length of the ecg sequence
        
        # Get top (normalized) features (excludes signal duration and demo feats)
        # print("all feates@@@@@@@@@111111111")
        # print(all_feats)
        # top_feats = all_feats[all_feats.filename == row.Patient][feature_names[:nb_feats]].values
        # print("top feates@@@@@@@@@222222222") 
        # print(top_feats)
        '''change!!!
        # First, convert any infs to nans
        top_feats[np.isinf(top_feats)] = np.nan
        # Replace NaNs with feature means
        top_feats[np.isnan(top_feats)] = feat_means[None][np.isnan(top_feats)]
        # Normalize wide features
        feats_normalized = (top_feats - feat_means) / feat_stds
        # Use zeros (normalized mean) if cannot find patient features
        if not len(feats_normalized):
            feats_normalized = np.zeros(nb_feats)[None]
        '''
        # Apply band pass filter
        if filter_bandwidth is not None:
            data = apply_filter(data, filter_bandwidth)
            
        # Polarity check, per selected channel
        for ch_idx in polarity_check:
            try:
                # Get BioSPPy ECG object, using specified channel
                ecg_object = ecg.ecg(signal=data[ch_idx], sampling_rate=fs, show=False)

                # Get rpeaks and beat templates
                rpeaks = ecg_object['rpeaks']
                templates, rpeaks = extract_templates(data[ch_idx], rpeaks)

                # Polarity check (based on extremes of median templates)
                templates_min = np.min(np.median(templates, axis=1)) 
                templates_max = np.max(np.median(templates, axis=1))

                if np.abs(templates_min) > np.abs(templates_max):
                    # Flip polarity
                    data[ch_idx] *= -1
                    templates *= -1
            except:
                continue
        
        data = normalize(data)

        lbl = row[classes].values.astype(int)
        # print("lbl")
        # print(lbl)
        # Add just enough padding to allow window
        pad = np.abs(np.min(seq_len - window, 0))
        if pad > 0:
            data = np.pad(data, ((0,0),(0,pad+1)))
            seq_len = data.shape[-1] # get the new length of the ecg sequence
        
        starts = np.random.randint(seq_len - self.window + 1, size=self.nb_windows) # get start indices of ecg segment        
        ecg_segs = np.array([data[:,start:start+self.window] for start in starts])
        #change!!!
        return ecg_segs, lbl       

def load_challenge_data(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()    
    sampling_rate = int(header[0].split()[2])    
    mat_file = header_file.replace('.hea', '.mat')
    x = loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    
    # Standardize sampling rate
    if sampling_rate > fs:
        recording = decimate(recording, int(sampling_rate / fs))
    elif sampling_rate < fs:
        recording = resample(recording, int(recording.shape[-1] * (fs / sampling_rate)), axis=1)
    
    return recording, header

def normalize(seq, smooth=1e-8):
    ''' Normalize each sequence between -1 and 1 '''
    return 2 * (seq - np.min(seq, axis=1)[None].T) / (np.max(seq, axis=1) - np.min(seq, axis=1) + smooth)[None].T - 1

def extract_templates(signal, rpeaks, before=0.2, after=0.4, fs=500):
    # convert delimiters to samples
    before = int(before * fs)
    after = int(after * fs)

    # Sort R-Peaks in ascending order
    rpeaks = np.sort(rpeaks)

    # Get number of sample points in waveform
    length = len(signal)

    # Create empty list for templates
    templates = []

    # Create empty list for new rpeaks that match templates dimension
    rpeaks_new = np.empty(0, dtype=int)

    # Loop through R-Peaks
    for rpeak in rpeaks:

        # Before R-Peak
        a = rpeak - before
        if a < 0:
            continue

        # After R-Peak
        b = rpeak + after
        if b > length:
            break

        # Append template list
        templates.append(signal[a:b])

        # Append new rpeaks list
        rpeaks_new = np.append(rpeaks_new, rpeak)

    # Convert list to numpy array
    templates = np.array(templates).T

    return templates, rpeaks_new    

def apply_filter(signal, filter_bandwidth, fs=500):
        # Calculate filter order
        order = int(0.3 * fs)
        # Filter signal
        signal, _, _ = filter_signal(signal=signal, ftype='FIR', band='bandpass',
                                     order=order, frequency=filter_bandwidth, 
                                     sampling_rate=fs)
        return signal
        
            
    
