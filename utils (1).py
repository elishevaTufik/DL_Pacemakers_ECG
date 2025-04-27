import argparse
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from wfdb import processing

import torch
import torch.nn as nn
import torch.nn.functional as F

# Parameters
debug = False
patience = 10
batch_size = 128 #* torch.cuda.device_count()
window = 15*500
dropout_rate = 0.2
deepfeat_sz = 64
padding = 'zero' # 'zero', 'qrs', or 'none'
fs = 500
filter_bandwidth = [3, 45]
polarity_check = []
model_name = 'ctn'

# Transformer parameters
d_model = 256   # embedding size
nhead = 8       # number of heads
d_ff = 2048     # feed forward layer size
num_layers = 8  # number of encoding layers

do_train = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ch_idx = 1
nb_demo = 2
nb_feats = 20
thrs_per_class = False
class_weights = None
#change!!!
classes = sorted(['10370003','426783006'])
char2dir = {
        'Q' : 'Training_2',
        'A' : 'Training_WFDB',
        'E' : 'WFDB',
        'S' : 'WFDB',
        'H' : 'WFDB',
        'I' : 'WFDB'
    }

# Load all features dataframe
data_df = pd.read_csv('underSampling_with_folds.csv', index_col=0)
all_feats = pd.concat([pd.read_csv(f, index_col=0) for f in list(Path('feats/').glob(f'*/*all_feats_ch_{ch_idx}.zip'))])    

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
lead2idx = dict(zip(leads, range(len(leads))))

dx_mapping_scored = pd.read_csv('eval/dx_mapping_scored.csv')
snomed2dx = dict(zip(dx_mapping_scored['SNOMED CT Code'].values, dx_mapping_scored['Dx']))

beta = 2
num_classes = len(classes)

weights_file = 'eval/weights.csv'
normal_class = '426783006'
normal_index = classes.index(normal_class)
normal_lbl = [0. if i != normal_index else 1. for i in range(num_classes)]
equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

# Get feature names in order of importance (remove duration and demo)
feature_names = list(np.load('top_feats.npy'))
feature_names.remove('full_waveform_duration')
feature_names.remove('Age')
feature_names.remove('Gender_Male')

# Compute top feature means and stds
# Get top feats (exclude signal duration)
feats = all_feats[feature_names[:nb_feats]].values

# First, convert any infs to nans
feats[np.isinf(feats)] = np.nan

# Store feature means and stds
feat_means = np.nanmean(feats, axis=0)
feat_stds = np.nanstd(feats, axis=0)

def get_age(hdrs):
    ''' Get list of ages as integers from list of hdrs '''
    hs = []
    for h in hdrs:
        res = re.search(r': (\d+)\n', h)
        if res is None:
            hs.append(0)
        else:
            hs.append(float(res.group(1)))
    return np.array(hs)    
