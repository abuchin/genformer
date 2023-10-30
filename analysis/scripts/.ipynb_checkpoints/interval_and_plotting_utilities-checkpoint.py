import argparse
import collections
import gzip
import math
import os
import random
import shutil
import subprocess
import sys
import time
sys.path.insert(0, '/home/jupyter/genformer/analysis/scripts')
import numpy as np
import pandas as pd
import tensorflow as tf


import pybedtools as pybt
import tabix as tb

from datetime import datetime
from tensorflow import strings as tfs
from tensorflow.keras import initializers as inits
from scipy import stats
from scipy.signal import find_peaks

pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt


def one_hot(sequence, replaceN = 'all_1'):
    '''
    Convert input sequence to one hot encoded numpy array

    Return:
        - np array of one hot encoded sequence
    '''

    if replaceN not in ['all_1', 'all_0', 'random']:
        raise ValueError('N_rep must be one of all_1, all_0, random')
    
    np_sequence = np.array(list(sequence.upper()))
    ## initialize empty numpy array
    length = len(sequence)
    one_hot_out = np.zeros((length, 4))

    one_hot_out[np_sequence == 'A'] = [1, 0, 0, 0]
    one_hot_out[np_sequence == 'T'] = [0, 1, 0, 0]
    one_hot_out[np_sequence == 'C'] = [0, 0, 1, 0]
    one_hot_out[np_sequence == 'G'] = [0, 0, 0, 1]

    replace = 4 * [0]
    if replaceN == 'all_0':
        replace = 4 * [1]
    if replaceN == 'random':
        rand = np.random.randint(4, size = 1)[0]
        replace[rand] = 1
    one_hot_out[np_sequence == 'N'] = replace

    return one_hot_out
    
    
 ################################################################################
def process_bedgraph(interval, df):
    '''
    Extract coverage info for input pybedtools interval object

    Return:
        numpy array of interval length where each pos is 5' read coverage
    '''
    
    df['start'] = df['start'].astype('int64') - int(interval.start)
    df['end'] = df['end'].astype('int64') - int(interval.start)
    
    ## ensure within bounds of interval
    df.loc[df.start < 0, 'start'] = 0
    df.loc[df.end > len(interval), 'end'] = len(interval)
    
    per_base = np.zeros(len(interval), dtype=np.float64)
    
    num_intervals = df['start'].to_numpy().shape[0]

    output = np.asarray(get_per_base_score_f(
                    df['start'].to_numpy().astype(np.int_),
                    df['end'].to_numpy().astype(np.int_),
                    df['score'].to_numpy().astype(np.float64),
                    per_base), dtype = np.float64)
    
    

    
    ### bin at the desired output res
    return output


def get_per_base_score_f(start, end, score, base):
    num_bins = start.shape[0]
    for k in range(num_bins):
        base[start[k]:end[k]] = score[k]
    return base
    

def return_atac_interval(atac_bedgraph,
                         chrom,interval_start,interval_end,num_bins,resolution):
    
    interval_str = '\t'.join([chrom, 
                              str(interval_start),
                              str(interval_end)])
    interval_bed = pybt.BedTool(interval_str, from_string=True)
    interval = interval_bed[0]
    
    atac_bedgraph_bed = tb.open(atac_bedgraph)
    ### atac processing ######################################-
    atac_subints= atac_bedgraph_bed.query(chrom,
                                          interval_start,
                                          interval_end)
    atac_subints_df = pd.DataFrame([rec for rec in atac_subints])


    # if malformed line without score then disard
    if (len(atac_subints_df.index) == 0):
        atac_bedgraph_out = np.array([0.0] * (target_length))
    else:
        atac_subints_df.columns = ['chrom', 'start', 'end', 'score']
        atac_bedgraph_out = process_bedgraph(
            interval, atac_subints_df)
        
    atac_processed = atac_bedgraph_out
    atac_processed = np.reshape(atac_processed, [num_bins,resolution])
    atac_processed = np.sum(atac_processed,axis=1,keepdims=True)
    
    return atac_processed
    
    
    
def resize_interval(interval_str,size):
    
    chrom = interval_str.split(':')[0]
    start=int(interval_str.split(':')[1].split('-')[0].replace(',',''))
    stop=int(interval_str.split(':')[1].split('-')[1].replace(',',''))


    new_start = (int(start)+int(stop))//2 - int(size)//2
    new_stop = (int(start)+int(stop))//2 + int(size)//2

    return chrom,new_start,new_stop



def plot_tracks(tracks,start, end, height=1.5):
    fig, axes = plt.subplots(len(tracks)+1, 1, figsize=(24, height * (len(tracks)+1)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(start, end, num=len(y[0])), y[0],color=y[1])
        ax.set_title(title)
    plt.tight_layout()