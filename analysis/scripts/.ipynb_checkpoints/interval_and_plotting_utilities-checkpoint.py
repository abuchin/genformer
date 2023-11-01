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
from kipoiseq import Interval
import pyfaidx
import kipoiseq

import src.models.aformer_atac as aformer
from src.layers.layers import *
import src.metrics as metrics
from src.optimizers import *
import src.schedulers as schedulers


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
    
    atac_processed = tf.constant(atac_processed,dtype=tf.float32)
    
    diff = tf.math.sqrt(tf.nn.relu(atac_processed - 10000.0 * tf.ones(atac_processed.shape)))
    atac_processed = tf.clip_by_value(atac_processed, clip_value_min=0.0, clip_value_max=10000.0) + diff
    
    return atac_processed
    
    
    
def resize_interval(interval_str,size):
    
    chrom = interval_str.split(':')[0]
    start=int(interval_str.split(':')[1].split('-')[0].replace(',',''))
    stop=int(interval_str.split(':')[1].split('-')[1].replace(',',''))


    new_start = (int(start)+int(stop))//2 - int(size)//2
    new_stop = (int(start)+int(stop))//2 + int(size)//2

    return chrom,new_start,new_stop

def plot_tracks(tracks,start, end, y_lim, height=1.5):
    fig, axes = plt.subplots(len(tracks)+1, 1, figsize=(24, height * (len(tracks)+1)), sharex=True)
    for ax, (title, y) in zip(axes, tracks.items()):
        ax.fill_between(np.linspace(start, end, num=len(y[0])), y[0],color=y[1])
        ax.set_title(title)
        ax.set_ylim((0,y_lim))
    plt.tight_layout()
    

class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()
    
def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def load_np(tf_activity):
    in_file = open(tf_activity,'r')
    lst = []
    for line in in_file.readlines():
        lst.append(np.log(1.0+float(line.rstrip('\n').split('\t')[-1])))
    np_arr = np.asarray(lst)
    in_file.close()
    return np_arr
    
    
class model:
    
    def __init__(self, model_checkpoint):
        self.model = aformer.aformer(kernel_transformation='relu_kernel_transformation',
                                dropout_rate=0.20,
                                pointwise_dropout_rate=0.10,
                                input_length=524288,
                                output_length=4096,
                                final_output_length=896,
                                num_heads=8,
                                numerical_stabilizer=0.0000001,
                                nb_random_features=256,
                                max_seq_length=4096,
                                norm=True,
                                BN_momentum=0.90,
                                normalize = True,
                                 use_rot_emb = True,
                                num_transformer_layers=7,
                                final_point_scale=6,
                                filter_list_seq=[768,896,1024,1152,1280,1536],
                                filter_list_atac=[32,64],
                                num_tfs=1629,
                                tf_dropout_rate=0.01)


        test = tf.ones((1,524288,4),dtype=tf.bfloat16),tf.ones((1,131072,1),dtype=tf.bfloat16),tf.ones((1,1,1629),dtype=tf.bfloat16)#,tf.ones((1,1,1536))
        self.model(test,training=False)
        print('ran test input')
        self.model.load_weights(model_checkpoint)
        print('loaded weights')
    
    def predict_on_batch(self, inputs):
        return self.model.predict_on_batch(inputs)

    def contribution_input_grad(self, model_inputs, gradient_mask):
        seq,atac,tf_activity = model_inputs

        gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)
        gradient_mask_mass = tf.reduce_sum(gradient_mask)

        with tf.GradientTape() as input_grad_tape:
            input_grad_tape.watch(seq)
            input_grad_tape.watch(atac)
            prediction, att_matrices = self.model.predict_on_batch(model_inputs)

            prediction = tf.cast(prediction,dtype=tf.float32)
            gradient_mask = tf.cast(gradient_mask,dtype=tf.float32)

            prediction_mask = tf.reduce_sum(gradient_mask *
                                            prediction) / gradient_mask_mass


        input_grads = input_grad_tape.gradient(prediction_mask, model_inputs)

        input_grads_seq = input_grads[0] 
        input_grads_atac = input_grads[1]

        seq_grads = tf.reduce_sum(input_grads_seq[0,:,:] * seq[0,:,:],
                                  axis=1)

        atac_grads = input_grads_atac[0,:,] * atac[0,:,]

        return seq_grads, atac_grads, input_grads_seq[0,:,:], prediction, att_matrices


    
def return_all_inputs(interval, atac_dataset, SEQUENCE_LENGTH,
                      num_bins, resolution,tf_arr,crop_size,output_length,
                      fasta_extractor,mask_indices):

    chrom,start,stop = resize_interval(interval,SEQUENCE_LENGTH)
    atac_arr = return_atac_interval(atac_dataset,chrom,
                                          start,stop,num_bins,resolution)
    tf_activity=load_np(tf_arr)
    tf_activity=tf.constant(tf_activity,dtype=tf.bfloat16)
    interval = kipoiseq.Interval(chrom, start, stop)
    sequence_one_hot = tf.constant(one_hot_encode(fasta_extractor.extract(interval)),dtype=tf.bfloat16)

    mask_start = int(mask_indices.split('-')[0])
    mask_end = int(mask_indices.split('-')[1])
    
    atac_mask = np.ones((SEQUENCE_LENGTH//128,1))
    for k in tf.range(mask_start,mask_end):
        atac_mask[k,0] = 0.0
    atac_mask = tf.constant(atac_mask,dtype=tf.float32)
    atac_mask = tf.reshape(tf.tile(atac_mask, [1,32]),[-1])
    atac_mask = tf.expand_dims(atac_mask,axis=1)

    masked_atac = atac_arr * atac_mask

    diff = tf.math.sqrt(tf.nn.relu(masked_atac - 10000.0 * tf.ones(masked_atac.shape)))
    masked_atac = tf.clip_by_value(masked_atac, clip_value_min=0.0, clip_value_max=10000.0) + diff
    
    masked_atac_reshape = tf.reduce_sum(tf.reshape(masked_atac, [-1,32]),axis=1,keepdims=True)
    masked_atac_reshape = tf.slice(masked_atac_reshape,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])
    
    target_atac = tf.reduce_sum(tf.reshape(atac_arr, [-1,32]),axis=1,keepdims=True)

    diff = tf.math.sqrt(tf.nn.relu(target_atac - 50000.0 * tf.ones(target_atac.shape)))
    target_atac = tf.clip_by_value(target_atac, clip_value_min=0.0, clip_value_max=50000.0) + diff

    target_atac = tf.slice(target_atac,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])
    
    inputs = tf.expand_dims(sequence_one_hot,axis=0), \
                tf.cast(tf.expand_dims(masked_atac,axis=0),dtype=tf.bfloat16), \
                    tf.expand_dims(tf.expand_dims(tf_activity,axis=0),axis=0)
    return inputs,target_atac,masked_atac_reshape