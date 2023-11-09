import time
import os
import subprocess
import sys
sys.path.insert(1, '/home/jupyter/datasets/genformer')
import re
import argparse
import collections
import gzip
import math
import shutil
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
import random

import seaborn as sns
%matplotlib inline
import logging
os.environ['TPU_LOAD_LIBRARY']='0'
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf

import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision
from scipy.stats.stats import pearsonr  
from scipy.stats.stats import spearmanr  
## custom modules
import src.models.aformer_atac as aformer
from src.layers.layers import *
import src.metrics as metrics
from src.optimizers import *
import src.schedulers as schedulers

import training_utils_atac as training_utils

from scipy import stats
import kipoiseq

import analysis.scripts.interval_and_plotting_utilities as utils

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



def deserialize_test(serialized_example, g, use_tf_activity, input_length = 196608,
                   max_shift = 10, output_length_ATAC = 49152, output_length = 1536,
                   crop_size = 320, output_res = 128, mask_indices,
                   mask_size = 896, log_atac = True, use_atac = True, use_seq = True):
    """Deserialize bytes stored in TFRecordFile."""
    ## parse out feature map
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'peaks': tf.io.FixedLenFeature([], tf.string),
        'peaks_center': tf.io.FixedLenFeature([], tf.string),
        'tf_activity': tf.io.FixedLenFeature([], tf.string),
        'interval': tf.io.FixedLenFeature([], tf.string)
    }
    ### stochastic sequence shift and gaussian noise
    seq_shift=5

    input_seq_length = input_length + max_shift

    ## now parse out the actual data
    data = tf.io.parse_example(serialized_example, feature_map)
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 seq_shift,input_length))
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float16),
                           [output_length_ATAC,1])
    atac = tf.cast(atac,dtype=tf.float32)
    peaks = tf.ensure_shape(tf.io.parse_tensor(data['peaks'],
                                              out_type=tf.int32),
                           [output_length])
    peaks_center = tf.ensure_shape(tf.io.parse_tensor(data['peaks_center'],
                                              out_type=tf.int32),
                           [output_length])
    tf_activity = tf.ensure_shape(tf.io.parse_tensor(data['tf_activity'],
                                              out_type=tf.float16),
                           [1629])
    tf_activity = tf.cast(tf_activity,dtype=tf.float32)
    tf_activity = tf.expand_dims(tf_activity,axis=0)
    tf_activity = tf_activity + tf.math.abs(g.normal(tf_activity.shape,
                                               mean=0.0,
                                               stddev=0.001,
                                               dtype=tf.float32))
    
    interval_id = tf.ensure_shape(tf.io.parse_tensor(data['interval'],
                                              out_type=tf.float16),
                           [])

    peaks_sum = tf.reduce_sum(peaks_center)
    seq_seed = tf.reduce_sum(sequence[:,0])
    # set up a semi-random seem based on the number of
    # peaks and adenosines in the window
    randomish_seed = peaks_sum + tf.cast(seq_seed,dtype=tf.int32)

    ## here we set up the target variable peaks
    ## to make calculating loss easier, adjust it
    ## here to the cropped window size
    peaks = tf.expand_dims(peaks,axis=1)
    peaks_crop = tf.slice(peaks,
                     [crop_size,0],
                     [output_length-2*crop_size,-1])


    ## here we set up the center of the peaks
    ## use the center of the peaks to set up the region for masking
    peaks_center = tf.expand_dims(peaks_center,axis=1)
    peaks_c_crop = tf.slice(peaks_center,
                     [crop_size,0],
                     [output_length-2*crop_size,-1])

    atac_target = atac ## store the target

    
    st=tf.SparseTensor(
        indices=mask_indices,
        values=[1.0]*len(mask_indices),
        dense_shape=[output_length])
    dense_peak_mask=tf.sparse.to_dense(st)
    dense_peak_mask_store = dense_peak_mask
    dense_peak_mask=1.0-dense_peak_mask ### masking regions here are set to 1. so invert the mask to actually use
    dense_peak_mask = tf.expand_dims(dense_peak_mask,axis=1)

    out_length_cropped = output_length-2*crop_size
    edge_append = tf.ones((crop_size,1),dtype=tf.float32)
    atac_mask = tf.ones(out_length_cropped // num_mask_bins,dtype=tf.float32)
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    atac_mask = tf.tile(atac_mask, [1,num_mask_bins])
    atac_mask = tf.reshape(atac_mask, [-1])
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    atac_mask_store = 1.0 - atac_mask
    full_atac_mask = tf.concat([edge_append,atac_mask,edge_append],axis=0)
    full_comb_mask = tf.math.floor((dense_peak_mask + full_atac_mask)/2)
    full_comb_mask_store = 1.0 - full_comb_mask
    full_comb_mask_store = full_comb_mask_store[crop_size:-crop_size,:]
    tiling_req = output_length_ATAC // output_length
    full_comb_mask = tf.expand_dims(tf.reshape(tf.tile(full_comb_mask, [1,tiling_req]),[-1]),axis=1)
    masked_atac = atac * full_comb_mask

    if log_atac:
        masked_atac = tf.math.log1p(masked_atac)

    diff = tf.math.sqrt(tf.nn.relu(masked_atac - 10000.0 * tf.ones(masked_atac.shape)))
    masked_atac = tf.clip_by_value(masked_atac, clip_value_min=0.0, clip_value_max=10000.0) + diff

    atac_out = tf.reduce_sum(tf.reshape(atac_target, [-1,tiling_req]),axis=1,keepdims=True)
    diff = tf.math.sqrt(tf.nn.relu(atac_out - 50000.0 * tf.ones(atac_out.shape)))
    atac_out = tf.clip_by_value(atac_out, clip_value_min=0.0, clip_value_max=50000.0) + diff
    atac_out = tf.slice(atac_out,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])

    peaks_gathered = tf.reduce_max(tf.reshape(peaks_crop, [(output_length-2*crop_size) // 4, -1]),
                                   axis=1,keepdims=True)
    mask_gathered = tf.reduce_max(tf.reshape(full_comb_mask_store, [(output_length-2*crop_size) // 4, -1]),
                                   axis=1,keepdims=True)

    if not use_atac:
        print('not using atac')
        masked_atac = tf.math.abs(g.normal(masked_atac.shape,
                               mean=0.0,
                               stddev=1.0,
                               dtype=tf.float32))
    if not use_seq:
        sequence = tf.random.experimental.stateless_shuffle(sequence,
                                                            seed=[1,randomish_seed+12])
        
        
    rev_seq = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
    rev_seq = tf.reverse(rev_seq, axis=[0])
    masked_atac_rev = tf.reverse(masked_atac,axis=[0])
    full_comb_mask_store_rev = tf.reverse(full_comb_mask_store,axis=[0])
    atac_out_rev = tf.reverse(atac_out,axis=[0])
    

    return tf.cast(tf.ensure_shape(sequence,[input_length,4]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(rev_seq,[input_length,4]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(masked_atac, [output_length_ATAC,1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(masked_atac_rev, [output_length_ATAC,1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(full_comb_mask_store, [output_length-crop_size*2,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(full_comb_mask_store_rev, [output_length-crop_size*2,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(mask_gathered, [(output_length-crop_size*2) // 4,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(peaks_gathered, [(output_length-2*crop_size) // 4,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(atac_out,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(atac_out_rev,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(tf_activity, [1,1629]),dtype=tf.bfloat16), \
                tf.cast(interval_id,dtype=tf.int32)



def return_dataset(gcs_path, batch, input_length, output_length_ATAC,
                   output_length, crop_size, output_res, max_shift, options,
                   num_parallel, mask_indices,
                   random_mask_size, log_atac, use_atac, use_seq, seed,
                   use_tf_activity, g):
    """
    return a tf dataset object for given gcs path
    """
    wc = "*.tfr"


    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                    split,
                                                    wc)))

        #random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files,shuffle=False)
    dataset = tf.data.TFRecordDataset(files,
                                          compression_type='ZLIB',
                                          num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)
    dataset = dataset.map(lambda record: deserialize_test(record, g, use_tf_activity,
                                                             input_length, max_shift,
                                                             output_length_ATAC, output_length,
                                                             crop_size, output_res,mask_indices,
                                                             atac_mask_dropout, random_mask_size,
                                                             log_atac, use_atac, use_seq),
                      deterministic=True,
                      num_parallel_calls=num_parallel)

    return dataset.batch(batch).repeat(2).prefetch(tf.data.AUTOTUNE)


def return_distributed_iterators(gcs_path, global_batch_size,
                                 input_length, max_shift, output_length_ATAC,
                                 output_length, crop_size, output_res,
                                 num_parallel_calls, strategy,
                                 options,random_mask_size, mask_indices,
                                 log_atac, use_atac, use_seq, seed,
                                 use_tf_activity, g):

    test_data = return_dataset(gcs_path, global_batch_size, input_length,
                             output_length_ATAC, output_length, crop_size,
                             output_res, max_shift, options, num_parallel_calls,
                               mask_indices, random_mask_size,
                             log_atac, use_atac, use_seq, seed, use_tf_activity, g)

    test_dist = strategy.experimental_distribute_dataset(test_data)
    test_data_it = iter(test_dist)

    return test_data_it



def return_test_build(model,strategy):

    @tf.function(reduce_retracing=True)
    def dist_test_step(inputs):
        sequence,rev_seq,atac,rev_atac,mask,mask_rev,mask_gathered,peaks,target,target_rev,tf_activity,interval_id=inputs

        input_tuple = sequence,atac,tf_activity

        output_profile = model(input_tuple,
                               training=False)
        output_profile = tf.cast(output_profile,dtype=tf.float32) # ensure cast to float32
        mask_indices = tf.where(mask[0,:,0] == 1)[:,0]
        target_atac = tf.gather(target, mask_indices,axis=1)
        output_atac = tf.gather(output_profile, mask_indices,axis=1)
        
        input_tuple_rev = rev_seq,rev_atac,tf_activity
        output_profile = model(input_tuple,
                               training=False)
        output_profile = tf.cast(output_profile,dtype=tf.float32) # ensure cast to float32
        mask_indices = tf.where(mask_rev[0,:,0] == 1)[:,0]
        target_atac_rev = tf.gather(target_rev, mask_indices,axis=1)
        output_atac_rev = tf.gather(output_profile, mask_indices,axis=1)

        
        target_atac_mean = (target_atac + target_atac_rev) / 2.0
        output_atac_mean = (output_atac + output_atac_rev) / 2.0
        
        interval_id = interval_id + tf.constant([2042, 2043, 2044, 2045, 2046, 2047, 
                                                 2048, 2049, 2050, 2051, 2052, 2053],
                                                dtype=tf.int32)
        
        
        return target_atac_mean, output_atac_mean, interval_id


    def build_step(iterator): #input_batch, model, optimizer, organism, gradient_clip):
        @tf.function(reduce_retracing=True)
        def val_step(inputs):
            sequence,rev_seq,atac,rev_atac,mask,mask_rev,mask_gathered,peaks,target,target_rev,tf_activity,interval_id=inputs

            input_tuple = sequence,atac,tf_activity

            output_profile = model(input_tuple,
                                    training=False)

        #for _ in tf.range(1): ## for loop within @tf.fuction for improved TPU performance
        strategy.run(val_step, args=(next(iterator),))

    return dist_test_step, build_step