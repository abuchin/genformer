import time
import os
import subprocess
import sys
import re
import argparse
import collections
import gzip
import math
import shutil
import matplotlib.pyplot as plt
import wandb
import numpy as np
from datetime import datetime
import random

import multiprocessing
#import logging
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision
import src.metrics as metrics ## switch to src
import src.optimizers
import src.schedulers
import pandas as pd
import src.utils
import seaborn as sns
from scipy.stats.stats import pearsonr, spearmanr
from scipy.stats import linregress
from scipy import stats
import keras.backend as kb

import scipy.special
import scipy.stats
import scipy.ndimage

from src.losses import poisson_multinomial

import numpy as np
from sklearn import metrics as sklearn_metrics

from tensorflow.keras import initializers as inits
from scipy.stats import zscore
import tensorflow_probability as tfp

tf.keras.backend.set_floatx('float32')

def return_train_val_functions(model, train_steps, optimizer,
                               strategy, metric_dict, global_batch_size,
                               gradient_clip,loss_type,total_weight):
    metric_dict["train_loss"] = tf.keras.metrics.Mean("train_loss",
                                                 dtype=tf.float32)
    metric_dict["val_loss"] = tf.keras.metrics.Mean("val_loss",
                                                  dtype=tf.float32)

    metric_dict['ATAC_PearsonR_tr'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['ATAC_R2_tr'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})

    metric_dict['ATAC_PearsonR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['ATAC_R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})

    if loss_type == 'poisson_multinomial':
        def loss_fn(y_true,y_pred, total_weight=total_weight,
                    epsilon=1e-6,rescale=True):
            return poisson_multinomial(y_true, y_pred, total_weight,epsilon,rescale=True)
    elif loss_type == 'poisson':
        loss_fn = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)
    else:
        raise ValueError('loss_type not implemented')

    @tf.function(reduce_retracing=True)
    def dist_train_step(inputs):
        #def train_step(inputs):
        print('tracing training step!')
        sequence,atac,mask,target,tf_activity =inputs

        input_tuple = sequence, atac, tf_activity

        with tf.GradientTape() as tape:

            output_profile = model(input_tuple,
                                    training=True)
            output_profile = tf.cast(output_profile,dtype=tf.float32) # ensure cast to float32

            mask_indices = tf.where(mask[0,:,0] == 1)[:,0]

            target_atac = tf.gather(target, mask_indices,axis=1)
            output_atac = tf.gather(output_profile, mask_indices,axis=1)

            loss = tf.reduce_mean(loss_fn(target_atac,
                                          output_atac)) *\
                        (1.0/global_batch_size)

        gradients = tape.gradient(loss, model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients,
                                              gradient_clip)
        optimizer.apply_gradients(zip(gradients,
                                       model.trainable_variables))
        metric_dict["train_loss"].update_state(loss)

        metric_dict['ATAC_PearsonR_tr'].update_state(target_atac,
                                                  output_atac)
        metric_dict['ATAC_R2_tr'].update_state(target_atac,
                                            output_atac)

    @tf.function(reduce_retracing=True)
    def dist_val_step(inputs):
        #def val_step(inputs):
        print('tracing validation step!')
        sequence,atac,mask,target,tf_activity=inputs

        input_tuple = sequence,atac,tf_activity

        output_profile = model(input_tuple,
                                            training=False)
        output_profile = tf.cast(output_profile,dtype=tf.float32) # ensure cast to float32

        mask_indices = tf.where(mask[0,:,0] == 1)[:,0]

        target_atac = tf.gather(target, mask_indices,axis=1)
        output_atac = tf.gather(output_profile, mask_indices,axis=1)
        loss = tf.reduce_mean(loss_fn(target_atac,
                                                  output_atac)) *\
                    (1.0/global_batch_size)
        metric_dict['ATAC_PearsonR'].update_state(target_atac,
                                                  output_atac)
        metric_dict['ATAC_R2'].update_state(target_atac,
                                            output_atac)
        metric_dict["val_loss"].update_state(loss)

        return target_atac, output_atac


    def build_step(iterator): #input_batch, model, optimizer, organism, gradient_clip):
        @tf.function(reduce_retracing=True)
        def val_step(inputs):
            sequence,atac,mask,target,tf_activity=inputs

            input_tuple = sequence,atac,tf_activity

            output_profile = model(input_tuple,
                                    training=False)

        #for _ in tf.range(1): ## for loop within @tf.fuction for improved TPU performance
        strategy.run(val_step, args=(next(iterator),))

    return dist_train_step,dist_val_step, build_step, metric_dict


def deserialize_tr(serialized_example, g, use_tf_activity,
                   input_length = 196608, max_shift = 10, output_length_ATAC = 49152,
                   output_length = 1536, crop_size = 320, output_res = 128,
                   atac_mask_dropout = 0.15, mask_size = 896, log_atac = True,
                   use_atac = True, use_seq = True, atac_corrupt_rate = 20):
    """Deserialize bytes stored in TFRecordFile."""
    ## parse out feature map
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'peaks': tf.io.FixedLenFeature([], tf.string),
        'peaks_center': tf.io.FixedLenFeature([], tf.string),
        'tf_activity': tf.io.FixedLenFeature([], tf.string)
    }
    '''
    generate random numbers for data augmentation
      rev_comp: whether or not to reverse/complement sequence + signal
      randomish_seed: hacky workaround to previous issue with random atac masking
    '''
    rev_comp = tf.math.round(g.uniform([], 0, 1)) #switch for random reverse complementation
    atac_mask_int = g.uniform([], 0, atac_corrupt_rate, dtype=tf.int32) ## atac increased corruption rate
    randomish_seed = g.uniform([], 0, 100000000,dtype=tf.int32) # hacky work-around to ensure randomish stateless operations

    shift = g.uniform(shape=(),
                      minval=0,
                      maxval=max_shift,
                      dtype=tf.int32)
    for k in range(max_shift):
        if k == shift:
            interval_end = input_length + k
            seq_shift = k
        else:
            seq_shift=0

    input_seq_length = input_length + max_shift

    ''' now parse out the actual data '''
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
    if not use_tf_activity:
        print('not using tf activity')
        tf_activity = tf.zeros_like(tf_activity)
    tf_activity = tf_activity + tf.math.abs(g.normal(tf_activity.shape,
                                               mean=0.0,
                                               stddev=0.001,
                                               dtype=tf.float32))
    '''scale'''
    #percentile99 = (tfp.stats.percentile(tf_activity, q=99.0, axis=1) + 1.0e-04)
    #tf_activity = tf_activity / percentile99


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

    '''
    here set up masking of one of the peaks. If there are no peaks, then mask the middle of the input sequence window
    '''
    atac_target = atac ## store the target ATAC

    ### here set up the ATAC masking
    num_mask_bins = mask_size // output_res ## calculate the number of adjacent bins that will be masked in each region

    center = (output_length-2*crop_size)//2
    ### here set up masking of one of the peaks
    mask_indices_temp = tf.where(peaks_c_crop[:,0] > 0)[:,0]
    mask_indices_temp = tf.random.experimental.stateless_shuffle(mask_indices_temp,seed=[4+randomish_seed,5])
    if tf.size(mask_indices_temp) > 0:
        ridx = tf.concat([mask_indices_temp],axis=0)   ### concatenate the middle in case theres no peaks
        start_index = ridx[0] - num_mask_bins // 2 + crop_size
        end_index = ridx[0] + 1 + num_mask_bins // 2 + crop_size
        indices = tf.range(start_index, end_index)
        mask = (indices >= 0) & (indices < output_length)
        filtered_indices = tf.boolean_mask(indices, mask)
        mask_indices = tf.cast(tf.reshape(filtered_indices, [-1, 1]), dtype=tf.int64)
        st=tf.SparseTensor(
            indices=mask_indices,
            values=tf.ones([tf.shape(mask_indices)[0]], dtype=tf.float32),
            dense_shape=[output_length])
        dense_peak_mask=tf.sparse.to_dense(st)
        dense_peak_mask_store = dense_peak_mask
        dense_peak_mask=1.0-dense_peak_mask ### masking regions here are set to 1. so invert the mask to actually use
    else:
        dense_peak_mask = tf.ones([output_length],dtype=tf.float32)
    dense_peak_mask = tf.expand_dims(dense_peak_mask,axis=1)

    out_length_cropped = output_length-2*crop_size
    if out_length_cropped % num_mask_bins != 0:
        raise ValueError('ensure that masking region size divided by output res is a factor of the cropped output length')
    edge_append = tf.ones((crop_size,1),dtype=tf.float32) ## since we only mask over the center 896, base calcs on the cropped size
    atac_mask = tf.ones(out_length_cropped // num_mask_bins,dtype=tf.float32)

    '''now compute the random atac seq dropout, which is done in addition to the randomly selected peak '''
    if ((atac_mask_int == 0)):
        atac_mask_dropout = 3 * atac_mask_dropout
    atac_mask=tf.nn.experimental.stateless_dropout(atac_mask,
                                              rate=(atac_mask_dropout),
                                              seed=[0,randomish_seed-5]) / (1. / (1.0-(atac_mask_dropout)))
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    atac_mask = tf.tile(atac_mask, [1,num_mask_bins])
    atac_mask = tf.reshape(atac_mask, [-1])
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    atac_mask_store = 1.0 - atac_mask ### store the actual masked regions after inverting the mask

    full_atac_mask = tf.concat([edge_append,atac_mask,edge_append],axis=0)
    full_comb_mask = tf.math.floor((dense_peak_mask + full_atac_mask)/2)
    full_comb_mask_store = 1.0 - full_comb_mask

    full_comb_mask_full_store = full_comb_mask_store
    if crop_size > 0:
        full_comb_mask_store = full_comb_mask_store[crop_size:-crop_size,:] # store the cropped mask

    tiling_req = output_length_ATAC // output_length ### how much do we need to tile the atac signal to desired length
    full_comb_mask = tf.expand_dims(tf.reshape(tf.tile(full_comb_mask, [1,tiling_req]),[-1]),axis=1)

    masked_atac = atac * full_comb_mask

    if log_atac:
        masked_atac = tf.math.log1p(masked_atac)

    diff = tf.math.sqrt(tf.nn.relu(masked_atac - 150.0 * tf.ones(masked_atac.shape)))
    masked_atac = tf.clip_by_value(masked_atac, clip_value_min=0.0, clip_value_max=150.0) + diff

    ''' randomly reverse complement the sequence, and reverse targets + peaks + mask'''
    if rev_comp == 1:
        sequence = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
        sequence = tf.reverse(sequence, axis=[0])
        atac_target = tf.reverse(atac_target,axis=[0])
        masked_atac = tf.reverse(masked_atac,axis=[0])
        peaks_crop=tf.reverse(peaks_crop,axis=[0])
        full_comb_mask_store=tf.reverse(full_comb_mask_store,axis=[0])


    atac_out = tf.reduce_sum(tf.reshape(atac_target, [-1,tiling_req]),axis=1,keepdims=True)
    diff = tf.math.sqrt(tf.nn.relu(atac_out - 2000.0 * tf.ones(atac_out.shape)))
    atac_out = tf.clip_by_value(atac_out, clip_value_min=0.0, clip_value_max=2000.0) + diff
    atac_out = tf.slice(atac_out,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])


    ''' in case we want to run ablation without these inputs'''
    if not use_atac:
        print('not using atac')
        masked_atac = tf.zeros_like(masked_atac)

    if not use_seq:
        print('not using sequence')
        sequence = tf.random.experimental.stateless_shuffle(sequence,
                                                              seed=[randomish_seed+1,randomish_seed+3])

    return tf.cast(tf.ensure_shape(sequence,[input_length,4]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(masked_atac, [output_length_ATAC,1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(full_comb_mask_store, [output_length-crop_size*2,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(atac_out,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(tf_activity, [1,1629]),dtype=tf.bfloat16)


def deserialize_val(serialized_example, g, use_tf_activity, input_length = 196608,
                   max_shift = 10, output_length_ATAC = 49152, output_length = 1536,
                   crop_size = 320, output_res = 128, atac_mask_dropout = 0.15,
                   mask_size = 896, log_atac = True, use_atac = True, use_seq = True):
    """Deserialize bytes stored in TFRecordFile."""
    ## parse out feature map
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'peaks': tf.io.FixedLenFeature([], tf.string),
        'peaks_center': tf.io.FixedLenFeature([], tf.string),
        'tf_activity': tf.io.FixedLenFeature([], tf.string)
    }

    input_seq_length = input_length + max_shift

    ## now parse out the actual data
    data = tf.io.parse_example(serialized_example, feature_map)
    peaks = tf.ensure_shape(tf.io.parse_tensor(data['peaks'],
                                              out_type=tf.int32),
                           [output_length])
    peaks_center = tf.ensure_shape(tf.io.parse_tensor(data['peaks_center'],
                                              out_type=tf.int32),
                           [output_length])


    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float16),
                           [output_length_ATAC,1])
    atac = tf.cast(atac,dtype=tf.float32)
    tf_activity = tf.ensure_shape(tf.io.parse_tensor(data['tf_activity'],
                                              out_type=tf.float16),
                           [1629])
    tf_activity = tf.cast(tf_activity,dtype=tf.float32)
    tf_activity = tf.expand_dims(tf_activity,axis=0)
    if not use_tf_activity:
        print('not using tf activity')
        tf_activity = tf.zeros_like(tf_activity)
    tf_activity = tf_activity + tf.math.abs(g.normal(tf_activity.shape,
                                               mean=0.0,
                                               stddev=0.001,
                                               dtype=tf.float32))

    # set up a semi-random seem based on the number of
    # peaks and atac signal in the window
    peaks_sum = tf.reduce_sum(peaks_center)
    randomish_seed = peaks_sum + tf.cast(tf.reduce_sum(atac),dtype=tf.int32)

    rev_comp = tf.random.stateless_uniform(shape=[], minval=0,
                                            maxval=2,
                                            seed=[randomish_seed+5,randomish_seed+6],
                                            dtype=tf.int32)

    shift = tf.random.stateless_uniform(shape=(),
                      minval=0,
                      maxval=max_shift,
                      seed=[randomish_seed+1,randomish_seed+2],
                      dtype=tf.int32)
    for k in range(max_shift):
        if k == shift:
            interval_end = input_length + k
            seq_shift = k
        else:
            seq_shift=0

    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 seq_shift,input_length))

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
    # add some small amount of gaussian noise to the input

    ### here set up the ATAC masking
    num_mask_bins = mask_size // output_res # the number of adjacent bins to mask

    center = (output_length-2*crop_size)//2 # the center of the window
    ### here set up masking of one of the peaks
    mask_indices_temp = tf.where(peaks_c_crop[:,0] > 0)[:,0]
    mask_indices_temp = tf.random.experimental.stateless_shuffle(mask_indices_temp,seed=[1+randomish_seed,5])
    if tf.size(mask_indices_temp) > 0:
        ridx = tf.concat([mask_indices_temp],axis=0)   ### concatenate the middle in case theres no peaks
        start_index = ridx[0] - num_mask_bins // 2 + crop_size
        end_index = ridx[0] + 1 + num_mask_bins // 2 + crop_size
        indices = tf.range(start_index, end_index)
        mask = (indices >= 0) & (indices < output_length)
        filtered_indices = tf.boolean_mask(indices, mask)
        mask_indices = tf.cast(tf.reshape(filtered_indices, [-1, 1]), dtype=tf.int64)
        st=tf.SparseTensor(
            indices=mask_indices,
            values=tf.ones([tf.shape(mask_indices)[0]], dtype=tf.float32),
            dense_shape=[output_length])
        dense_peak_mask=tf.sparse.to_dense(st)
        dense_peak_mask_store = dense_peak_mask
        dense_peak_mask=1.0-dense_peak_mask ### masking regions here are set to 1. so invert the mask to actually use
    else:
        dense_peak_mask = tf.ones([output_length],dtype=tf.float32)
    dense_peak_mask = tf.expand_dims(dense_peak_mask,axis=1)

    out_length_cropped = output_length-2*crop_size
    edge_append = tf.ones((crop_size,1),dtype=tf.float32)
    atac_mask = tf.ones(out_length_cropped // num_mask_bins,dtype=tf.float32)

    atac_mask=tf.nn.experimental.stateless_dropout(atac_mask,
                                              rate=(atac_mask_dropout),
                                              seed=[randomish_seed+1,randomish_seed+10]) / (1. / (1.0-(atac_mask_dropout)))
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    atac_mask = tf.tile(atac_mask, [1,num_mask_bins])
    atac_mask = tf.reshape(atac_mask, [-1])
    atac_mask = tf.expand_dims(atac_mask,axis=1)
    atac_mask_store = 1.0 - atac_mask
    full_atac_mask = tf.concat([edge_append,atac_mask,edge_append],axis=0)
    full_comb_mask = tf.math.floor((dense_peak_mask + full_atac_mask)/2)
    full_comb_mask_store = 1.0 - full_comb_mask
    if crop_size > 0:
        full_comb_mask_store = full_comb_mask_store[crop_size:-crop_size,:] # store the cropped mask
    tiling_req = output_length_ATAC // output_length
    full_comb_mask = tf.expand_dims(tf.reshape(tf.tile(full_comb_mask, [1,tiling_req]),[-1]),axis=1)
    masked_atac = atac * full_comb_mask

    if log_atac:
        masked_atac = tf.math.log1p(masked_atac)

    diff = tf.math.sqrt(tf.nn.relu(masked_atac - 150.0 * tf.ones(masked_atac.shape)))
    masked_atac = tf.clip_by_value(masked_atac, clip_value_min=0.0, clip_value_max=150.0) + diff


    if not use_atac:
        print('not using atac')
        masked_atac = tf.zeros_like(masked_atac)
    if not use_seq:
        print('not using sequence')
        sequence = tf.random.experimental.stateless_shuffle(sequence,
                                                            seed=[1,randomish_seed+12])
    if rev_comp == 1:
        sequence = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
        sequence = tf.reverse(sequence, axis=[0])
        atac_target = tf.reverse(atac_target,axis=[0])
        masked_atac = tf.reverse(masked_atac,axis=[0])
        peaks_crop=tf.reverse(peaks_crop,axis=[0])
        full_comb_mask_store=tf.reverse(full_comb_mask_store,axis=[0])

    atac_out = tf.reduce_sum(tf.reshape(atac_target, [-1,tiling_req]),axis=1,keepdims=True)
    diff = tf.math.sqrt(tf.nn.relu(atac_out - 2000.0 * tf.ones(atac_out.shape)))
    atac_out = tf.clip_by_value(atac_out, clip_value_min=0.0, clip_value_max=2000.0) + diff
    atac_out = tf.slice(atac_out,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])

    return tf.cast(tf.ensure_shape(sequence,[input_length,4]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(masked_atac, [output_length_ATAC,1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(full_comb_mask_store, [output_length-crop_size*2,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(atac_out,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(tf_activity, [1,1629]),dtype=tf.bfloat16)


def return_dataset(gcs_path, split, batch, input_length, output_length_ATAC,
                   output_length, crop_size, output_res, max_shift, options,
                   num_parallel, num_epoch, atac_mask_dropout,
                   random_mask_size, log_atac, use_atac, use_seq, seed,
                   atac_corrupt_rate, validation_steps,
                   use_tf_activity, g):
    """
    return a tf dataset object for given gcs path
    """
    wc = "*.tfr"

    if split == 'train':
        list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                    split,
                                                    wc)))
        #random.shuffle(list_files)
        files = tf.data.Dataset.list_files(list_files,seed=seed,shuffle=True)

        dataset = tf.data.TFRecordDataset(files,
                                          compression_type='ZLIB',
                                          num_parallel_reads=num_parallel)
        dataset = dataset.with_options(options)

        dataset = dataset.map(lambda record: deserialize_tr(record,
                                                            g, use_tf_activity,
                                                            input_length, max_shift,
                                                            output_length_ATAC, output_length,
                                                            crop_size, output_res,
                                                            atac_mask_dropout, random_mask_size,
                                                            log_atac, use_atac, use_seq,
                                                            atac_corrupt_rate),


                              deterministic=False,
                              num_parallel_calls=num_parallel)

        return dataset.repeat((num_epoch*2)).batch(batch).prefetch(tf.data.AUTOTUNE)

    else:
        list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                    split,
                                                    wc)))

        #random.shuffle(list_files)
        files = tf.data.Dataset.list_files(list_files,shuffle=False)
        dataset = tf.data.TFRecordDataset(files,
                                          compression_type='ZLIB',
                                          num_parallel_reads=num_parallel)
        dataset = dataset.with_options(options)
        dataset = dataset.map(lambda record: deserialize_val(record, g, use_tf_activity,
                                                             input_length, max_shift,
                                                             output_length_ATAC, output_length,
                                                             crop_size, output_res,
                                                             atac_mask_dropout, random_mask_size,
                                                             log_atac, use_atac, use_seq),
                      deterministic=True,
                      num_parallel_calls=num_parallel)

        return dataset.take(batch*validation_steps).batch(batch).repeat((num_epoch*2)).prefetch(tf.data.AUTOTUNE)


def return_distributed_iterators(gcs_path, gcs_path_ho, global_batch_size,
                                 input_length, max_shift, output_length_ATAC,
                                 output_length, crop_size, output_res,
                                 num_parallel_calls, num_epoch, strategy,
                                 options, options_val,
                                 atac_mask_dropout, atac_mask_dropout_val,
                                 random_mask_size,
                                 log_atac, use_atac, use_seq, seed,seed_val,
                                 atac_corrupt_rate,
                                 validation_steps, use_tf_activity, g,g_val):



    tr_data = return_dataset(gcs_path, "train", global_batch_size, input_length,
                             output_length_ATAC, output_length, crop_size,
                             output_res, max_shift, options, num_parallel_calls,
                             num_epoch, atac_mask_dropout, random_mask_size,
                             log_atac, use_atac, use_seq, seed,
                             atac_corrupt_rate, validation_steps, use_tf_activity, g)

    val_data_ho = return_dataset(gcs_path_ho, "valid", global_batch_size, input_length,
                                 output_length_ATAC, output_length, crop_size,
                                 output_res, max_shift, options_val, num_parallel_calls, num_epoch,
                                 atac_mask_dropout_val, random_mask_size, log_atac,
                                 use_atac, use_seq, seed_val, atac_corrupt_rate,
                                 validation_steps, use_tf_activity, g_val)

    val_dist_ho=strategy.experimental_distribute_dataset(val_data_ho)
    val_data_ho_it = iter(val_dist_ho)

    train_dist = strategy.experimental_distribute_dataset(tr_data)
    tr_data_it = iter(train_dist)

    return tr_data_it, val_data_ho_it


def early_stopping(current_val_loss,
                   logged_val_losses,
                   current_pearsons,
                   logged_pearsons,
                   current_epoch,
                   best_epoch,
                   save_freq,
                   patience,
                   patience_counter,
                   min_delta,
                   model,
                   save_directory,
                   saved_model_basename):
    """early stopping function
    Args:
        current_val_loss: current epoch val loss
        logged_val_losses: previous epochs val losses
        current_epoch: current epoch number
        save_freq: frequency(in epochs) with which to save checkpoints
        patience: # of epochs to continue w/ stable/increasing val loss
                  before terminating training loop
        patience_counter: # of epochs over which val loss hasn't decreased
        min_delta: minimum decrease in val loss required to reset patience
                   counter
        model: model object
        save_directory: cloud bucket location to save model
        model_parameters: log file of all model parameters
        saved_model_basename: prefix for saved model dir
    Returns:
        stop_criteria: bool indicating whether to exit train loop
        patience_counter: # of epochs over which val loss hasn't decreased
        best_epoch: best epoch so far
    """
    print('check whether early stopping/save criteria met')
    if (current_epoch % save_freq) == 0:
        print('Saving model...')
        model_name = save_directory + "/" + \
                        saved_model_basename + "/iteration_" + \
                            str(current_epoch) + "/saved_model"
        model.save_weights(model_name)### check if min_delta satisfied
    try:
        best_loss = min(logged_val_losses[:-1])
        best_pearsons=max(logged_pearsons[:-1])

    except ValueError:
        best_loss = current_val_loss
        best_pearsons = current_pearsons

    stop_criteria = False
    ## if min delta satisfied then log loss

    if (current_val_loss >= (best_loss - min_delta)):# and (current_pearsons <= best_pearsons):
        patience_counter += 1
        if patience_counter >= patience:
            stop_criteria=True
    else:

        best_epoch = np.argmin(logged_val_losses)
        ## save current model

        patience_counter = 0
        stop_criteria = False

    return stop_criteria, patience_counter, best_epoch


def parse_args(parser):
    """Loads in command line arguments
    """

    parser.add_argument('--tpu_name', dest = 'tpu_name',
                        help='tpu_name')
    parser.add_argument('--tpu_zone', dest = 'tpu_zone',
                        help='tpu_zone')
    parser.add_argument('--wandb_project',
                        dest='wandb_project',
                        help ='wandb_project')
    parser.add_argument('--wandb_user',
                        dest='wandb_user',
                        help ='wandb_user')
    parser.add_argument('--wandb_sweep_name',
                        dest='wandb_sweep_name',
                        help ='wandb_sweep_name')
    parser.add_argument('--gcs_project', dest = 'gcs_project',
                        help='gcs_project')
    parser.add_argument('--gcs_path',
                        dest='gcs_path',
                        help= 'google bucket containing preprocessed data')
    parser.add_argument('--gcs_path_holdout',
                        dest='gcs_path_holdout',
                        help= 'google bucket containing preprocessed data')
    parser.add_argument('--num_parallel', dest = 'num_parallel',
                        type=int, default=multiprocessing.cpu_count(),
                        help='thread count for tensorflow record loading')
    parser.add_argument('--batch_size', dest = 'batch_size',
                        default=1,
                        type=int, help='batch_size')
    parser.add_argument('--num_epochs', dest = 'num_epochs',
                        type=int, help='num_epochs')
    parser.add_argument('--train_examples', dest = 'train_examples',
                        type=int)
    parser.add_argument('--val_examples_ho', dest = 'val_examples_ho',
                        type=int, help='val_examples_ho')
    parser.add_argument('--patience', dest = 'patience',
                        type=int, help='patience for early stopping')
    parser.add_argument('--min_delta', dest = 'min_delta',
                        type=float, help='min_delta for early stopping')
    parser.add_argument('--model_save_dir',
                        dest='model_save_dir',
                        type=str)
    parser.add_argument('--model_save_basename',
                        dest='model_save_basename',
                        type=str)
    parser.add_argument('--max_shift',
                    dest='max_shift',
                        default=10,
                    type=int)
    parser.add_argument('--output_res',
                    dest='output_res',
                        default=128,
                    type=int)
    parser.add_argument('--lr_base',
                        dest='lr_base',
                        default="1.0e-03",
                        help='lr_base')
    parser.add_argument('--decay_frac',
                        dest='decay_frac',
                        type=str,
                        help='decay_frac')
    parser.add_argument('--warmup_frac',
                        dest = 'warmup_frac',
                        default=0.0,
                        type=float, help='warmup_frac')
    parser.add_argument('--input_length',
                        dest='input_length',
                        type=int,
                        default=196608,
                        help= 'input_length')
    parser.add_argument('--output_length',
                        dest='output_length',
                        type=int,
                        default=1536,
                        help= 'output_length')
    parser.add_argument('--output_length_ATAC',
                        dest='output_length_ATAC',
                        type=int,
                        default=1536,
                        help= 'output_length_ATAC')
    parser.add_argument('--final_output_length',
                        dest='final_output_length',
                        type=int,
                        default=896,
                        help= 'final_output_length')
    parser.add_argument('--num_transformer_layers',
                        dest='num_transformer_layers',
                        type=str,
                        default="6",
                        help= 'num_transformer_layers')
    parser.add_argument('--filter_list_seq',
                        dest='filter_list_seq',
                        default="768,896,1024,1152,1280,1536",
                        help='filter_list_seq')
    parser.add_argument('--filter_list_atac',
                        dest='filter_list_atac',
                        default="32,64",
                        help='filter_list_atac')
    parser.add_argument('--epsilon',
                        dest='epsilon',
                        default=1.0e-16,
                        type=float,
                        help= 'epsilon')
    parser.add_argument('--gradient_clip',
                        dest='gradient_clip',
                        type=str,
                        default="1.0",
                        help= 'gradient_clip')
    parser.add_argument('--dropout_rate',
                        dest='dropout_rate',
                        default="0.40",
                        help= 'dropout_rate')
    parser.add_argument('--pointwise_dropout_rate',
                        dest='pointwise_dropout_rate',
                        default="0.05",
                        help= 'pointwise_dropout_rate')
    parser.add_argument('--num_heads',
                        dest='num_heads',
                        default="8",
                        help= 'num_heads')
    parser.add_argument('--num_random_features',
                        dest='num_random_features',
                        type=str,
                        default="256",
                        help= 'num_random_features')
    parser.add_argument('--BN_momentum',
                        dest='BN_momentum',
                        type=float,
                        default=0.80,
                        help= 'BN_momentum')
    parser.add_argument('--kernel_transformation',
                        dest='kernel_transformation',
                        type=str,
                        default="relu_kernel_transformation",
                        help= 'kernel_transformation')
    parser.add_argument('--savefreq',
                        dest='savefreq',
                        type=int,
                        help= 'savefreq')
    parser.add_argument('--checkpoint_path',
                        dest='checkpoint_path',
                        type=str,
                        default="gs://picard-testing-176520/enformer_performer/models/enformer_performer_230120_196k_load_init-True_freeze-False_LR1-1e-06_LR2-0.0001_T-6_F-1536_D-0.4_K-relu_kernel_transformation_MP-True_AD-0.05/iteration_10",
                        help= 'checkpoint_path')
    parser.add_argument('--load_init',
                        dest='load_init',
                        type=str,
                        default="True",
                        help= 'load_init')
    parser.add_argument('--normalize',
                        dest='normalize',
                        type=str,
                        default="True",
                        help= 'normalize')
    parser.add_argument('--norm',
                        dest='norm',
                        type=str,
                        default="True",
                        help= 'norm')
    parser.add_argument('--atac_mask_dropout',
                        dest='atac_mask_dropout',
                        type=float,
                        default=0.05,
                        help= 'atac_mask_dropout')
    parser.add_argument('--atac_mask_dropout_val',
                        dest='atac_mask_dropout_val',
                        type=float,
                        default=0.05,
                        help= 'atac_mask_dropout_val')
    parser.add_argument('--final_point_scale',
                        dest='final_point_scale',
                        type=str,
                        default="6",
                        help= 'final_point_scale')
    parser.add_argument('--rectify',
                        dest='rectify',
                        type=str,
                        default="True",
                        help= 'rectify')
    parser.add_argument('--optimizer',
                        dest='optimizer',
                        type=str,
                        default="adam",
                        help= 'optimizer')
    parser.add_argument('--log_atac',
                        dest='log_atac',
                        type=str,
                        default="True",
                        help= 'log_atac')
    parser.add_argument('--use_atac',
                        dest='use_atac',
                        type=str,
                        default="True",
                        help= 'use_atac')
    parser.add_argument('--use_seq',
                        dest='use_seq',
                        type=str,
                        default="True",
                        help= 'use_seq')
    parser.add_argument('--random_mask_size',
                        dest='random_mask_size',
                        type=str,
                        default="1152",
                        help= 'random_mask_size')
    parser.add_argument('--seed',
                        dest='seed',
                        type=int,
                        default=42,
                        help= 'seed')
    parser.add_argument('--val_data_seed',
                        dest='val_data_seed',
                        type=int,
                        default=25,
                        help= 'val_data_seed')
    parser.add_argument('--atac_corrupt_rate',
                        dest='atac_corrupt_rate',
                        type=str,
                        default="20",
                        help= 'atac_corrupt_rate')
    parser.add_argument('--use_tf_activity',
                        dest='use_tf_activity',
                        type=str,
                        default="False",
                        help= 'use_tf_activity')
    parser.add_argument('--num_epochs_to_start',
                        dest='num_epochs_to_start',
                        type=str,
                        default="0",
                        help= 'num_epochs_to_start')
    parser.add_argument('--loss_type',
                        dest='loss_type',
                        type=str,
                        default="poisson_multinomial",
                        help= 'loss_type')
    parser.add_argument('--total_weight_loss',
                        dest='total_weight_loss',
                        type=str,
                        default="0.15",
                        help= 'total_weight_loss')
    parser.add_argument('--use_rot_emb',
                        dest='use_rot_emb',
                        type=str,
                        default="True",
                        help= 'use_rot_emb')
    parser.add_argument('--best_val_loss',
                        dest='best_val_loss',
                        type=float,
                        default=0.09113)
    args = parser.parse_args()
    return parser

def one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    vocabulary = tf.constant(['A', 'C', 'G', 'T'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=0)

    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))

    out = tf.one_hot(table.lookup(input_characters),
                      depth = 4,
                      dtype=tf.float32)
    return out

def rev_comp_one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))
    input_characters = tf.reverse(input_characters,[0])

    vocabulary = tf.constant(['T', 'G', 'C', 'A'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=0)

    out = tf.one_hot(table.lookup(input_characters),
                      depth = 4,
                      dtype=tf.float32)
    return out


def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def tf_tpu_initialize(tpu_name,zone):
    """Initialize TPU and return global batch size for loss calculation
    Args:
        tpu_name
    Returns:
        distributed strategy
    """

    try:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_name,zone=zone)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)

    except ValueError: # no TPU found, detect GPUs
        strategy = tf.distribute.get_strategy()

    return strategy

def make_plots(y_trues,
               y_preds,
               num_points):


    results_df = pd.DataFrame()
    results_df['true'] = y_trues
    results_df['pred'] = y_preds

    results_df['true_log'] = np.log2(1.0+results_df['true'])
    results_df['pred_log'] = np.log2(1.0+results_df['pred'])

    true=results_df[['true']].to_numpy()[:,0]
    pred=results_df[['pred']].to_numpy()[:,0]
    true_log=results_df[['true_log']].to_numpy()[:,0]
    pred_log=results_df[['pred_log']].to_numpy()[:,0]

    try:
        overall_corr=results_df['true'].corr(results_df['pred'])
        overall_corr_log=results_df['true_log'].corr(results_df['pred_log'])
        #cell_specific_corrs_sp=results_df[['true','pred']].corr(method='spearman').unstack().iloc[:,1].tolist()
    except np.linalg.LinAlgError as err:
        overall_corr = 0.0
        overall_corr_log = 0.0

    fig_overall,ax_overall=plt.subplots(figsize=(6,6))

    ## scatter plot for 50k points max
    idx = np.random.choice(np.arange(len(true)), num_points, replace=False)

    data = np.vstack([true[idx],
                      pred[idx]])

    min_true = min(true)
    max_true = max(true)

    min_pred = min(pred)
    max_pred = max(pred)

    try:
        kernel = stats.gaussian_kde(data)(data)
        sns.scatterplot(
            x=true_log[idx],
            y=pred_log[idx],
            c=kernel,
            cmap="viridis")
        ax_overall.set_xlim(min_true,max_true)
        ax_overall.set_ylim(min_pred,max_pred)
        plt.xlabel("log-true")
        plt.ylabel("log-pred")
        plt.title("overall atac corr")
    except np.linalg.LinAlgError as err:
        sns.scatterplot(
            x=true_log[idx],
            y=pred_log[idx],
            cmap="viridis")
        ax_overall.set_xlim(min_true,max_true)
        ax_overall.set_ylim(min_pred,max_pred)
        plt.xlabel("log-true")
        plt.ylabel("log-pred")
        plt.title("overall atac corr")
    except ValueError:
        sns.scatterplot(
            x=true_log[idx],
            y=pred_log[idx],
            cmap="viridis")
        plt.xlabel("log-true")
        plt.ylabel("log-pred")
        plt.title("overall atac corr")

    return fig_overall, overall_corr,overall_corr_log
