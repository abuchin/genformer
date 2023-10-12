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

from src.losses import poisson_multinomial

import scipy.special
import scipy.stats
import scipy.ndimage

import numpy as np
from sklearn import metrics as sklearn_metrics

from tensorflow.keras import initializers as inits
from scipy.stats import zscore
import tensorflow_probability as tfp

tf.keras.backend.set_floatx('float32')

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

def return_train_val_functions(model,
                               optimizers_in,
                               strategy,
                               metric_dict,
                               global_batch_size,
                               gradient_clip,
                               rna_scale):

    optimizer1,optimizer2=optimizers_in

    metric_dict["corr_stats"] = metrics.correlation_stats_gene_centered(name='corr_stats')
    metric_dict["corr_stats_ho"] = metrics.correlation_stats_gene_centered(name='corr_stats_ho')

    metric_dict["train_loss"] = tf.keras.metrics.Mean("train_loss",
                                                 dtype=tf.float32)
    metric_dict["train_loss_atac"] = tf.keras.metrics.Mean("train_loss_atac",
                                                 dtype=tf.float32)
    metric_dict["train_loss_rna"] = tf.keras.metrics.Mean("train_loss_rna",
                                                 dtype=tf.float32)
    metric_dict["val_loss"] = tf.keras.metrics.Mean("val_loss",
                                                  dtype=tf.float32)
    metric_dict["val_loss_atac"] = tf.keras.metrics.Mean("val_loss_atac",
                                                  dtype=tf.float32)
    metric_dict["val_loss_rna"] = tf.keras.metrics.Mean("val_loss_rna",
                                                  dtype=tf.float32)

    metric_dict['RNA_PearsonR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['RNA_R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})
    metric_dict['RNA_PearsonR_ho'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['RNA_R2_ho'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})
    metric_dict['ATAC_PearsonR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['ATAC_R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})
    metric_dict['ATAC_PearsonR_ho'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
    metric_dict['ATAC_R2_ho'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})

    @tf.function(reduce_retracing=True)
    def dist_train_step(inputs):
        print('tracing training step!')
        sequence,atac,mask,mask_gathered,peaks,target_atac,target_rna,assay_type,weights,tf_activity =inputs
        input_tuple = sequence, atac, target_rna,tf_activity,assay_type

        with tf.GradientTape() as tape:
            conv_performer_vars = model.stem_conv.trainable_variables + \
                        model.stem_res_conv.trainable_variables + \
                        model.stem_pool.trainable_variables + \
                        model.conv_tower.trainable_variables + \
                        model.stem_conv_atac.trainable_variables + model.stem_res_conv_atac.trainable_variables + \
                        model.stem_pool_atac.trainable_variables + model.conv_tower_atac.trainable_variables + \
                        model.tf_activity_fc.trainable_variables + \
                        model.performer.trainable_variables

            output_heads = model.final_pointwise_conv.trainable_variables + \
                           model.final_dense_profile_atac.trainable_variables + \
                           model.assay_type_fc.trainable_variables + \
                           model.final_dense_profile_rna.trainable_variables

            vars_all = conv_performer_vars + output_heads
            for var in vars_all:
                tape.watch(var)

            output_atac,output_rna = model(input_tuple,
                                           training=True)

            mask_indices = tf.where(mask[0,:,0] == 1)[:,0]

            target_atac = tf.gather(target_atac[:,:,0], mask_indices,axis=1)
            output_atac = tf.gather(output_atac[:,:,0], mask_indices,axis=1)

            atac_loss = tf.reduce_mean(poisson_multinomial(target_atac,
                                                           output_atac,
                                                           total_weight=0.15,
                                                           rescale=True)) *\
                                                           (1.0/global_batch_size)

            rna_loss = tf.reduce_mean(poisson_multinomial(target_rna[:,:,0],
                                                          output_rna[:,:,0],
                                                          total_weight=0.15,
                                                          rescale=True) * weights) *\
                                                          (1.0/global_batch_size)
            loss = atac_loss * (1.0-rna_scale) + rna_loss * rna_scale

        gradients = tape.gradient(loss, vars_all)
        gradients, _ = tf.clip_by_global_norm(gradients,
                                              gradient_clip)

        optimizer1.apply_gradients(zip(gradients[:len(conv_performer_vars)],
                                       conv_performer_vars))
        optimizer2.apply_gradients(zip(gradients[len(conv_performer_vars):],
                                       output_heads))
        metric_dict["train_loss"].update_state(loss)
        metric_dict["train_loss_rna"].update_state(rna_loss)
        metric_dict["train_loss_atac"].update_state(atac_loss)

    @tf.function(reduce_retracing=True)
    def dist_val_step(inputs):
        print('tracing validation step!')
        sequence,atac,mask,mask_gathered,peaks,target_atac,target_rna,assay_type,weights,tf_activity =inputs
        input_tuple = sequence, atac, target_rna,tf_activity,assay_type

        output_atac,output_rna = model(input_tuple,
                               training=False)

        mask_indices = tf.where(mask[0,:,0] == 1)[:,0]

        target_atac = tf.gather(target_atac[:,:,0], mask_indices,axis=1)
        output_atac = tf.gather(output_atac[:,:,0], mask_indices,axis=1)

        atac_loss = tf.reduce_mean(poisson_multinomial(target_atac,
                                                       output_atac,
                                                       total_weight=0.15,
                                                       rescale=True)) *\
                                                       (1.0/global_batch_size)

        rna_loss = tf.reduce_mean(poisson_multinomial(target_rna[:,:,0],
                                                      output_rna[:,:,0],
                                                      total_weight=0.15,
                                                      rescale=True)*weights) *\
                                                      (1.0/global_batch_size)
        loss = atac_loss * (1.0-rna_scale) + rna_loss * rna_scale

        metric_dict['ATAC_PearsonR'].update_state(target_atac,
                                                  output_atac)
        metric_dict['ATAC_R2'].update_state(target_atac,
                                            output_atac)
        metric_dict["val_loss"].update_state(loss)
        metric_dict["val_loss_rna"].update_state(rna_loss)
        metric_dict["val_loss_atac"].update_state(atac_loss)

        return target_rna[:,:,0], output_rna[:,:,0], assay_type

    def build_step(iterator):
        @tf.function(reduce_retracing=True)
        def val_step(inputs):
            sequence,atac,mask,mask_gathered,peaks,target_atac,target_rna,assay_type,weights,tf_activity = inputs
            input_tuple = sequence, atac, target_rna,tf_activity, assay_type

            output_atac,output_rna= model(input_tuple,
                                           training=False)
        strategy.run(val_step, args=(next(iterator),))

    return dist_train_step,dist_val_step, build_step, metric_dict

def deserialize_tr(serialized_example, g, use_tf_activity, input_length = 196608,
                   max_shift = 10, output_length_ATAC = 49152, output_length = 1536,
                   crop_size = 320, output_res = 128, atac_mask_dropout = 0.15,
                   mask_size = 896, log_atac = True, use_atac = True,
                   use_seq = True, seq_corrupt_rate = 20, atac_corrupt_rate = 20):
    """Deserialize bytes stored in TFRecordFile."""
    ## parse out feature map
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'rna': tf.io.FixedLenFeature([], tf.string),
        'rna_assay_type': tf.io.FixedLenFeature([], tf.string),
        'tss_tokens': tf.io.FixedLenFeature([], tf.string),
        'peaks': tf.io.FixedLenFeature([], tf.string),
        'peaks_center': tf.io.FixedLenFeature([], tf.string),
        'tf_activity': tf.io.FixedLenFeature([], tf.string)
    }
    '''
    generate random numbers for data augmentation
      rev_comp: whether or not to reverse/complement sequence + signal
      seq_mask_int: whether we will randomly also mask the sequence underlying
                    masked ATAC regions
      randomish_seed: hacky workaround to previous issue with random atac masking
    '''
    rev_comp = tf.math.round(g.uniform([], 0, 1)) #switch for random reverse complementation
    seq_mask_int = g.uniform([], 0, seq_corrupt_rate, dtype=tf.int32) ## sequence corruption rate
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
                                              out_type=tf.float32),
                           [output_length_ATAC,1])
    rna = tf.ensure_shape(tf.io.parse_tensor(data['rna'],
                                              out_type=tf.float32),
                           [output_length,1])
    rna_assay_type = tf.ensure_shape(tf.io.parse_tensor(data['rna_assay_type'],
                                              out_type=tf.int32),
                                 [])
    rna_assay_type = tf.expand_dims(rna_assay_type,axis=0)
    peaks = tf.ensure_shape(tf.io.parse_tensor(data['peaks'],
                                              out_type=tf.int32),
                           [output_length])
    peaks_center = tf.ensure_shape(tf.io.parse_tensor(data['peaks_center'],
                                              out_type=tf.int32),
                           [output_length])
    tf_activity = tf.ensure_shape(tf.io.parse_tensor(data['tf_activity'],
                                              out_type=tf.float32),
                           [1629])
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

    ''' mask of one of the peaks. If no peaks, mask middle of the window '''
    atac_target = atac ## store the target ATAC

    ### here set up the ATAC masking
    num_mask_bins = mask_size // output_res ## calculate the number of adjacent bins that will be masked in each region


    center = (output_length-2*crop_size)//2
    ### here set up masking of one of the peaks
    mask_indices_temp = tf.where(peaks_c_crop[:,0] > 0)[:,0]
    ridx = tf.concat([tf.random.experimental.stateless_shuffle(mask_indices_temp,seed=[4+randomish_seed,5]),
                      tf.constant([center],dtype=tf.int64)],axis=0)   ### concatenate the middle in case theres no peaks
    mask_indices = [[ridx[0]+x+crop_size] for x in range(-num_mask_bins//2,1+(num_mask_bins//2))]

    st=tf.SparseTensor(
        indices=mask_indices,
        values=[1.0]*len(mask_indices),
        dense_shape=[output_length])
    dense_peak_mask=tf.sparse.to_dense(st)
    dense_peak_mask_store = dense_peak_mask
    dense_peak_mask=1.0-dense_peak_mask ### masking regions here are set to 1. so invert the mask to actually use
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
    full_comb_mask_store = full_comb_mask_store[crop_size:-crop_size,:] # store the cropped mask
    tiling_req = output_length_ATAC // output_length ### how much do we need to tile the atac signal to desired length
    full_comb_mask = tf.expand_dims(tf.reshape(tf.tile(full_comb_mask, [1,tiling_req]),[-1]),axis=1)

    masked_atac = atac * full_comb_mask

    if log_atac:
        masked_atac = tf.math.log1p(masked_atac)

    diff = tf.math.sqrt(tf.nn.relu(masked_atac - 10000.0 * tf.ones(masked_atac.shape)))
    masked_atac = tf.clip_by_value(masked_atac, clip_value_min=0.0, clip_value_max=10000.0) + diff

    ''' here set up the random sequence masking '''
    if ((seq_mask_int == 0) and (atac_mask_int != 0)):
        seq_mask = 1.0 - full_comb_mask_full_store
        tiling_req_seq = input_length // output_length
        seq_mask = tf.expand_dims(tf.reshape(tf.tile(seq_mask, [1,tiling_req_seq]),[-1]),axis=1)
        masked_seq = sequence * seq_mask + tf.random.experimental.stateless_shuffle(sequence,
                                                                                    seed=[randomish_seed+30,randomish_seed])*(1.0-seq_mask)
    else:
        seq_mask = 1.0 - full_comb_mask_full_store
        tiling_req_seq = input_length // output_length
        seq_mask = tf.expand_dims(tf.reshape(tf.tile(seq_mask, [1,tiling_req_seq]),[-1]),axis=1)
        masked_seq = sequence

    ''' randomly reverse complement the sequence, and reverse targets + peaks + mask'''
    if rev_comp == 1:
        masked_seq = tf.gather(masked_seq, [3, 2, 1, 0], axis=-1)
        masked_seq = tf.reverse(masked_seq, axis=[0])
        atac_target = tf.reverse(atac_target,axis=[0])
        masked_atac = tf.reverse(masked_atac,axis=[0])
        peaks_crop=tf.reverse(peaks_crop,axis=[0])
        full_comb_mask_store=tf.reverse(full_comb_mask_store,axis=[0])
        rna = tf.reverse(rna, axis=[0])

    atac_out = tf.reduce_sum(tf.reshape(atac_target, [-1,tiling_req]),axis=1,keepdims=True)
    diff = tf.math.sqrt(tf.nn.relu(atac_out - 50000.0 * tf.ones(atac_out.shape)))
    atac_out = tf.clip_by_value(atac_out, clip_value_min=0.0, clip_value_max=50000.0) + diff
    atac_out = tf.slice(atac_out,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])
    rna_out = tf.slice(rna,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])
    diff = tf.math.sqrt(tf.nn.relu(rna_out - 10000.0 * tf.ones(rna_out.shape)))
    rna_out = tf.clip_by_value(rna_out, clip_value_min=0.0, clip_value_max=10000.0) + diff
    rna_out=tf.where(tf.math.is_inf(rna_out), tf.zeros_like(rna_out), rna_out)

    peaks_gathered = tf.reduce_max(tf.reshape(peaks_crop, [(output_length-2*crop_size) // 4, -1]),
                                   axis=1,keepdims=True)
    mask_gathered = tf.reduce_max(tf.reshape(full_comb_mask_store, [(output_length-2*crop_size) // 4, -1]),
                                   axis=1,keepdims=True)

    ''' in case we want to run ablation without these inputs'''
    if not use_atac:
        print('not using atac')
        masked_atac = tf.math.abs(g.normal(masked_atac.shape,
                               mean=0.0,
                               stddev=1.0,
                               dtype=tf.float32))
    if not use_seq:
        print('not using sequence')
        masked_seq = tf.random.experimental.stateless_shuffle(masked_seq,
                                                              seed=[randomish_seed+1,randomish_seed+3])



    ### add variable for loss weighting for different assay cell_types
    rna_lookup = {0:1.0, 1:2.0,2:0.40, 3: 1.5,
                                 4: 0.70,5:1.25,6:0.20,7:0.15}
    keys_tensor = tf.constant(list(rna_lookup.keys()))
    vals_tensor = tf.constant(list(rna_lookup.values()))
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value=1.0)
    weighting_factor=table.lookup(rna_assay_type)

    return tf.cast(tf.ensure_shape(masked_seq,[input_length,4]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(masked_atac, [output_length_ATAC,1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(full_comb_mask_store, [output_length-crop_size*2,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(mask_gathered, [(output_length-crop_size*2) // 4,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(peaks_gathered, [(output_length-2*crop_size) // 4,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(atac_out,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(rna_out,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(rna_assay_type,[1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(weighting_factor,[1]),dtype=tf.float32), \
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
        'rna': tf.io.FixedLenFeature([], tf.string),
        'rna_assay_type': tf.io.FixedLenFeature([], tf.string),
        'tss_tokens': tf.io.FixedLenFeature([], tf.string),
        'peaks': tf.io.FixedLenFeature([], tf.string),
        'peaks_center': tf.io.FixedLenFeature([], tf.string),
        'tf_activity': tf.io.FixedLenFeature([], tf.string)
    }
    ### stochastic sequence shift and gaussian noise
    seq_shift=5

    input_seq_length = input_length + max_shift

    ## now parse out the actual data
    data = tf.io.parse_example(serialized_example, feature_map)
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 seq_shift,input_length))
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float32),
                           [output_length_ATAC,1])
    rna = tf.ensure_shape(tf.io.parse_tensor(data['rna'],
                                              out_type=tf.float32),
                           [output_length,1])
    rna_assay_type = tf.ensure_shape(tf.io.parse_tensor(data['rna_assay_type'],
                                              out_type=tf.int32),
                                 [])
    rna_assay_type = tf.expand_dims(rna_assay_type,axis=0)
    peaks = tf.ensure_shape(tf.io.parse_tensor(data['peaks'],
                                              out_type=tf.int32),
                           [output_length])
    peaks_center = tf.ensure_shape(tf.io.parse_tensor(data['peaks_center'],
                                              out_type=tf.int32),
                           [output_length])
    tf_activity = tf.ensure_shape(tf.io.parse_tensor(data['tf_activity'],
                                              out_type=tf.float32),
                           [1629])
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
    #
    #tf_activity = tf_activity / percentile99

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

    ### here set up the ATAC masking
    num_mask_bins = mask_size // output_res # the number of adjacent bins to mask

    center = (output_length-2*crop_size)//2 # the center of the window

    ### here set up masking of one of the peaks
    mask_indices_temp = tf.where(peaks_c_crop[:,0] > 0)[:,0]
    ridx = tf.concat([tf.random.experimental.stateless_shuffle(mask_indices_temp,seed=[4+randomish_seed,5]),
                      tf.constant([center],dtype=tf.int64)],axis=0)   ### concatenate the middle in case theres no peaks
    mask_indices = [[ridx[0]+x+crop_size] for x in range(-num_mask_bins//2,1+(num_mask_bins//2))]

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
    rna_out = tf.slice(rna,
                        [crop_size,0],
                        [output_length-2*crop_size,-1])
    diff = tf.math.sqrt(tf.nn.relu(rna_out - 10000.0 * tf.ones(rna_out.shape)))
    rna_out = tf.clip_by_value(rna_out, clip_value_min=0.0, clip_value_max=10000.0) + diff
    rna_out=tf.where(tf.math.is_inf(rna_out), tf.zeros_like(rna_out), rna_out)

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

    ### add variable for loss weighting for different assay cell_types
    rna_lookup = {0:1.0, 1:2.0,2:0.40, 3: 1.5,
                                 4: 0.70,5:1.25,6:0.20,7:0.15}
    keys_tensor = tf.constant(list(rna_lookup.keys()))
    vals_tensor = tf.constant(list(rna_lookup.values()))
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value=-1.0)
    weighting_factor=table.lookup(rna_assay_type)

    return tf.cast(tf.ensure_shape(sequence,[input_length,4]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(masked_atac, [output_length_ATAC,1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(full_comb_mask_store, [output_length-crop_size*2,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(mask_gathered, [(output_length-crop_size*2) // 4,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(peaks_gathered, [(output_length-2*crop_size) // 4,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(atac_out,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(rna_out,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(rna_assay_type,[1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(weighting_factor,[1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(tf_activity, [1,1629]),dtype=tf.bfloat16)

def deserialize_val_TSS(serialized_example, g, use_tf_activity, input_length = 196608,
                   max_shift = 10, output_length_ATAC = 49152, output_length = 1536,
                   crop_size = 320, output_res = 128, atac_mask_dropout = 0.15,
                   mask_size = 896, log_atac = True, use_atac = True, use_seq = True):
    """Deserialize bytes stored in TFRecordFile."""
    ## parse out feature map
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'rna': tf.io.FixedLenFeature([], tf.string),
        'rna_assay_type': tf.io.FixedLenFeature([], tf.string),
        'tss_tokens': tf.io.FixedLenFeature([], tf.string),
        'peaks': tf.io.FixedLenFeature([], tf.string),
        'peaks_center': tf.io.FixedLenFeature([], tf.string),
        'tf_activity': tf.io.FixedLenFeature([], tf.string),
        'gene_token': tf.io.FixedLenFeature([], tf.string),
        'cell_type': tf.io.FixedLenFeature([], tf.string)
    }
    ### stochastic sequence shift and gaussian noise
    seq_shift=5

    input_seq_length = input_length + max_shift

    ## now parse out the actual data
    data = tf.io.parse_example(serialized_example, feature_map)
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 seq_shift,input_length))
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float32),
                           [output_length_ATAC,1])

    gene_token= tf.io.parse_tensor(data['processed_gene_token'],
                                   out_type=tf.int32)

    cell_type = tf.io.parse_tensor(data['cell_type'],
                                  out_type=tf.int32)

    rna = tf.ensure_shape(tf.io.parse_tensor(data['rna'],
                                              out_type=tf.float32),
                           [output_length,1])
    rna_assay_type = tf.ensure_shape(tf.io.parse_tensor(data['rna_assay_type'],
                                              out_type=tf.int32),
                                 [])
    rna_assay_type = tf.expand_dims(rna_assay_type,axis=0)
    peaks = tf.ensure_shape(tf.io.parse_tensor(data['peaks'],
                                              out_type=tf.int32),
                           [output_length])
    peaks_center = tf.ensure_shape(tf.io.parse_tensor(data['peaks_center'],
                                              out_type=tf.int32),
                           [output_length])
    tf_activity = tf.ensure_shape(tf.io.parse_tensor(data['tf_activity'],
                                              out_type=tf.float32),
                           [1629])
    tf_activity = tf.expand_dims(tf_activity,axis=0)
    if not use_tf_activity:
        tf_activity = tf.zeros_like(tf_activity)
    tf_activity = tf_activity + tf.math.abs(g.normal(tf_activity.shape,
                                               mean=0.0,
                                               stddev=0.005,
                                               dtype=tf.float32))

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

    ### here set up the ATAC masking
    num_mask_bins = mask_size // output_res # the number of adjacent bins to mask

    center = (output_length-2*crop_size)//2 # the center of the window

    ### here set up masking of one of the peaks
    mask_indices_temp = tf.where(peaks_c_crop[:,0] > 0)[:,0]
    ridx = tf.concat([tf.random.experimental.stateless_shuffle(mask_indices_temp,seed=[4+randomish_seed,5]),
                      tf.constant([center],dtype=tf.int64)],axis=0)   ### concatenate the middle in case theres no peaks
    mask_indices = [[ridx[0]+x+crop_size] for x in range(-num_mask_bins//2,1+(num_mask_bins//2))]

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
    rna_out = tf.slice(rna,
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

    return tf.cast(tf.ensure_shape(sequence,[input_length,4]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(masked_atac, [output_length_ATAC,1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(full_comb_mask_store, [output_length-crop_size*2,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(mask_gathered, [(output_length-crop_size*2) // 4,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(peaks_gathered, [(output_length-2*crop_size) // 4,1]),dtype=tf.int32), \
                tf.cast(tf.ensure_shape(atac_out,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(rna_out,[output_length-crop_size*2,1]),dtype=tf.float32), \
                tf.cast(tf.ensure_shape(rna_assay_type,[1,1]),dtype=tf.bfloat16), \
                tf.cast(tf.ensure_shape(tf_activity, [1,1629]),dtype=tf.bfloat16),\
                gene_token, cell_type


def return_dataset(gcs_path, split, tss_bool, batch, input_length, output_length_ATAC,
                   output_length, crop_size, output_res, max_shift, options,
                   num_parallel, num_epoch, atac_mask_dropout,
                   random_mask_size, log_atac, use_atac, use_seq, seed,
                   seq_corrupt_rate, atac_corrupt_rate, validation_steps,
                   use_tf_activity, g):

    """
    return a tf dataset object for given gcs path
    """

    if split == 'train':
        wc = "*.tfr"
        list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                    split,
                                                    wc)))

        files = tf.data.Dataset.list_files(list_files,shuffle=True,seed=seed)

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
                                                            seq_corrupt_rate, atac_corrupt_rate),
                              deterministic=False,
                              num_parallel_calls=num_parallel)

        return dataset.repeat((num_epoch*2)).batch(batch).prefetch(tf.data.AUTOTUNE)


    else:
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
                                 options, atac_mask_dropout, random_mask_size,
                                 log_atac, use_atac, use_seq, seed,
                                 seq_corrupt_rate, atac_corrupt_rate,
                                 validation_steps, use_tf_activity, g):

    """
    returns train + val dictionaries of distributed iterators
    for given heads_dictionary
    """

    tr_data = return_dataset(gcs_path, "train", False, global_batch_size,
                             input_length, output_length_ATAC, output_length,
                             crop_size, output_res, max_shift, options,
                             num_parallel_calls, num_epoch, atac_mask_dropout,
                             random_mask_size, log_atac, use_atac, use_seq,
                             seed, seq_corrupt_rate,atac_corrupt_rate, validation_steps,
                             use_tf_activity, g)

    val_data = return_dataset(gcs_path_ho, "valid", False, global_batch_size,
                             input_length, output_length_ATAC, output_length,
                             crop_size, output_res, max_shift, options,
                             num_parallel_calls, num_epoch, atac_mask_dropout,
                             random_mask_size, log_atac, use_atac, use_seq,
                             seed, seq_corrupt_rate,atac_corrupt_rate, validation_steps,
                             use_tf_activity, g)
    '''
    val_data_ho = return_dataset(gcs_path_ho,
                             "valid",
                             False,
                             global_batch_size,
                             input_length,
                             output_length_ATAC,
                             output_length,
                             crop_size,
                             output_res,
                             max_shift,
                             options,
                             num_parallel_calls,
                             num_epoch,
                             atac_mask_dropout,
                             mask_size,
                             log_atac,
                             use_atac,
                             use_seq,
                             g)

    val_data_TSS = return_dataset(gcs_path_TSS,
                             "valid",
                             True,
                             global_batch_size,
                             input_length,
                             output_length_ATAC,
                             output_length,
                             crop_size,
                             output_res,
                             max_shift,
                             options,
                             num_parallel_calls,
                             num_epoch,
                             0.0,
                             mask_size,
                             log_atac,
                             use_atac,
                             use_seq,
                             g)


    val_data_TSS_ho = return_dataset(gcs_path_TSS_holdout,
                             "valid",
                             True,
                             global_batch_size,
                             input_length,
                             output_length_ATAC,
                             output_length,
                             crop_size,
                             output_res,
                             max_shift,
                             options,
                             num_parallel_calls,
                             num_epoch,
                             0.0,
                             mask_size,
                             log_atac,
                             use_atac,
                             use_seq,
                             g)
    '''

    train_dist = strategy.experimental_distribute_dataset(tr_data)
    val_dist= strategy.experimental_distribute_dataset(val_data)
    #val_dist_ho=strategy.experimental_distribute_dataset(val_data_ho)
    #val_dist_TSS= strategy.experimental_distribute_dataset(val_data_TSS)
    #val_dist_TSS_ho = strategy.experimental_distribute_dataset(val_data_TSS_ho)

    tr_data_it = iter(train_dist)
    val_data_it = iter(val_dist)
    #val_data_ho_it = iter(val_dist_ho)
    #val_data_TSS_it = iter(val_dist_TSS)
    #val_data_TSS_ho_it = iter(val_dist_TSS_ho)


    return tr_data_it,val_data_it#,val_data_ho_it,val_data_TSS_it,val_data_TSS_ho_it


def make_plots(y_trues,
               y_preds,
               cell_types,
               gene_map, num_points):

    results_df = pd.DataFrame()
    results_df['true'] = y_trues
    results_df['pred'] = y_preds
    results_df['gene_encoding'] =gene_map
    results_df['cell_type_encoding'] = cell_types

    results_df=results_df.groupby(['gene_encoding', 'cell_type_encoding']).agg({'true': 'sum', 'pred': 'sum'})
    results_df['true'] = np.log2(1.0+results_df['true'])
    results_df['pred'] = np.log2(1.0+results_df['pred'])

    results_df['true_zscore']=results_df.groupby(['cell_type_encoding']).true.transform(lambda x : zscore(x))
    results_df['pred_zscore']=results_df.groupby(['cell_type_encoding']).pred.transform(lambda x : zscore(x))


    true_zscore=results_df[['true_zscore']].to_numpy()[:,0]
    pred_zscore=results_df[['pred_zscore']].to_numpy()[:,0]

    try:
        cell_specific_corrs=results_df.groupby('cell_type_encoding')[['true_zscore','pred_zscore']].corr(method='pearson').unstack().iloc[:,1].tolist()
        cell_specific_corrs_raw=results_df.groupby('cell_type_encoding')[['true','pred']].corr(method='pearson').unstack().iloc[:,1].tolist()
    except np.linalg.LinAlgError as err:
        cell_specific_corrs = [0.0] * len(np.unique(cell_types))

    try:
        gene_specific_corrs=results_df.groupby('gene_encoding')[['true_zscore','pred_zscore']].corr(method='pearson').unstack().iloc[:,1].tolist()
        gene_specific_corrs_raw=results_df.groupby('gene_encoding')[['true','pred']].corr(method='pearson').unstack().iloc[:,1].tolist()
    except np.linalg.LinAlgError as err:
        gene_specific_corrs = [0.0] * len(np.unique(gene_map))

    corrs_overall = np.nanmean(cell_specific_corrs), \
                        np.nanmean(gene_specific_corrs), \
                            np.nanmean(cell_specific_corrs_raw), \
                                np.nanmean(gene_specific_corrs_raw)


    fig_overall,ax_overall=plt.subplots(figsize=(6,6))

    ## scatter plot for 50k points max

    try:
        idx = np.random.choice(np.arange(len(true_zscore)), num_points, replace=False)
    except ValueError:
        print('subsampling 10 points only. figure out why!')
        idx = np.random.choice(np.arange(len(true_zscore)), 10, replace=False)

    data = np.vstack([true_zscore[idx],
                      pred_zscore[idx]])

    min_true = min(true_zscore)
    max_true = max(true_zscore)

    min_pred = min(pred_zscore)
    max_pred = max(pred_zscore)


    try:
        kernel = stats.gaussian_kde(data)(data)
        sns.scatterplot(
            x=true_zscore[idx],
            y=pred_zscore[idx],
            c=kernel,
            cmap="viridis")
        ax_overall.set_xlim(min_true,max_true)
        ax_overall.set_ylim(min_pred,max_pred)
        plt.xlabel("log-true")
        plt.ylabel("log-pred")
        plt.title("overall gene corr")
    except np.linalg.LinAlgError as err:
        sns.scatterplot(
            x=true_zscore[idx],
            y=pred_zscore[idx],
            cmap="viridis")
        ax_overall.set_xlim(min_true,max_true)
        ax_overall.set_ylim(min_pred,max_pred)
        plt.xlabel("log-true")
        plt.ylabel("log-pred")
        plt.title("overall gene corr")
    except ValueError:
        sns.scatterplot(
            x=true_zscore[idx],
            y=pred_zscore[idx],
            cmap="viridis")
        plt.xlabel("log-true")
        plt.ylabel("log-pred")
        plt.title("overall gene corr")

    fig_gene_spec,ax_gene_spec=plt.subplots(figsize=(6,6))
    sns.histplot(x=np.asarray(gene_specific_corrs), bins=50)
    plt.xlabel("log-log pearsons")
    plt.ylabel("count")
    plt.title("single gene cross cell-type correlations")

    fig_cell_spec,ax_cell_spec=plt.subplots(figsize=(6,6))
    sns.histplot(x=np.asarray(cell_specific_corrs), bins=50)
    plt.xlabel("log-log pearsons")
    plt.ylabel("count")
    plt.title("single cell-type cross gene correlations")

        ### by coefficient variation breakdown
    figures = fig_cell_spec, fig_gene_spec, fig_overall

    return figures, corrs_overall



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
    parser.add_argument('--gcs_path_TSS',
                        dest='gcs_path_TSS',
                        help= 'google bucket containing preprocessed data')
    parser.add_argument('--gcs_path_TSS_holdout',
                        dest='gcs_path_TSS_holdout',
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
    parser.add_argument('--val_examples', dest = 'val_examples',
                        type=int, help='val_examples')
    parser.add_argument('--val_examples_ho', dest = 'val_examples_ho',
                        type=int, help='val_examples_ho')
    parser.add_argument('--val_examples_TSS', dest = 'val_examples_TSS',
                        type=int, help='val_examples_TSS')
    parser.add_argument('--val_examples_TSS_ho', dest = 'val_examples_TSS_ho',
                        type=int, help='val_examples_TSS_ho')
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
    parser.add_argument('--lr_base1',
                        dest='lr_base1',
                        default="1.0e-03",
                        help='lr_base1')
    parser.add_argument('--lr_base2',
                        dest='lr_base2',
                        default="1.0e-03",
                        help='lr_base2')
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
                        default="gs://picard-testing-176520/genformer_atac_pretrain/models/aformer_524k_load-False_LR1-0.0001_LR2-0.0001_T-7_TF-False_2023-10-01_15:07:58/iteration_26",
                        help= 'checkpoint_path')
    parser.add_argument('--load_init_FT',
                        dest='load_init_FT',
                        type=str,
                        default="True",
                        help= 'load_init_FT')
    parser.add_argument('--load_init_FULL',
                        dest='load_init_FULL',
                        type=str,
                        default="True",
                        help= 'load_init_FULL')
    parser.add_argument('--use_rot_emb',
                        dest='use_rot_emb',
                        type=str,
                        default="True",
                        help= 'use_rot_emb')
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
    parser.add_argument('--freeze_conv_layers',
                        dest='freeze_conv_layers',
                        type=str,
                        default="False",
                        help= 'freeze_conv_layers')
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
    parser.add_argument('--atac_mask_dropout',
                        dest='atac_mask_dropout',
                        type=float,
                        default=0.05,
                        help= 'atac_mask_dropout')
    parser.add_argument('--final_point_scale',
                        dest='final_point_scale',
                        type=str,
                        default="6",
                        help= 'final_point_scale')
    parser.add_argument('--rna_scale',
                        dest='rna_scale',
                        type=str,
                        default="5.0",
                        help= 'rna_scale')
    parser.add_argument('--rectify',
                        dest='rectify',
                        type=str,
                        default="True",
                        help= 'rectify')
    parser.add_argument('--log_atac',
                        dest='log_atac',
                        type=str,
                        default="True",
                        help= 'log_atac')
    parser.add_argument('--random_mask_size',
                        dest='random_mask_size',
                        type=str,
                        default="1024",
                        help= 'random_mask_size')
    parser.add_argument('--seed',
                        dest='seed',
                        type=int,
                        default=42,
                        help= 'seed')
    parser.add_argument('--seq_corrupt_rate',
                        dest='seq_corrupt_rate',
                        type=str,
                        default="20",
                        help= 'seq_corrupt_rate')
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
