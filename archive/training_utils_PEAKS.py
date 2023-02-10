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
import src.schedulers
from src.losses import regular_mse,abs_mse,poisson
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

import numpy as np
from sklearn import metrics as sklearn_metrics

from tensorflow.keras import initializers as inits

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


def get_initializers(checkpoint_path):
    
    inside_checkpoint=tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_path))
    reader = tf.train.load_checkpoint(checkpoint_path)

    initializers_dict = {'stem_conv_k': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/0/w/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_conv_b': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/0/b/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_k': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_b': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_g': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_b': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_m': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE')[0,0:]),
                         'stem_res_conv_BN_v': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE')[0,0:]),
                         'stem_pool': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/2/_logit_linear/w/.ATTRIBUTES/VARIABLE_VALUE'))}

    for i in range(6):
        var_name_stem = 'module/_trunk/_layers/1/_layers/' + str(i) + '/_layers/' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        conv1_k = var_name_stem + '0/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE'
        conv1_b = var_name_stem + '0/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_g = var_name_stem + '0/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_b = var_name_stem + '0/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_m = var_name_stem + '0/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_v = var_name_stem + '0/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE'
        conv2_k = var_name_stem + '1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE'
        conv2_b = var_name_stem + '1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_g = var_name_stem + '1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_b = var_name_stem + '1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_m = var_name_stem + '1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_v = var_name_stem + '1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE'
        pool = var_name_stem + '2/_logit_linear/w/.ATTRIBUTES/VARIABLE_VALUE'
        all_vars = [conv1_k,
                    conv1_b,
                    BN1_g,
                    BN1_b,
                    BN1_m,
                    BN1_v,
                    conv2_k,
                    conv2_b,
                    BN2_g,
                    BN2_b,
                    BN2_m,
                    BN2_v,
                    pool]

        out_dict = {'conv1_k_' + str(i): inits.Constant(reader.get_tensor(conv1_k)),
                    'conv1_b_' + str(i): inits.Constant(reader.get_tensor(conv1_b)),
                    'BN1_g_' + str(i): inits.Constant(reader.get_tensor(BN1_g)),
                    'BN1_b_' + str(i): inits.Constant(reader.get_tensor(BN1_b)),
                    'BN1_m_' + str(i): inits.Constant(reader.get_tensor(BN1_m)[0,0:]),
                    'BN1_v_' + str(i): inits.Constant(reader.get_tensor(BN1_v)[0,0:]),
                    'conv2_k_' + str(i): inits.Constant(reader.get_tensor(conv2_k)),
                    'conv2_b_' + str(i): inits.Constant(reader.get_tensor(conv2_b)),
                    'BN2_g_' + str(i): inits.Constant(reader.get_tensor(BN2_g)),
                    'BN2_b_' + str(i): inits.Constant(reader.get_tensor(BN2_b)),
                    'BN2_m_' + str(i): inits.Constant(reader.get_tensor(BN2_m)[0,0:]),
                    'BN2_v_' + str(i): inits.Constant(reader.get_tensor(BN2_v)[0,0:]),
                    'pool_' + str(i): inits.Constant(reader.get_tensor(pool))}
        initializers_dict.update(out_dict)
    return initializers_dict

"""
having trouble w/ providing organism/step inputs to train/val steps w/o
triggering retracing/metadata resource exhausted errors, so defining 
them separately for hg, mm 
to do: simplify to two functions w/ organism + mini_batch_step_inputs
consolidate into single simpler function
"""


def corr_coef(x, y, eps = 1.0e-07):
    x2 = tf.math.square(x)
    y2 = tf.math.square(y)
    xy = x * y
    ex = tf.reduce_mean(x, axis = 1)
    ey = tf.reduce_mean(y, axis = 1)
    exy = tf.reduce_mean(xy, axis = 1)
    ex2 = tf.reduce_mean(x2, axis = 1)
    ey2 = tf.reduce_mean(y2, axis = 1)
    r = (exy - ex * ey) / ((tf.math.sqrt(ex2 - tf.math.square(ex) + eps) * tf.math.sqrt(ey2 - tf.math.square(ey) + eps)) + eps)
    return tf.reduce_mean(r, axis = -1)


def return_train_val_functions(model,
                               optimizers_in,
                               strategy,
                               metric_dict,
                               train_steps, 
                               val_steps,
                               val_steps_ho,
                               global_batch_size,
                               gradient_clip,
                               batch_size,
                               loss_fn_main='poisson',
                               use_peaks=True,
                               use_coef_var=False,
                               use_atac=False):
    """Returns distributed train and validation functions for
    a given list of organisms
    Args:
        model: model object
        optimizer: optimizer object
        metric_dict: empty dictionary to populate with organism
                     specific metrics
        train_steps: number of train steps to take in single epoch
        val_steps: number of val steps to take in single epoch
        global_batch_size: # replicas * batch_size_per_replica
        gradient_clip: gradient clip value to be applied in case of adam/adamw optimizer
    Returns:
        distributed train function
        distributed val function
        metric_dict: dict of tr_loss,val_loss, correlation_stats metrics
                     for input organisms
    
    return distributed train and val step functions for given organism
    train_steps is the # steps in a single epoch
    val_steps is the # steps to fully iterate over validation set
    """
    metric_dict["hg_tr"] = tf.keras.metrics.Mean("hg_tr_loss",
                                                 dtype=tf.float32)
    metric_dict["hg_val"] = tf.keras.metrics.Mean("hg_val_loss",
                                                  dtype=tf.float32)
    
    metric_dict["hg_corr_stats"] = metrics.correlation_stats_gene_centered(name='hg_corr_stats')
    
    metric_dict["hg_corr_stats_ho"] = metrics.correlation_stats_gene_centered(name='hg_corr_stats_ho')
    
    metric_dict["hg_val_ho"] = tf.keras.metrics.Mean("hg_val_loss_ho", dtype=tf.float32)
    
    if loss_fn_main == 'mse':
        loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    elif loss_fn_main == 'poisson':
        loss_fn = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)
    else:
        raise ValueError('loss_fn_not_implemented')

    optimizer1,optimizer2=optimizers_in
    
    def dist_train_step(iterator):
        @tf.function(jit_compile=True)
        def train_step(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.bfloat16)
            atac=tf.cast(inputs['atac'],dtype=tf.bfloat16)
            tss_tokens=tf.cast(inputs['tss_tokens'],dtype=tf.bfloat16)
            peaks_sequences = tf.cast(inputs['peaks_sequences'],dtype=tf.bfloat16)
            coef_var= tf.cast(inputs['coef_var'],dtype=tf.float32)


            if not use_atac:
                atac = tf.zeros_like(atac)
            
            if not use_peaks:
                peaks_sequences = tf.zeros_like(peaks_sequences)
                
            if not use_coef_var:
                coef_var = tf.ones_like(coef_var)
                
            exons=tf.cast(inputs['exons'],dtype=tf.bfloat16)
            target=tf.cast(inputs['target'],dtype=tf.float32)
            
            input_tuple = sequence,tss_tokens,exons, peaks_sequences, atac
            
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                conv_vars = model.stem_conv.trainable_variables + \
                            model.stem_res_conv.trainable_variables + \
                            model.stem_pool.trainable_variables + \
                            model.conv_tower_seq.trainable_variables
                for i in range(6):
                    conv_vars = conv_vars + model.conv_tower_peaks.layers[i].layers[2].trainable_variables

                remaining_vars = model.peaks_module.trainable_variables + \
                                    model.conv_mix_block.trainable_variables + \
                                    model.shared_transformer.trainable_variables + \
                                    model.final_pointwise_rna.trainable_variables + \
                                    model.rna_head.trainable_variables 
                
                vars_subset = conv_vars + remaining_vars
                
                for var in vars_subset:
                    tape.watch(var)
                    
                output = model(input_tuple,
                                training=True)
                output = tf.cast(output,dtype=tf.float32)
                loss = loss_fn(target,output)
                loss = tf.math.reduce_sum(loss, axis=-1) * coef_var
                loss = tf.math.reduce_sum(loss) * (1. / global_batch_size)

            gradients = tape.gradient(loss, vars_subset)
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip)
            
            optimizer1.apply_gradients(zip(gradients[:len(conv_vars)], 
                                           conv_vars))
            optimizer2.apply_gradients(zip(gradients[len(conv_vars):], 
                                           remaining_vars))

            metric_dict["hg_tr"].update_state(loss)
            
        for _ in tf.range(train_steps): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(train_step, args=(next(iterator),))

            
    def dist_val_step(iterator):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.bfloat16)
            atac=tf.cast(inputs['atac'],dtype=tf.bfloat16)
            exons=tf.cast(inputs['exons'],dtype=tf.bfloat16)
            target=tf.cast(inputs['target'],dtype=tf.float32)
            tss_tokens=tf.cast(inputs['tss_tokens'],dtype=tf.bfloat16)
            peaks_sequences = tf.cast(inputs['peaks_sequences'],dtype=tf.bfloat16)
            coef_var= tf.cast(inputs['coef_var'],dtype=tf.float32)
            
            if not use_atac:
                atac = tf.zeros_like(atac)
            
            if not use_peaks:
                peaks_sequences = tf.zeros_like(peaks_sequences)
                
            if not use_coef_var:
                coef_var = tf.ones_like(coef_var)

            
            input_tuple = sequence,tss_tokens,exons, peaks_sequences, atac
            
            
            cell_type = inputs['cell_type']
            gene_map = inputs['gene_encoded']

            output = model(input_tuple,
                            training=False)
            output = tf.cast(output,dtype=tf.float32)
            
            loss = loss_fn(target,output)
            loss = tf.math.reduce_sum(loss, axis=-1) * coef_var
            loss = tf.math.reduce_sum(loss) * (1. / global_batch_size)

            metric_dict["hg_val"].update_state(loss)
            
            return target, output, cell_type, gene_map
            
        ta_pred_h = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store preds
        ta_true_h = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store vals
        ta_celltype_h = tf.TensorArray(tf.int32, size=0, dynamic_size=True) # tensor array to store preds
        ta_genemap_h = tf.TensorArray(tf.int32, size=0, dynamic_size=True)            

        for _ in tf.range(val_steps): ## for loop within @tf.fuction for improved TPU performance
            target_rep, output_rep, cell_type_rep, gene_map_rep = strategy.run(val_step,
                                                                               args=(next(iterator),))
            
            target_reshape = tf.reshape(strategy.gather(target_rep, axis=0), [-1]) # reshape to 1D
            output_reshape = tf.reshape(strategy.gather(output_rep, axis=0), [-1])
            cell_type_reshape = tf.reshape(strategy.gather(cell_type_rep, axis=0), [-1])
            gene_map_reshape = tf.reshape(strategy.gather(gene_map_rep, axis=0), [-1])

            ta_pred_h = ta_pred_h.write(_, output_reshape)
            ta_true_h = ta_true_h.write(_, target_reshape)
            ta_celltype_h = ta_celltype_h.write(_, cell_type_reshape)
            ta_genemap_h = ta_genemap_h.write(_, gene_map_reshape)
            
        metric_dict["hg_corr_stats"].update_state(ta_true_h.concat(),
                                                  ta_pred_h.concat(),
                                                  ta_celltype_h.concat(),
                                                  ta_genemap_h.concat())
            
        ta_pred_h.close()
        ta_true_h.close()
        ta_celltype_h.close()
        ta_genemap_h.close()
            
    def dist_val_step_ho(iterator):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.bfloat16)
            atac=tf.cast(inputs['atac'],dtype=tf.bfloat16)
            tss_tokens=tf.cast(inputs['tss_tokens'],dtype=tf.bfloat16)
            peaks_sequences = tf.cast(inputs['peaks_sequences'],dtype=tf.bfloat16)
            exons=tf.cast(inputs['exons'],dtype=tf.bfloat16)
            target=tf.cast(inputs['target'],dtype=tf.float32)
            coef_var= tf.cast(inputs['coef_var'],dtype=tf.float32)
            
            if not use_atac:
                atac = tf.zeros_like(atac)
            
            if not use_peaks:
                peaks_sequences = tf.zeros_like(peaks_sequences)
                
            if not use_coef_var:
                coef_var = tf.ones_like(coef_var)
            
            input_tuple = sequence,tss_tokens,exons, peaks_sequences, atac
            
            
            cell_type = inputs['cell_type']
            gene_map = inputs['gene_encoded']

            output = model(input_tuple,
                            training=False)
            output = tf.cast(output,dtype=tf.float32)
            loss = loss_fn(target,output)
            loss = tf.math.reduce_sum(loss, axis=-1) * coef_var
            loss = tf.math.reduce_sum(loss) * (1. / global_batch_size)
            
            return target, output, cell_type, gene_map
            
        ta_pred_ho = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store preds
        ta_true_ho = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store vals
        ta_celltype_ho = tf.TensorArray(tf.int32, size=0, dynamic_size=True) # tensor array to store preds
        ta_genemap_ho = tf.TensorArray(tf.int32, size=0, dynamic_size=True)            

        for _ in tf.range(val_steps_ho): ## for loop within @tf.fuction for improved TPU performance
            target_rep, output_rep, cell_type_rep, gene_map_rep = strategy.run(val_step,
                                                                               args=(next(iterator),))
            
            target_reshape = tf.reshape(strategy.gather(target_rep, axis=0), [-1]) # reshape to 1D
            output_reshape = tf.reshape(strategy.gather(output_rep, axis=0), [-1])
            cell_type_reshape = tf.reshape(strategy.gather(cell_type_rep, axis=0), [-1])
            gene_map_reshape = tf.reshape(strategy.gather(gene_map_rep, axis=0), [-1])

            ta_pred_ho = ta_pred_ho.write(_, output_reshape)
            ta_true_ho = ta_true_ho.write(_, target_reshape)
            ta_celltype_ho = ta_celltype_ho.write(_, cell_type_reshape)
            ta_genemap_ho = ta_genemap_ho.write(_, gene_map_reshape)
            
        metric_dict["hg_corr_stats_ho"].update_state(ta_true_ho.concat(),
                                                  ta_pred_ho.concat(),
                                                  ta_celltype_ho.concat(),
                                                  ta_genemap_ho.concat())
            
        ta_pred_ho.close()
        ta_true_ho.close()
        ta_celltype_ho.close()
        ta_genemap_ho.close()
            
        return dist_train_step, dist_val_step, dist_val_step_ho, metric_dict

        
    def build_step(iterator):
        @tf.function(jit_compile=True)
        def build_step(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.bfloat16)
            atac=tf.cast(inputs['atac'],dtype=tf.bfloat16)
            tss_tokens=tf.cast(inputs['tss_tokens'],dtype=tf.bfloat16)
            peaks_sequences = tf.cast(inputs['peaks_sequences'],dtype=tf.bfloat16)
            exons=tf.cast(inputs['exons'],dtype=tf.bfloat16)
            target=tf.cast(inputs['target'],dtype=tf.bfloat16)
            
            input_tuple = sequence,tss_tokens,exons, peaks_sequences, atac

            output = model(input_tuple,
                            training=False)

        for _ in tf.range(1): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(build_step,
                         args=(next(iterator),))
            

    return dist_train_step,dist_val_step,dist_val_step_ho,\
            build_step, metric_dict

@tf.function
def random_encode(input_tuple):
    sequence, randint = input_tuple
    if randint == 0:
        return one_hot(sequence)
    else:
        return rev_comp_one_hot(sequence)
    

def deserialize(serialized_example,
                input_length, 
                output_length,
                peaks_length_target,
                peaks_center_length,
                number_peaks,
                output_res,
                num_TFs,max_shift):
    """
    Deserialize bytes stored in TFRecordFile.
    """
    feature_map = {
        'atac': tf.io.FixedLenFeature([], tf.string),
        'exons': tf.io.FixedLenFeature([],tf.string),
        'peaks_sequences': tf.io.FixedLenFeature([],tf.string),
        'sequence': tf.io.FixedLenFeature([],tf.string),
        'TPM': tf.io.FixedLenFeature([],tf.string),
        'TF_expression': tf.io.FixedLenFeature([], tf.string),
        'tss_tokens': tf.io.FixedLenFeature([], tf.string),
        'coef_var': tf.io.FixedLenFeature([], tf.string)
    }
    
    data = tf.io.parse_example(serialized_example, feature_map)
    
    ### stochastic sequence shift and gaussian noise
    rev_comp = tf.math.round(tf.random.uniform([], 0, 1))
    
    shift = tf.random.uniform(shape=(), minval=0, maxval=max_shift, dtype=tf.int32)
    for k in range(max_shift):
        if k == shift:
            interval_end = input_length + k
            seq_shift = k
        else:
            seq_shift=0
    
    input_seq_length = input_length + max_shift

    tss_tokens = tf.ensure_shape(tf.io.parse_tensor(data['tss_tokens'],
                                              out_type=tf.int32),
                            [input_seq_length,])
    
    tss_tokens = tf.cast(tf.slice(tss_tokens, [seq_shift], [input_length]),dtype=tf.float32)
    tss_tokens = tf.reshape(tss_tokens, [output_length,output_res])
    tss_tokens = tf.reduce_max(tss_tokens,axis=1,keepdims=True)


    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float32),
                           [input_seq_length,])
    atac = tf.slice(atac, [seq_shift],[input_length])
    atac = tf.reshape(atac, [output_length,output_res])
    atac = tf.reduce_sum(atac,axis=1,keepdims=True)
    atac = tf.math.log(1.0 + atac)
    
    exons = tf.ensure_shape(tf.io.parse_tensor(data['exons'],
                                              out_type=tf.int32),
                            [input_seq_length,])
    exons = tf.cast(tf.slice(exons, [seq_shift],[input_length]),dtype=tf.float32)
    exons = tf.reshape(exons, [output_length,output_res])
    exons = tf.reduce_max(exons,axis=1,keepdims=True)
    
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 seq_shift,input_length))
    
    ### process peaks
    # first we want to randomly select the input peaks, let's say top 2000 out of 5000
    
    split_test=tf.strings.split(
        data['peaks_sequences'], sep='|', maxsplit=-1, name=None
    )
    split_test = split_test[:-1]
    
    padding_amount = ((peaks_length_target // peaks_center_length) - tf.shape(split_test)[0])

    paddings = [[0,padding_amount]]

    split_test = tf.pad(split_test,
                        paddings, "CONSTANT", constant_values=tf.constant("NNNNNNNNNNNNNNNN"))
    
    idxs = tf.range(tf.shape(split_test)[0])
    ridxs = tf.random.shuffle(idxs)[:number_peaks]
    random_sample = tf.gather(split_test, ridxs)
    
    peaks_sequences = tf.strings.reduce_join(random_sample)
    peaks_sequences = one_hot(peaks_sequences)
    

    #rev_comp = tf.math.round(tf.random.uniform([], 0, 1))
    if rev_comp == 1:
        atac = tf.reverse(atac,[0])
        tss_tokens = tf.reverse(tss_tokens,[0])
        sequence = rev_comp_one_hot(tf.strings.substr(data['sequence'],
                                                      seq_shift,input_length))
        exons = tf.reverse(exons, [0])
        peaks_sequences = rev_comp_one_hot(tf.strings.reduce_join(random_sample))
    


    TPM = tf.io.parse_tensor(data['TPM'],out_type=tf.float32)
    
    target = log2(1.0 + tf.math.maximum(0.0,TPM))
    target = tf.transpose(tf.reshape(target,[-1]))
    
    coef_var = tf.io.parse_tensor(data['coef_var'],out_type=tf.float32)

    return {
        'sequence': tf.ensure_shape(sequence,[input_length,4]),
        'atac': tf.ensure_shape(atac, [output_length,1]),
        'target': target,
        'coef_var': coef_var,
        'peaks_sequences': tf.ensure_shape(peaks_sequences,[number_peaks*peaks_center_length,4]),
        'tss_tokens': tf.ensure_shape(tss_tokens,[output_length,1]),
        'exons': tf.ensure_shape(exons,[output_length,1])
    }


def deserialize_val(serialized_example,input_length, 
                    output_length,
                    peaks_center_length,
                    number_peaks,
                    output_res,
                num_TFs,max_shift):
    """
    Deserialize bytes stored in TFRecordFile.
    """
    feature_map = {
        'atac': tf.io.FixedLenFeature([], tf.string),
        'exons': tf.io.FixedLenFeature([],tf.string),
        'sequence': tf.io.FixedLenFeature([],tf.string),
        'TPM': tf.io.FixedLenFeature([],tf.string),
        'cell_type': tf.io.FixedLenFeature([],tf.string),
        'coef_var': tf.io.FixedLenFeature([],tf.string),
        'gene_encoded': tf.io.FixedLenFeature([],tf.string),
        'TF_expression': tf.io.FixedLenFeature([], tf.string),
        'peaks_sequences': tf.io.FixedLenFeature([], tf.string),
        'tss_tokens': tf.io.FixedLenFeature([], tf.string)
    }
    
    data = tf.io.parse_example(serialized_example, feature_map)

    ### stochastic sequence shift and gaussian noise
    shift = max_shift // 2
    input_seq_length = input_length + max_shift
    interval_end = input_length + shift

    tss_tokens = tf.ensure_shape(tf.io.parse_tensor(data['tss_tokens'],
                                              out_type=tf.int32),
                            [input_seq_length,])
    tss_tokens = tf.cast(tf.slice(tss_tokens, [shift],[input_length]),dtype=tf.float32)
    tss_tokens = tf.reshape(tss_tokens, [output_length,output_res])
    tss_tokens = tf.reduce_max(tss_tokens,axis=1,keepdims=True)
    
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float32),
                           [input_seq_length,])
    atac = tf.slice(atac, [shift],[input_length])
    atac = tf.reshape(atac, [output_length,output_res])
    atac = tf.reduce_sum(atac,axis=1,keepdims=True)
    atac = tf.math.log(1.0 + atac)

    exons = tf.ensure_shape(tf.io.parse_tensor(data['exons'],
                                              out_type=tf.int32),
                            [input_seq_length,])
    exons = tf.cast(tf.slice(exons, [shift],[input_length]),dtype=tf.float32)
    exons = tf.reshape(exons, [output_length,output_res])
    exons = tf.reduce_max(exons,axis=1,keepdims=True)

    cell_type = tf.io.parse_tensor(data['cell_type'],out_type=tf.int32)
    gene_encoded = tf.io.parse_tensor(data['gene_encoded'],out_type=tf.int32)
    
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 shift,input_length))

    ### process peaks
    # first we want to randomly select the input peaks, let's say top 2000 out of 5000
    split_test=tf.strings.split(
        data['peaks_sequences'], sep='|', maxsplit=-1, name=None
    )
    split_test = split_test[:-1]
    
    idxs = tf.range(tf.shape(split_test)[0])
    ridxs = tf.random.shuffle(idxs)[:number_peaks]
    random_sample = tf.gather(split_test, ridxs)
    
    randints=tf.math.round(tf.random.uniform(shape=[random_sample.shape[0],],minval=0,maxval=1))
    
    peaks_sequences=tf.map_fn(random_encode,(random_sample,randints),fn_output_signature=tf.float32)
    

    TPM = tf.io.parse_tensor(data['TPM'],out_type=tf.float32)
    if tf.math.is_nan(TPM):
        TPM = 0.0
    if tf.math.less(TPM, 0.0):
        TPM = 0.0
    
    target = log2(1.0 + tf.math.maximum(0.0,TPM))
    
    target = tf.transpose(tf.reshape(target,[-1]))
    
    coef_var = tf.io.parse_tensor(data['coef_var'],out_type=tf.float32)
    

    return {
        'sequence': tf.ensure_shape(sequence,[input_length,4]),
        'atac': tf.ensure_shape(atac, [output_length,1]),
        'target': tf.transpose(tf.reshape(target,[-1])),
        'cell_type': tf.transpose(tf.reshape(cell_type,[-1])),
        'gene_encoded': tf.transpose(tf.reshape(gene_encoded,[-1])),
        'peaks_sequences': tf.ensure_shape(peaks_sequences,[number_peaks, peaks_center_length, 4]),
        'tss_tokens': tf.ensure_shape(tss_tokens,[output_length,1]),
        'exons': tf.ensure_shape(exons,[output_length,1]),
        'coef_var': coef_var
    }
                    
def return_dataset(gcs_path,
                   split,
                   organism,
                   batch,
                   input_length,
                   output_length,
                   peaks_length_target,
                   peaks_center_length,
                   number_peaks,
                   output_res,
                   max_shift,
                   options,
                   num_parallel,
                   num_epoch,
                   num_TFs):
    """
    return a tf dataset object for given gcs path
    """
    wc = str(organism) + "*.tfr"
    
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                split,
                                                wc)))
    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)
    
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize(record,
                                                     input_length,
                                                     output_length,
                                                     peaks_length_target,
                                                     peaks_center_length,
                                                     number_peaks,
                                                     output_res,
                                                     num_TFs,
                                                     max_shift),
                          deterministic=False,
                          num_parallel_calls=num_parallel)

    return dataset.repeat(num_epoch).batch(batch,drop_remainder=True).prefetch(1)


def return_dataset_val(gcs_path,
                       split,
                       organism,
                       batch,
                       input_length,
                       output_length,
                       peaks_length_target,
                       peaks_center_length,
                       number_peaks,
                       output_res,
                       max_shift,
                       options,
                       num_parallel,
                       num_epoch,
                       num_TFs):
    """
    return a tf dataset object for given gcs path
    """

    wc = str(organism) + "*.tfr"
    
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                split,
                                                wc)))
    

    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)

    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize_val(record,
                                                         input_length,
                                                         output_length,
                                                         peaks_length_target,
                                                         peaks_center_length,
                                                         number_peaks,
                                                         output_res,
                                                         num_TFs,
                                                         max_shift),
                          deterministic=False,
                          num_parallel_calls=num_parallel)

    return dataset.repeat(num_epoch).batch(batch, drop_remainder=True).prefetch(1)


 
def one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    vocabulary = tf.constant(['A', 'C', 'G', 'T'])
    mapping = tf.constant([0, 1, 2, 3])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=5) # makes N correspond to all 0s

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
    table = tf.lookup.StaticHashTable(init, default_value=1)

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out




def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator