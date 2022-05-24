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
from wandb.keras import WandbCallback
import multiprocessing

import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision
import src.metrics as metrics ## switch to src 
import src.schedulers
from src.losses import regular_mse
import src.optimizers
import src.schedulers
import src.utils



def tf_tpu_initialize(tpu_name):
    """Initialize TPU and return global batch size for loss calculation
    Args:
        tpu_name
    Returns:
        distributed strategy
    """
    
    try: 
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_name)
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)

    except ValueError: # no TPU found, detect GPUs
        strategy = tf.distribute.get_strategy()

    return strategy


"""
having trouble w/ providing organism/step inputs to train/val steps w/o
triggering retracing/metadata resource exhausted errors, so defining 
them separately for hg, mm 
to do: simplify to two functions w/ organism + mini_batch_step_inputs
consolidate into single simpler function
"""

def return_train_val_functions_hg(model,
                                   optimizer,
                                   strategy,
                                   metric_dict,
                                   train_steps, 
                                   val_steps, 
                                   global_batch_size,
                                   gradient_clip):
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
    #print('parsing_metric_dict')
    ## define the dictionary of metrics
    metric_dict["hg_tr"] = tf.keras.metrics.Mean("hg_tr_loss",
                                                 dtype=tf.float32)
    metric_dict["hg_val"] = tf.keras.metrics.Mean("hg_val_loss",
                                                  dtype=tf.float32)
    metric_dict["hg_corr_stats"] = metrics.correlation_stats(name='hg_corr_stats')

    @tf.function
    def dist_train_step(iterator):
        def train_step_hg(inputs):
            target=inputs['target']
            model_inputs=inputs['inputs']
            with tf.GradientTape() as tape:
                outputs = tf.cast(model(model_inputs,training=True)[0]["hg"],
                                  dtype=tf.float32)
                loss = tf.reduce_sum(regular_mse(outputs,target), 
                                     axis=0) * (1. / global_batch_size)

            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip) #comment this back in if using adam or adamW
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            metric_dict["hg_tr"].update_state(loss)

        for _ in tf.range(train_steps): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(train_step_hg, args=(next(iterator),))

    @tf.function
    def dist_val_step(iterator):
        def val_step_hg(inputs):
            target=inputs['target']
            tss_tokens = inputs['tss_tokens']
            model_inputs=inputs['inputs']          
            outputs = tf.cast(model(model_inputs,training=False)[0]["hg"],
                              dtype=tf.float32)

            loss = tf.reduce_sum(regular_mse(outputs, target), 
                                 axis=0) * (1. / global_batch_size)

            metric_dict["hg_val"].update_state(loss)
            
            outputs_reshape = tf.reshape(outputs, [-1]) # reshape to 1D
            targets_reshape = tf.reshape(target, [-1])
            tss_reshape = tf.reshape(tss_tokens, [-1])
            
            keep_indices = tf.reshape(tf.where(tf.equal(tss_reshape, 1)), [-1]) # figure out where TSS are
            targets_sub = tf.gather(targets_reshape, indices=keep_indices)
            outputs_sub = tf.gather(outputs_reshape, indices=keep_indices)
            tss_sub = tf.gather(tss_reshape, indices=keep_indices)
        
            return outputs_sub, targets_sub, tss_sub
        
    
        ta_pred = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store preds
        ta_true = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store vals
        ta_tss = tf.TensorArray(tf.int32, size=0, dynamic_size=True) # tensor array to store TSS indices
        
        for _ in tf.range(val_steps): ## for loop within @tf.fuction for improved TPU performance
            outputs_rep, targets_rep,tss_tokens_rep = strategy.run(val_step_hg,
                                                                   args=(next(iterator),))
            
            outputs_reshape = tf.reshape(strategy.gather(outputs_rep, axis=0), [-1]) # reshape to 1D
            targets_reshape = tf.reshape(strategy.gather(targets_rep, axis=0), [-1])
            tss_reshape = tf.reshape(strategy.gather(tss_tokens_rep, axis=0), [-1])

            ta_pred = ta_pred.write(_, outputs_reshape)
            ta_true = ta_true.write(_, targets_reshape)
            ta_tss = ta_tss.write(_, tss_reshape)

        metric_dict["hg_corr_stats"].update_state(ta_pred.concat(), 
                                                  ta_true.concat(), 
                                                  ta_tss.concat()) # compute corr stats
        ta_pred.close()
        ta_true.close()
        ta_tss.close()
        
    return dist_train_step, dist_val_step, metric_dict

def return_train_val_functions_hg_mm(model,
                                   optimizer,
                                   strategy,
                                   metric_dict,
                                   train_steps, 
                                   val_steps, 
                                   global_batch_size,
                                   gradient_clip):
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
    metric_dict["hg_corr_stats"] = metrics.correlation_stats(name='hg_corr_stats')
    metric_dict["mm_tr"] = tf.keras.metrics.Mean("mm_tr_loss", 
                                                     dtype=tf.float32)
    metric_dict["mm_val"] = tf.keras.metrics.Mean("mm_val_loss", 
                                                      dtype=tf.float32)
    metric_dict["mm_corr_stats"] = metrics.correlation_stats(name='mm_corr_stats')

    @tf.function
    def dist_train_step(hg_iterator, mm_iterator):

        def train_step_hg(inputs):
            target=inputs['target']
            model_inputs=inputs['inputs']
            with tf.GradientTape() as tape:
                outputs = tf.cast(model(model_inputs,training=True)[0]["hg"],
                                  dtype=tf.float32)
                loss = tf.reduce_sum(regular_mse(outputs,target), 
                                     axis=0) * (1. / global_batch_size)
            gradients = tape.gradient(loss, model.trainable_variables)
            #gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip) #comment this back in if using adam or adamw
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            metric_dict["hg_tr"].update_state(loss)

        def train_step_mm(inputs):
            target=inputs['target']
            model_inputs=inputs['inputs']
            with tf.GradientTape() as tape:
                outputs = tf.cast(model(model_inputs,training=True)[0]["mm"],
                                  dtype=tf.float32)
                loss = tf.reduce_sum(regular_mse(outputs,target), 
                                     axis=0) * (1. / global_batch_size)

            gradients = tape.gradient(loss, model.trainable_variables)
            #gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip) #comment this back in if using adam or adamw
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            metric_dict["mm_tr"].update_state(loss)

        for _ in tf.range(train_steps): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(train_step_hg, args=(next(hg_iterator),))
            strategy.run(train_step_mm, args=(next(mm_iterator),))

    @tf.function
    def dist_val_step(hg_iterator,
                      mm_iterator):
        def val_step_hg(inputs):
            target=inputs['target']
            tss_tokens = inputs['tss_tokens']
            model_inputs=inputs['inputs']          
            outputs = tf.cast(model(model_inputs,training=False)[0]["hg"],
                              dtype=tf.float32)

            loss = tf.reduce_sum(regular_mse(outputs, target), 
                                 axis=0) * (1. / global_batch_size)

            metric_dict["hg_val"].update_state(loss)

            outputs_reshape = tf.reshape(outputs, [-1]) # reshape to 1D
            targets_reshape = tf.reshape(target, [-1])
            tss_reshape = tf.reshape(tss_tokens, [-1])
            
            keep_indices = tf.reshape(tf.where(tf.equal(tss_reshape, 1)), [-1]) # figure out where TSS are
            targets_sub = tf.gather(targets_reshape, indices=keep_indices)
            outputs_sub = tf.gather(outputs_reshape, indices=keep_indices)
            tss_sub = tf.gather(tss_reshape, indices=keep_indices)
        
            return outputs_sub, targets_sub, tss_sub

        def val_step_mm(inputs):
            target=inputs['target']
            tss_tokens = inputs['tss_tokens']
            model_inputs=inputs['inputs']          
            outputs = tf.cast(model(model_inputs,training=False)[0]["mm"],
                              dtype=tf.float32)

            loss = tf.reduce_sum(regular_mse(outputs, target), 
                                 axis=0) * (1. / global_batch_size)

            metric_dict["mm_val"].update_state(loss)

            outputs_reshape = tf.reshape(outputs, [-1]) # reshape to 1D
            targets_reshape = tf.reshape(target, [-1])
            tss_reshape = tf.reshape(tss_tokens, [-1])
            
            keep_indices = tf.reshape(tf.where(tf.equal(tss_reshape, 1)), [-1]) # figure out where TSS are
            targets_sub = tf.gather(targets_reshape, indices=keep_indices)
            outputs_sub = tf.gather(outputs_reshape, indices=keep_indices)
            tss_sub = tf.gather(tss_reshape, indices=keep_indices)
        
            return outputs_sub, targets_sub, tss_sub

        ta_pred_h = tf.TensorArray(tf.float32,size=0, dynamic_size=True, clear_after_read=True)
        ta_true_h = tf.TensorArray(tf.float32,size=0, dynamic_size=True, clear_after_read=True)
        ta_tss_h = tf.TensorArray(tf.int32,size=0, dynamic_size=True, clear_after_read=True)

        ta_pred_m = tf.TensorArray(tf.float32,size=0, dynamic_size=True, clear_after_read=True)
        ta_true_m = tf.TensorArray(tf.float32,size=0, dynamic_size=True, clear_after_read=True)
        ta_tss_m = tf.TensorArray(tf.int32,size=0, dynamic_size=True, clear_after_read=True)

        for _ in tf.range(val_steps): ## for loop within @tf.fuction for improved TPU performance
            ### all human tensors
            outputs_rep_h,targets_rep_h,tss_tokens_rep_h = strategy.run(val_step_hg,
                                                                        args=(next(hg_iterator),))
            outputs_reshape_h = tf.reshape(strategy.gather(outputs_rep_h, axis=0), [-1]) # reshape to 1D
            targets_reshape_h = tf.reshape(strategy.gather(targets_rep_h, axis=0), [-1])
            tss_reshape_h = tf.reshape(strategy.gather(tss_tokens_rep_h, axis=0), [-1])

            ta_pred_h = ta_pred_h.write(_, outputs_reshape_h)
            ta_true_h = ta_true_h.write(_, targets_reshape_h)
            ta_tss_h = ta_tss_h.write(_, tss_reshape_h)

            ### all mouse tensors
            outputs_rep_m,targets_rep_m,tss_tokens_rep_m = strategy.run(val_step_mm,
                                                                        args=(next(mm_iterator),))
            outputs_reshape_m = tf.reshape(strategy.gather(outputs_rep_m, axis=0), [-1]) # reshape to 1D
            targets_reshape_m = tf.reshape(strategy.gather(targets_rep_m, axis=0), [-1])
            tss_reshape_m = tf.reshape(strategy.gather(tss_tokens_rep_m, axis=0), [-1])

            ta_pred_m = ta_pred_m.write(_, outputs_reshape_m)
            ta_true_m = ta_true_m.write(_, targets_reshape_m)
            ta_tss_m = ta_tss_m.write(_, tss_reshape_m)
        metric_dict["hg_corr_stats"].update_state(ta_pred_h.concat(), 
                                                  ta_true_h.concat(), 
                                                  ta_tss_h.concat()) # compute corr stats
        metric_dict["mm_corr_stats"].update_state(ta_pred_m.concat(), 
                                                  ta_true_m.concat(), 
                                                  ta_tss_m.concat()) # compute corr stats
        ta_pred_h.close()
        ta_true_h.close()
        ta_tss_h.close()
        ta_pred_m.close()
        ta_true_m.close()
        ta_tss_m.close()
    return dist_train_step, dist_val_step, metric_dict


    
"""
helper functions for returning distributed iter for specific organism
"""
    
def deserialize(serialized_example, input_length, output_length):
    """
    Deserialize bytes stored in TFRecordFile.
    """
    feature_map = {
        'inputs': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([],tf.string),
        'tss_tokens': tf.io.FixedLenFeature([],tf.string)
    }

    data = tf.io.parse_example(serialized_example, feature_map)

    return {
        'inputs': tf.ensure_shape(tf.io.parse_tensor(data['inputs'],
                                                     out_type=tf.float32),
                                  [input_length,5]),
        'target': tf.ensure_shape(tf.io.parse_tensor(data['target'],
                                                     out_type=tf.float32),
                                  [output_length,]),
        'tss_tokens': tf.ensure_shape(tf.io.parse_tensor(data['tss_tokens'],
                                                     out_type=tf.int32),
                                  [output_length,])
    }

def deserialize_validation(serialized_example, input_length, output_length,
                           output_length_pre,crop_size,out_length):
    """
    Deserialize bytes stored in TFRecordFile.
    """
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([],tf.string),
        'target': tf.io.FixedLenFeature([],tf.string),
        'tss_tokens': tf.io.FixedLenFeature([],tf.string),
        'interval': tf.io.FixedLenFeature([],tf.string),
        'genes_list': tf.io.FixedLenFeature([],tf.string),
        'name': tf.io.FixedLenFeature([],tf.string)
    }

    data = tf.io.parse_example(serialized_example, feature_map)

    return {
        'inputs': tf.concat([tf.expand_dims(tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                                                              out_type=tf.float32), 
                                                           [input_length,]), 1),
                                    one_hot(data['sequence'])], axis=1),
        'target': tf.slice(tf.ensure_shape(tf.io.parse_tensor(data['target'],
                                                                      out_type=tf.float32), 
                                                   [output_length_pre,]),
                           [crop_size],
                           [out_length]),
        'tss_tokens': tf.slice(tf.ensure_shape(tf.io.parse_tensor(data['tss_tokens'],
                                                                  out_type=tf.int32),
                                               [output_length_pre,]),
                               [crop_size],
                               [out_length]),
        'genes_list': data['genes_list'],
        'interval': data['interval'],
        'name': data['name']
    }

                    
def return_dataset(gcs_path,
                   split,
                   organism,
                   batch,
                   input_length,
                   output_length,
                   options,
                   num_parallel,
                   num_epoch):
    """
    return a tf dataset object for given gcs path
    """
    if organism == 'mm':
        num_epoch = num_epoch + 30
    wc = str(organism) + "*.tfrecords"
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                split,
                                                wc)))
    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)
    
    #dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, 
    ##                                                             compression_type='ZLIB',
    #                                                             buffer_size=10000000,
    #                                                             num_parallel_reads=num_parallel),
    #                            num_parallel_calls=num_parallel,
    #                            deterministic=False,
    #                            cycle_length=num_parallel,
    #                            block_length=1)
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      buffer_size=10000000,
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize(record,
                                                     input_length,
                                                     output_length),
                          deterministic=False,
                          num_parallel_calls=num_parallel)


    return dataset.repeat(num_epoch).batch(batch,drop_remainder=True).prefetch(1)


def return_dataset_validation(gcs_path,
                   organism,
                   batch,
                   input_length,
                   output_length_pre,
                   output_length,
                    crop_size,
                   options,
                   num_parallel,
                   num_epoch):
    """
    return a tf dataset object for given gcs path
    """
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path)))
    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)
    
    #dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, 
    ##                                                             compression_type='ZLIB',
    #                                                             buffer_size=10000000,
    #                                                             num_parallel_reads=num_parallel),
    #                            num_parallel_calls=num_parallel,
    #                            deterministic=False,
    #                            cycle_length=num_parallel,
    #                            block_length=1)
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      buffer_size=10000000,
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize_validation(record,
                                                     input_length,
                                                     output_length,
                                                                output_length_pre,
                                                                crop_size,
                                                               output_length),
                          deterministic=False,
                          num_parallel_calls=num_parallel)


    return dataset.repeat(num_epoch).batch(batch,drop_remainder=True).prefetch(1)

def return_distributed_iterators(heads_dict,
                                 gcs_path,
                                 global_batch_size,
                                 input_length,
                                 output_length,
                                 num_parallel_calls,
                                 num_epoch,
                                 strategy,
                                 options):
    """ 
    returns train + val dictionaries of distributed iterators
    for given heads_dictionary
    """
    with strategy.scope():
        data_it_tr_list = []
        data_it_val_list = []

        for org,index in heads_dict.items():
            tr_data = return_dataset(gcs_path,
                                     "train",org, 
                                     global_batch_size,
                                     input_length,
                                     output_length,
                                     options,
                                     num_parallel_calls,
                                     num_epoch)
            val_data = return_dataset(gcs_path,
                                     "val",org, 
                                     global_batch_size,
                                     input_length,
                                     output_length,
                                     options,
                                     num_parallel_calls,
                                     num_epoch)

            train_dist = strategy.experimental_distribute_dataset(tr_data)
            #train_dist = train_dist.with_options(options)
            val_dist= strategy.experimental_distribute_dataset(val_data)
            #val_dist=val_dist.with_options(options)

            tr_data_it = iter(train_dist)
            val_data_it = iter(val_dist)
            data_it_tr_list.append(tr_data_it)
            data_it_val_list.append(val_data_it)
        data_dict_tr = dict(zip(heads_dict.keys(), data_it_tr_list))
        data_dict_val = dict(zip(heads_dict.keys(), data_it_val_list))

        return data_dict_tr, data_dict_val



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
    ### check if min_delta satisfied
    best_loss = min(logged_val_losses[:-1])
    best_pearsons=max(logged_pearsons[:-1])
    stop_criteria = False
    
    ## if min delta satisfied then log loss
    
    if (current_val_loss >= (best_loss - min_delta)) and (current_pearsons <= best_pearsons):
        patience_counter += 1
        if patience_counter >= patience:
            stop_criteria=True
    else:

        best_epoch = np.argmin(logged_val_losses)
        ## save current model
        if (current_epoch % save_freq) == 0:
            print('Saving model...')
            model_name = save_directory + "/" + \
                            saved_model_basename + "/iteration_" + \
                                str(current_epoch)
            model.save_weights(model_name)
            ### write to logging file in saved model dir to model parameters and current epoch info
            
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
    
    # data loading parameters
    parser.add_argument('--gcs_path',
                        dest='gcs_path',
                        help= 'google bucket containing preprocessed data')
    parser.add_argument('--output_heads',
                        dest='output_heads',
                        type=str,
                        help= 'list of organisms(hg,mm,rm)')
    parser.add_argument('--num_parallel', dest = 'num_parallel',
                        type=int, default=multiprocessing.cpu_count(),
                        help='thread count for tensorflow record loading')
    parser.add_argument('--input_length',
                        dest='input_length',
                        type=int,
                        help= 'input_length')
    parser.add_argument('--output_res',
                        dest='output_res',
                        type=int,
                        help= 'output_res')
    parser.add_argument('--output_length',
                        dest='output_length',
                        type=int,
                        help= 'output_length')
    
    ## training loop parameters
    parser.add_argument('--batch_size', dest = 'batch_size',
                        type=int, help='batch_size')
    parser.add_argument('--num_epochs', dest = 'num_epochs',
                        type=int, help='num_epochs')
    parser.add_argument('--warmup_frac', dest = 'warmup_frac',
                        type=float, help='warmup_frac')
    parser.add_argument('--train_steps', dest = 'train_steps',
                        type=int, help='train_steps')
    parser.add_argument('--val_steps', dest = 'val_steps',
                        type=int, help='val_steps')
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
## parameters to sweep over
    parser.add_argument('--lr_schedule',
                        dest = 'lr_schedule',
                        type=str)
    parser.add_argument('--lr_base',
                        dest='lr_base',
                        help='lr_base')
    parser.add_argument('--min_lr',
                        dest='min_lr',
                        help= 'min_lr')
    parser.add_argument('--epsilon',
                        dest='epsilon',
                        default=1.0e-08,
                        type=float,
                        help= 'epsilon')
    parser.add_argument('--rectify',
                        dest='rectify',
                        default=True,
                        help= 'rectify')
    parser.add_argument('--optimizer',
                        dest='optimizer',
                        help= 'optimizer, one of adafactor, adam, or adamW')
    parser.add_argument('--gradient_clip',
                        dest='gradient_clip',
                        type=str,
                        help= 'gradient_clip')
    parser.add_argument('--precision',
                        dest='precision',
                        type=str,
                        help= 'bfloat16 or float32') ### need to implement this actually
    parser.add_argument('--weight_decay_frac',
                        dest='weight_decay_frac',
                        type=str,
                        help= 'weight_decay_frac')
    parser.add_argument('--sync_period',
                        type=int,
                        dest='sync_period',
                        help= 'sync_period')
    parser.add_argument('--slow_step_frac',
                        type=float,
                        dest='slow_step_frac',
                        help= 'slow_step_frac')
    
    
    # network hyperparameters
    parser.add_argument('--conv_channel_list',
                        dest='conv_channel_list',
                        help= 'conv_channel_list')
    parser.add_argument('--dropout',
                        dest='dropout',
                        help= 'dropout')
    parser.add_argument('--num_transformer_layers',
                        dest='num_transformer_layers',
                        help= 'num_transformer_layers')
    parser.add_argument('--num_heads',
                        dest='num_heads',
                        help= 'num_heads')
    parser.add_argument('--momentum',
                        dest='momentum',
                        type=str,
                        help= 'batch norm momentum')
    parser.add_argument('--num_random_features',
                        dest='num_random_features',
                        type=str,
                        help= 'num_random_features')
    parser.add_argument('--kernel_transformation',
                        dest='kernel_transformation',
                        help= 'kernel_transformation')
    parser.add_argument('--hidden_size',
                        dest='hidden_size',
                        type=str,
                        help= 'hidden size for transformer' + \
                                'should be equal to last conv layer filters')
    parser.add_argument('--conv_filter_size',
                        dest='conv_filter_size',
                        help= 'conv_filter_size')
    parser.add_argument('--dim',
                        dest='dim',
                        type=int,
                        help= 'mask_pos_dim')
    parser.add_argument('--max_seq_length',
                        dest='max_seq_length',
                        type=int,
                        help= 'max_seq_length')
    parser.add_argument('--rel_pos_bins',
                        dest='rel_pos_bins',
                        type=int,
                        help= 'rel_pos_bins')



    args = parser.parse_args()
    return parser
    
    
    
def one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    vocabulary = tf.constant(['A', 'T', 'C', 'G', 'N'])
    mapping = tf.constant([0, 1, 2, 3, 4])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=0)

    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out