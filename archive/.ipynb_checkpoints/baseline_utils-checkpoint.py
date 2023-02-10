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

tf.keras.backend.set_floatx('float32')



def return_distributed_iterators_linear(heads_dict,
                                 gcs_path,
                                 gcs_path_val_ho,
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 output_type,
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
            if org == 'hg':
                num_tf = 1637
            else:
                num_tf = 1366
            tr_data = return_dataset(gcs_path,
                                     "train",org, 
                                     global_batch_size,
                                     input_length,
                                     max_shift,
                                     output_type,
                                     options,
                                     num_parallel_calls,
                                     num_epoch,
                                     num_tf)
            
            val_data = return_dataset_val(gcs_path,
                                         "val",org, 
                                         global_batch_size,
                                         input_length,
                                         max_shift,
                                         output_type,
                                         options,
                                         num_parallel_calls,
                                         num_epoch,
                                         num_tf)
            
            

    
    
    
def return_dataset(gcs_path,
                   split,
                   organism,
                   batch,
                   input_length,
                   max_shift,
                   output_type,
                   options,
                   num_parallel,
                   num_epoch,
                   num_TFs,
                   strategy):
    """
    return a tf dataset object for given gcs path
    """
    wc = str(organism).upper() + "*.tfr"
    
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                split,
                                                wc)))
    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)
    
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      #buffer_size=1048576,
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)

    dataset = dataset.map(lambda record: deserialize(record,
                                                     input_length,
                                                     num_TFs,
                                                     max_shift),
                          deterministic=False,
                          num_parallel_calls=num_parallel)

    dataset= dataset.batch(batch).prefetch(1)
    
    dist = strategy.experimental_distribute_dataset(dataset)

    data_it = iter(dist)
    
    return data_it


def deserialize(serialized_example,input_length, 
                num_TFs,max_shift):
    """
    Deserialize bytes stored in TFRecordFile.
    """
    feature_map = {
        'atac': tf.io.FixedLenFeature([], tf.string),
        'exons': tf.io.FixedLenFeature([],tf.string),
        'sequence': tf.io.FixedLenFeature([],tf.string),
        'TPM': tf.io.FixedLenFeature([],tf.string),
        'TPM_uqn': tf.io.FixedLenFeature([],tf.string),
        'cell_type': tf.io.FixedLenFeature([],tf.string),
        'gene_mean': tf.io.FixedLenFeature([],tf.string),
        'gene_encoded': tf.io.FixedLenFeature([],tf.string),
        'gene_std': tf.io.FixedLenFeature([],tf.string),
        'TF_acc': tf.io.FixedLenFeature([], tf.string),
        'tss_tokens': tf.io.FixedLenFeature([], tf.string),
        'interval': tf.io.FixedLenFeature([],tf.string)
    }
    
    data = tf.io.parse_example(serialized_example, feature_map)

    ### stochastic sequence shift and gaussian noise
    shift = 150
    input_seq_length = input_length + max_shift
    interval_end = input_length + shift
    
    ### rev_comp
    rev_comp = random.randrange(0,2)
    
    tss_tokens = tf.ensure_shape(tf.io.parse_tensor(data['tss_tokens'],
                                              out_type=tf.int32),
                            [input_seq_length,])
    
    tss_tokens = tf.cast(tf.slice(tss_tokens, [shift],[input_length]),dtype=tf.float32)
    
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float32),
                           [input_seq_length,])
    atac = tf.slice(atac, [shift],[input_length])
    atac = tf.nn.dropout(atac, rate=0.01)
    atac = atac + tf.math.abs(tf.random.normal(tss_tokens.shape, 
                                   mean=0.0, 
                                   stddev=1.0e-04, dtype=tf.float32))
    atac=tf.expand_dims(atac,1)
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 shift,input_length))
    
    TPM = tf.io.parse_tensor(data['TPM'],out_type=tf.float32)
    
    TSS_GC = tss_tokens * tf.transpose(tf.ensure_shape(sequence,
                                          [input_length,4]))
    
    gc_count = tf.reduce_sum(TSS_GC[1:3,:])
    
    gc_pct = gc_count  / tf.reduce_sum(tss_tokens)
    atac_summed = tf.reduce_sum(tss_tokens * atac[:,0]) / tf.reduce_sum(tss_tokens)

    target = log2(1.0 + tf.math.maximum(0.0,TPM))

    cell_type = tf.io.parse_tensor(data['cell_type'],out_type=tf.int32)
    #print(data['cell_type'])
    gene_encoded = tf.io.parse_tensor(data['gene_encoded'],out_type=tf.int32)
    
    return {
        'gc_pct': gc_pct,
        'target': target,
        'atac_summed': atac_summed,
        'num_tss_tokens':tf.reduce_sum(tss_tokens),
        'cell_type': tf.transpose(tf.reshape(cell_type,[-1])),
        'gene_encoded': tf.transpose(tf.reshape(gene_encoded,[-1])),
    }

def log2(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

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
