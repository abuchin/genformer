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
import time
from datetime import datetime
import random

import seaborn as sns
import logging
from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
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
#import src.aformer_TF as aformer
from src.layers.layers import *
import src.metrics as metrics
from src.optimizers import *
import src.schedulers as schedulers
import src.utils as utils

import src.enformer_convs as enformer_convs

import training_utils_PEAKS as training_utils

from scipy import stats



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

strategy=tf_tpu_initialize("node-6",zone="us-central1-a")

with strategy.scope():

    inits = training_utils.get_initializers("/home/jupyter/dev/BE_CD69_paper_2022/enformer_fine_tuning/checkpoint/sonnet_weights")
    model = enformer_convs.enformer_convs(load_init=True,inits=inits)


list_files = (tf.io.gfile.glob(os.path.join("gs://picard-testing-176520/all_ATAC_peaks_genomewide",
                                            "*.tfr")))

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

def random_encode(input_tuple):
    sequence, randint = input_tuple
    if randint == 0:
        return one_hot(sequence)
    else:
        return rev_comp_one_hot(sequence)
    
#@tf.function
def deserialize(serialized_example,
                peaks_center_length,
                number_peaks):
    """
    Deserialize bytes stored in TFRecordFile.
    """
    feature_map = {
        'peaks_sequences': tf.io.FixedLenFeature([],tf.string)
    }
    
    data = tf.io.parse_example(serialized_example, feature_map)
    
    #random_generator = tf.random.get_global_generator()

    ### process peaks
    # first we want to randomly select the input peaks, let's say top 2000 out of 5000
    split_test=tf.strings.split(
        data['peaks_sequences'], sep='|', maxsplit=-1, name=None
    )
    split_test = split_test[:-1]


    idxs = tf.range(tf.shape(split_test)[0])
    
    ridxs = tf.random.shuffle(idxs)[:number_peaks]
    
    random_sample = tf.gather(split_test, ridxs)
    
    randints=tf.math.round(tf.random.uniform(shape=[number_peaks,],minval=0,maxval=1))
    
    peaks_sequences=tf.map_fn(random_encode,(random_sample,randints),fn_output_signature=tf.float32)
    peaks_sequences=tf.reshape(peaks_sequences,[number_peaks*peaks_center_length,4])
    
    return {
        'peaks_sequences': tf.ensure_shape(peaks_sequences,[number_peaks*peaks_center_length,4])
    }

with strategy.scope():
    for ind_file in list_files:
        files = tf.data.Dataset.list_files(ind_file)

        dataset = tf.data.TFRecordDataset(files,
                                          compression_type='ZLIB',
                                          num_parallel_reads=1)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy=\
            tf.data.experimental.AutoShardPolicy.OFF
        options.deterministic=False

        dataset = dataset.map(lambda record: deserialize(record,
                                                         128,
                                                         8000),
                              deterministic=False,
                              num_parallel_calls=1)
        dataset = dataset.with_options(options)



        dataset= dataset.repeat(1).batch(8)
        dataset_dist = strategy.experimental_distribute_dataset(dataset)

        out=next(iter(dataset_dist))

        @tf.function
        def run_step(inputs):
            return model(inputs['peaks_sequences'])

        model_out=strategy.run(run_step,
                               args=(out,))
        for k,val in enumerate(model_out.values):
            name=ind_file.split('/')[-1].split('.')[0] + "-" + str(k)
            np.savetxt('enformer_base_line_GAP_peaks/all_datasets_8000peaks_max/' + name + '.out', val.numpy()[0,:], delimiter=',')
    
    