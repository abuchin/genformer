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
import src.aformer_TF_gc_separated as aformer
#import src.aformer_TF as aformer
from src.layers.layers import *
import src.metrics as metrics
from src.optimizers import *
import src.schedulers as schedulers
import src.utils as utils

import baseline_utils as utils

from scipy import stats

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='node-15')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    options = tf.data.Options()
    #options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    options.deterministic=False
    #options.experimental_threading.max_intra_op_parallelism = 1
    mixed_precision.set_global_policy('mixed_bfloat16')
    tf.config.optimizer.set_jit(True)
    #options.num_devices = 64

    BATCH_SIZE_PER_REPLICA = 8
    NUM_REPLICAS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_REPLICAS
    
    
with strategy.scope():
    ### processing the training datasets
    '''
    train_dist = utils.return_dataset("gs://picard-testing-176520/16k_genecentered_blacklist0.50_atacnormalized",
                                       "train",
                                       "hg",
                                       GLOBAL_BATCH_SIZE,
                                       16384,
                                       300,
                                       "logTPM",
                                       options,
                                       4,
                                       1,
                                       1637,
                                       strategy)
    
    out_file = open('gc_acc_out_train.tsv','w')
    continue_write=True
    while continue_write:
        try:
            out = next(train_dist)
            out_parsed = strategy.experimental_local_results(out)
            try:
                for i in range(8):
                    for k in range(8):
                        val = out_parsed[i]
                        gc_pct = str(val['gc_pct'].numpy()[k])
                        atac_summed = str(val['atac_summed'].numpy()[k])
                        target = str(val['target'].numpy()[k])
                        gene_encoded = str(val['gene_encoded'].numpy()[k][0])
                        cell_type_encoded = str(val['cell_type'].numpy()[k][0])
                        line = '\t'.join([gc_pct,atac_summed,target, gene_encoded,cell_type_encoded]) + '\n'
                        out_file.write(line)
            except IndexError:
                continue
        except StopIteration:
            continue_write=False
            out_file.close()
    #subprocess.call("gsutil cp gc_acc_out_train.tsv gs://picard-testing-176520",shell=True)
    #subprocess.call("rm gc_acc_out_train.tsv", shell=True)
    '''
    val_dist = utils.return_dataset("gs://picard-testing-176520/16k_genecentered_blacklist0.50_atacnormalized",
                                       "val",
                                       "hg",
                                       GLOBAL_BATCH_SIZE,
                                       16384,
                                       300,
                                       "logTPM",
                                       options,
                                       4,
                                       1,
                                       1637,
                                       strategy)
    
    out_file = open('gc_acc_out_val.tsv','w')
    continue_write=True
    while continue_write:
        try:
            out = next(val_dist)
            out_parsed = strategy.experimental_local_results(out)
            try:
                for i in range(8):
                    for k in range(8):
                        val = out_parsed[i]
                        gc_pct = str(val['gc_pct'].numpy()[k])
                        atac_summed = str(val['atac_summed'].numpy()[k])
                        target = str(val['target'].numpy()[k])
                        gene_encoded = str(val['gene_encoded'].numpy()[k][0])
                        cell_type_encoded = str(val['cell_type'].numpy()[k][0])
                        line = '\t'.join([gc_pct,atac_summed,target, gene_encoded,cell_type_encoded]) + '\n'
                        out_file.write(line)
            except IndexError:
                continue
        except StopIteration:
            continue_write=False
            out_file.close()

            
    val_dist_holdout = utils.return_dataset("gs://picard-testing-176520/16k_genecentered_blacklist0.50_atacnormalized/val_holdout",
                                       "val",
                                       "hg",
                                       GLOBAL_BATCH_SIZE,
                                       16384,
                                       300,
                                       "logTPM",
                                       options,
                                       4,
                                       1,
                                       1637,
                                       strategy)
    
    out_file = open('gc_acc_out_valholdout.tsv','w')
    continue_write=True
    while continue_write:
        try:
            out = next(val_dist_holdout)
            out_parsed = strategy.experimental_local_results(out)
            try:
                for i in range(8):
                    for k in range(8):
                        val = out_parsed[i]
                        gc_pct = str(val['gc_pct'].numpy()[k])
                        atac_summed = str(val['atac_summed'].numpy()[k])
                        target = str(val['target'].numpy()[k])
                        gene_encoded = str(val['gene_encoded'].numpy()[k][0])
                        cell_type_encoded = str(val['cell_type'].numpy()[k][0])
                        line = '\t'.join([gc_pct,atac_summed,target, gene_encoded,cell_type_encoded]) + '\n'
                        out_file.write(line)
            except IndexError:
                continue
        except StopIteration:
            continue_write=False
            out_file.close()
