import time
import os
import subprocess
import sys
sys.path.insert(1, '/home/javed/genformer')
import re
import argparse
import collections
import gzip
import math
import shutil

import numpy as np
import time
from datetime import datetime
import random


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

import pandas as pd

import analysis.scripts.test_set_analysis_ATAC as utils

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=sys.argv[1])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)

g = tf.random.Generator.from_seed(5)
with strategy.scope():
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy=\
        tf.data.experimental.AutoShardPolicy.DATA
    options.deterministic=False
    options.experimental_threading.max_intra_op_parallelism=1
    mixed_precision.set_global_policy('mixed_bfloat16')
    #options.num_devices = 64

    BATCH_SIZE_PER_REPLICA = 4 # batch size 24, use LR ~ 2.5 e -04
    NUM_REPLICAS = strategy.num_replicas_in_sync
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_REPLICAS
    mask_indices = list(range(2041,2055))
    
    test_data_it = utils.return_distributed_iterators(sys.argv[2],GLOBAL_BATCH_SIZE, 524288,
                                                       5, 131072, 4096, 1600,
                                                       128, 8, strategy, options, 1792, 
                                                       mask_indices, "False", "True", "True", 
                                                       5, "False", g)
    
    model = aformer.aformer(kernel_transformation='relu_kernel_transformation',
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
    
    test_step,build_step = utils.return_test_build(model,strategy)
    build_step(test_data_it)
    model.load_weights("gs://" + sys.argv[3] + "/saved_model")
    print('built model') 
    pred_list = []
    true_list = []
    id_list = []
    cell_type_list = []

    pred_sum_list = []
    true_sum_list = []
    id_single_list = []
    cell_type_single_list = []
    for k in range(5000):
        true, pred,interval_id,cell_type,true_sum,pred_sum,interval_id_single,cell_type_single = strategy.run(test_step, args=(next(test_data_it),))
        
        for x in strategy.experimental_local_results(true):
            true_list.append(tf.reshape(x, [-1]))
        for x in strategy.experimental_local_results(pred):
            pred_list.append(tf.reshape(x, [-1]))
        for x in strategy.experimental_local_results(interval_id):
            id_list.append(tf.reshape(x, [-1]))
        for x in strategy.experimental_local_results(cell_type):
            cell_type_list.append(tf.reshape(x, [-1]))

        for x in strategy.experimental_local_results(true_sum):
            true_sum_list.append(tf.reshape(x, [-1]))
        for x in strategy.experimental_local_results(pred_sum):
            pred_sum_list.append(tf.reshape(x, [-1]))
        for x in strategy.experimental_local_results(interval_id_single):
            id_single_list.append(tf.reshape(x, [-1]))
        for x in strategy.experimental_local_results(cell_type_single):
            cell_type_single_list.append(tf.reshape(x, [-1]))
    

    df = pd.DataFrame(list(zip(tf.concat(pred_list,axis=0).numpy(),
                                tf.concat(true_list,axis=0).numpy(),
                                    tf.concat(id_list,axis=0).numpy(),
                                    tf.concat(cell_type_list,axis=0).numpy())), 
                        columns=['pred_list', 'true_list', 'id_list','cell_type_list'])
    print(df['pred_list'].corr(df['true_list']))
    df.to_csv('test_set_peak_predictions.tsv',sep='\t',index=False,header=True)


    df_sum = pd.DataFrame(list(zip(tf.concat(pred_sum_list,axis=0).numpy(),
                                tf.concat(true_sum_list,axis=0).numpy(),
                                    tf.concat(id_single_list,axis=0).numpy(),
                                        tf.concat(cell_type_single_list,axis=0).numpy())),
                        columns=['pred_list', 'true_list', 'id_list','cell_type_list'])
    print(df_sum['pred_list'].corr(df_sum['true_list']))
    df_sum.to_csv('test_set_peak_sum_predictions.tsv',sep='\t',index=False,header=True)
