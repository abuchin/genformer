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



def get_initializers_enformer_conv(checkpoint_path,
                                   from_enformer_bool,
                                   num_convs):

    inside_checkpoint=tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_path))
    reader = tf.train.load_checkpoint(checkpoint_path)


    if from_enformer_bool:
        initializers_dict = {'stem_conv_k': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/0/w/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_conv_b': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/0/b/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_res_conv_k': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/2/w/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_res_conv_b': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/2/b/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_res_conv_BN_g': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/scale/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_res_conv_BN_b': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/offset/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_res_conv_BN_m': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/moving_mean/average/.ATTRIBUTES/VARIABLE_VALUE')[0,0:]),
                             'stem_res_conv_BN_v': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/1/_module/_layers/0/moving_variance/average/.ATTRIBUTES/VARIABLE_VALUE')[0,0:]),
                             'stem_pool': inits.Constant(reader.get_tensor('module/_trunk/_layers/0/_layers/2/_logit_linear/w/.ATTRIBUTES/VARIABLE_VALUE'))}

        for i in range(num_convs):
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

            out_dict = {'conv1_k_' + str(i): inits.Constant(reader.get_tensor(conv1_k)),
                        'conv1_b_' + str(i): inits.Constant(reader.get_tensor(conv1_b)),
                        'BN1_g_' + str(i): inits.Constant(reader.get_tensor(BN1_g)),
                        'BN1_b_' + str(i): inits.Constant(reader.get_tensor(BN1_b)),
                        'BN1_m_' + str(i): inits.Constant(reader.get_tensor(BN1_m)),
                        'BN1_v_' + str(i): inits.Constant(reader.get_tensor(BN1_v)),
                        'conv2_k_' + str(i): inits.Constant(reader.get_tensor(conv2_k)),
                        'conv2_b_' + str(i): inits.Constant(reader.get_tensor(conv2_b)),
                        'BN2_g_' + str(i): inits.Constant(reader.get_tensor(BN2_g)),
                        'BN2_b_' + str(i): inits.Constant(reader.get_tensor(BN2_b)),
                        'BN2_m_' + str(i): inits.Constant(reader.get_tensor(BN2_m)),
                        'BN2_v_' + str(i): inits.Constant(reader.get_tensor(BN2_v)),
                        'pool_' + str(i): inits.Constant(reader.get_tensor(pool))}
            initializers_dict.update(out_dict)
    else:

        initializers_dict = {'stem_conv_k': inits.Constant(reader.get_tensor('stem_conv/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_conv_b': inits.Constant(reader.get_tensor('stem_conv/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_res_conv_k': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_res_conv_b': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_res_conv_BN_g': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_res_conv_BN_b': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_res_conv_BN_m': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_res_conv_BN_v': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE')),
                             'stem_pool': inits.Constant(reader.get_tensor('stem_pool/_logit_linear/kernel/.ATTRIBUTES/VARIABLE_VALUE'))}

        for i in range(num_convs):
            var_name_stem = 'conv_tower/layer_with_weights-' + str(i) + '/layer_with_weights-'



            conv1_k = var_name_stem + '0/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
            conv1_b = var_name_stem + '0/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
            BN1_g = var_name_stem + '0/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE'
            BN1_b = var_name_stem + '0/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE'
            BN1_m = var_name_stem + '0/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
            BN1_v = var_name_stem + '0/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'

            conv2_k = var_name_stem + '1/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
            conv2_b = var_name_stem + '1/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
            BN2_g = var_name_stem + '1/_layer/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE'
            BN2_b = var_name_stem + '1/_layer/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE'
            BN2_m = var_name_stem + '1/_layer/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
            BN2_v = var_name_stem + '1/_layer/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'
            pool = var_name_stem + '2/_logit_linear/kernel/.ATTRIBUTES/VARIABLE_VALUE'


            out_dict = {'conv1_k_' + str(i): inits.Constant(reader.get_tensor(conv1_k)),
                        'conv1_b_' + str(i): inits.Constant(reader.get_tensor(conv1_b)),
                        'BN1_g_' + str(i): inits.Constant(reader.get_tensor(BN1_g)),
                        'BN1_b_' + str(i): inits.Constant(reader.get_tensor(BN1_b)),
                        'BN1_m_' + str(i): inits.Constant(reader.get_tensor(BN1_m)),
                        'BN1_v_' + str(i): inits.Constant(reader.get_tensor(BN1_v)),
                        'conv2_k_' + str(i): inits.Constant(reader.get_tensor(conv2_k)),
                        'conv2_b_' + str(i): inits.Constant(reader.get_tensor(conv2_b)),
                        'BN2_g_' + str(i): inits.Constant(reader.get_tensor(BN2_g)),
                        'BN2_b_' + str(i): inits.Constant(reader.get_tensor(BN2_b)),
                        'BN2_m_' + str(i): inits.Constant(reader.get_tensor(BN2_m)),
                        'BN2_v_' + str(i): inits.Constant(reader.get_tensor(BN2_v)),
                        'pool_' + str(i): inits.Constant(reader.get_tensor(pool))}
            initializers_dict.update(out_dict)
    return initializers_dict

def get_initializers_enformer_performer(checkpoint_path,
                                        num_transformer_layers):

    inside_checkpoint=tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_path))
    reader = tf.train.load_checkpoint(checkpoint_path)

    initializers_dict = {'stem_conv_k': inits.Constant(reader.get_tensor('stem_conv/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_conv_b': inits.Constant(reader.get_tensor('stem_conv/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_k': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_b': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_g': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_b': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_m': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_v': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}


    out_dict = {'stem_conv_atac_k': inits.Constant(reader.get_tensor('stem_conv_atac/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_conv_atac_b': inits.Constant(reader.get_tensor('stem_conv_atac/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_k': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_b': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_g': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_b': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_m': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_v': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)


    out_dict = {'final_point_k': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_b': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_g': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_b': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_m': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_v': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)


    initializers_dict['stem_pool'] = inits.Constant(reader.get_tensor('stem_pool/_logit_linear/kernel/.ATTRIBUTES/VARIABLE_VALUE'))
    initializers_dict['stem_pool_atac'] = inits.Constant(reader.get_tensor('stem_pool_atac/_logit_linear/kernel/.ATTRIBUTES/VARIABLE_VALUE'))

    ## load in convolutional weights
    for i in range(6):
        var_name_stem = 'conv_tower/layer_with_weights-' + str(i) + '/layer_with_weights-' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        conv1_k = var_name_stem + '0/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        conv1_b = var_name_stem + '0/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_g = var_name_stem + '0/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_b = var_name_stem + '0/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_m = var_name_stem + '0/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_v = var_name_stem + '0/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'

        conv2_k = var_name_stem + '1/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        conv2_b = var_name_stem + '1/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_g = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_b = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_m = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_v = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'
        pool = var_name_stem + '2/_logit_linear/kernel/.ATTRIBUTES/VARIABLE_VALUE'

        out_dict = {'conv1_k_' + str(i): inits.Constant(reader.get_tensor(conv1_k)),
                    'conv1_b_' + str(i): inits.Constant(reader.get_tensor(conv1_b)),
                    'BN1_g_' + str(i): inits.Constant(reader.get_tensor(BN1_g)),
                    'BN1_b_' + str(i): inits.Constant(reader.get_tensor(BN1_b)),
                    'BN1_m_' + str(i): inits.Constant(reader.get_tensor(BN1_m)),
                    'BN1_v_' + str(i): inits.Constant(reader.get_tensor(BN1_v)),
                    'conv2_k_' + str(i): inits.Constant(reader.get_tensor(conv2_k)),
                    'conv2_b_' + str(i): inits.Constant(reader.get_tensor(conv2_b)),
                    'BN2_g_' + str(i): inits.Constant(reader.get_tensor(BN2_g)),
                    'BN2_b_' + str(i): inits.Constant(reader.get_tensor(BN2_b)),
                    'BN2_m_' + str(i): inits.Constant(reader.get_tensor(BN2_m)),
                    'BN2_v_' + str(i): inits.Constant(reader.get_tensor(BN2_v))}

        out_dict['pool_' + str(i)] = inits.Constant(reader.get_tensor(pool))
        initializers_dict.update(out_dict)


    ## load in convolutional weights ATAC
    for i in range(2):
        var_name_stem = 'conv_tower_atac/layer_with_weights-' + str(i) + '/layer_with_weights-' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        conv1_k = var_name_stem + '0/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        conv1_b = var_name_stem + '0/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_g = var_name_stem + '0/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_b = var_name_stem + '0/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_m = var_name_stem + '0/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_v = var_name_stem + '0/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'

        conv2_k = var_name_stem + '1/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        conv2_b = var_name_stem + '1/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_g = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_b = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_m = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_v = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'
        pool = var_name_stem + '2/_logit_linear/kernel/.ATTRIBUTES/VARIABLE_VALUE'

        out_dict = {'conv_at1_k_' + str(i): inits.Constant(reader.get_tensor(conv1_k)),
                    'conv_at1_b_' + str(i): inits.Constant(reader.get_tensor(conv1_b)),
                    'BN_at1_g_' + str(i): inits.Constant(reader.get_tensor(BN1_g)),
                    'BN_at1_b_' + str(i): inits.Constant(reader.get_tensor(BN1_b)),
                    'BN_at1_m_' + str(i): inits.Constant(reader.get_tensor(BN1_m)),
                    'BN_at1_v_' + str(i): inits.Constant(reader.get_tensor(BN1_v)),
                    'conv_at2_k_' + str(i): inits.Constant(reader.get_tensor(conv2_k)),
                    'conv_at2_b_' + str(i): inits.Constant(reader.get_tensor(conv2_b)),
                    'BN_at2_g_' + str(i): inits.Constant(reader.get_tensor(BN2_g)),
                    'BN_at2_b_' + str(i): inits.Constant(reader.get_tensor(BN2_b)),
                    'BN_at2_m_' + str(i): inits.Constant(reader.get_tensor(BN2_m)),
                    'BN_at2_v_' + str(i): inits.Constant(reader.get_tensor(BN2_v))}

        out_dict['pool_at_' + str(i)] = inits.Constant(reader.get_tensor(pool))
        initializers_dict.update(out_dict)


    initializers_dict['performer_encoder_LN_b'] = inits.Constant(reader.get_tensor("performer/layer_norm/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"))
    initializers_dict['performer_encoder_LN_g'] = inits.Constant(reader.get_tensor("performer/layer_norm/layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"))

    for i in range(num_transformer_layers):
        var_name_stem = 'performer/layers/' + str(i) + '/' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        #if not stable_variant:
        LN_b=var_name_stem + 'layer_norm/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        LN_g=var_name_stem + 'layer_norm/layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        out_dict = {'LN_b' + str(i): inits.Constant(reader.get_tensor(LN_b)),
                    'LN_g' + str(i): inits.Constant(reader.get_tensor(LN_g))}
        initializers_dict.update(out_dict)

        SA_k=var_name_stem + "self_attention/key_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_q=var_name_stem + "self_attention/query_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_v=var_name_stem + "self_attention/value_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_O=var_name_stem + "self_attention/output_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"

        FFN_narr_k=var_name_stem + "FFN/FFN_dense_narrow/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_narr_b=var_name_stem + "FFN/FFN_dense_narrow/bias/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_wide_k=var_name_stem + "FFN/FFN_dense_wide/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_wide_b=var_name_stem + "FFN/FFN_dense_wide/bias/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_LN_b=var_name_stem + "FFN/FFN_layer_norm/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_LN_g=var_name_stem + "FFN/FFN_layer_norm/layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"


        out_dict = {'SA_k' + str(i): inits.Constant(reader.get_tensor(SA_k)),
                    'SA_q' + str(i): inits.Constant(reader.get_tensor(SA_q)),
                    'SA_v' + str(i): inits.Constant(reader.get_tensor(SA_v)),
                    'SA_O' + str(i): inits.Constant(reader.get_tensor(SA_O)),
                    'FFN_narr_k' + str(i): inits.Constant(reader.get_tensor(FFN_narr_k)),
                    'FFN_narr_b' + str(i): inits.Constant(reader.get_tensor(FFN_narr_b)),
                    'FFN_wide_k' + str(i): inits.Constant(reader.get_tensor(FFN_wide_k)),
                    'FFN_wide_b' + str(i): inits.Constant(reader.get_tensor(FFN_wide_b)),
                    'FFN_LN_b' + str(i): inits.Constant(reader.get_tensor(FFN_LN_b)),
                    'FFN_LN_g' + str(i): inits.Constant(reader.get_tensor(FFN_LN_g))}

        initializers_dict.update(out_dict)

    #if pos_embedding_learned:
    #    out_dict = {'pos_embedding_learned': inits.Constant(reader.get_tensor('pos_embedding_learned/embeddings/.ATTRIBUTES/VARIABLE_VALUE'))}
    #    initializers_dict.update(out_dict)


    return initializers_dict

def get_initializers_enformer_performer_full(checkpoint_path,
                                            num_transformer_layers):

    inside_checkpoint=tf.train.list_variables(tf.train.latest_checkpoint(checkpoint_path))
    reader = tf.train.load_checkpoint(checkpoint_path)

    initializers_dict = {'stem_conv_k': inits.Constant(reader.get_tensor('stem_conv/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_conv_b': inits.Constant(reader.get_tensor('stem_conv/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_k': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_b': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_g': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_b': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_m': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_v': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}


    out_dict = {'stem_conv_atac_k': inits.Constant(reader.get_tensor('stem_conv_atac/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_conv_atac_b': inits.Constant(reader.get_tensor('stem_conv_atac/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_k': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_b': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_g': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_b': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_m': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_v': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)


    out_dict = {'final_point_k': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_b': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_g': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_b': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_m': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_v': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)


    initializers_dict['stem_pool'] = inits.Constant(reader.get_tensor('stem_pool/_logit_linear/kernel/.ATTRIBUTES/VARIABLE_VALUE'))
    initializers_dict['stem_pool_atac'] = inits.Constant(reader.get_tensor('stem_pool_atac/_logit_linear/kernel/.ATTRIBUTES/VARIABLE_VALUE'))

    ## load in convolutional weights
    for i in range(6):
        var_name_stem = 'conv_tower/layer_with_weights-' + str(i) + '/layer_with_weights-' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        conv1_k = var_name_stem + '0/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        conv1_b = var_name_stem + '0/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_g = var_name_stem + '0/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_b = var_name_stem + '0/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_m = var_name_stem + '0/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_v = var_name_stem + '0/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'

        conv2_k = var_name_stem + '1/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        conv2_b = var_name_stem + '1/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_g = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_b = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_m = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_v = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'
        pool = var_name_stem + '2/_logit_linear/kernel/.ATTRIBUTES/VARIABLE_VALUE'

        out_dict = {'conv1_k_' + str(i): inits.Constant(reader.get_tensor(conv1_k)),
                    'conv1_b_' + str(i): inits.Constant(reader.get_tensor(conv1_b)),
                    'BN1_g_' + str(i): inits.Constant(reader.get_tensor(BN1_g)),
                    'BN1_b_' + str(i): inits.Constant(reader.get_tensor(BN1_b)),
                    'BN1_m_' + str(i): inits.Constant(reader.get_tensor(BN1_m)),
                    'BN1_v_' + str(i): inits.Constant(reader.get_tensor(BN1_v)),
                    'conv2_k_' + str(i): inits.Constant(reader.get_tensor(conv2_k)),
                    'conv2_b_' + str(i): inits.Constant(reader.get_tensor(conv2_b)),
                    'BN2_g_' + str(i): inits.Constant(reader.get_tensor(BN2_g)),
                    'BN2_b_' + str(i): inits.Constant(reader.get_tensor(BN2_b)),
                    'BN2_m_' + str(i): inits.Constant(reader.get_tensor(BN2_m)),
                    'BN2_v_' + str(i): inits.Constant(reader.get_tensor(BN2_v))}

        out_dict['pool_' + str(i)] = inits.Constant(reader.get_tensor(pool))
        initializers_dict.update(out_dict)


    ## load in convolutional weights ATAC
    for i in range(2):
        var_name_stem = 'conv_tower_atac/layer_with_weights-' + str(i) + '/layer_with_weights-' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        conv1_k = var_name_stem + '0/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        conv1_b = var_name_stem + '0/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_g = var_name_stem + '0/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_b = var_name_stem + '0/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_m = var_name_stem + '0/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
        BN1_v = var_name_stem + '0/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'

        conv2_k = var_name_stem + '1/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE'
        conv2_b = var_name_stem + '1/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_g = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_b = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_m = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/moving_mean/.ATTRIBUTES/VARIABLE_VALUE'
        BN2_v = var_name_stem + '1/_layer/layer_with_weights-0/batch_norm/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'
        pool = var_name_stem + '2/_logit_linear/kernel/.ATTRIBUTES/VARIABLE_VALUE'

        out_dict = {'conv_at1_k_' + str(i): inits.Constant(reader.get_tensor(conv1_k)),
                    'conv_at1_b_' + str(i): inits.Constant(reader.get_tensor(conv1_b)),
                    'BN_at1_g_' + str(i): inits.Constant(reader.get_tensor(BN1_g)),
                    'BN_at1_b_' + str(i): inits.Constant(reader.get_tensor(BN1_b)),
                    'BN_at1_m_' + str(i): inits.Constant(reader.get_tensor(BN1_m)),
                    'BN_at1_v_' + str(i): inits.Constant(reader.get_tensor(BN1_v)),
                    'conv_at2_k_' + str(i): inits.Constant(reader.get_tensor(conv2_k)),
                    'conv_at2_b_' + str(i): inits.Constant(reader.get_tensor(conv2_b)),
                    'BN_at2_g_' + str(i): inits.Constant(reader.get_tensor(BN2_g)),
                    'BN_at2_b_' + str(i): inits.Constant(reader.get_tensor(BN2_b)),
                    'BN_at2_m_' + str(i): inits.Constant(reader.get_tensor(BN2_m)),
                    'BN_at2_v_' + str(i): inits.Constant(reader.get_tensor(BN2_v))}

        out_dict['pool_at_' + str(i)] = inits.Constant(reader.get_tensor(pool))
        initializers_dict.update(out_dict)


    initializers_dict['performer_encoder_LN_b'] = inits.Constant(reader.get_tensor("performer/layer_norm/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"))
    initializers_dict['performer_encoder_LN_g'] = inits.Constant(reader.get_tensor("performer/layer_norm/layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"))

    for i in range(num_transformer_layers):
        var_name_stem = 'performer/layers/' + str(i) + '/' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        #if not stable_variant:
        LN_b=var_name_stem + 'layer_norm/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        LN_g=var_name_stem + 'layer_norm/layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        out_dict = {'LN_b' + str(i): inits.Constant(reader.get_tensor(LN_b)),
                    'LN_g' + str(i): inits.Constant(reader.get_tensor(LN_g))}
        initializers_dict.update(out_dict)

        SA_k=var_name_stem + "self_attention/key_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_q=var_name_stem + "self_attention/query_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_v=var_name_stem + "self_attention/value_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_O=var_name_stem + "self_attention/output_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"

        FFN_narr_k=var_name_stem + "FFN/FFN_dense_narrow/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_narr_b=var_name_stem + "FFN/FFN_dense_narrow/bias/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_wide_k=var_name_stem + "FFN/FFN_dense_wide/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_wide_b=var_name_stem + "FFN/FFN_dense_wide/bias/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_LN_b=var_name_stem + "FFN/FFN_layer_norm/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_LN_g=var_name_stem + "FFN/FFN_layer_norm/layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"


        out_dict = {'SA_k' + str(i): inits.Constant(reader.get_tensor(SA_k)),
                    'SA_q' + str(i): inits.Constant(reader.get_tensor(SA_q)),
                    'SA_v' + str(i): inits.Constant(reader.get_tensor(SA_v)),
                    'SA_O' + str(i): inits.Constant(reader.get_tensor(SA_O)),
                    'FFN_narr_k' + str(i): inits.Constant(reader.get_tensor(FFN_narr_k)),
                    'FFN_narr_b' + str(i): inits.Constant(reader.get_tensor(FFN_narr_b)),
                    'FFN_wide_k' + str(i): inits.Constant(reader.get_tensor(FFN_wide_k)),
                    'FFN_wide_b' + str(i): inits.Constant(reader.get_tensor(FFN_wide_b)),
                    'FFN_LN_b' + str(i): inits.Constant(reader.get_tensor(FFN_LN_b)),
                    'FFN_LN_g' + str(i): inits.Constant(reader.get_tensor(FFN_LN_g))}

        initializers_dict.update(out_dict)

    #if pos_embedding_learned:
    #out_dict = {'pos_embedding_learned': inits.Constant(reader.get_tensor('pos_embedding_learned/embeddings/.ATTRIBUTES/VARIABLE_VALUE'))}
    #initializers_dict.update(out_dict)

    return initializers_dict
