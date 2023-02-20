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

import numpy as np
from sklearn import metrics as sklearn_metrics

from tensorflow.keras import initializers as inits
from scipy.stats import zscore

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


def get_initializers_enformer_conv(checkpoint_path):
    
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
    
    print(inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')))

    initializers_dict = {'stem_conv_k': inits.Constant(reader.get_tensor('stem_conv/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_conv_b': inits.Constant(reader.get_tensor('stem_conv/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_k': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_b': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_g': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_b': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_m': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_BN_v': inits.Constant(reader.get_tensor('stem_res_conv/_layer/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}
    

    out_dict = {'stem_conv_atac_k': inits.Constant(reader.get_tensor('stem_conv_atac/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_conv_atac_b': inits.Constant(reader.get_tensor('stem_conv_atac/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_k': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_b': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_g': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_b': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_m': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'stem_res_conv_atac_BN_v': inits.Constant(reader.get_tensor('stem_res_conv_atac/_layer/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)
    
    
    out_dict = {'final_point_k': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_b': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_g': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_b': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_m': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE')),
                         'final_point_BN_v': inits.Constant(reader.get_tensor('final_pointwise_conv/layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE'))}
    initializers_dict.update(out_dict)
    

    initializers_dict['stem_pool'] = inits.Constant(reader.get_tensor('stem_pool/_logit_linear/kernel/.ATTRIBUTES/VARIABLE_VALUE'))

    ## load in convolutional weights
    for i in range(6):
        var_name_stem = 'conv_tower/layer_with_weights-' + str(i) + '/layer_with_weights-' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

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
                    'BN2_v_' + str(i): inits.Constant(reader.get_tensor(BN2_v))}

        out_dict['pool_' + str(i)] = inits.Constant(reader.get_tensor(pool))
        initializers_dict.update(out_dict)
        
        
    ## load in convolutional weights
    
    
    initializers_dict['performer_encoder_LN_b'] = inits.Constant(reader.get_tensor("performer/layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"))
    initializers_dict['performer_encoder_LN_g'] = inits.Constant(reader.get_tensor("performer/layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"))
    
    for i in range(num_transformer_layers):
        var_name_stem = 'performer/layers/' + str(i) + '/' #0/moving_mean/_counter/.ATTRIBUTES/VARIABLE_VALUE'

        LN_b=var_name_stem + 'layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE'
        LN_g=var_name_stem + 'layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE'
        
        SA_k=var_name_stem + "self_attention/key_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_q=var_name_stem + "self_attention/query_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_v=var_name_stem + "self_attention/value_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        SA_O=var_name_stem + "self_attention/output_dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        
        FFN_narr_k=var_name_stem + "FFN/FFN_dense_narrow/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_narr_b=var_name_stem + "FFN/FFN_dense_narrow/bias/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_wide_k=var_name_stem + "FFN/FFN_dense_wide/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_wide_b=var_name_stem + "FFN/FFN_dense_wide/bias/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_LN_b=var_name_stem + "FFN/FFN_layer_norm/beta/.ATTRIBUTES/VARIABLE_VALUE"
        FFN_LN_g=var_name_stem + "FFN/FFN_layer_norm/gamma/.ATTRIBUTES/VARIABLE_VALUE"
    

        out_dict = {'LN_b' + str(i): inits.Constant(reader.get_tensor(LN_b)),
                    'LN_g' + str(i): inits.Constant(reader.get_tensor(LN_g)),
                    'SA_k' + str(i): inits.Constant(reader.get_tensor(SA_k)),
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
        
    return initializers_dict



def extract_batch_norm_stats(model):
    gamma_names=[]
    gamma_vars=[]
    gamma_layer_num=[]
    beta_names=[]
    beta_vars=[]
    beta_layer_num=[]
    moving_means_names=[]
    moving_means_vars=[]
    moving_means_layer_num=[]
    moving_vars_names=[]
    moving_vars_vars=[]
    moving_vars_layer_num=[]

    all_vars=[(k,x.name,x) for k,x in enumerate(model.stem_res_conv.variables)]

    for gamma_tuple in all_vars:
        if 'sync_batch_normalization' in gamma_tuple[1]:

            specific_var = gamma_tuple[1].split('/')[1].split(':')[0]
            layer_num=0
            var_name = specific_var + "_1"
            if (('gamma' in var_name) or 'moving_variance' in var_name):
                vals = list(np.log(1.0+gamma_tuple[-1].values[0].numpy()))
            else:
                vals = list(np.log(1.0+np.abs(gamma_tuple[-1].values[0].numpy())) * \
                                    np.sign(gamma_tuple[-1].values[0].numpy()))
            #print(vals)
            names = []
            layer_nums=[]
            for variable_val in vals:
                names.append(var_name)
                layer_nums.append(layer_num)
            if 'gamma' in var_name:
                gamma_names += names
                gamma_vars += vals
                gamma_layer_num += layer_nums
            if 'beta' in var_name:
                beta_names += names
                beta_vars += vals
                beta_layer_num += layer_nums
            if 'moving_mean' in var_name:
                moving_means_names += names
                moving_means_vars += vals
                moving_means_layer_num += layer_nums
            if 'moving_variance' in var_name:
                moving_vars_names += names
                moving_vars_vars += vals    
                moving_vars_layer_num += layer_nums


    all_vars=[(k,x.name,x) for k,x in enumerate(model.conv_tower.variables)]

    for gamma_tuple in all_vars:
        if 'sync_batch_normalization' in gamma_tuple[1]:
            specific_var = gamma_tuple[1].split('/')[1].split(':')[0]
            layer_num= int(gamma_tuple[1].split('/')[0].split('_')[-1])+1

            var_name = specific_var + "_" + str(layer_num)
            if (('gamma' in var_name) or 'moving_variance' in var_name):
                vals = list(np.log(1.0+gamma_tuple[-1].values[0].numpy()))
            else:
                vals = list(np.log(1.0+np.abs(gamma_tuple[-1].values[0].numpy())) * \
                                    np.sign(gamma_tuple[-1].values[0].numpy()))

            names = []
            layer_nums=[]
            for variable_val in vals:
                names.append(var_name)
                layer_nums.append(layer_num)
            if 'gamma' in var_name:
                gamma_names += names
                gamma_vars += vals
                gamma_layer_num += layer_nums
            if 'beta' in var_name:
                beta_names += names
                beta_vars += vals
                beta_layer_num += layer_nums
            if 'moving_mean' in var_name:
                moving_means_names += names
                moving_means_vars += vals
                moving_means_layer_num += layer_nums
            if 'moving_variance' in var_name:
                moving_vars_names += names
                moving_vars_vars += vals    
                moving_vars_layer_num += layer_nums
    

    gamma_df=pd.DataFrame(list(zip(gamma_names, gamma_vars,gamma_layer_num)), columns =['layer', 'values','layer_num'])
    gamma_df=gamma_df.sort_values(by="layer_num",ascending=False)
    fig_gamma,ax_gamma=plt.subplots(figsize=(6,6))
    sns.kdeplot(data=gamma_df, x="values", hue="layer")
    plt.xlabel("log(1.0+gamma)")
    plt.ylabel("count")
    plt.title("batch_norm_gamma")
    
    beta_df=pd.DataFrame(list(zip(beta_names, beta_vars,beta_layer_num)), columns =['layer', 'values','layer_num'])
    beta_df=beta_df.sort_values(by="layer_num",ascending=False)
    fig_beta,ax_beta=plt.subplots(figsize=(6,6))
    sns.kdeplot(data=beta_df, x="values", hue="layer")
    plt.xlabel("log(1.0+|beta|)*sign(beta)")
    plt.ylabel("count")
    plt.title("batch_norm_beta")

    moving_means_df=pd.DataFrame(list(zip(moving_means_names, moving_means_vars,moving_means_layer_num)), columns =['layer', 'values','layer_num'])
    moving_means_df=moving_means_df.sort_values(by="layer_num",ascending=False)
    fig_moving_means,ax_moving_means=plt.subplots(figsize=(6,6))
    sns.kdeplot(data=moving_means_df, x="values", hue="layer")
    plt.xlabel("log(1.0+|moving_mean|)*sign(moving_mean)")
    plt.ylabel("count")
    plt.title("batch_norm_moving_mean")
    
    moving_vars_df=pd.DataFrame(list(zip(moving_vars_names, moving_vars_vars,moving_vars_layer_num)), columns =['layer', 'values','layer_num'])
    moving_vars_df=moving_vars_df.sort_values(by="layer_num",ascending=False)
    fig_moving_vars,ax_moving_vars=plt.subplots(figsize=(6,6))
    sns.kdeplot(data=moving_vars_df, x="values", hue="layer")
    plt.xlabel("log(1.0+moving_variance)")
    plt.ylabel("count")
    plt.title("batch_norm_moving_var")
    
    return fig_gamma, fig_beta, fig_moving_means,fig_moving_vars


def return_train_val_functions(model,
                               train_steps,
                               val_steps,
                               val_steps_TSS,
                               optimizers_in,
                               strategy,
                               metric_dict,
                               global_batch_size,
                               gradient_clip,
                               cage_scale,
                               predict_masked_atac_bool):
    

    loss_fn = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)

    optimizer1,optimizer2=optimizers_in
    
    metric_dict["corr_stats"] = metrics.correlation_stats_gene_centered(name='corr_stats')
    metric_dict["train_loss"] = tf.keras.metrics.Mean("train_loss",
                                                 dtype=tf.float32)
    metric_dict["val_loss"] = tf.keras.metrics.Mean("val_loss",
                                                  dtype=tf.float32)
    if predict_masked_atac_bool: 
        metric_dict['ATAC_PearsonR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
        metric_dict['ATAC_R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})
        
        metric_dict['ATAC_PearsonR_baseline'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
        metric_dict['ATAC_R2_baseline'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})
        
        metric_dict['CAGE_PearsonR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
        metric_dict['CAGE_R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})
    else:
        metric_dict['CAGE_PearsonR'] = metrics.MetricDict({'PearsonR': metrics.PearsonR(reduce_axis=(0,1))})
        metric_dict['CAGE_R2'] = metrics.MetricDict({'R2': metrics.R2(reduce_axis=(0,1))})
        

    
    def dist_train_step(iterator):    
        @tf.function(jit_compile=True)
        def train_step(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.bfloat16)
            atac=tf.cast(inputs['atac'],dtype=tf.bfloat16)
            global_acc=tf.cast(inputs['global_acc'],dtype=tf.bfloat16)
            target=tf.cast(inputs['target'],dtype=tf.float32)

            input_tuple = sequence, atac, global_acc

            with tf.GradientTape() as tape:
                conv_vars = model.stem_conv.trainable_variables + \
                            model.stem_res_conv.trainable_variables + \
                            model.stem_pool.trainable_variables + \
                            model.conv_tower.trainable_variables
                
                performer_vars = model.performer.trainable_variables + \
                                    model.final_pointwise_conv.trainable_variables + \
                                    model.stem_conv_atac.trainable_variables + \
                                    model.stem_res_conv_atac.trainable_variables + \
                                    model.final_dense.trainable_variables +\
                                    model.global_acc_proc.trainable_variables
                
                vars_all = conv_vars + performer_vars
                
                output = model(input_tuple,
                               training=True)

                output = tf.cast(output,dtype=tf.float32)
                
                if predict_masked_atac_bool: 
                    atac_loss = loss_fn(target[:,:,0],output[:,:,0])
                    cage_loss = loss_fn(target[:,:,1],output[:,:,1]) * cage_scale
                    loss = tf.math.reduce_mean(atac_loss) + tf.math.reduce_mean(cage_loss)
                else:
                    cage_loss = loss_fn(target,output) * cage_scale
                    loss = tf.math.reduce_mean(cage_loss)

                loss = loss * (1. / global_batch_size)

            gradients = tape.gradient(loss, vars_all)
            gradients, _ = tf.clip_by_global_norm(gradients, 
                                                  gradient_clip)
            optimizer1.apply_gradients(zip(gradients[:len(conv_vars)], 
                                           conv_vars))
            optimizer2.apply_gradients(zip(gradients[len(conv_vars):], 
                                           performer_vars))
            metric_dict["train_loss"].update_state(loss)
        
        for _ in tf.range(train_steps):
            strategy.run(train_step,
                         args=(next(iterator),))

    def dist_val_step(iterator):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.bfloat16)
            target=tf.cast(inputs['target'],dtype=tf.float32)
            atac=tf.cast(inputs['atac'],dtype=tf.bfloat16)
            global_acc=tf.cast(inputs['global_acc'],dtype=tf.bfloat16)
            
            input_tuple = sequence,atac,global_acc

            output = model(input_tuple,
                           training=False)
            output = tf.cast(output,dtype=tf.float32)
            
            if predict_masked_atac_bool: 
                atac_loss = loss_fn(target[:,:,0],output[:,:,0])
                cage_loss = loss_fn(target[:,:,1],output[:,:,1]) * cage_scale
                loss = tf.math.reduce_mean(atac_loss) + tf.math.reduce_mean(cage_loss)
                metric_dict['CAGE_PearsonR'].update_state(target[:,:,1:], 
                                                          output[:,:,1:])
                metric_dict['CAGE_R2'].update_state(target[:,:,1:], 
                                                    output[:,:,1:])
                metric_dict['ATAC_PearsonR'].update_state(target[:,:,0:1], 
                                                          output[:,:,0:1])
                metric_dict['ATAC_R2'].update_state(target[:,:,0:1], 
                                                    output[:,:,0:1])

                metric_dict['ATAC_PearsonR_baseline'].update_state(target[:,:,0:1], 
                                                                   tf.slice(tf.cast(atac,dtype=tf.float32),
                                                                            [0,320,0],[-1,896,-1]))
                metric_dict['ATAC_R2_baseline'].update_state(target[:,:,0:1],
                                                             tf.slice(tf.cast(atac,dtype=tf.float32),
                                                                      [0,320,0],[-1,896,-1]))
                
                
            else:
                cage_loss = loss_fn(target,output) * cage_scale
                loss = tf.math.reduce_mean(cage_loss)
                metric_dict['CAGE_PearsonR'].update_state(target, 
                                                          output)
                metric_dict['CAGE_R2'].update_state(target, 
                                                    output)

            loss = tf.math.reduce_mean(loss) * (1. / global_batch_size)

            metric_dict["val_loss"].update_state(loss)

        for _ in tf.range(val_steps): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(val_step,
                         args=(next(iterator),))

    
    def dist_val_step_TSS(iterator): #input_batch, model, optimizer, organism, gradient_clip):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.bfloat16)
            target=tf.cast(inputs['target'],dtype=tf.float32)
            atac=tf.cast(inputs['atac'],dtype=tf.bfloat16)
            global_acc=tf.cast(inputs['global_acc'],dtype=tf.bfloat16)
            input_tuple = sequence, atac,global_acc

            output = model(input_tuple,
                           training=False)
            output = tf.cast(output,dtype=tf.float32)
            
            
            tss_tokens = tf.cast(inputs['tss_tokens'],dtype=tf.float32)
            gene_token = inputs['gene_token']
            cell_type = inputs['cell_type']
            
            if predict_masked_atac_bool: 
                pred = tf.reduce_sum(tf.cast(output,dtype=tf.float32)[:,:,1:] * tss_tokens,axis=1)
                true = tf.reduce_sum(target[:,:,1:] * tss_tokens,axis=1)
            else:
                pred = tf.reduce_sum(tf.cast(output,dtype=tf.float32) * tss_tokens,axis=1)
                true = tf.reduce_sum(target * tss_tokens,axis=1)
            
            return pred,true,gene_token,cell_type


        ta_pred = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store preds
        ta_true = tf.TensorArray(tf.float32, size=0, dynamic_size=True) # tensor array to store vals
        ta_celltype = tf.TensorArray(tf.int32, size=0, dynamic_size=True) # tensor array to store preds
        ta_genemap = tf.TensorArray(tf.int32, size=0, dynamic_size=True)        

        for _ in tf.range(val_steps_TSS): ## for loop within @tf.fuction for improved TPU performance

            pred_rep, true_rep, gene_rep, cell_type_rep = strategy.run(val_step,
                                                                       args=(next(iterator),))
            
            pred_reshape = tf.reshape(strategy.gather(pred_rep, axis=0), [-1]) # reshape to 1D
            true_reshape = tf.reshape(strategy.gather(true_rep, axis=0), [-1])
            cell_type_reshape = tf.reshape(strategy.gather(cell_type_rep, axis=0), [-1])
            gene_map_reshape = tf.reshape(strategy.gather(gene_rep, axis=0), [-1])
            
            ta_pred = ta_pred.write(_, pred_reshape)
            ta_true = ta_true.write(_, true_reshape)
            ta_celltype = ta_celltype.write(_, cell_type_reshape)
            ta_genemap = ta_genemap.write(_, gene_map_reshape)
        metric_dict["corr_stats"].update_state(ta_true.concat(),
                                                  ta_pred.concat(),
                                                  ta_celltype.concat(),
                                                  ta_genemap.concat())
        ta_true.close()
        ta_pred.close()
        ta_celltype.close()
        ta_genemap.close()
    
    def build_step(iterator): #input_batch, model, optimizer, organism, gradient_clip):
        @tf.function(jit_compile=True)
        def val_step(inputs):
            sequence=tf.cast(inputs['sequence'],dtype=tf.bfloat16)
            atac=tf.cast(inputs['atac'],dtype=tf.bfloat16)
            target=tf.cast(inputs['target'],dtype=tf.float32)
            global_acc=tf.cast(inputs['global_acc'],dtype=tf.bfloat16)         
            input_tuple = sequence,atac,global_acc

            output = model(input_tuple,
                           training=False)

        for _ in tf.range(1): ## for loop within @tf.fuction for improved TPU performance
            strategy.run(val_step, args=(next(iterator),))
    

    return dist_train_step,dist_val_step, dist_val_step_TSS,build_step, metric_dict


def deserialize_tr(serialized_example,
                   input_length,
                   max_shift,
                   output_length,
                   crop_size,
                   output_res,
                   predict_masked_atac_bool,
                   g):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'cage': tf.io.FixedLenFeature([], tf.string),
        'cell_specific_conv_arr': tf.io.FixedLenFeature([], tf.string)
    }
    ### stochastic sequence shift and gaussian noise

    rev_comp = tf.math.round(g.uniform([], 0, 1))

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


    data = tf.io.parse_example(serialized_example, feature_map)
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 seq_shift,input_length))
    
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float32),
                           [output_length,1])
    atac = atac + tf.math.abs(g.normal(atac.shape,
                                               mean=0.0,
                                               stddev=0.10,
                                               dtype=tf.float32))
    
    masked_atac=tf.nn.experimental.stateless_dropout(atac, rate=0.25, seed=[0,seq_shift]) / (1. / (1.0 - 0.25))
    
    cage = tf.ensure_shape(tf.io.parse_tensor(data['cage'],
                                              out_type=tf.float32),
                           [output_length - 2*crop_size,1])
    cage = cage
    #diff = tf.math.sqrt(tf.nn.relu(cage - 384.0 * tf.ones(cage.shape)))
    #cage = tf.clip_by_value(cage, clip_value_min=0.0, clip_value_max=384.0) + diff
    
    
    if rev_comp == 1:
        sequence = tf.gather(sequence, [3, 2, 1, 0], axis=-1)
        sequence = tf.reverse(sequence, axis=[0])
        atac = tf.reverse(atac,axis=[0])
        masked_atac = tf.reverse(masked_atac,axis=[0])
        cage = tf.reverse(cage,axis=[0])
        
        
    if predict_masked_atac_bool:
        atac_out = tf.slice(atac,
                            [crop_size,0],
                            [output_length-2*crop_size,-1])
        target = tf.concat([atac_out,cage],axis=1)
        
    global_acc = tf.ensure_shape(tf.io.parse_tensor(data['cell_specific_conv_arr'],
                                              out_type=tf.float32),
                           [1,1536])
    #global_acc=tf.expand_dims(global_acc,axis=0)
    global_acc = tf.math.asinh(global_acc)
    global_acc = (global_acc - tf.math.reduce_mean(global_acc)) / tf.math.reduce_std(global_acc)
    global_acc=tf.nn.experimental.stateless_dropout(global_acc, rate=0.05, seed=[0,seq_shift]) / (1. / (1.0 - 0.05))
    global_acc = global_acc + tf.math.abs(g.normal(global_acc.shape,
                                               mean=0.0,
                                               stddev=0.10,
                                               dtype=tf.float32))
        
    if predict_masked_atac_bool:
    
        return {'sequence': tf.ensure_shape(sequence,
                                            [input_length,4]),
                'atac': tf.ensure_shape(masked_atac,
                                          [output_length,1]),
                'global_acc': tf.ensure_shape(global_acc,
                                          [1,1536]),
                'target': tf.ensure_shape(target,
                                          [output_length-crop_size*2,2])}
    else:
        return {'sequence': tf.ensure_shape(sequence,
                                            [input_length,4]),
                'atac': tf.ensure_shape(atac,
                                          [output_length,1]),
                'global_acc': tf.ensure_shape(global_acc,
                                          [1,1536]),
                'target': tf.ensure_shape(cage,
                                          [output_length-crop_size*2,1])}

def deserialize_val(serialized_example,input_length,max_shift,output_length,crop_size,output_res,predict_masked_atac_bool):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'cage': tf.io.FixedLenFeature([], tf.string),
        'cell_specific_conv_arr': tf.io.FixedLenFeature([], tf.string)
    }
    
    seq_shift = 5
    input_seq_length = input_length + max_shift

    ### rev_comp
    #rev_comp = random.randrange(0,2)

    data = tf.io.parse_example(serialized_example, feature_map)
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 seq_shift,input_length))
    
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float32),
                           [output_length,1])
    
    masked_atac=tf.nn.experimental.stateless_dropout(atac, rate=0.25, seed=[0,seq_shift]) / (1. / (1.0 - 0.25))

    cage = tf.ensure_shape(tf.io.parse_tensor(data['cage'],
                                              out_type=tf.float32),
                           [output_length - 2*crop_size,1])
    
    if predict_masked_atac_bool:
        atac_out = tf.slice(atac,
                            [crop_size,0],
                            [output_length-2*crop_size,-1])
        target = tf.concat([atac_out,cage],axis=1)
        
    global_acc = tf.ensure_shape(tf.io.parse_tensor(data['cell_specific_conv_arr'],
                                              out_type=tf.float32),
                           [1,1536])
    global_acc = tf.math.asinh(global_acc)
    global_acc = (global_acc - tf.math.reduce_mean(global_acc)) / tf.math.reduce_std(global_acc)
    
    if predict_masked_atac_bool:
        return {'sequence': tf.ensure_shape(sequence,
                                            [input_length,4]),
                'atac': tf.ensure_shape(masked_atac,
                                          [output_length,1]),
                'global_acc': tf.ensure_shape(global_acc,
                                          [1,1536]),
                'target': tf.ensure_shape(target,
                                          [output_length-crop_size*2,2])}
    else:
        return {'sequence': tf.ensure_shape(sequence,
                                            [input_length,4]),
                'atac': tf.ensure_shape(atac,
                                          [output_length,1]),
                'global_acc': tf.ensure_shape(global_acc,
                                          [1,1536]),
                'target': tf.ensure_shape(cage,
                                          [output_length-crop_size*2,1])}

def deserialize_val_TSS(serialized_example,input_length,max_shift,output_length,crop_size,output_res,predict_masked_atac_bool):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        'cage': tf.io.FixedLenFeature([], tf.string),
        'tss_tokens': tf.io.FixedLenFeature([], tf.string),
        'gene_token': tf.io.FixedLenFeature([], tf.string),
        'cell_type': tf.io.FixedLenFeature([], tf.string),
        'cell_specific_conv_arr': tf.io.FixedLenFeature([], tf.string)
    }
    
    seq_shift = 5
    input_seq_length = input_length + max_shift

    ### rev_comp
    #rev_comp = random.randrange(0,2)

    data = tf.io.parse_example(serialized_example, feature_map)
    sequence = one_hot(tf.strings.substr(data['sequence'],
                                 seq_shift,input_length))
    
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                                              out_type=tf.float32),
                           [output_length,1])
    masked_atac=tf.nn.experimental.stateless_dropout(atac, rate=0.25, seed=[0,seq_shift]) / (1. / (1.0 - 0.25))
    #diff = tf.math.sqrt(tf.nn.relu(atac - 64.0 * tf.ones(atac.shape)))
    #atac = tf.clip_by_value(atac, clip_value_min=0.0, clip_value_max=64.0) + diff
    
    cage = tf.ensure_shape(tf.io.parse_tensor(data['cage'],
                                              out_type=tf.float32),
                           [output_length - 2*crop_size,1])
    #diff = tf.math.sqrt(tf.nn.relu(cage - 384.0 * tf.ones(cage.shape)))
    #cage = tf.clip_by_value(cage, clip_value_min=0.0, clip_value_max=384.0) + diff

    tss_tokens = tf.io.parse_tensor(data['tss_tokens'],
                                  out_type=tf.int32)

    #tss_tokens = tf.expand_dims(tss_tokens,axis=1)

    
    gene_token= tf.io.parse_tensor(data['gene_token'],
                                   out_type=tf.int32)

    cell_type = tf.io.parse_tensor(data['cell_type'],
                                  out_type=tf.int32)
    
    global_acc = tf.ensure_shape(tf.io.parse_tensor(data['cell_specific_conv_arr'],
                                              out_type=tf.float32),
                           [1,1536])
    #global_acc=tf.expand_dims(global_acc,axis=0)
    global_acc = tf.math.asinh(global_acc)
    global_acc = (global_acc - tf.math.reduce_mean(global_acc)) / tf.math.reduce_std(global_acc)

    if predict_masked_atac_bool:
        atac_out = tf.slice(atac,
                            [crop_size,0],
                            [output_length-2*crop_size,-1])
        target = tf.concat([atac_out,cage],axis=1)
        
    if predict_masked_atac_bool:
        return {'sequence': tf.ensure_shape(sequence,
                                            [input_length,4]),
                'atac': tf.ensure_shape(masked_atac,
                                          [output_length,1]),
                'target': tf.ensure_shape(target,
                                          [output_length-crop_size*2,2]),
                'tss_tokens': tf.ensure_shape(tss_tokens,
                                          [output_length-2*crop_size,1]),
                'global_acc': tf.ensure_shape(global_acc,
                                          [1,1536]),
                'gene_token':gene_token,
                'cell_type': cell_type}
    else:
        return {'sequence': tf.ensure_shape(sequence,
                                            [input_length,4]),
                'atac': tf.ensure_shape(atac,
                                          [output_length,1]),
                'target': tf.ensure_shape(cage,
                                          [output_length-crop_size*2,1]),
                'tss_tokens': tf.ensure_shape(tss_tokens,
                                          [output_length-2*crop_size,1]),
                'global_acc': tf.ensure_shape(global_acc,
                                          [1,1536]),
                'gene_token':gene_token,
                'cell_type': cell_type}

                    
def return_dataset(gcs_path,
                   split,
                   tss_bool,
                   batch,
                   input_length,
                   output_length,
                   crop_size,
                   output_res,
                   max_shift,
                   options,
                   num_parallel,
                   num_epoch,
                   predict_masked_atac_bool,
                   g):
    """
    return a tf dataset object for given gcs path
    """
    wc = "*.tfr"
    #print(split)
    
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                split,
                                                wc)))
    #print(list_files)
    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)
    
    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)
    dataset = dataset.with_options(options)
    #print(list_files)
    if split == 'train':
        
        dataset = dataset.map(lambda record: deserialize_tr(record,
                                                            input_length,
                                                            max_shift,
                                                            output_length,
                                                            crop_size,
                                                            output_res,
                                                            predict_masked_atac_bool,
                                                            g),
                              deterministic=False,
                              num_parallel_calls=num_parallel)
        
    else:
        if tss_bool:
            dataset = dataset.map(lambda record: deserialize_val_TSS(record,
                                                                 input_length,
                                                                 max_shift,
                                                                 output_length,
                                                                 crop_size,
                                                                 output_res,
                                                                    predict_masked_atac_bool),
                                  deterministic=False,
                                  num_parallel_calls=num_parallel)
        else:
            dataset = dataset.map(lambda record: deserialize_val(record,
                                                                 input_length,
                                                                 max_shift,
                                                                 output_length,
                                                                 crop_size,
                                                                 output_res,
                                                                predict_masked_atac_bool),
                                  deterministic=False,
                                  num_parallel_calls=num_parallel)

    return dataset.repeat(num_epoch).batch(batch,drop_remainder=True).prefetch(1)



def return_distributed_iterators(gcs_path,
                                 gcs_path_TSS,
                                 global_batch_size,
                                 input_length,
                                 max_shift,
                                 output_length,
                                 crop_size,
                                 output_res,
                                 num_parallel_calls,
                                 num_epoch,
                                 strategy,
                                 options,
                                 predict_masked_atac_bool,
                                 g):
    """ 
    returns train + val dictionaries of distributed iterators
    for given heads_dictionary
    """
    
    
    tr_data = return_dataset(gcs_path,
                             "train",
                             False,
                             global_batch_size,
                             input_length,
                             output_length,
                             crop_size,
                             output_res,
                             max_shift,
                             options,
                             num_parallel_calls,
                             num_epoch,
                             predict_masked_atac_bool,
                             g)

    val_data = return_dataset(gcs_path,
                              "valid",
                              False,
                              global_batch_size,
                              input_length,
                              output_length,
                              crop_size,
                              output_res,
                              max_shift,
                              options,
                              num_parallel_calls,
                              num_epoch,
                              predict_masked_atac_bool,
                              g)
    
    val_data_TSS = return_dataset(gcs_path_TSS,
                              "valid",
                              True,
                              global_batch_size,
                              input_length,
                              output_length,
                              crop_size,
                              output_res,
                              max_shift,
                              options,
                              num_parallel_calls,
                              num_epoch,
                                  predict_masked_atac_bool,
                              g)

    train_dist = strategy.experimental_distribute_dataset(tr_data)
    val_dist= strategy.experimental_distribute_dataset(val_data)
    val_dist_TSS= strategy.experimental_distribute_dataset(val_data_TSS)


    tr_data_it = iter(train_dist)
    val_data_it = iter(val_dist)
    val_data_TSS_it = iter(val_dist_TSS)


    return tr_data_it,val_data_it,val_data_TSS_it


def make_plots(y_trues,
               y_preds, 
               cell_types, 
               gene_map):

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
    except np.linalg.LinAlgError as err:
        cell_specific_corrs = [0.0] * len(np.unique(cell_types))

    try: 
        gene_specific_corrs=results_df.groupby('gene_encoding')[['true_zscore','pred_zscore']].corr(method='pearson').unstack().iloc[:,1].tolist()
    except np.linalg.LinAlgError as err:
        gene_specific_corrs = [0.0] * len(np.unique(gene_map))
    
    corrs_overall = np.nanmean(cell_specific_corrs), \
                        np.nanmean(gene_specific_corrs)
                        
        
    fig_overall,ax_overall=plt.subplots(figsize=(6,6))
    
    ## scatter plot for 50k points max
    idx = np.random.choice(np.arange(len(true_zscore)), 100, replace=False)
    
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
    parser.add_argument('--gcs_path_TSS',
                        dest='gcs_path_TSS',
                        help= 'google bucket containing preprocessed data')
    parser.add_argument('--num_parallel', dest = 'num_parallel',
                        type=int, default=tf.data.AUTOTUNE,
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
    parser.add_argument('--val_examples_TSS', dest = 'val_examples_TSS',
                        type=int, help='val_examples_TSS')
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
                        help='filter_list')
    parser.add_argument('--hidden_size',
                        dest='hidden_size',
                        default="1536",
                        help='hidden_size')
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
    parser.add_argument('--multitask_checkpoint_path',
                        dest='multitask_checkpoint_path',
                        type=str,
                        default="gs://picard-testing-176520/enformer_performer/models/enformer_performer_230120_196k_load_init-True_freeze-False_LR1-1e-06_LR2-0.0001_T-6_F-1536_D-0.4_K-relu_kernel_transformation_MP-True_AD-0.05/iteration_10",
                        help= 'multitask_checkpoint_path')
    parser.add_argument('--load_init',
                        dest='load_init',
                        type=str,
                        default="True",
                        help= 'load_init')
    parser.add_argument('--freeze_conv_layers',
                        dest='freeze_conv_layers',
                        type=str,
                        default="False",
                        help= 'freeze_conv_layers')
    parser.add_argument('--use_rot_emb',
                        dest='use_rot_emb',
                        type=str,
                        default="True",
                        help= 'use_rot_emb')
    parser.add_argument('--use_mask_pos',
                        dest='use_mask_pos',
                        type=str,
                        default="False",
                        help= 'use_mask_pos')
    parser.add_argument('--normalize',
                        dest='normalize',
                        type=str,
                        default="True",
                        help= 'normalize')
    parser.add_argument('--predict_masked_atac_bool',
                        dest='predict_masked_atac_bool',
                        type=str,
                        default="True",
                        help= 'predict_masked_atac_bool')
    parser.add_argument('--norm',
                        dest='norm',
                        type=str,
                        default="True",
                        help= 'norm')
    parser.add_argument('--wd_1',
                        dest='wd_1',
                        type=float,
                        default=0.01,
                        help= 'wd_1')
    parser.add_argument('--wd_2',
                        dest='wd_2',
                        type=float,
                        default=0.01,
                        help= 'wd_2')
    parser.add_argument('--cage_scale',
                        dest='cage_scale',
                        type=str,
                        default="5.0",
                        help= 'cage_scale')
    parser.add_argument('--rectify',
                        dest='rectify',
                        type=str,
                        default="True",
                        help= 'rectify')
    parser.add_argument('--inits_type',
                        dest='inits_type',
                        type=str,
                        default="enformer_conv",
                        help= 'inits_type')
    parser.add_argument('--optimizer',
                        dest='optimizer',
                        type=str,
                        default="adabelief",
                        help= 'optimizer')
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


