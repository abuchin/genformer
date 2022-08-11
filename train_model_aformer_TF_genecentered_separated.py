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

import logging
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision

## custom modules
import src.aformer_TF_gc_separated as aformer
import src.metrics as metrics
import src.optimizers as optimizers
import src.schedulers as schedulers
import src.utils as utils

import training_utils_aformer_TF_genecentered_separated as training_utils
import seaborn as sns
from scipy.stats.stats import pearsonr  
from scipy import stats

 ## reformat 
# ===========================================================================#

def main():
    # ============== arg parse ==============================================# 
    parser = argparse.ArgumentParser(
        description='process input for genformer training loop')
    parser = training_utils.parse_args(parser)
    args = parser.parse_args()
    
    #================ init ==================================================# 
    
    ### make sure gcloud auth set to picard-testing-176520
        
    ### make sure TPU started

    # ============== define sweep options ==================== #
    sweep_config = {
            "name" : args.wandb_sweep_name,
            'method': "grid",
            'metric': {
                'name': 'hg_val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'lr_schedule':{
                    'values': [args.lr_schedule]
                },
                'dropout': {
                    'values': [float(x) for x in args.dropout.split(',')]
                },
                'lr_base': {
                    'values':[float(x) for x in args.lr_base.split(',')]
                },
                'min_lr': {
                    'values':[float(x) for x in args.min_lr.split(',')]
                },
                'optimizer': {
                    'values': [args.optimizer]
                },
                'gradient_clip': {
                    'values': [float(x) for x in args.gradient_clip.split(',')]
                },
                'precision': {
                    'values': [args.precision]
                },
                'weight_decay_frac': {
                    'values': [float(x) for x in args.weight_decay_frac.split(',')]
                },
                'conv_channel_list':{
                    'values': [[int(x) for x in args.conv_channel_list.split(',')]]
                },
                'conv_filter_size_1_seq':{
                    'values': [int(x) for x in args.conv_filter_size_1_seq.split(',')]
                },
                'conv_filter_size_2_seq':{
                    'values': [int(x) for x in args.conv_filter_size_2_seq.split(',')]
                },
                'conv_filter_size_1_atac':{
                    'values': [int(x) for x in args.conv_filter_size_1_atac.split(',')]
                },
                'conv_filter_size_2_atac':{
                    'values': [int(x) for x in args.conv_filter_size_2_atac.split(',')]
                },
                'bottleneck_units':{
                    'values': [int(x) for x in args.bottleneck_units.split(',')]
                },
                'bottleneck_units_tf':{
                    'values': [int(x) for x in args.bottleneck_units_tf.split(',')]
                },
                'num_transformer_layers':{
                    'values': [int(x) for x in args.num_transformer_layers.split(',')]
                },
                'num_heads':{
                    'values': [int(x) for x in args.num_heads.split(',')]
                },
                'momentum': {
                    'values':[float(x) for x in args.momentum.split(',')]
                },
                'hidden_size': {
                    'values':[int(x) for x in args.hidden_size.split(',')]
                },
                'num_random_features': {
                    'values':[int(x) for x in args.num_random_features.split(',')]
                },
                'kernel_transformation': {
                    'values':[args.kernel_transformation]
                },
                'organisms': {
                    'values':args.output_heads.split(';')
                },
                'dim': {
                    'values':[args.dim]
                },
                'max_seq_length': {
                    'values':[args.max_seq_length]
                },
                'rel_pos_bins': {
                    'values':[args.rel_pos_bins]
                },
                'epsilon': {
                    'values':[args.epsilon]
                },
                'rectify': {
                    'values':[args.rectify]
                },
                'kernel_regularizer': {
                    'values':[float(x) for x in args.kernel_regularizer.split(',')]
                },
                'use_fft_prior': {
                    'values':[x == 'True' for x in args.use_fft_prior.split(',')]
                },
                'fft_prior_scale': {
                    'values':[float(x) for x in args.fft_prior_scale.split(',')]
                },
                'freq_limit': {
                    'values':[int(x) for x in args.freq_limit.split(',')]
                }
            }
    }

    
    def sweep_train(config_defaults=None):
        # Set default values
        # Specify the other hyperparameters to the configuration, if any

        ## tpu initialization
        strategy = training_utils.tf_tpu_initialize(args.tpu_name)
        if args.precision == 'mixed_bfloat16':
            mixed_precision.set_global_policy('mixed_bfloat16')
        
        ## rest must be w/in strategy scope
        with strategy.scope():
            config_defaults = {
                "lr_base": 0.01 ### will be overwritten
            }
            
            ### log training parameters
            wandb.init(config=config_defaults, 
                       project= args.wandb_project, 
                       entity=args.wandb_user)
            #wandb.init(mode="disabled")
            wandb.config.tpu=args.tpu_name
            wandb.config.gcs_path=args.gcs_path
            wandb.config.input_length=args.input_length
            wandb.config.target_unit=args.target_unit
            wandb.config.max_shift=args.max_shift

            wandb.config.output_heads=args.output_heads
            wandb.config.num_epochs=args.num_epochs
            wandb.config.train_steps=args.train_steps
            wandb.config.val_steps_h=args.val_steps_h
            wandb.config.val_steps_m=args.val_steps_m
            wandb.config.batch_size=args.batch_size
            wandb.config.warmup_frac=args.warmup_frac
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            wandb.config.precision=args.precision
            wandb.config.sync_period=args.sync_period
            wandb.config.slow_step_frac=args.slow_step_frac
            wandb.config.use_rot_emb=args.use_rot_emb
            wandb.config.use_mask_pos=args.use_mask_pos

            wandb.config.max_seq_length=args.max_seq_length
            
            num_convs = len(wandb.config.conv_channel_list) + 1
            wandb.run.name = 'test'
            '''
            TPU init options
            '''
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            options.deterministic=False
            #options.experimental_threading.max_intra_op_parallelism = 1
            mixed_precision.set_global_policy('mixed_bfloat16')
            tf.config.optimizer.set_jit(True)

            BATCH_SIZE_PER_REPLICA = args.batch_size
            NUM_REPLICAS = strategy.num_replicas_in_sync
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_REPLICAS
            
            data_it_tr_list = []
            data_it_val_list = []
            
            ### create dataset iterators
            heads_dict = {}
            orgs = wandb.config.organisms.split(',')
            for k, org in enumerate(orgs):
                heads_dict[org] = int(k)

            data_dict_tr, data_dict_val = training_utils.return_distributed_iterators(heads_dict,
                                                                                      wandb.config.gcs_path,
                                                                                      GLOBAL_BATCH_SIZE,
                                                                                      wandb.config.input_length,
                                                                                      wandb.config.max_shift,
                                                                                      wandb.config.target_unit,
                                                                                      args.num_parallel,
                                                                                      args.num_epochs,
                                                                                      strategy,
                                                                                      options)
            

            rot_emb_bool = False
            if wandb.config.use_rot_emb == 'True':
                rot_emb_bool = True
            mask_bool = False
            if wandb.config.use_mask_pos == 'True':
                mask_bool = True
            #print(rot_emb_bool)
            #print(mask_bool)
            if ((mask_bool and rot_emb_bool)):
                raise ValueError('choose one of rotary or mask')
            #print(((not mask_bool) and (not rot_emb_bool)))
            if ((not mask_bool) and (not rot_emb_bool)):
                raise ValueError('choose one of rotary or mask')
                
            
            model = aformer.aformer(kernel_transformation=wandb.config.kernel_transformation,
                                        dropout_rate=wandb.config.dropout,
                                        input_length=wandb.config.input_length,
                                        num_heads=wandb.config.num_heads,
                                        numerical_stabilizer=0.0000001,
                                        nb_random_features=wandb.config.num_random_features,
                                        hidden_size=wandb.config.hidden_size,
                                        d_model=wandb.config.hidden_size,
                                        dim=wandb.config.dim,
                                        rel_pos_bins=wandb.config.rel_pos_bins,
                                        max_seq_length=wandb.config.max_seq_length,
                                        widening = 2, 
                                        conv_filter_size_1_seq=wandb.config.conv_filter_size_1_seq,
                                        conv_filter_size_2_seq=wandb.config.conv_filter_size_2_seq,
                                        conv_filter_size_1_atac=wandb.config.conv_filter_size_1_atac,
                                        conv_filter_size_2_atac=wandb.config.conv_filter_size_2_atac,
                                        transformer_depth=wandb.config.num_transformer_layers,
                                        momentum=wandb.config.momentum,
                                        channels_list=wandb.config.conv_channel_list,
                                        kernel_regularizer=wandb.config.kernel_regularizer,
                                        bottleneck_units=wandb.config.bottleneck_units,
                                        bottleneck_units_tf=wandb.config.bottleneck_units_tf,
                                        heads_dict=heads_dict,
                                        use_rot_emb = rot_emb_bool,
                                        use_mask_pos = mask_bool)
    

            optimizer = optimizers.AdafactorOptimizer()
            optimizer=tfa.optimizers.Lookahead(optimizer,
                                               sync_period=wandb.config.sync_period,
                                               slow_step_size=wandb.config.slow_step_frac)
            
            metric_dict = {}
            if len(orgs) == 1:
                train_step, val_step, metric_dict = training_utils.return_train_val_functions_hg(model,
                                                                                              optimizer,
                                                                                              strategy,
                                                                                              metric_dict, 
                                                                                              wandb.config.train_steps,
                                                                                              wandb.config.val_steps_h,
                                                                                              wandb.config.val_steps_m,
                                                                                              GLOBAL_BATCH_SIZE,
                                                                                              wandb.config.gradient_clip,
                                                                                              wandb.config.use_fft_prior,
                                                                                              wandb.config.freq_limit,
                                                                                              wandb.config.fft_prior_scale)
            else:
                train_step, val_step, metric_dict = training_utils.return_train_val_functions_hg_mm(model,
                                                                                              optimizer,
                                                                                              strategy,
                                                                                              metric_dict, 
                                                                                              wandb.config.train_steps,
                                                                                              wandb.config.val_steps_h,
                                                                                              wandb.config.val_steps_m,
                                                                                              GLOBAL_BATCH_SIZE,
                                                                                              wandb.config.gradient_clip,
                                                                                              wandb.config.use_fft_prior,
                                                                                              wandb.config.freq_limit,
                                                                                              wandb.config.fft_prior_scale)

            
            ### main training loop
            global_step = 0
            val_losses = []
            val_pearsons = []
            val_R2 = []
            patience_counter = 0
            stop_criteria = False
            best_epoch = 0
            for epoch_i in range(1, wandb.config.num_epochs):
                start = time.time()
                if len(orgs) == 1:
                    train_step(data_dict_tr['hg'])
                    val_step(data_dict_val['hg'])
                else:
                    train_step(data_dict_tr['hg'],
                               data_dict_tr['mm'])
                    val_step(data_dict_val['hg'],
                             data_dict_val['mm'])
                y_trues = metric_dict['hg_corr_stats'].result()['y_trues'].numpy()
                y_preds = metric_dict['hg_corr_stats'].result()['y_preds'].numpy()
                cell_types = metric_dict['hg_corr_stats'].result()['cell_types'].numpy()
                gene_map = metric_dict['hg_corr_stats'].result()['gene_map'].numpy()

                val_losses.append(metric_dict['hg_val'].result().numpy())
                val_pearsons.append(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy())

                overall_corr, low_corr,high_corr,cell_corr, gene_corr, overall_fig,l_fig,h_fig,cell_fig,gene_fig = training_utils.make_plots(y_trues,y_preds,cell_types,gene_map, 'hg')
                

                wandb.log({'hg_train_loss': metric_dict['hg_tr'].result().numpy(),
                           'hg_val_loss': metric_dict['hg_val'].result().numpy(),
                           'hg_overall_rho': overall_corr,
                           'hg_low_rho': low_corr,
                           'hg_high_rho': high_corr,
                           'hg_mean_cell_rho': cell_corr,
                           'hg_mean_gene_rho': gene_corr},
                          step=epoch_i)
                wandb.log({"overall_gene_level":overall_fig},step=epoch_i)
                wandb.log({"low_gene_level":l_fig},step=epoch_i)
                wandb.log({"high_gene_level":h_fig},step=epoch_i)
                wandb.log({"cell level corr":cell_fig},step=epoch_i)
                wandb.log({"gene level corrs/var":gene_fig},step=epoch_i)

                
                end = time.time()
                duration = (end - start) / 60.

                
                if (epoch_i > 2):
                    stop_criteria,patience_counter,best_epoch = training_utils.early_stopping(current_val_loss=val_losses[-1],
                                                                                              logged_val_losses=val_losses,
                                                                                              current_pearsons=val_pearsons[-1],
                                                                                              logged_pearsons=val_pearsons,
                                                                                              current_epoch=epoch_i,
                                                                                              best_epoch=best_epoch,
                                                                                              save_freq=args.savefreq,
                                                                                              patience=wandb.config.patience,
                                                                                              patience_counter=patience_counter,
                                                                                              min_delta=wandb.config.min_delta,
                                                                                              model=model,
                                                                                              save_directory=wandb.config.model_save_dir,
                                                                                              saved_model_basename=wandb.config.model_save_basename)
                print('completed epoch ' + str(epoch_i))
                print('duration(mins): ' + str(duration))
                print('hg_train_loss: ' + str(metric_dict['hg_tr'].result().numpy()))
                print('hg_val_loss: ' + str(metric_dict['hg_val'].result().numpy()))
                print('hg_val_pearson: ' + str(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy()))
                print('hg_val_R2: ' + str(metric_dict['hg_corr_stats'].result()['R2'].numpy()))
                print('patience counter at: ' + str(patience_counter))

                for key, item in metric_dict.items():
                    item.reset_state()
                if stop_criteria:
                    print('early stopping at: epoch ' + str(epoch_i))
                    break
                    
            print('saving model at: epoch ' + str(epoch_i))
            print('best model was at: epoch ' + str(best_epoch))
            model.save_weights(wandb.config.model_save_dir + "/" + wandb.config.model_save_basename + "_" + wandb.run.name + "/final/saved_model")
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

##########################################################################
if __name__ == '__main__':
    main()
        