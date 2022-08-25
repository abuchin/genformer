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
os.environ['TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE']='False'
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

import training_utils_aformer_TF_genecentered_separated_SINGLE as training_utils
import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr  
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
                'input_length': {
                    'values': [int(x) for x in args.input_length.split(',')]
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
                #'max_seq_length': {
                #    'values':[args.max_seq_length]
                #},
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
                'freq_limit_scale': {
                    'values':[float(x) for x in args.freq_limit_scale.split(',')]
                },
                'beta1': {
                    'values':[float(x) for x in args.beta1.split(',')]
                },
                'beta2': {
                    'values':[float(x) for x in args.beta2.split(',')]
                },
                'use_tf_acc': {
                    'values':[x == 'True' for x in args.use_tf_acc.split(',')]
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
            wandb.Table.MAX_ROWS = 2000000
            #wandb.init(mode="disabled")
            wandb.config.tpu=args.tpu_name
            wandb.config.gcs_path=args.gcs_path
            wandb.config.gcs_path_val_ho=args.gcs_path_val_ho
            wandb.config.input_length=args.input_length
            wandb.config.target_unit=args.target_unit
            wandb.config.max_shift=args.max_shift

            wandb.config.output_heads=args.output_heads
            wandb.config.num_epochs=args.num_epochs
            wandb.config.train_steps=args.train_steps
            wandb.config.val_steps_h=args.val_steps_h
            wandb.config.val_steps_ho=args.val_steps_ho
            wandb.config.batch_size=args.batch_size
            wandb.config.warmup_frac=args.warmup_frac
            wandb.config.total_steps=args.total_steps
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            wandb.config.precision=args.precision
            wandb.config.sync_period=args.sync_period
            wandb.config.slow_step_frac=args.slow_step_frac
            wandb.config.use_rot_emb=args.use_rot_emb
            wandb.config.use_mask_pos=args.use_mask_pos
            
            wandb.config.use_tf_acc=args.use_tf_acc

            #wandb.config.max_seq_length=args.max_seq_length
            
            num_convs = len(wandb.config.conv_channel_list) + 1
            wandb.run.name = 'test'
            '''
            TPU init options
            '''
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            options.deterministic=False
            #options.experimental_threading.max_intra_op_parallelism = 1
            mixed_precision.set_global_policy('mixed_bfloat16')
            tf.config.optimizer.set_jit(True)


            NUM_REPLICAS = strategy.num_replicas_in_sync
            
            
            if wandb.config.input_length == 16384:
                wandb.config.update({"gcs_path": "gs://picard-testing-176520/16k_genecentered_blacklist0.50_atacnormalized/jurkat_single"},
                                   allow_val_change=True)
                wandb.config.update({"gcs_path_val_ho": "gs://picard-testing-176520/16k_genecentered_blacklist0.50_atacnormalized/val_holdout/val"},
                                   allow_val_change=True)
                wandb.config.update({"model_save_dir": "gs://picard-testing-176520/16k_genecentered_blacklist0.50_atacnormalized/jurkat_single/jurkat_single_model"},
                                   allow_val_change=True)
                wandb.config.update({"max_seq_length": 128},
                                   allow_val_change=True)
                BATCH_SIZE_PER_REPLICA=args.batch_size
                wandb.config.update({"train_steps": args.train_steps},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_h" : args.val_steps_h},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_ho" : args.val_steps_ho},
                                   allow_val_change=True)
                wandb.config.update({"total_steps": args.total_steps},
                                   allow_val_change=True)
            elif wandb.config.input_length == 32768:
                
                wandb.config.update({"gcs_path": "gs://picard-testing-176520/32k_genecentered_blacklist0.50_atacnormalized/jurkat_single"},
                                   allow_val_change=True)
                wandb.config.update({"gcs_path_val_ho": "gs://picard-testing-176520/32k_genecentered_blacklist0.50_atacnormalized/val_holdout/val"},
                                   allow_val_change=True)
                wandb.config.update({"model_save_dir": "gs://picard-testing-176520/32k_genecentered_blacklist0.50_atacnormalized/jurkat_single/jurkat_single_model"},
                                   allow_val_change=True)
                wandb.config.update({"max_seq_length": 512},
                                   allow_val_change=True)
                BATCH_SIZE_PER_REPLICA=64
                wandb.config.update({"train_steps": 7},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_h" : 1},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_ho" : 29},
                                   allow_val_change=True)
                wandb.config.update({"total_steps": 100},
                                   allow_val_change=True)
                
            elif wandb.config.input_length == 65536:
                
                wandb.config.update({"gcs_path": "gs://picard-testing-176520/65k_genecentered_blacklist0.50_atacnormalized/jurkat_single"},
                                   allow_val_change=True)
                wandb.config.update({"gcs_path_val_ho": "gs://picard-testing-176520/65k_genecentered_blacklist0.50_atacnormalized/val_holdout/val"},
                                   allow_val_change=True)
                wandb.config.update({"model_save_dir": "gs://picard-testing-176520/65k_genecentered_blacklist0.50_atacnormalized/jurkat_single/jurkat_single_model"},
                                   allow_val_change=True)
                wandb.config.update({"max_seq_length": 512},
                                   allow_val_change=True)
                BATCH_SIZE_PER_REPLICA=48
                wandb.config.update({"train_steps": 3},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_h" : 1},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_ho" : 10},
                                   allow_val_change=True)
                wandb.config.update({"total_steps": 150},
                                   allow_val_change=True)
                
            elif wandb.config.input_length == 131072:
                
                wandb.config.update({"gcs_path": "gs://picard-testing-176520/131k_genecentered_blacklist0.50_atacnormalized/jurkat_single"},
                                   allow_val_change=True)
                wandb.config.update({"gcs_path_val_ho": "gs://picard-testing-176520/131k_genecentered_blacklist0.50_atacnormalized/val_holdout/val"},
                                   allow_val_change=True)
                wandb.config.update({"model_save_dir": "gs://picard-testing-176520/131k_genecentered_blacklist0.50_atacnormalized/jurkat_single/jurkat_single_model"},
                                   allow_val_change=True)
                wandb.config.update({"max_seq_length": 1024},
                                   allow_val_change=True)
                BATCH_SIZE_PER_REPLICA=16
                wandb.config.update({"train_steps": 7},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_h" : 1},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_ho" : 27},
                                   allow_val_change=True)
                wandb.config.update({"total_steps": 350},
                                   allow_val_change=True)
                
            elif wandb.config.input_length == 196608:
                
                wandb.config.update({"gcs_path": "gs://picard-testing-176520/196k_genecentered_blacklist0.50_atacnormalized/jurkat_single"},
                                   allow_val_change=True)
                wandb.config.update({"gcs_path_val_ho": "gs://picard-testing-176520/196k_genecentered_blacklist0.50_atacnormalized/val_holdout/val"},
                                   allow_val_change=True)
                wandb.config.update({"model_save_dir": "gs://picard-testing-176520/196k_genecentered_blacklist0.50_atacnormalized/jurkat_single/jurkat_single_model"},
                                   allow_val_change=True)
                wandb.config.update({"max_seq_length": 1536},
                                   allow_val_change=True)
                BATCH_SIZE_PER_REPLICA=12
                wandb.config.update({"train_steps": 9},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_h" : 2},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_ho" : 39},
                                   allow_val_change=True)
                wandb.config.update({"total_steps": 450},
                                   allow_val_change=True)
                
            elif wandb.config.input_length == 262144:
                wandb.config.update({"gcs_path": "gs://picard-testing-176520/262k_genecentered_blacklist0.50_atacnormalized/jurkat_single"},
                                   allow_val_change=True)
                wandb.config.update({"gcs_path_val_ho": "gs://picard-testing-176520/262k_genecentered_blacklist0.50_atacnormalized/val_holdout/val"},
                                   allow_val_change=True)
                wandb.config.update({"model_save_dir": "gs://picard-testing-176520/262k_genecentered_blacklist0.50_atacnormalized/jurkat_single/jurkat_single_model"},
                                   allow_val_change=True)
                wandb.config.update({"max_seq_length": 2048},
                                   allow_val_change=True)
                BATCH_SIZE_PER_REPLICA=6
                #print(BATCH_SIZE_PER_REPLICA)
                wandb.config.update({"train_steps": 17},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_h" : 3},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_ho" : 77},
                                   allow_val_change=True)
                wandb.config.update({"total_steps": 850},
                                   allow_val_change=True)
            elif wandb.config.input_length == 393216:
                wandb.config.update({"gcs_path": "gs://picard-testing-176520/393k_genecentered_blacklist0.50_atacnormalized/jurkat_single"},
                                   allow_val_change=True)
                wandb.config.update({"gcs_path_val_ho": "gs://picard-testing-176520/393k_genecentered_blacklist0.50_atacnormalized/val_holdout/val"},
                                   allow_val_change=True)
                wandb.config.update({"model_save_dir": "gs://picard-testing-176520/393k_genecentered_blacklist0.50_atacnormalized/jurkat_single/jurkat_single_model"},
                                   allow_val_change=True)
                wandb.config.update({"max_seq_length": 3072},
                                   allow_val_change=True)
                BATCH_SIZE_PER_REPLICA=2
                #print(BATCH_SIZE_PER_REPLICA)
                wandb.config.update({"train_steps": 51},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_h" : 8},
                                   allow_val_change=True)
                wandb.config.update({"val_steps_ho" : 230},
                                   allow_val_change=True)
                wandb.config.update({"total_steps": 2550},
                                   allow_val_change=True)
            else:
                raise ValueError('input a valid length')
            
            
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_REPLICAS
            print('global batch size:', GLOBAL_BATCH_SIZE)
            data_it_tr_list = []
            data_it_val_list = []
            
            ### create dataset iterators
            heads_dict = {}
            orgs = wandb.config.organisms.split(',')
            for k, org in enumerate(orgs):
                heads_dict[org] = int(k)

            data_dict_tr, data_dict_val,val_ho_it = training_utils.return_distributed_iterators(heads_dict,
                                                                                      wandb.config.gcs_path,
                                                                                      wandb.config.gcs_path_val_ho,
                                                                                      GLOBAL_BATCH_SIZE,
                                                                                      wandb.config.input_length,
                                                                                      wandb.config.max_shift,
                                                                                      wandb.config.target_unit,
                                                                                      args.num_parallel,
                                                                                      args.num_epochs,
                                                                                      strategy,
                                                                                      options)
            
            print('val ho it')
            print(val_ho_it)
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
            
    
            if wandb.config.optimizer == "adabelief":
                optimizer = tfa.optimizers.AdaBelief(learning_rate=wandb.config.lr_base,
                                                     weight_decay=wandb.config.weight_decay_frac,
                                                     warmup_proportion=wandb.config.warmup_frac,
                                                     epsilon=wandb.config.epsilon,
                                                     rectify=wandb.config.rectify,
                                                     min_lr=wandb.config.min_lr,
                                                     total_steps=wandb.config.total_steps)
                    
                optimizer=tfa.optimizers.Lookahead(optimizer,
                                                   sync_period=wandb.config.sync_period,
                                                   slow_step_size=wandb.config.slow_step_frac)
            elif wandb.config.optimizer == 'adafactor':
                optimizer = optimizers.AdafactorOptimizer()
                optimizer=tfa.optimizers.Lookahead(optimizer,
                                                   sync_period=wandb.config.sync_period,
                                                   slow_step_size=wandb.config.slow_step_frac)
            elif (wandb.config.optimizer.lower() == 'adamw') :
                ## learning rate_scheduler
                scheduler= tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=wandb.config.lr_base,
                    decay_steps=wandb.config.total_steps, alpha=(wandb.config.min_lr / wandb.config.lr_base))
                scheduler=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base,
                                             warmup_steps=wandb.config.warmup_frac * wandb.config.total_steps,
                                             decay_schedule_fn=scheduler)
                                            
                scheduler_wd= tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=wandb.config.weight_decay_frac,
                    decay_steps=wandb.config.total_steps, alpha=0.0)
                scheduler_wd=optimizers.WarmUp(initial_learning_rate=wandb.config.weight_decay_frac,
                                             warmup_steps=int(wandb.config.warmup_frac * wandb.config.total_steps),
                                             decay_schedule_fn=scheduler)
                
                optimizer = tfa.optimizers.AdamW(learning_rate=scheduler,
                                                 beta_1=wandb.config.beta1,
                                                 beta_2=wandb.config.beta2,
                                                 weight_decay=scheduler_wd)
                                                 
                optimizer=tfa.optimizers.Lookahead(optimizer,
                                                   sync_period=wandb.config.sync_period,
                                                   slow_step_size=wandb.config.slow_step_frac)
            else:
                raise ValueError("optimizer not implemented")

            
            metric_dict = {}
            
            freq_limit = int(wandb.config.input_length * wandb.config.freq_limit_scale)
            print('freq_limit:', freq_limit)
            if len(orgs) == 1:
                
                train_step, val_step, dist_val_step_ho, metric_dict = training_utils.return_train_val_functions_hg(model,
                                                                                              optimizer,
                                                                                              strategy,
                                                                                              metric_dict, 
                                                                                              wandb.config.train_steps,
                                                                                              wandb.config.val_steps_h,
                                                                                              wandb.config.val_steps_ho,
                                                                                              GLOBAL_BATCH_SIZE,
                                                                                              wandb.config.gradient_clip,
                                                                                              wandb.config.use_fft_prior,
                                                                                              freq_limit,
                                                                                              wandb.config.fft_prior_scale,
                                                                                              wandb.config.use_tf_acc)
            else:
                train_step, val_step, dist_val_step_ho, metric_dict = training_utils.return_train_val_functions_hg_mm(model,
                                                                                              optimizer,
                                                                                              strategy,
                                                                                              metric_dict, 
                                                                                              wandb.config.train_steps,
                                                                                              wandb.config.val_steps_h,
                                                                                              wandb.config.val_steps_ho,
                                                                                              GLOBAL_BATCH_SIZE,
                                                                                              wandb.config.gradient_clip,
                                                                                              wandb.config.use_fft_prior,
                                                                                              freq_limit,
                                                                                              wandb.config.fft_prior_scale,
                                                                                              wandb.config.use_tf_acc)

                
            print(wandb.config)
            
            ### main training loop
            global_step = 0
            val_losses = []
            val_pearsons = []
            val_R2 = []
            patience_counter = 0
            stop_criteria = False
            best_epoch = 0
            
            
            for epoch_i in range(1, wandb.config.num_epochs + 1):
                start = time.time()
                if len(orgs) == 1:
                    lr, it = train_step(data_dict_tr['hg'])

                else:
                    lr, it = train_step(data_dict_tr['hg'],
                                        data_dict_tr['mm'])

                end = time.time()
                duration = (end - start) / 60.
                print('completed epoch ' + str(epoch_i))
                print('hg_train_loss: ' + str(metric_dict['hg_tr'].result().numpy()))
                print('training duration(mins): ' + str(duration))
                
                start = time.time()
                if len(orgs) == 1:
                    val_step(data_dict_val['hg'])
                else:
                    val_step(data_dict_val['hg'])
                    
                val_losses.append(metric_dict['hg_val'].result().numpy())
                val_pearsons.append(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy())
                
                print('hg_val_loss: ' + str(metric_dict['hg_val'].result().numpy()))
                print('hg_val_pearson: ' + str(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy()))
                print('hg_val_R2: ' + str(metric_dict['hg_corr_stats'].result()['R2'].numpy()))
                


                y_trues = metric_dict['hg_corr_stats'].result()['y_trues'].numpy()
                y_preds = metric_dict['hg_corr_stats'].result()['y_preds'].numpy()
                cell_types = metric_dict['hg_corr_stats'].result()['cell_types'].numpy()
                gene_map = metric_dict['hg_corr_stats'].result()['gene_map'].numpy()
                
                
                overall_corr,overall_corr_sp,low_corr,low_corr_sp,high_corr, high_corr_sp, cell_corr,cell_corr_sp, gene_corr,gene_corr_sp,cell_fig,gene_fig, cells_df,genes_df,h_var_corr,h_var_corr_sp,low_var_corr,low_var_corr_sp= training_utils.make_plots(y_trues,y_preds,cell_types,gene_map, 'hg',args.cell_type_map_file, args.gene_map_file, args.gene_symbol_map_file,args.high_variance_list)


                wandb.log({'hg_train_loss': metric_dict['hg_tr'].result().numpy(),
                           'hg_val_loss': metric_dict['hg_val'].result().numpy(),
                           'hg_overall_rho': overall_corr,
                           'hg_overall_rho_sp': overall_corr_sp,
                           'hg_low_var': low_var_corr,
                           'hg_low_var_sp': low_var_corr_sp,
                           'hg_high_var': h_var_corr,
                           'hg_high_var_sp': h_var_corr_sp,
                           'hg_median_cell_rho': cell_corr,
                           'hg_median_cell_rho_sp': cell_corr_sp},
                          step=epoch_i)
                
                if epoch_i % 3 == 0:
    
                    dist_val_step_ho(val_ho_it)

                    print('completed dist_val_step_ho')
                    print('hg_val_pearson_ho: ' + str(metric_dict['hg_corr_stats_ho'].result()['pearsonR'].numpy()))
                    print('hg_val_R2_ho: ' + str(metric_dict['hg_corr_stats_ho'].result()['R2'].numpy()))

                    y_trues_ho = metric_dict['hg_corr_stats_ho'].result()['y_trues'].numpy()
                    y_preds_ho = metric_dict['hg_corr_stats_ho'].result()['y_preds'].numpy()
                    cell_types_ho = metric_dict['hg_corr_stats_ho'].result()['cell_types'].numpy()
                    gene_map_ho = metric_dict['hg_corr_stats_ho'].result()['gene_map'].numpy()

                    overall_corr_ho,overall_corr_sp_ho,low_corr_ho,low_corr_sp_ho,high_corr_ho, high_corr_sp_ho, cell_corr_ho,cell_corr_sp_ho, gene_corr_ho,gene_corr_sp_ho,cell_fig_ho,gene_fig_ho,cells_df_ho,genes_df_ho,h_var_corr_ho,h_var_corr_sp_ho,low_var_corr_ho,low_var_corr_sp_ho = training_utils.make_plots(y_trues_ho,y_preds_ho,cell_types_ho,gene_map_ho, 'hg_ho',args.cell_type_map_file, args.gene_map_file, args.gene_symbol_map_file,args.high_variance_list)
                    
                    wandb.log({'hg_overall_rho_ho': overall_corr_ho,
                               'hg_overall_rho_sp_ho': overall_corr_sp_ho,
                               'hg_low_var_ho': low_var_corr_ho,
                               'hg_low_var_sp_ho': low_var_corr_sp_ho,
                               'hg_high_var_ho': h_var_corr_ho,
                               'hg_high_var_sp_ho': h_var_corr_sp_ho,
                               'hg_median_cell_rho_ho': cell_corr_ho,
                               'hg_median_cell_rho_sp_ho': cell_corr_sp_ho},
                              step=epoch_i)
                    
                    cells_table = wandb.Table(dataframe=cells_df_ho)
                    genes_table = wandb.Table(dataframe=genes_df_ho)
                    
                    wandb.log({"cells_correlations": cells_table},step=epoch_i)
                    wandb.log({"genes_correlations": genes_table},step=epoch_i)
                    wandb.log({"cell level corr":cell_fig_ho},step=epoch_i)
                    wandb.log({"gene level corrs/var":gene_fig_ho},step=epoch_i)
                
                
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

                print('patience counter at: ' + str(patience_counter))
                for key, item in metric_dict.items():
                    item.reset_state()
                if stop_criteria:
                    print('early stopping at: epoch ' + str(epoch_i))
                    break
                
                end = time.time()
                duration = (end - start) / 60.
                print('validation duration(mins): ' + str(duration))
                    
            print('saving model at: epoch ' + str(epoch_i))
            #print('best model was at: epoch ' + str(best_epoch))
            model.save_weights(wandb.config.model_save_dir + "/" + wandb.config.model_save_basename + "_" + wandb.run.name + "/final/saved_model")
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

##########################################################################
if __name__ == '__main__':
    main()
        
