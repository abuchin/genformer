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


import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import tensorflow_addons as tfa
from tensorflow import strings as tfs
from tensorflow.keras import mixed_precision

## custom modules
import src.genformer1 as genformer
import src.metrics as metrics
import src.optimizers as optimizers
import src.schedulers as schedulers
import src.utils as utils

import training_utils
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
                'conv_filter_size_1':{
                    'values': [int(x) for x in args.conv_filter_size_1.split(',')]
                },
                'conv_filter_size_2':{
                    'values': [int(x) for x in args.conv_filter_size_2.split(',')]
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
                    'values':[args.output_heads]
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
                    'values':[args.kernel_regularizer]
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
            wandb.config.output_length=args.output_length
            wandb.config.output_res=args.output_res
            wandb.config.output_heads=args.output_heads
            wandb.config.num_epochs=args.num_epochs
            wandb.config.train_steps=args.train_steps
            wandb.config.val_steps=args.val_steps
            wandb.config.batch_size=args.batch_size
            wandb.config.warmup_frac=args.warmup_frac
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            wandb.config.precision=args.precision
            wandb.config.sync_period=args.sync_period
            wandb.config.slow_step_frac=args.slow_step_frac
            wandb.config.quant_type=args.quant_type

            wandb.config.dim=args.dim
            wandb.config.rel_pos_bins=args.rel_pos_bins
            wandb.config.max_seq_length=args.max_seq_length
            
            num_convs = len(wandb.config.conv_channel_list) + 1
            wandb.run.name = '_'.join([str(wandb.config.model_save_basename),
                                      'I' + str(wandb.config.input_length),
                                      'O' + str(wandb.config.output_length),
                                      'R' + str(wandb.config.output_res),
                                      'T' + str(wandb.config.num_transformer_layers),
                                      'H' + str(wandb.config.num_heads),
                                      'C' + str(num_convs),
                                      'F.1' + str(wandb.config.conv_filter_size_1),
                                      'F.2' + str(wandb.config.conv_filter_size_2),
                                      'D' + str(wandb.config.dropout),
                                      'HS' + str(wandb.config.hidden_size)])
            
            '''
            TPU init options
            '''
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            options.deterministic=False
            options.experimental_threading.max_intra_op_parallelism = 1

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
            print(heads_dict)
            data_dict_tr, data_dict_val = training_utils.return_distributed_iterators(heads_dict,
                                                                                      wandb.config.gcs_path,
                                                                                      GLOBAL_BATCH_SIZE,
                                                                                      wandb.config.input_length,
                                                                                      wandb.config.output_length,
                                                                                      args.num_parallel,
                                                                                      args.num_epochs,
                                                                                      strategy,
                                                                                      options,
                                                                                      wandb.config.quant_type,
                                                                                      args.input_dim)

            

                    
            ### define model
            model = genformer.genformer(kernel_transformation=wandb.config.kernel_transformation,
                                        dropout_rate=wandb.config.dropout,
                                        final_out_length=wandb.config.output_length,
                                        num_heads=wandb.config.num_heads,
                                        numerical_stabilizer=0.0000001,
                                        nb_random_features=wandb.config.num_random_features,
                                        hidden_size=wandb.config.hidden_size,
                                        d_model=wandb.config.hidden_size,
                                        dim=wandb.config.dim,
                                        rel_pos_bins=wandb.config.rel_pos_bins,
                                        max_seq_length=wandb.config.max_seq_length,
                                        widening = 2, 
                                        conv_filter_size=wandb.config.conv_filter_size_1,
                                        #conv_filter_size_2=wandb.config.conv_filter_size_2,
                                        transformer_depth=wandb.config.num_transformer_layers,
                                        momentum=wandb.config.momentum,
                                        channels_list=wandb.config.conv_channel_list,
                                        kernel_regularizer=wandb.config.kernel_regularizer,
                                        heads_dict=heads_dict)
                                        #input_dim=args.input_dim)
    
            
            ## choose an optimizer
        
            #### create distributed train + val steps
            #if wandb.config.optimizer in ['adamW', 'adam']:
            if args.lr_schedule == 'cosine_decay_w_warmup':
                learning_rate_fn = schedulers.cosine_decay_w_warmup(peak_lr = wandb.config.lr_base,
                                                                    initial_learning_rate=wandb.config.min_lr,
                                                                    total_steps=wandb.config.train_steps * wandb.config.num_epochs,
                                                                    warmup_steps=wandb.config.warmup_frac * wandb.config.train_steps * wandb.config.num_epochs,
                                                                    hold_peak_rate_steps=wandb.config.train_steps)
                weight_decay_fn = schedulers.cosine_decay_w_warmup(peak_lr = wandb.config.lr_base * wandb.config.weight_decay_frac,
                                                                   initial_learning_rate=wandb.config.min_lr * wandb.config.weight_decay_frac,
                                                                   total_steps=wandb.config.train_steps * wandb.config.num_epochs,
                                                                   warmup_steps=wandb.config.warmup_frac * wandb.config.train_steps * wandb.config.num_epochs,
                                                                   hold_peak_rate_steps=wandb.config.train_steps)
            else:
                ValueError('schedule not implemented yet')
                
            if wandb.config.optimizer == 'adamW':
                optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate_fn,
                                                 weight_decay=weight_decay_fn)
            elif wandb.config.optimizer == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
            elif wandb.config.optimizer == 'adabelief':
                optimizer = tfa.optimizers.AdaBelief(learning_rate=wandb.config.lr_base,
                                                     weight_decay=wandb.config.weight_decay_frac,
                                                     warmup_proportion=wandb.config.warmup_frac,
                                                     epsilon=wandb.config.epsilon,
                                                     rectify=wandb.config.rectify,
                                                     min_lr=wandb.config.min_lr,
                                                     total_steps=wandb.config.train_steps*wandb.config.num_epochs)
                
                optimizer = tfa.optimizers.Lookahead(optimizer, 
                                                     sync_period=wandb.config.sync_period, 
                                                     slow_step_size=wandb.config.slow_step_frac)


            #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif wandb.config.optimizer == 'adafactor':
                optimizer = optimizers.AdafactorOptimizer(multiply_by_parameter_scale=False,
                                                          learning_rate = learning_rate_fn)
            else:
                ValueError('optimizer not implemented')
            
            metric_dict = {}
            if orgs == ["hg"]:
                train_step, val_step, metric_dict = training_utils.return_train_val_functions_hg(model,
                                                                                              optimizer,
                                                                                              strategy,
                                                                                              metric_dict, 
                                                                                              wandb.config.train_steps,
                                                                                              wandb.config.val_steps,
                                                                                              GLOBAL_BATCH_SIZE,
                                                                                              wandb.config.gradient_clip, 
                                                                                                 args.quant_type)
            elif (("hg" in orgs) and ("mm" in orgs)):
                train_step, val_step, metric_dict = training_utils.return_train_val_functions_hg_mm(model,
                                                                                              optimizer,
                                                                                              strategy,
                                                                                              metric_dict, 
                                                                                              wandb.config.train_steps,
                                                                                              wandb.config.val_steps,
                                                                                              GLOBAL_BATCH_SIZE,
                                                                                              wandb.config.gradient_clip,
                                                                                              wandb.config.quant_type)
                
            elif (("hg" in orgs) and ("mm" in orgs) and ("rm" in orgs)):
                train_step, val_step, metric_dict = training_utils.return_train_val_functions_hg_mm_rm(model,
                                                                                              optimizer,
                                                                                              strategy,
                                                                                              metric_dict, 
                                                                                              wandb.config.train_steps,
                                                                                              wandb.config.val_steps,
                                                                                              GLOBAL_BATCH_SIZE,
                                                                                              wandb.config.gradient_clip,
                                                                                              wandb.config.quant_type)
            
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
                if orgs == ["hg"]:
                    train_step(data_dict_tr['hg'])
                    val_step(data_dict_val['hg'])
                    y_trues = metric_dict['hg_corr_stats'].result()['y_trues'].numpy()
                    y_preds = metric_dict['hg_corr_stats'].result()['y_preds'].numpy()
                    cell_types = metric_dict['hg_corr_stats'].result()['cell_types'].numpy()
                    gene_map = metric_dict['hg_corr_stats'].result()['gene_map'].numpy()
                    
                    val_losses.append(metric_dict['hg_val'].result().numpy())
                    val_pearsons.append(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy())

                    overall_corr, cell_corr, gene_corr, overall_fig,cell_fig,gene_fig = training_utils.make_plots(y_trues,y_preds, cell_types,gene_map, 'hg')
                    
                    wandb.log({'hg_train_loss': metric_dict['hg_tr'].result().numpy(),
                               'hg_val_loss': metric_dict['hg_val'].result().numpy(),
                               'hg_bin_level_rho': metric_dict['hg_corr_stats'].result()['pearsonR'].numpy(),
                               'hg_gene_level_rho': overall_corr,
                               'hg_mean_cell_rho': cell_corr,
                               'hg_mean_gene_rho': gene_corr},
                              step=epoch_i)
                    wandb.log({"overall_gene_level":overall_fig},step=epoch_i)
                    wandb.log({"cell level corr":cell_fig},step=epoch_i)
                    wandb.log({"gene level corrs/var":gene_fig},step=epoch_i)
                    
                    
                elif (("hg" in orgs) and ("mm" in orgs)):
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

                    overall_corr, cell_corr, gene_corr, overall_fig,cell_fig,gene_fig = training_utils.make_plots(y_trues, 
                                                                                                                  y_preds, 
                                                                                                                  cell_types,
                                                                                                                  gene_map, 'hg')
                    
                    wandb.log({'hg_train_loss': metric_dict['hg_tr'].result().numpy(),
                               'hg_val_loss': metric_dict['hg_val'].result().numpy(),
                               'hg_bin_level_rho': metric_dict['hg_corr_stats'].result()['pearsonR'].numpy(),
                               'hg_gene_level_rho': overall_corr,
                               'hg_mean_cell_rho': cell_corr,
                               'hg_mean_gene_rho': gene_corr},
                              step=epoch_i)
                    wandb.log({"hg_overall_gene_level":overall_fig},step=epoch_i)
                    wandb.log({"hg_cell_level_corr":cell_fig},step=epoch_i)
                    wandb.log({"hg_gene_level_corrs/var":gene_fig},step=epoch_i)
                    
                    y_trues = metric_dict['mm_corr_stats'].result()['y_trues'].numpy()
                    y_preds = metric_dict['mm_corr_stats'].result()['y_preds'].numpy()
                    cell_types = metric_dict['mm_corr_stats'].result()['cell_types'].numpy()
                    gene_map = metric_dict['mm_corr_stats'].result()['gene_map'].numpy()
                    
                    val_losses.append(metric_dict['mm_val'].result().numpy())
                    val_pearsons.append(metric_dict['mm_corr_stats'].result()['pearsonR'].numpy())

                    overall_corr, cell_corr, gene_corr, overall_fig,cell_fig,gene_fig = training_utils.make_plots(y_trues,
                                                                                                                  y_preds, 
                                                                                                                  cell_types,
                                                                                                                  gene_map, 'mm')
                    
                    wandb.log({'mm_train_loss': metric_dict['mm_tr'].result().numpy(),
                               'mm_val_loss': metric_dict['mm_val'].result().numpy(),
                               'mm_bin_level_rho': metric_dict['mm_corr_stats'].result()['pearsonR'].numpy(),
                               'mm_gene_level_rho': overall_corr,
                               'mm_mean_cell_rho': cell_corr,
                               'mm_mean_gene_rho': gene_corr},
                              step=epoch_i)
                    wandb.log({"mm_overall_gene_level":overall_fig},step=epoch_i)
                    wandb.log({"mm_cell_level_corr":cell_fig},step=epoch_i)
                    wandb.log({"mm_gene_level_corrs/var":gene_fig},step=epoch_i)
                    
                elif (("hg" in orgs) and ("mm" in orgs) and ("rm" in orgs)):
                    train_step(data_dict_tr['hg'],
                               data_dict_tr['mm'],
                               data_dict_tr['rm'])
                    val_step(data_dict_val['hg'],
                             data_dict_val['mm'],
                             data+dict_val['rm'])
                    y_trues = metric_dict['hg_corr_stats'].result()['y_trues'].numpy()
                    y_preds = metric_dict['hg_corr_stats'].result()['y_preds'].numpy()
                    cell_types = metric_dict['hg_corr_stats'].result()['cell_types'].numpy()
                    gene_map = metric_dict['hg_corr_stats'].result()['gene_map'].numpy()
                    
                    val_losses.append(metric_dict['hg_val'].result().numpy())
                    val_pearsons.append(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy())
                    
                    overall_corr, cell_corr, gene_corr, overall_fig,cell_fig,gene_fig = training_utils.make_plots(y_trues, 
                                                                                                                  y_preds, 
                                                                                                                  cell_types,
                                                                                                                  gene_map, 'hg')
                    
                    wandb.log({'hg_train_loss': metric_dict['hg_tr'].result().numpy(),
                               'hg_val_loss': metric_dict['hg_val'].result().numpy(),
                               'hg_bin_level_rho_overall': metric_dict['hg_corr_stats'].result()['pearsonR'].numpy(),
                               'hg_gene_level_rho_overall': overall_corr,
                               'hg_mean_cell_rho': cell_corr,
                               'hg_mean_gene_rho': gene_corr},
                              step=epoch_i)
                    wandb.log({"hg_overall_gene_level":overall_fig},step=epoch_i)
                    wandb.log({"hg_cell_level_corr":cell_fig},step=epoch_i)
                    wandb.log({"hg_gene_level_corrs/var":gene_fig},step=epoch_i)
                    print('hg_tr', metric_dict['hg_tr'].result().numpy())
                    print('hg_val', metric_dict['hg_val'].result().numpy())
                    print('hg_pearsonR_bin', metric_dict['hg_corr_stats'].result()['pearsonR'].numpy())
                    
                    y_trues = metric_dict['mm_corr_stats'].result()['y_trues'].numpy()
                    y_preds = metric_dict['mm_corr_stats'].result()['y_preds'].numpy()
                    cell_types = metric_dict['mm_corr_stats'].result()['cell_types'].numpy()
                    gene_map = metric_dict['mm_corr_stats'].result()['gene_map'].numpy()
                

                    overall_corr, cell_corr, gene_corr, overall_fig,cell_fig,gene_fig = training_utils.make_plots(y_trues,
                                                                                                                  y_preds, 
                                                                                                                  cell_types,
                                                                                                                  gene_map, 'mm')
                    
                    wandb.log({'mm_train_loss': metric_dict['mm_tr'].result().numpy(),
                               'mm_val_loss': metric_dict['mm_val'].result().numpy(),
                               'mm_bin_level_rho_overall': metric_dict['mm_corr_stats'].result()['pearsonR'].numpy(),
                               'mm_gene_level_rho_overall': overall_corr,
                               'mm_mean_cell_rho': cell_corr,
                               'mm_mean_gene_rho': gene_corr},
                              step=epoch_i)
                    wandb.log({"mm_overall_gene_level":overall_fig},step=epoch_i)
                    wandb.log({"mm_cell_level_corr":cell_fig},step=epoch_i)
                    wandb.log({"mm_gene_level_corrs/var":gene_fig},step=epoch_i)
                    
                    print('mm_tr', metric_dict['mm_tr'].result().numpy())
                    print('mm_val', metric_dict['mm_val'].result().numpy())
                    print('mm_pearsonR_bin', metric_dict['mm_corr_stats'].result()['pearsonR'].numpy())
                      
                    y_trues = metric_dict['rm_corr_stats'].result()['y_trues'].numpy()
                    y_preds = metric_dict['rm_corr_stats'].result()['y_preds'].numpy()
                    cell_types = metric_dict['rm_corr_stats'].result()['cell_types'].numpy()
                    gene_map = metric_dict['rm_corr_stats'].result()['gene_map'].numpy()
                

                    overall_corr, cell_corr, gene_corr, overall_fig,cell_fig,gene_fig = training_utils.make_plots(y_trues,
                                                                                                                  y_preds, 
                                                                                                                  cell_types,
                                                                                                                  gene_map, 'rm')
                    
                    wandb.log({'rm_train_loss': metric_dict['rm_tr'].result().numpy(),
                               'rm_val_loss': metric_dict['rm_val'].result().numpy(),
                               'rm_bin_level_rho_overall': metric_dict['rm_corr_stats'].result()['pearsonR'].numpy(),
                               'rm_gene_level_rho_overall': overall_corr,
                               'rm_mean_cell_rho': cell_corr,
                               'rm_mean_gene_rho': gene_corr},
                              step=epoch_i)
                    wandb.log({"rm_overall_gene_level":overall_fig},step=epoch_i)
                    wandb.log({"rm_cell_level_corr":cell_fig},step=epoch_i)
                    wandb.log({"rm_gene_level_corrs/var":gene_fig},step=epoch_i)
                    print('rm_tr', metric_dict['rm_tr'].result().numpy())
                    print('rm_val', metric_dict['rm_val'].result().numpy())
                    print('rm_pearsonR_bin', metric_dict['rm_corr_stats'].result()['pearsonR'].numpy())
                    
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
        