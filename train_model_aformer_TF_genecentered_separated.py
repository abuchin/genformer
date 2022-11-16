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
import pandas as pd
from datetime import datetime
import random

#import logging
#from silence_tensorflow import silence_tensorflow
#silence_tensorflow()
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

import training_utils_aformer_TF_genecentered_separated as training_utils
import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr  
from scipy import stats

def parse_bool_str(input_str):
    if input_str == 'False':
        return False
    else:
        return True
    

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
                    'values': [args.input_length]
                },
                'dropout_rate': {
                    'values': [float(x) for x in args.dropout_rate.split(',')]
                },
                'attention_dropout_rate': {
                    'values': [float(x) for x in args.attention_dropout_rate.split(',')]
                },
                'tf_dropout_rate': {
                    'values': [float(x) for x in args.tf_dropout_rate.split(',')]
                },
                'pointwise_dropout_rate': {
                    'values': [float(x) for x in args.pointwise_dropout_rate.split(',')]
                },
                'lr_base1': {
                    'values':[float(x) for x in args.lr_base1.split(',')]
                },
                'lr_base2': {
                    'values':[float(x) for x in args.lr_base2.split(',')]
                },
                'gradient_clip': {
                    'values': [float(x) for x in args.gradient_clip.split(',')]
                },
                'decay_frac': {
                    'values': [float(x) for x in args.decay_frac.split(',')]
                },
                'weight_decay_frac': {
                    'values': [float(x) for x in args.weight_decay_frac.split(',')]
                },
                'shared_transformer_depth':{
                    'values': [int(x) for x in args.shared_transformer_depth.split(',')]
                },
                'pre_transf_channels':{
                    'values': [int(x) for x in args.pre_transf_channels.split(',')]
                },
                'num_heads':{
                    'values': [int(x) for x in args.num_heads.split(',')]
                },
                'TF_inputs':{
                    'values': [int(x) for x in args.TF_inputs.split(',')]
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
                'epsilon': {
                    'values':[args.epsilon]
                },
                'load_init': {
                    'values':[parse_bool_str(x) for x in args.load_init.split(',')]
                },
                'freeze_conv_layers': {
                    'values':[parse_bool_str(x) for x in args.freeze_conv_layers.split(',')]
                },
                'use_tf_module': {
                    'values':[parse_bool_str(x) for x in args.use_tf_module.split(',')]
                },
                'filter_list_seq': {
                    'values': [[int(x) for x in args.filter_list_seq.split(',')]]
                },
                #'filter_list_atac': {
                #    'values': [[int(x) for x in args.filter_list_atac.split(',')]]
                #},
                'dim_reduce_length': {
                    'values': [args.dim_reduce_length]
                },
                'loss_type': {
                    'values':[str(x) for x in args.loss_type.split(',')]
                }
                
            }
    }

    
    def sweep_train(config_defaults=None):
        # Set default values
        # Specify the other hyperparameters to the configuration, if any

        ## tpu initialization
        strategy = training_utils.tf_tpu_initialize(args.tpu_name,args.tpu_zone)
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
            wandb.config.num_epochs=args.num_epochs
            wandb.config.train_examples=args.train_examples
            wandb.config.val_examples=args.val_examples
            wandb.config.val_examples_ho=args.val_examples_ho
            wandb.config.batch_size=args.batch_size
            wandb.config.warmup_frac=args.warmup_frac
            wandb.config.total_steps=args.total_steps
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            wandb.config.max_shift=args.max_shift
            
            run_name = '_'.join(['input_length-' + str(wandb.config.input_length),
                                        'load_init-' + str(wandb.config.load_init),
                                       'freeze-' + str(wandb.config.freeze_conv_layers),
                                       'TF_in-' + str(wandb.config.use_tf_module),
                                       'LR-' + str(wandb.config.lr_base),
                                       'ST-' + str(wandb.config.shared_transformer_depth),
                                       'TD-' + str(wandb.config.pre_transf_channels),
                                       'D-' + str(wandb.config.dropout_rate),
                                       'AD-' + str(wandb.config.attention_dropout_rate),
                                       'HS-' + str(wandb.config.hidden_size)])
            wandb.run.name = run_name
            base_name = wandb.config.model_save_basename + "_" + run_name

            '''
            TPU init options
            '''

            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.FILE
            options.deterministic=False
            #options.experimental_threading.max_intra_op_parallelism=1
            mixed_precision.set_global_policy('mixed_bfloat16')
            #options.experimental_slack = True


            NUM_REPLICAS = strategy.num_replicas_in_sync
            BATCH_SIZE_PER_REPLICA=wandb.config.batch_size
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS
            print('global batch size:', GLOBAL_BATCH_SIZE)
            
            num_train=wandb.config.train_examples
            num_val=wandb.config.val_examples
            num_val_ho=wandb.config.val_examples_ho#4192000

            wandb.config.update({"train_steps": num_train // (GLOBAL_BATCH_SIZE)},
                                allow_val_change=True)
            wandb.config.update({"val_steps" : num_val // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"val_steps_ho" : num_val_ho // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"total_steps": num_train // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            

            data_dict_tr,data_dict_val,data_dict_val_ho = \
                    training_utils.return_distributed_iterators(wandb.config.gcs_path,
                                                                wandb.config.gcs_path_val_ho,
                                                                GLOBAL_BATCH_SIZE,
                                                                wandb.config.input_length,
                                                                wandb.config.dim_reduce_length,
                                                                128,
                                                                wandb.config.max_shift,
                                                                args.num_parallel,
                                                                args.num_epochs,
                                                                strategy,
                                                                options)

            print('created dataset iterators')
            if wandb.config.load_init:
                inits=training_utils.get_initializers(args.enformer_checkpoint_path)
                wandb.config.update({"filter_list_seq": [768, 896, 1024, 1152, 1280, 1536]},
                                    allow_val_change=True)
            else:
                inits=None

            model = aformer.aformer(kernel_transformation=wandb.config.kernel_transformation,
                                    dropout_rate=wandb.config.dropout_rate,
                                    attention_dropout_rate=wandb.config.attention_dropout_rate,
                                    tf_dropout_rate=wandb.config.tf_dropout_rate,
                                    pointwise_dropout_rate=wandb.config.pointwise_dropout_rate,
                                    input_length=wandb.config.input_length,
                                    dim_reduce_length=wandb.config.dim_reduce_length,
                                    num_heads=wandb.config.num_heads,
                                    numerical_stabilizer=0.0000001,
                                    nb_random_features=wandb.config.num_random_features,
                                    hidden_size=wandb.config.hidden_size,
                                    d_model=wandb.config.hidden_size,
                                    dim=wandb.config.hidden_size // wandb.config.num_heads,
                                    max_seq_length=768,
                                    rel_pos_bins=768,
                                    norm=True,
                                    use_rot_emb = True,
                                    use_mask_pos = False,
                                    normalize = True,
                                    shared_transformer_depth=wandb.config.shared_transformer_depth,
                                    pre_transf_channels=wandb.config.pre_transf_channels,
                                    TF_inputs=wandb.config.TF_inputs,
                                    inits=inits,
                                    load_init=wandb.config.load_init,
                                    freeze_conv_layers=wandb.config.freeze_conv_layers,
                                    filter_list_seq=wandb.config.filter_list_seq)
            
            if args.checkpoint_path is not None:
                print('loading checkpointed model')
                options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
                checkpoint = tf.train.Checkpoint(module=model)#,options=options)
                tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
                latest = tf.train.latest_checkpoint(args.checkpoint_path)
                checkpoint.restore(latest,options=options).assert_existing_objects_matched()

            print('initialized model')
            scheduler1= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base1,
                decay_steps=wandb.config.total_steps, alpha=wandb.config.decay_frac)
            scheduler1=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base1,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps,
                                         decay_schedule_fn=scheduler1)
            
            scheduler1wd= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base1 * wandb.config.weight_decay_frac,
                decay_steps=wandb.config.total_steps, alpha=wandb.config.decay_frac)
            scheduler1wd=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base1 * wandb.config.weight_decay_frac,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps,
                                         decay_schedule_fn=scheduler1wd)
            
            optimizer1 = tfa.optimizers.AdamW(learning_rate=scheduler1,
                                              weight_decay=scheduler1wd)
            #####
            scheduler2= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base2,
                decay_steps=wandb.config.total_steps, alpha=wandb.config.decay_frac)
            scheduler2=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base2,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps,
                                         decay_schedule_fn=scheduler2)
            scheduler2wd= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base2 * wandb.config.weight_decay_frac,
                decay_steps=wandb.config.total_steps, alpha=wandb.config.decay_frac)
            scheduler2wd=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base2 * wandb.config.weight_decay_frac,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps,
                                         decay_schedule_fn=scheduler2wd)
            
            optimizer2 = tfa.optimizers.AdamW(learning_rate=scheduler2,
                                              weight_decay=scheduler2wd)
            #####
            optimizers_in = optimizer1,optimizer2
            

            metric_dict = {}
            train_step,val_step,val_step_ho,build_step,metric_dict = \
                training_utils.return_train_val_functions(model,
                                                      optimizers_in,
                                                      strategy,
                                                      metric_dict,
                                                      wandb.config.train_steps,
                                                      wandb.config.val_steps,
                                                      wandb.config.val_steps_ho,
                                                      GLOBAL_BATCH_SIZE,
                                                      wandb.config.gradient_clip,
                                                      wandb.config.batch_size,
                                                      loss_fn_main=wandb.config.loss_type,
                                                      use_tf=wandb.config.use_tf_module)
                

            print('finished loading training/val loop functions')
            global_step = 0
            val_losses = []
            val_pearsons = []
            val_R2 = []
            patience_counter = 0
            stop_criteria = False
            best_epoch = 0
            for epoch_i in range(1, wandb.config.num_epochs+1):
                if epoch_i == 1:
                    print('building model')
                    build_step(data_dict_val)
                    print('built model')
                    total_params = 0
                    for k in model.trainable_variables:
                        var = k.values[0]
                        total_params += tf.size(var)
                    print('total params: ' + str(total_params)) 
                
                print('starting epoch_', str(epoch_i))
                start = time.time()
                train_step(data_dict_tr)
                
                end = time.time()
                duration = (end - start) / 60.
                print('completed epoch ' + str(epoch_i))
                print('hg_train_loss: ' + str(metric_dict['hg_tr'].result().numpy()))
                wandb.log({'hg_train_loss': metric_dict['hg_tr'].result().numpy()},
                          step=epoch_i)

                print('training duration(mins): ' + str(duration))
                
                start = time.time()
                ##### validation
                val_step(data_dict_val)
                
                print('hg_val_loss: ' + str(metric_dict['hg_val'].result().numpy()))
                val_losses.append(metric_dict['hg_val'].result().numpy())
                wandb.log({'hg_val_loss': metric_dict['hg_val'].result().numpy()},
                          step=epoch_i)
                
            
                val_pearsons.append(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy())
                print('hg_RNA_pearson: ' + str(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy()))
                print('hg_RNA_R2: ' + str(metric_dict['hg_corr_stats'].result()['R2'].numpy()))

                y_trues = metric_dict['hg_corr_stats'].result()['y_trues'].numpy()
                y_preds = metric_dict['hg_corr_stats'].result()['y_preds'].numpy()
                cell_types = metric_dict['hg_corr_stats'].result()['cell_types'].numpy()
                gene_map = metric_dict['hg_corr_stats'].result()['gene_map'].numpy()

                corrs_overall= training_utils.make_plots(y_trues,
                                                         y_preds,
                                                         cell_types,
                                                         gene_map,
                                                         val_holdout=False)
                
                overall_gene_level_corr_sp, \
                    cell_spec_median_corrs_sp, \
                    cell_spec_median_corrs, \
                    gene_spec_median_corrs_sp, \
                    gene_spec_median_corrs = corrs_overall
                
                print('returned correlation metrics from make plots function')
                print('hg_median_gene_rho_sp: ' + str(gene_spec_median_corrs_sp))

                wandb.log({'hg_overall_rho_sp': overall_gene_level_corr_sp,
                           'hg_median_cell_rho_sp': cell_spec_median_corrs_sp,
                           'hg_median_cell_rho': cell_spec_median_corrs,
                           'hg_median_gene_rho_sp': gene_spec_median_corrs_sp,
                           'hg_median_gene_rho': gene_spec_median_corrs},
                          step=epoch_i)
                #####
                
                
                ##### validation HOLDOUT
                val_step_ho(data_dict_val_ho)
                print('hg_RNA_pearson_HO: ' + str(metric_dict['hg_corr_stats_ho'].result()['pearsonR'].numpy()))
                print('hg_RNA_R2_HO: ' + str(metric_dict['hg_corr_stats_ho'].result()['R2'].numpy()))

                y_trues = metric_dict['hg_corr_stats_ho'].result()['y_trues'].numpy()
                y_preds = metric_dict['hg_corr_stats_ho'].result()['y_preds'].numpy()
                cell_types = metric_dict['hg_corr_stats_ho'].result()['cell_types'].numpy()
                gene_map = metric_dict['hg_corr_stats_ho'].result()['gene_map'].numpy()

                figures,corrs_overall= training_utils.make_plots(y_trues,y_preds,
                                                                 cell_types,gene_map,
                                                                 val_holdout=True,
                                                                 cell_type_map_overall=args.cell_type_map,
                                                                 gene_map_overall=args.gene_map_overall,
                                                                 gene_map_var_breakdown=args.gene_map_var_breakdown)
                
                
                fig_cell_spec, fig_gene_spec, fig_overall,fig_var_breakdown=figures 

                overall_gene_level_corr_sp, \
                    cell_spec_median_corrs_sp, \
                    cell_spec_median_corrs, \
                    gene_spec_median_corrs_sp, \
                    gene_spec_median_corrs = corrs_overall
                
                print('returned correlation metrics from make plots function')
                print('hg_median_gene_rho_sp_ho: ' + str(gene_spec_median_corrs_sp))

                wandb.log({'hg_overall_rho_sp_ho': overall_gene_level_corr_sp,
                           'hg_median_cell_rho_sp_ho': cell_spec_median_corrs_sp,
                           'hg_median_cell_rho_ho': cell_spec_median_corrs,
                           'hg_median_gene_rho_sp_ho': gene_spec_median_corrs_sp,
                           'hg_median_gene_rho_ho': gene_spec_median_corrs},
                          step=epoch_i)
                wandb.log({'hg_OVERALL_correlation': fig_overall,
                           'hg_variance_breakdown': fig_var_breakdown,
                           'hg_cross_dataset_dist': fig_gene_spec,
                           'hg_cross_gene_dist': fig_cell_spec},
                          step=epoch_i)
                end = time.time()
                duration = (end - start) / 60.

                print('validation duration(mins): ' + str(duration))
                
                if (epoch_i > 1):
                    stop_criteria,patience_counter,best_epoch = \
                        training_utils.early_stopping(current_val_loss=val_losses[-1],
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
                                                        saved_model_basename=base_name)
                #plt.close('all')
                print('patience counter at: ' + str(patience_counter))
                for key, item in metric_dict.items():
                    item.reset_state()
                if stop_criteria:
                    print('early stopping at: epoch ' + str(epoch_i))
                    break
                    
            print('saving model at: epoch ' + str(epoch_i))
            print('best model was at: epoch ' + str(best_epoch))
            model.save_weights(wandb.config.model_save_dir + "/" + base_name + "/final/saved_model")

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

##########################################################################
if __name__ == '__main__':
    main()
        
