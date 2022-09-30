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
import src.aformer_dual as aformer
import src.metrics as metrics
import src.optimizers as optimizers
import src.schedulers as schedulers
import src.utils as utils

import training_utils_aformer_TF_expression as training_utils
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
                'dropout_rate': {
                    'values': [float(x) for x in args.dropout_rate.split(',')]
                },
                'attention_dropout_rate': {
                    'values': [float(x) for x in args.attention_dropout_rate.split(',')]
                },
                'lr_base': {
                    'values':[float(x) for x in args.lr_base.split(',')]
                },
                'gradient_clip': {
                    'values': [float(x) for x in args.gradient_clip.split(',')]
                },
                'weight_decay_frac': {
                    'values': [float(x) for x in args.weight_decay_frac.split(',')]
                },
                'transformer_depth_1':{
                    'values': [int(x) for x in args.transformer_depth_1.split(',')]
                },
                'transformer_depth_2':{
                    'values': [int(x) for x in args.transformer_depth_2.split(',')]
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
                'dim': {
                    'values':[args.dim]
                },
                'epsilon': {
                    'values':[args.epsilon]
                },
                'load_init': {
                    'values':[parse_bool_str(x) for x in args.load_init.split(',')]
                },
                'train_mode': {
                    'values':[args.train_mode]
                },
                'freeze_conv_layers': {
                    'values':[parse_bool_str(x) for x in args.freeze_conv_layers.split(',')]
                },
                'use_tf_module': {
                    'values':[parse_bool_str(x) for x in args.use_tf_module.split(',')]
                },
                'rna_loss_scale': {
                    'values':[float(x) for x in args.rna_loss_scale.split(',')]
                }
                
            }
    }

    
    def sweep_train(config_defaults=None):
        # Set default values
        # Specify the other hyperparameters to the configuration, if any

        ## tpu initialization
        strategy = training_utils.tf_tpu_initialize(args.tpu_name)
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
            wandb.config.train_steps=args.train_steps
            wandb.config.val_steps=args.val_steps
            wandb.config.val_steps_ho=args.val_steps_ho
            wandb.config.batch_size=args.batch_size
            wandb.config.warmup_frac=args.warmup_frac
            wandb.config.total_steps=args.total_steps
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            wandb.config.max_shift=args.max_shift
            
            wandb.run.name = '_'.join(['load_init-' + str(wandb.config.load_init),
                                       str(wandb.config.train_mode),
                                       'freeze-' + str(wandb.config.freeze_conv_layers),
                                       'TF_in-' + str(wandb.config.use_tf_module),
                                       'LR-' + str(wandb.config.lr_base),
                                       'T.1-' + str(wandb.config.transformer_depth_1),
                                       'T.2-' + str(wandb.config.transformer_depth_2),
                                       'ST-' + str(wandb.config.shared_transformer_depth),
                                       'TD-' + str(wandb.config.pre_transf_channels),
                                       'D-' + str(wandb.config.dropout_rate),
                                       'AD-' + str(wandb.config.attention_dropout_rate),
                                       'HS-' + str(wandb.config.hidden_size),
                                       'RNA_loss_scale-' + str(wandb.config.rna_loss_scale)])

            '''
            TPU init options
            '''
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.FILE
            options.deterministic=False
            options.experimental_threading.max_intra_op_parallelism=1
            mixed_precision.set_global_policy('mixed_bfloat16')
            #options.experimental_slack = True


            NUM_REPLICAS = strategy.num_replicas_in_sync
            BATCH_SIZE_PER_REPLICA=wandb.config.batch_size
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS
            print('global batch size:', GLOBAL_BATCH_SIZE)
            num_train=977500
            num_val=153000
            num_val_ho=96#4192000

            wandb.config.update({"train_steps": num_train // (GLOBAL_BATCH_SIZE*3)},
                                allow_val_change=True)
            wandb.config.update({"val_steps" : num_val // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"val_steps_ho" : num_val_ho // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"total_steps": 100 * num_train // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            

            cell_type_map_df = pd.read_csv(args.cell_type_map_file,sep='\t',header=None)
            cell_type_map_df.columns = ['cell_type', 
                                        'cell_type_encoding']
            gene_map_df = pd.read_csv(args.gene_map_file,sep='\t')

            gene_map_df.columns = ['ensembl_id', 
                                'gene_encoding']
            
            gene_symbol_df = pd.read_csv(args.gene_symbol_map_file,sep='\t')
            gene_symbol_df.columns = ['ensembl_id',
                                      'symbol']
            
            data_dict_tr,data_dict_val = \
                    training_utils.return_distributed_iterators(wandb.config.gcs_path,
                                                                wandb.config.gcs_path_val_ho,
                                                                GLOBAL_BATCH_SIZE,
                                                                196608,
                                                                1536,
                                                                128,
                                                                wandb.config.max_shift,
                                                                args.num_parallel,
                                                                args.num_epochs,
                                                                strategy,
                                                                options)
            
            if wandb.config.load_init:
                inits=training_utils.get_initializers(args.checkpoint_path)
            else:
                inits=None

            model = aformer.aformer(kernel_transformation=wandb.config.kernel_transformation,
                                    dropout_rate=wandb.config.dropout_rate,
                                    attention_dropout_rate=wandb.config.attention_dropout_rate,
                                    input_length=196608,
                                    atac_output_length=896,
                                    num_heads=wandb.config.num_heads,
                                    numerical_stabilizer=0.0000001,
                                    nb_random_features=wandb.config.num_random_features,
                                    hidden_size=wandb.config.hidden_size,
                                    d_model=wandb.config.hidden_size,
                                    dim=wandb.config.dim,
                                    transformer_depth_1=wandb.config.transformer_depth_1,
                                    transformer_depth_2=wandb.config.transformer_depth_2,
                                    shared_transformer_depth=wandb.config.shared_transformer_depth,
                                    pre_transf_channels=wandb.config.pre_transf_channels,
                                    TF_inputs=wandb.config.TF_inputs,
                                    inits=inits,
                                    load_init=wandb.config.load_init,
                                    freeze_conv_layers=wandb.config.freeze_conv_layers)

            scheduler= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base,
                decay_steps=wandb.config.total_steps, alpha=0.10)
            scheduler=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps,
                                         decay_schedule_fn=scheduler)

            scheduler_wd= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.weight_decay_frac,
                decay_steps=wandb.config.total_steps, alpha=0.0)
            scheduler_wd=optimizers.WarmUp(initial_learning_rate=wandb.config.weight_decay_frac,
                                         warmup_steps=int(wandb.config.warmup_frac*wandb.config.total_steps),
                                         decay_schedule_fn=scheduler)

            optimizer = tfa.optimizers.AdamW(learning_rate=scheduler,
                                             weight_decay=scheduler_wd)
            
            
            metric_dict = {}
            if wandb.config.use_tf_module:
                train_step_atac, val_step_atac,\
                    train_step_rna,val_step_rna,\
                        train_step_both,val_step_both,\
                            metric_dict = \
                training_utils.return_train_val_functions(model,
                                                          optimizer,
                                                          strategy,
                                                          metric_dict,
                                                          wandb.config.train_steps,
                                                          wandb.config.val_steps,
                                                          wandb.config.val_steps_ho,
                                                          GLOBAL_BATCH_SIZE,
                                                          wandb.config.gradient_clip,
                                                          rna_loss_scale=wandb.config.rna_loss_scale)
            else:
                train_step_atac, val_step_atac,\
                    train_step_rna,val_step_rna,\
                        train_step_both,val_step_both,\
                            metric_dict = \
                training_utils.return_train_val_functions_notf(model,
                                                               optimizer,
                                                               strategy,
                                                               metric_dict,
                                                               wandb.config.train_steps,
                                                               wandb.config.val_steps,
                                                               wandb.config.val_steps_ho,
                                                               GLOBAL_BATCH_SIZE,
                                                               wandb.config.gradient_clip,
                                                               rna_loss_scale=wandb.config.rna_loss_scale)


            global_step = 0
            val_losses = []
            val_pearsons = []
            val_R2 = []
            patience_counter = 0
            stop_criteria = False
            best_epoch = 0
            train_mode = wandb.config.train_mode
            
            for epoch_i in range(1, wandb.config.num_epochs+1):
                print('starting epoch_', str(epoch_i))
                start = time.time()
                if train_mode == 'atac_only':
                    train_step_atac(data_dict_tr)
                elif train_mode == 'rna_only':
                    train_step_rna(data_dict_tr)
                else:
                    train_step_both(data_dict_tr)
                
                end = time.time()
                duration = (end - start) / 60.
                print('completed epoch ' + str(epoch_i))
                print('hg_train_loss: ' + str(metric_dict['hg_tr'].result().numpy()))
                wandb.log({'hg_train_loss': metric_dict['hg_tr'].result().numpy()},
                          step=epoch_i)

                print('training duration(mins): ' + str(duration))
                
                start = time.time()
                if train_mode == 'atac_only':
                    val_step_atac(data_dict_val)
                    val_pearsons.append(metric_dict['hg_pearsonsR_ATAC'].result()['PearsonR'].numpy())
                    pearsons_atac = metric_dict['hg_pearsonsR_ATAC'].result()['PearsonR'].numpy()
                    pearsons_R2= metric_dict['hg_R2_ATAC'].result()['R2'].numpy()
                    print('hg_ATAC_pearsons: ' + str(pearsons_atac))
                    print('hg_ATAC_R2: ' + str(pearsons_R2))
                    print('returned correlation metrics from make plots function')
                    wandb.log({'hg_ATAC_pearsons': pearsons_atac,
                               'hg_ATAC_R2': pearsons_R2},
                              step=epoch_i)
                elif train_mode == 'rna_only':
                    val_step_rna(data_dict_val)
                    val_pearsons.append(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy())
                    print('hg_RNA_pearson: ' + str(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy()))
                    print('hg_RNA_R2: ' + str(metric_dict['hg_corr_stats'].result()['R2'].numpy()))
                else:
                    val_step_both(data_dict_val)
                    val_pearsons.append(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy()) ## for joint we care more about RNA
                    print('hg_ATAC_pearsons: ' + str(metric_dict['hg_pearsonsR_ATAC'].result()['PearsonR'].numpy()))
                    print('hg_ATAC_R2: ' + str(metric_dict['hg_R2_ATAC'].result()['R2'].numpy()))
                    print('hg_RNA_pearson: ' + str(metric_dict['hg_corr_stats'].result()['pearsonR'].numpy()))
                    print('hg_RNA_R2: ' + str(metric_dict['hg_corr_stats'].result()['R2'].numpy()))
                    
                val_losses.append(metric_dict['hg_val'].result().numpy())
                
                print('hg_val_loss: ' + str(metric_dict['hg_val'].result().numpy()))
                wandb.log({'hg_val_loss': metric_dict['hg_val'].result().numpy()},
                          step=epoch_i)
                
                if train_mode != 'atac_only':
                    y_trues = metric_dict['hg_corr_stats'].result()['y_trues'].numpy()
                    y_preds = metric_dict['hg_corr_stats'].result()['y_preds'].numpy()
                    cell_types = metric_dict['hg_corr_stats'].result()['cell_types'].numpy()
                    gene_map = metric_dict['hg_corr_stats'].result()['gene_map'].numpy()

                    dataframes,figures,corrs_overall= \
                                                    training_utils.make_plots(y_trues,y_preds,
                                                                            cell_types,gene_map, 
                                                                            'hg',cell_type_map_df, 
                                                                            gene_map_df, 
                                                                            gene_symbol_df)
                    correlations_cells_df, correlations_genes_df=dataframes
                    fig_cell_spec, fig_gene_spec=figures 

                    overall_gene_level_corr_sp, gene_spec_median_corr_sp, cell_specific_corrs_sp=\
                        corrs_overall
                    print('returned correlation metrics from make plots function')
                    wandb.log({'hg_overall_rho_sp': overall_gene_level_corr_sp,
                               'hg_median_cell_rho_sp': cell_specific_corrs_sp,
                               'hg_median_gene_rho_sp': gene_spec_median_corr_sp},
                              step=epoch_i)
                end = time.time()
                duration = (end - start) / 60.

                print('validation duration(mins): ' + str(duration))
                
                if (epoch_i > 2):
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
                                                        saved_model_basename=wandb.config.model_save_basename)
                #plt.close('all')
                print('patience counter at: ' + str(patience_counter))
                for key, item in metric_dict.items():
                    item.reset_state()
                if stop_criteria:
                    print('early stopping at: epoch ' + str(epoch_i))
                    break
                    
            print('saving model at: epoch ' + str(epoch_i))
            print('best model was at: epoch ' + str(best_epoch))
            model.save_weights(wandb.config.model_save_dir + "/" + wandb.config.model_save_basename + "_" + wandb.run.name + "/final/saved_model")
            file_name = wandb.config.model_save_basename + "." + str(epoch_i) + ".val.out.tsv"
            df = pd.DataFrame({'y_trues_ho':y_trues_ho, 'y_preds_ho':y_preds_ho,
                                    'cell_types_ho': cell_types_ho, 'gene_map_ho': gene_map_ho})
            df.to_csv(file_name, sep='\t',header=True,index=False)
            command = "gsutil cp " + file_name + " " + wandb.config.model_save_dir
            subprocess.call(command,shell=True)

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

##########################################################################
if __name__ == '__main__':
    main()
        
