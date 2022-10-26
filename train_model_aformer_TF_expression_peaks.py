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
import src.aformer_dual_peaks as aformer
import src.metrics as metrics
import src.optimizers as optimizers
import src.schedulers as schedulers
import src.utils as utils

import training_utils_aformer_TF_expression_peaks as training_utils
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
                'lr_base3': {
                    'values':[float(x) for x in args.lr_base3.split(',')]
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
                'transformer_depth_rna':{
                    'values': [int(x) for x in args.transformer_depth_rna.split(',')]
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
                'train_mode': {
                    'values':[str(x) for x in args.train_mode.split(',')]
                },
                'freeze_conv_layers': {
                    'values':[parse_bool_str(x) for x in args.freeze_conv_layers.split(',')]
                },
                'use_tf_module': {
                    'values':[parse_bool_str(x) for x in args.use_tf_module.split(',')]
                },
                'rna_loss_scale': {
                    'values':[float(x) for x in args.rna_loss_scale.split(',')]
                },
                'filter_list': {
                    'values': [[int(x) for x in args.filter_list.split(',')]]
                },
                'atac_length_uncropped': {
                    'values': [args.atac_length_uncropped]
                },
                'atac_output_length': {
                    'values': [args.atac_output_length]
                },
                'lambda1': {
                    'values':[float(x) for x in args.lambda1.split(',')]
                },
                'lambda2': {
                    'values':[float(x) for x in args.lambda2.split(',')]
                },
                'lambda3': {
                    'values':[float(x) for x in args.lambda3.split(',')]
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
            
            wandb.config.atac_peaks_cropped = args.atac_peaks_cropped
            
            wandb.run.name = '_'.join(['input_length-' + str(wandb.config.input_length),
                                        'load_init-' + str(wandb.config.load_init),
                                       str(wandb.config.train_mode),
                                       'freeze-' + str(wandb.config.freeze_conv_layers),
                                       'TF_in-' + str(wandb.config.use_tf_module),
                                       'LR-' + str(wandb.config.lr_base),
                                       'ST-' + str(wandb.config.shared_transformer_depth),
                                       'RT.2-' + str(wandb.config.transformer_depth_rna),
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

            wandb.config.update({"train_steps": num_train // (GLOBAL_BATCH_SIZE * 4)},
                                allow_val_change=True)
            wandb.config.update({"val_steps" : num_val // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"val_steps_ho" : num_val_ho // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"total_steps": num_train // GLOBAL_BATCH_SIZE},
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

            data_dict_tr,data_dict_val,data_dict_val_ho = \
                    training_utils.return_distributed_iterators(wandb.config.gcs_path,
                                                                wandb.config.gcs_path_val_ho,
                                                                wandb.config.train_mode,
                                                                GLOBAL_BATCH_SIZE,
                                                                wandb.config.input_length,
                                                                wandb.config.atac_length_uncropped,
                                                                128,
                                                                wandb.config.max_shift,
                                                                args.num_parallel,
                                                                args.num_epochs,
                                                                strategy,
                                                                options)

            print('created dataset iterators')
            if wandb.config.load_init:
                inits=training_utils.get_initializers(args.enformer_checkpoint_path)
                wandb.config.update({"filter_list": [768, 896, 1024, 1152, 1280, 1536]},
                                    allow_val_change=True)
            else:
                inits=None

            model = aformer.aformer(kernel_transformation=wandb.config.kernel_transformation,
                                    dropout_rate=wandb.config.dropout_rate,
                                    attention_dropout_rate=wandb.config.attention_dropout_rate,
                                    tf_dropout_rate=wandb.config.tf_dropout_rate,
                                    pointwise_dropout_rate=wandb.config.pointwise_dropout_rate,
                                    input_length=wandb.config.input_length,
                                    atac_length_uncropped=wandb.config.atac_length_uncropped,
                                    atac_output_length=wandb.config.atac_output_length,
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
                                    transformer_depth_rna=wandb.config.transformer_depth_rna,
                                    shared_transformer_depth=wandb.config.shared_transformer_depth,
                                    pre_transf_channels=wandb.config.pre_transf_channels,
                                    TF_inputs=wandb.config.TF_inputs,
                                    inits=inits,
                                    load_init=wandb.config.load_init,
                                    freeze_conv_layers=wandb.config.freeze_conv_layers,
                                    filter_list=wandb.config.filter_list)
            
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
            scheduler3= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base3,
                decay_steps=wandb.config.total_steps, alpha=wandb.config.decay_frac)
            scheduler3=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base3,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps,
                                         decay_schedule_fn=scheduler3)
            scheduler3wd= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base3 * wandb.config.weight_decay_frac,
                decay_steps=wandb.config.total_steps, alpha=wandb.config.decay_frac)
            scheduler3wd=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base3 * wandb.config.weight_decay_frac,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps,
                                         decay_schedule_fn=scheduler3wd)
            optimizer3 = tfa.optimizers.AdamW(learning_rate=scheduler3,
                                              weight_decay=scheduler3wd)
            #####
            optimizers_in = optimizer1,optimizer2,optimizer3
            
            crop_size = (wandb.config.atac_length_uncropped - \
                             wandb.config.atac_output_length) // 2
            peaks_crop = (crop_size / wandb.config.atac_length_uncropped) * wandb.config.atac_peaks_cropped
            peaks_crop = int(peaks_crop)
            metric_dict = {}
            if wandb.config.use_tf_module:
                train_step_atac,val_step_atac,val_step_atac_ho,\
                    train_step_rna,val_step_rna,\
                        train_step_both,val_step_both,\
                            build_step, metric_dict = \
                training_utils.return_train_val_functions(model,
                                                          optimizers_in,
                                                          strategy,
                                                          metric_dict,
                                                          wandb.config.train_steps,
                                                          wandb.config.val_steps,
                                                          wandb.config.val_steps_ho,
                                                          GLOBAL_BATCH_SIZE,
                                                          wandb.config.gradient_clip,
                                                          wandb.config.atac_output_length,
                                                          crop_size,
                                                          wandb.config.atac_peaks_cropped,
                                                          peaks_crop,
                                                          wandb.config.batch_size,
                                                          wandb.config.lambda1,
                                                          wandb.config.lambda2,
                                                          wandb.config.lambda3,
                                                          loss_fn_main=wandb.config.loss_type,
                                                          rna_loss_scale=wandb.config.rna_loss_scale)
            else:
                train_step_atac,val_step_atac,val_step_atac_ho,\
                    train_step_rna,val_step_rna,\
                        train_step_both,val_step_both,\
                            build_step, metric_dict = \
                training_utils.return_train_val_functions_notf(model,
                                                               optimizers_in,
                                                               strategy,
                                                               metric_dict,
                                                               wandb.config.train_steps,
                                                               wandb.config.val_steps,
                                                               wandb.config.val_steps_ho,
                                                               GLOBAL_BATCH_SIZE,
                                                               wandb.config.gradient_clip,
                                                               wandb.config.atac_output_length,
                                                               crop_size,
                                                               wandb.config.atac_peaks_cropped,
                                                               peaks_crop,
                                                               wandb.config.batch_size,
                                                               wandb.config.lambda1,
                                                               wandb.config.lambda2,
                                                               wandb.config.lambda3,
                                                               loss_fn_main=wandb.config.loss_type,
                                                               rna_loss_scale=wandb.config.rna_loss_scale)
                

            print('finished loading training/val loop functions')
            global_step = 0
            val_losses = []
            val_pearsons = []
            val_R2 = []
            patience_counter = 0
            stop_criteria = False
            best_epoch = 0
            train_mode = wandb.config.train_mode
            print(wandb.config)
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
                print('hg_train_loss_bce: ' + str(metric_dict['hg_tr_bce'].result().numpy()))
                print('hg_train_loss_corr: ' + str(metric_dict['hg_tr_corr'].result().numpy()))
                print('hg_train_loss_main: ' + str(metric_dict['hg_tr_main'].result().numpy()))
                wandb.log({'hg_train_loss': metric_dict['hg_tr'].result().numpy(),
                          'hg_train_loss_bce': metric_dict['hg_tr_bce'].result().numpy(),
                          'hg_train_loss_corr': metric_dict['hg_tr_corr'].result().numpy(),
                          'hg_train_loss_main': metric_dict['hg_tr_main'].result().numpy()},
                          step=epoch_i)

                print('training duration(mins): ' + str(duration))
                
                start = time.time()
                if train_mode == 'atac_only':
                    val_step_atac(data_dict_val)

                    
                    
                    pearsons_atac = metric_dict['hg_pearsonsR_ATAC'].result()['PearsonR'].numpy()
                    
                    val_pearsons.append(pearsons_atac)
                    pearsons_R2= metric_dict['hg_R2_ATAC'].result()['R2'].numpy()
                    auprc = metric_dict['hg_val_AUPRC'].result().numpy()
                    print('hg_ATAC_pearsons: ' + str(pearsons_atac))
                    print('hg_ATAC_R2: ' + str(pearsons_R2))
                    print('hg_val_AUPRC: ' + str(auprc))

            
                    reg_true,reg_pred, peak_true,peak_pred,cell_types,intervals,count_sds = val_step_atac_ho(data_dict_val_ho)
                
                    cell_type_auprcs_median,cell_type_pearsons_median,ax_preds,ax_trues, fig_atac_ho, ax_preds_peak,ax_trues_peak = training_utils.make_atac_plots(reg_pred.numpy(),
                                                                                                 reg_true.numpy(),
                                                                                                 peak_pred.numpy(),
                                                                                                 peak_true.numpy(),
                                                                                                 count_sds.numpy(),
                                                                                                 cell_types.numpy(),
                                                                                                 intervals.numpy())
                    

                    print('cell_type_auprcs_median: ' + str(cell_type_auprcs_median))
                    print('cell_type_pearsons_median: ' + str(cell_type_pearsons_median))
                
                    pearsons_atac_ho = metric_dict['hg_pearsonsR_ATAC_ho'].result()['PearsonR'].numpy()
                    #val_pearsons.append(pearsons_atac_ho)
                    pearsons_R2_ho = metric_dict['hg_R2_ATAC_ho'].result()['R2'].numpy()
                    auprc_ho = metric_dict['hg_val_AUPRC_ho'].result().numpy()
                    print('hg_ATAC_pearsons_ho: ' + str(pearsons_atac_ho))
                    print('hg_ATAC_R2_ho: ' + str(pearsons_R2_ho))
                    print('hg_val_AUPRC_ho: ' + str(auprc_ho))
                    
                    print('returned correlation metrics from make plots function')
                    wandb.log({'hg_ATAC_pearsons': pearsons_atac,
                               'hg_ATAC_R2': pearsons_R2,
                               'hg_ATAC_auprc': auprc,
                               'hg_ATAC_pearsons_ho': pearsons_atac_ho,
                               'hg_ATAC_R2_ho': pearsons_R2_ho,
                               'hg_ATAC_auprc_ho': auprc_ho,
                               'cell_type_pearsons_median': cell_type_pearsons_median,
                               'cell_type_auprcs_median': cell_type_auprcs_median,
                               'pred atac high var':wandb.Image(ax_preds),
                               'true atac high var':wandb.Image(ax_trues),
                               'pred atac high var peaks':wandb.Image(ax_preds_peak),
                               'true atac high var peaks':wandb.Image(ax_trues_peak)},step=epoch_i)
                    print('wandb logging complete')
                              
                elif train_mode == 'rna_only':
                    val_step_rna(data_dict_val)
                    val_step_rna_ho(data_dict_val_ho)
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
                    
                val_losses.append(metric_dict['hg_val_ho'].result().numpy())
                
                print('hg_val_loss_ho: ' + str(metric_dict['hg_val_ho'].result().numpy()))
                wandb.log({'hg_val_loss_ho': metric_dict['hg_val_ho'].result().numpy()},
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
            #file_name = wandb.config.model_save_basename + "." + str(epoch_i) + ".val.out.tsv"
            #df = pd.DataFrame({'y_trues':y_trues, 'y_preds_ho':y_preds_ho,
            #                        'cell_types_ho': cell_types_ho, 'gene_map_ho': gene_map_ho})
            #df.to_csv(file_name, sep='\t',header=True,index=False)
            #command = "gsutil cp " + file_name + " " + wandb.config.model_save_dir
            #subprocess.call(command,shell=True)

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

##########################################################################
if __name__ == '__main__':
    main()
        
