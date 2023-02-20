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
import src.aformer_atac_cage as aformer
import src.metrics as metrics
import src.optimizers as optimizers
import src.schedulers as schedulers
import src.utils as utils

import training_utils_atac_cage as training_utils
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
                'output_length': {
                    'values': [args.output_length]
                },
                'final_output_length': {
                    'values': [args.final_output_length]
                },
                'output_res': {
                    'values': [args.output_res]
                },
                'dropout_rate': { 
                    'values': [float(x) for x in args.dropout_rate.split(',')]
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
                'cage_scale': {
                    'values': [float(x) for x in args.cage_scale.split(',')]
                },
                'decay_frac': {
                    'values': [float(x) for x in args.decay_frac.split(',')]
                },
                'num_transformer_layers':{
                    'values': [int(x) for x in args.num_transformer_layers.split(',')]
                },
                'num_heads':{
                    'values': [int(x) for x in args.num_heads.split(',')]
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
                'filter_list_seq': {
                    'values': [[int(x) for x in args.filter_list_seq.split(',')]]
                },
                'BN_momentum': {
                    'values': [args.BN_momentum]
                },
                'wd_1': {
                    'values': [args.wd_1]
                },
                'wd_2': {
                    'values': [args.wd_2]
                },
                'rectify': {
                    'values':[parse_bool_str(x) for x in args.rectify.split(',')]
                },
                'predict_masked_atac_bool': {
                    'values':[parse_bool_str(x) for x in args.predict_masked_atac_bool.split(',')]
                },
                'optimizer': {
                    'values':[args.optimizer]
                }
                
            }
    }

    
    def sweep_train(config_defaults=None):
        # Set default values
        # Specify the other hyperparameters to the configuration, if any

        ## tpu initialization
        strategy = training_utils.tf_tpu_initialize(args.tpu_name,args.tpu_zone)
        mixed_precision.set_global_policy('mixed_bfloat16')
        #g = tf.random.Generator.from_non_deterministic_state()
        ## rest must be w/in strategy scope
        g = tf.random.Generator.from_seed(datetime.now().timestamp())
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
            wandb.config.gcs_path_TSS=args.gcs_path_TSS
            wandb.config.num_epochs=args.num_epochs
            wandb.config.train_examples=args.train_examples
            wandb.config.val_examples=args.val_examples
            wandb.config.val_examples_TSS=args.val_examples_TSS
            wandb.config.batch_size=args.batch_size
            wandb.config.warmup_frac=args.warmup_frac
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            wandb.config.max_shift=args.max_shift
            wandb.config.inits_type=args.inits_type
            
            wandb.config.crop_size = (wandb.config.output_length - wandb.config.final_output_length) // 2
            
            
            run_name = '_'.join(["EP_baseline",
                                 "glob_acc",
                                  str(wandb.config.input_length)[:3] + 'k',
                                 'load-' + str(wandb.config.load_init),
                                 'frz-' + str(wandb.config.freeze_conv_layers),
                                 'LR1-' + str(wandb.config.lr_base1),
                                 'LR2-' + str(wandb.config.lr_base2),
                                 'T-' + str(wandb.config.num_transformer_layers),
                                 'F-' + str(wandb.config.hidden_size),
                                 'D-' + str(wandb.config.dropout_rate)])
            
            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = date_string.replace(' ','_')
            wandb.run.name = run_name + "_" + date_string
            base_name = wandb.config.model_save_basename + "_" + run_name
            
            
            '''
            TPU init options
            '''

            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.OFF
            options.deterministic=False
            #options.experimental_threading.max_intra_op_parallelism=1
            mixed_precision.set_global_policy('mixed_bfloat16')

            
            NUM_REPLICAS = strategy.num_replicas_in_sync
            BATCH_SIZE_PER_REPLICA=wandb.config.batch_size
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS
            print('global batch size:', GLOBAL_BATCH_SIZE)
            
            num_train=wandb.config.train_examples
            num_val=wandb.config.val_examples
            num_val_TSS=wandb.config.val_examples_TSS#4192000

            wandb.config.update({"train_steps": num_train // (GLOBAL_BATCH_SIZE)},
                                allow_val_change=True)
            wandb.config.update({"val_steps" : num_val // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"val_steps_TSS" : num_val_TSS // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"total_steps": num_train // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            

            data_train,data_val,data_val_TSS = \
                    training_utils.return_distributed_iterators(wandb.config.gcs_path,
                                                                wandb.config.gcs_path_TSS,
                                                                GLOBAL_BATCH_SIZE,
                                                                wandb.config.input_length,
                                                                wandb.config.max_shift,
                                                                wandb.config.output_length,
                                                                wandb.config.crop_size,
                                                                wandb.config.output_res,
                                                                args.num_parallel,
                                                                args.num_epochs,
                                                                strategy,
                                                                options,
                                                                wandb.config.predict_masked_atac_bool,
                                                                g)

            print('created dataset iterators')
            #if (wandb.config.load_init and os.path.isdir(args.multitask_checkpoint_path)):
            if wandb.config.inits_type == 'enformer_performer':
                print('loaded enformer performer weights')
                inits=training_utils.get_initializers_enformer_performer(args.multitask_checkpoint_path,
                                                                         wandb.config.num_transformer_layers)
            elif wandb.config.inits_type == 'enformer_conv':
                print('loaded enformer conv weights')
                inits=training_utils.get_initializers_enformer_conv(args.multitask_checkpoint_path)
            else:
                raise ValueError('inits type not found')
                
            wandb.config.update({"filter_list_seq": [768, 896, 1024, 1152, 1280, 1536]},
                                allow_val_change=True)
            #else:
            #    inits=None
            #    print('WARNING: supplied checkpoint directory does not exist')

            model = aformer.aformer(kernel_transformation=wandb.config.kernel_transformation,
                                    dropout_rate=wandb.config.dropout_rate,
                                    pointwise_dropout_rate=wandb.config.pointwise_dropout_rate,
                                    input_length=wandb.config.input_length,
                                    output_length=wandb.config.output_length,
                                    final_output_length=wandb.config.final_output_length,
                                    num_heads=wandb.config.num_heads,
                                    numerical_stabilizer=0.0000001,
                                    nb_random_features=wandb.config.num_random_features,
                                    hidden_size=wandb.config.hidden_size,
                                    d_model=wandb.config.hidden_size,
                                    dim=wandb.config.hidden_size // wandb.config.num_heads,
                                    max_seq_length=wandb.config.output_length,
                                    rel_pos_bins=wandb.config.output_length,
                                    norm=True,
                                    BN_momentum=wandb.config.BN_momentum,
                                    use_rot_emb = True,
                                    use_mask_pos = False,
                                    normalize = True,
                                    predict_masked_atac_bool=wandb.config.predict_masked_atac_bool,
                                    num_transformer_layers=wandb.config.num_transformer_layers,
                                    inits=inits,
                                    inits_type=wandb.config.inits_type,
                                    load_init=wandb.config.load_init,
                                    freeze_conv_layers=wandb.config.freeze_conv_layers,
                                    filter_list_seq=wandb.config.filter_list_seq)
            

            print('initialized model')

            if wandb.config.optimizer == 'adamw':
            
                scheduler1= tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=wandb.config.lr_base1,
                    decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=wandb.config.decay_frac)
                scheduler1=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base1,
                                             warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps*wandb.config.num_epochs,
                                             decay_schedule_fn=scheduler1)
                scheduler1_wd= tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=wandb.config.wd_1,
                    decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=wandb.config.decay_frac)
                scheduler1_wd=optimizers.WarmUp(initial_learning_rate=wandb.config.wd_1,
                                             warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps*wandb.config.num_epochs,
                                             decay_schedule_fn=scheduler1)

                optimizer1 = tfa.optimizers.AdamW(learning_rate=scheduler1,
                                                  weight_decay=scheduler1_wd,
                                                  epsilon=wandb.config.epsilon)
                #####
                scheduler2= tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=wandb.config.lr_base2,
                    decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=wandb.config.decay_frac)
                scheduler2=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base2,
                                             warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps*wandb.config.num_epochs,
                                             decay_schedule_fn=scheduler2)
                scheduler2_wd= tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=wandb.config.wd_2,
                    decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=wandb.config.decay_frac)
                scheduler2_wd=optimizers.WarmUp(initial_learning_rate=wandb.config.wd_2,
                                             warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps*wandb.config.num_epochs,
                                             decay_schedule_fn=scheduler2)

                optimizer2 = tfa.optimizers.AdamW(learning_rate=scheduler2,
                                                  weight_decay=scheduler2_wd,
                                                  epsilon=wandb.config.epsilon)
            elif wandb.config.optimizer == 'adabelief':
                optimizer1 = tfa.optimizers.AdaBelief(
                    learning_rate= wandb.config.lr_base1,
                    epsilon= wandb.config.epsilon,
                    weight_decay= wandb.config.wd_1,
                    rectify=wandb.config.rectify,
                    total_steps= wandb.config.total_steps*wandb.config.num_epochs,
                    warmup_proportion= wandb.config.warmup_frac,
                    min_lr= wandb.config.decay_frac * wandb.config.lr_base1
                )
                optimizer2 = tfa.optimizers.AdaBelief(
                    learning_rate= wandb.config.lr_base2,
                    epsilon= wandb.config.epsilon,
                    weight_decay= wandb.config.wd_2,
                    rectify=wandb.config.rectify,
                    total_steps= wandb.config.total_steps*wandb.config.num_epochs,
                    warmup_proportion= wandb.config.warmup_frac,
                    min_lr= wandb.config.decay_frac * wandb.config.lr_base2
                )
            else:
                raise ValueError('optimizer not found')


            optimizers_in = optimizer1,optimizer2
            
            metric_dict = {}

            train_step,val_step,val_step_TSS,\
                build_step, metric_dict = training_utils.return_train_val_functions(model,
                                                                                    wandb.config.train_steps,
                                                                                    wandb.config.val_steps,
                                                                                    wandb.config.val_steps_TSS,
                                                                                    optimizers_in,
                                                                                    strategy,
                                                                                    metric_dict,
                                                                                    GLOBAL_BATCH_SIZE,
                                                                                    wandb.config.gradient_clip,
                                                                                    wandb.config.cage_scale,
                                                                                    wandb.config.predict_masked_atac_bool)
                

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
                    build_step(data_val)
                    print('built model')
                    total_params = 0
                    for k in model.trainable_variables:
                        var = k.values[0]
                        total_params += tf.size(var)
                    print('total params: ' + str(total_params)) 
                
                print('starting epoch_', str(epoch_i))
                start = time.time()
                train_step(data_train)
                end = time.time()
                duration = (end - start) / 60.
                
                print('completed epoch ' + str(epoch_i))
                print('train_loss: ' + str(metric_dict['train_loss'].result().numpy()))
                wandb.log({'human_train_loss': metric_dict['train_loss'].result().numpy()},
                          step=epoch_i)
                print('training duration(mins): ' + str(duration))
                
                start = time.time()
                
                val_step(data_val)
                
                val_loss = metric_dict['val_loss'].result().numpy()
                cage_pearsons = metric_dict['CAGE_PearsonR'].result()['PearsonR'].numpy()
                cage_R2 = metric_dict['CAGE_R2'].result()['R2'].numpy()
                
                print('val_loss: ' + str(val_loss))
                print('human_CAGE_pearsons: ' + str(cage_pearsons))
                print('human_CAGE_R2: ' + str(cage_R2))
                
                val_losses.append(val_loss)
                val_pearsons.append(cage_pearsons)
                wandb.log({'human_val_loss': val_loss,
                           'human_CAGE_pearsons': cage_pearsons,
                           'human_CAGE_R2': cage_R2},step=epoch_i)
                
                if wandb.config.predict_masked_atac_bool:
                    atac_pearsons = metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy()
                    atac_R2 = metric_dict['ATAC_R2'].result()['R2'].numpy()
                    wandb.log({'human_ATAC_pearsons': atac_pearsons,
                               'human_ATAC_R2': atac_R2},step=epoch_i)
                    print('human_ATAC_pearsons: ' + str(atac_pearsons))
                    print('human_ATAC_R2: ' + str(atac_R2))
                    
                    atac_pearsons_baseline = metric_dict['ATAC_PearsonR_baseline'].result()['PearsonR'].numpy()
                    atac_R2_baseline = metric_dict['ATAC_R2_baseline'].result()['R2'].numpy()
                    wandb.log({'human_ATAC_baseline_pearsons': atac_pearsons_baseline,
                               'human_ATAC_baseline_R2': atac_R2_baseline},step=epoch_i)
                    print('human_ATAC_baseline_pearsons: ' + str(atac_pearsons_baseline))
                    print('human_ATAC_baseline_R2: ' + str(atac_R2_baseline))
                

                
                if epoch_i % 2 == 0: 
                    val_step_TSS(data_val_TSS)

                    val_pearson_TSS = metric_dict['corr_stats'].result()['pearsonR'].numpy()
                    val_R2_TSS = metric_dict['corr_stats'].result()['R2'].numpy()

                    y_trues = metric_dict['corr_stats'].result()['y_trues'].numpy()
                    y_preds = metric_dict['corr_stats'].result()['y_preds'].numpy()
                    cell_types = metric_dict['corr_stats'].result()['cell_types'].numpy()
                    gene_map = metric_dict['corr_stats'].result()['gene_map'].numpy()

                    print('making plots')
                    figures,corrs_overall= training_utils.make_plots(y_trues,y_preds,
                                                                     cell_types,gene_map)


                    fig_cell_spec, fig_gene_spec, fig_overall=figures 

                    cell_specific_corrs, gene_specific_corrs = corrs_overall

                    print('cell_specific_correlation: ' + str(cell_specific_corrs))
                    print('gene_specific_correlation: ' + str(gene_specific_corrs))

                    wandb.log({'gene_spec_mean_corrs': gene_specific_corrs,
                               'cell_spec_mean_corrs': cell_specific_corrs},
                              step=epoch_i)
                    wandb.log({'hg_OVERALL_TSS_predictions': fig_overall,
                               'cross_cell_dist': fig_cell_spec,
                               'cross_gene_dist': fig_gene_spec},
                              step=epoch_i)
                

                end = time.time()
                duration = (end - start) / 60.
                print('completed epoch ' + str(epoch_i) + ' validation')
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
                                                        saved_model_basename=wandb.config.model_save_basename + "_" + wandb.run.name)
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
        