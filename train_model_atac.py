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
import src.models.aformer_atac as aformer
import src.metrics as metrics
import src.optimizers as optimizers
import src.schedulers as schedulers
import src.utils as utils

import training_utils_atac as training_utils
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
                'output_length_ATAC': {
                    'values': [args.output_length_ATAC]
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
                'wd_1': {
                    'values':[float(x) for x in args.wd_1.split(',')]
                },
                'wd_2': {
                    'values':[float(x) for x in args.wd_2.split(',')]
                },
                'gradient_clip': {
                    'values': [float(x) for x in args.gradient_clip.split(',')]
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
                'filter_list_atac': {
                    'values': [[int(x) for x in args.filter_list_atac.split(',')]]
                },
                'BN_momentum': {
                    'values': [args.BN_momentum]
                },
                'atac_mask_dropout': {
                    'values': [args.atac_mask_dropout]
                },
                'rectify': {
                    'values':[parse_bool_str(x) for x in args.rectify.split(',')]
                },
                'optimizer': {
                    'values':[args.optimizer]
                },
                'log_atac': {
                    'values':[parse_bool_str(x) for x in args.log_atac.split(',')]
                },
                'use_atac': {
                    'values':[parse_bool_str(x) for x in args.use_atac.split(',')]
                },
                'use_seq': {
                    'values':[parse_bool_str(x) for x in args.use_seq.split(',')]
                },
                'sonnet_weights_bool': {
                    'values':[parse_bool_str(x) for x in args.sonnet_weights_bool.split(',')]
                },
                'random_mask_size': {
                    'values':[int(x) for x in args.random_mask_size.split(',')]
                },
                'final_point_scale': {
                    'values':[int(x) for x in args.final_point_scale.split(',')]
                },
                'seed': {
                    'values':[args.seed]
                },
                'bce_loss_scale': {
                    'values':[args.bce_loss_scale]
                },
                'atac_corrupt_rate': {
                    'values': [int(x) for x in args.atac_corrupt_rate.split(',')]
                },
                'seq_corrupt_rate': {
                    'values': [int(x) for x in args.seq_corrupt_rate.split(',')]
                },
                'use_pooling': {
                    'values': [str(x) for x in args.use_pooling.split(',')]
                }
            }
    }


    def sweep_train(config_defaults=None):
        # Set default values
        # Specify the other hyperparameters to the configuration, if any

        ## tpu initialization
        strategy = training_utils.tf_tpu_initialize(args.tpu_name,args.tpu_zone)
        mixed_precision.set_global_policy('mixed_bfloat16')
        g = tf.random.Generator.from_seed(args.seed)
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
            wandb.config.gcs_path_holdout=args.gcs_path_holdout
            wandb.config.num_epochs=args.num_epochs
            wandb.config.train_examples=args.train_examples
            wandb.config.val_examples_ho=args.val_examples_ho
            wandb.config.batch_size=args.batch_size
            wandb.config.warmup_frac=args.warmup_frac
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            wandb.config.max_shift=args.max_shift
            wandb.config.inits_type=args.inits_type

            wandb.config.crop_size = (wandb.config.output_length - wandb.config.final_output_length) // 2

            gcs_path = wandb.config.gcs_path

            run_name = '_'.join([str(int(wandb.config.input_length) / 1000)[:4].rstrip('.') + 'k',
                                 'load-' + str(wandb.config.load_init),
                                 'LR-' + str(wandb.config.lr_base),
                                 'T-' + str(wandb.config.num_transformer_layers),
                                 'D-' + str(wandb.config.dropout_rate)])

            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = date_string.replace(' ','_')
            wandb.run.name = run_name + "_" + date_string
            base_name = wandb.config.model_save_basename + "_" + wandb.run.name


            '''
            TPU init options
            '''

            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.DATA
            options.deterministic=False
            #options.experimental_threading.max_intra_op_parallelism=1
            mixed_precision.set_global_policy('mixed_bfloat16')
            #tf.autograph.set_verbosity(5)


            NUM_REPLICAS = strategy.num_replicas_in_sync
            BATCH_SIZE_PER_REPLICA=wandb.config.batch_size
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS
            print('global batch size:', GLOBAL_BATCH_SIZE)

            num_train=wandb.config.train_examples
            num_val_ho=wandb.config.val_examples_ho

            wandb.config.update({"train_steps": num_train // (GLOBAL_BATCH_SIZE)},
                                allow_val_change=True)
            wandb.config.update({"val_steps_ho" : num_val_ho // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"total_steps": num_train // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)


            out_iterators = \
                    training_utils.return_distributed_iterators(wandb.config.gcs_path,
                                                                wandb.config.gcs_path_holdout,
                                                                GLOBAL_BATCH_SIZE,
                                                                wandb.config.input_length,
                                                                wandb.config.max_shift,
                                                                wandb.config.output_length_ATAC,
                                                                wandb.config.output_length,
                                                                wandb.config.crop_size,
                                                                wandb.config.output_res,
                                                                args.num_parallel,
                                                                args.num_epochs,
                                                                strategy,
                                                                options,
                                                                wandb.config.atac_mask_dropout,
                                                                wandb.config.random_mask_size,
                                                                wandb.config.log_atac,
                                                                wandb.config.use_atac,
                                                                wandb.config.use_seq,
                                                                wandb.config.seed,
                                                                wandb.config.seq_corrupt_rate,
                                                                wandb.config.atac_corrupt_rate,
                                                                wandb.config.val_steps_ho,
                                                                g)

            train_human, data_val_ho = out_iterators

            loading_checkpoint_bool=False
            inits=None
            print('created dataset iterators')
            if wandb.config.load_init:
                if wandb.config.inits_type == 'enformer_performer':
                    print('loaded enformer performer weights')
                    inits=training_utils.get_initializers_enformer_performer(args.multitask_checkpoint_path,
                                                                             wandb.config.num_transformer_layers)
                elif wandb.config.inits_type == 'enformer_conv':
                    print('loaded enformer conv weights')
                    inits=training_utils.get_initializers_enformer_conv(args.multitask_checkpoint_path,
                                                                        wandb.config.sonnet_weights_bool,
                                                                        len(wandb.config.filter_list_seq))
                    wandb.config.update({"filter_list_seq": [768, 896, 1024, 1152, 1280, 1536]},
                                        allow_val_change=True)
                elif wandb.config.inits_type == 'enformer_performer_full':
                    wandb.config.update({"load_init": False},
                                        allow_val_change=True)
                    loading_checkpoint_bool=True

                else:
                    raise ValueError('inits type not found')

            print(wandb.config)
            model = aformer.aformer(kernel_transformation=wandb.config.kernel_transformation,
                                    dropout_rate=wandb.config.dropout_rate,
                                    pointwise_dropout_rate=wandb.config.pointwise_dropout_rate,
                                    input_length=wandb.config.input_length,
                                    output_length=wandb.config.output_length,
                                    final_output_length=wandb.config.final_output_length,
                                    num_heads=wandb.config.num_heads,
                                    numerical_stabilizer=0.0000001,
                                    nb_random_features=wandb.config.num_random_features,
                                    max_seq_length=wandb.config.output_length,
                                    rel_pos_bins=wandb.config.output_length,
                                    norm=True,
                                    BN_momentum=wandb.config.BN_momentum,
                                    use_rot_emb = True,
                                    use_mask_pos = False,
                                    normalize = True,
                                    num_transformer_layers=wandb.config.num_transformer_layers,
                                    inits=inits,
                                    inits_type=wandb.config.inits_type,
                                    load_init=wandb.config.load_init,
                                    final_point_scale=wandb.config.final_point_scale,
                                    freeze_conv_layers=wandb.config.freeze_conv_layers,
                                    filter_list_seq=wandb.config.filter_list_seq,
                                    filter_list_atac=wandb.config.filter_list_atac)


            print('initialized model')


            scheduler1= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base1,
                decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=wandb.config.decay_frac)
            scheduler1=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base1,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps*wandb.config.num_epochs,
                                         decay_schedule_fn=scheduler1)
            scheduler2= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base2,
                decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=wandb.config.decay_frac)
            scheduler2=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base2,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps*wandb.config.num_epochs,
                                         decay_schedule_fn=scheduler2)

            if wandb.config.optimizer == 'adam':
                optimizer1 = tf.keras.optimizers.Adam(learning_rate=scheduler1,
                                                      epsilon=wandb.config.epsilon)
                optimizer2 = tf.keras.optimizers.Adam(learning_rate=scheduler2,
                                                      epsilon=wandb.config.epsilon)

            elif wandb.config.optimizer == 'adamw':
                optimizer1 = tfa.optimizers.AdamW(learning_rate=scheduler1,
                                                     weight_decay=wandb.config.wd_1,
                                                     epsilon=wandb.config.epsilon,
                                                      exclude_from_weight_decay=['layer_norm',
                                                                                 'bias',
                                                                                 'embeddings',
                                                                                 'batch_norm'])
                optimizer2 = tfa.optimizers.AdamW(learning_rate=scheduler2,
                                                     weight_decay=wandb.config.wd_2,
                                                     epsilon=wandb.config.epsilon,
                                                      exclude_from_weight_decay=['layer_norm',
                                                                                 'bias',
                                                                                 'embeddings',
                                                                                 'batch_norm'])
            else:
                raise ValueError('optimizer not found')

            metric_dict = {}

            optimizers_in = optimizer1,optimizer2

            human_step, val_step, \
                build_step, metric_dict = training_utils.return_train_val_functions(model,
                                                                                    wandb.config.train_steps,
                                                                                    wandb.config.val_steps_ho,
                                                                                    optimizers_in,
                                                                                    strategy,
                                                                                    metric_dict,
                                                                                    GLOBAL_BATCH_SIZE,
                                                                                    wandb.config.gradient_clip,
                                                                                    wandb.config.bce_loss_scale)


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
                    build_step(data_val_ho)
                    if ((wandb.config.inits_type == 'enformer_performer_full') and (loading_checkpoint_bool==True)):
                        model.load_weights(args.multitask_checkpoint_path + "/saved_model")
                        print('built and loaded model')
                    total_params = 0
                    for k in model.trainable_variables:
                        var = k.values[0]
                        total_params += tf.size(var)
                    print('total params: ' + str(total_params))

                print('starting epoch_', str(epoch_i))
                start = time.time()

                for step in range(wandb.config.train_steps):
                    strategy.run(human_step, args=(next(train_human),))

                print('train_loss: ' + str(metric_dict['train_loss'].result().numpy()))
                wandb.log({'human_train_loss': metric_dict['train_loss'].result().numpy()},
                          step=epoch_i)
                train_loss_poisson = metric_dict['train_loss_poisson'].result().numpy()
                train_loss_bce = metric_dict['train_loss_bce'].result().numpy()

                wandb.log({'human_train_loss_poisson': metric_dict['train_loss_poisson'].result().numpy(),
                           'human_train_loss_bce': metric_dict['train_loss_bce'].result().numpy()},
                           step=epoch_i)

                atac_pearsons_tr = metric_dict['ATAC_PearsonR_tr'].result()['PearsonR'].numpy()
                atac_R2_tr = metric_dict['ATAC_R2_tr'].result()['R2'].numpy()
                atac_roc_tr = metric_dict['ATAC_ROC_tr'].result().numpy()
                atac_pr_tr = metric_dict['ATAC_PR_tr'].result().numpy()
                atac_TP_tr = metric_dict['ATAC_TP_tr'].result().numpy()
                atac_T_tr = metric_dict['ATAC_T_tr'].result().numpy()
                wandb.log({'human_ATAC_pearsons_tr': atac_pearsons_tr,
                           'human_ATAC_R2_tr': atac_R2_tr,
                           'human_ATAC_ROC_tr': atac_roc_tr,
                           'human_ATAC_pos_rate_tr': (atac_TP_tr/atac_T_tr),
                           'human_ATAC_PR_tr': atac_pr_tr},
                          step=epoch_i)

                end = time.time()
                duration = (end - start) / 60.

                print('completed epoch ' + str(epoch_i))
                print('training duration(mins): ' + str(duration))

                start = time.time()
                for k in range(wandb.config.val_steps_ho):
                    strategy.run(val_step, args=(next(data_val_ho),))

                val_loss = metric_dict['val_loss'].result().numpy()
                val_loss_poisson = metric_dict['val_loss_poisson'].result().numpy()
                val_loss_bce = metric_dict['val_loss_bce'].result().numpy()
                print('val_loss: ' + str(val_loss))
                print('val_loss_poisson: ' + str(val_loss_poisson))
                print('val_loss_bce: ' + str(val_loss_bce))
                val_losses.append(val_loss)

                wandb.log({'human_val_loss': metric_dict['val_loss'].result().numpy(),
                           'human_val_loss_poisson': metric_dict['val_loss_poisson'].result().numpy(),
                           'human_val_loss_bce': metric_dict['val_loss_bce'].result().numpy()},
                           step=epoch_i)

                atac_pearsons = metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy()
                atac_R2 = metric_dict['ATAC_R2'].result()['R2'].numpy()
                atac_roc = metric_dict['ATAC_ROC'].result().numpy()
                atac_pr = metric_dict['ATAC_PR'].result().numpy()
                atac_TP = metric_dict['ATAC_TP'].result().numpy()
                atac_T = metric_dict['ATAC_T'].result().numpy()

                val_pearsons.append(atac_pearsons)
                print('human_ATAC_pearsons: ' + str(atac_pearsons))
                print('human_ATAC_R2: ' + str(atac_R2))
                print('human_ATAC_PR: ' + str(atac_pr))
                print('human_ATAC_ROC: ' + str(atac_roc))

                wandb.log({'human_ATAC_pearsons': atac_pearsons,
                           'human_ATAC_R2': atac_R2,
                           'human_ATAC_ROC': atac_roc,
                           'human_ATAC_pos_rate': (atac_TP/atac_T),
                           'human_ATAC_PR': atac_pr},
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


    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

##########################################################################
if __name__ == '__main__':
    main()
