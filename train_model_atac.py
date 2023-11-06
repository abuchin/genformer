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
import src.load_weights_atac as load_weights_atac
import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy import stats

def parse_bool_str(input_str):
    if ((input_str == 'False') or (input_str == 'false')):
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
                'input_length': {'values': [args.input_length]},
                'output_length': {'values': [args.output_length]},
                'output_length_ATAC': {'values': [args.output_length_ATAC]},
                'final_output_length': {'values': [args.final_output_length]},
                'output_res': {'values': [args.output_res]},
                'dropout_rate': {'values': [float(x) for x in args.dropout_rate.split(',')]},
                'pointwise_dropout_rate': {'values': [float(x) for x in args.pointwise_dropout_rate.split(',')]},
                'lr_base': {'values':[float(x) for x in args.lr_base.split(',')]},
                'gradient_clip': {'values': [float(x) for x in args.gradient_clip.split(',')]},
                'decay_frac': {'values': [float(x) for x in args.decay_frac.split(',')]},
                'num_transformer_layers': {'values': [int(x) for x in args.num_transformer_layers.split(',')]},
                'num_heads': {'values': [int(x) for x in args.num_heads.split(',')]},
                'num_random_features': {'values':[int(x) for x in args.num_random_features.split(',')]},
                'kernel_transformation': {'values':[args.kernel_transformation]},
                'epsilon': {'values':[args.epsilon]},
                'load_init': {'values':[parse_bool_str(x) for x in args.load_init.split(',')]},
                'filter_list_seq': {'values': [[int(x) for x in args.filter_list_seq.split(',')]]},
                'filter_list_atac': {'values': [[int(x) for x in args.filter_list_atac.split(',')]]},
                'BN_momentum': {'values': [args.BN_momentum]},
                'atac_mask_dropout': {'values': [args.atac_mask_dropout]},
                'atac_mask_dropout_val': {'values': [args.atac_mask_dropout_val]},
                'rectify': {'values':[parse_bool_str(x) for x in args.rectify.split(',')]},
                'log_atac': {'values':[parse_bool_str(x) for x in args.log_atac.split(',')]},
                'use_atac': {'values':[parse_bool_str(x) for x in args.use_atac.split(',')]},
                'use_seq': {'values':[parse_bool_str(x) for x in args.use_seq.split(',')]},
                'random_mask_size': {'values':[int(x) for x in args.random_mask_size.split(',')]},
                'final_point_scale': {'values':[int(x) for x in args.final_point_scale.split(',')]},
                'seed': {'values':[args.seed]},
                'atac_corrupt_rate': {'values': [int(x) for x in args.atac_corrupt_rate.split(',')]},
                'seq_corrupt_rate': {'values': [int(x) for x in args.seq_corrupt_rate.split(',')]},
                'use_tf_activity': {'values': [parse_bool_str(x) for x in args.use_tf_activity.split(',')]},
                'num_epochs_to_start': {'values': [int(x) for x in args.num_epochs_to_start.split(',')]},
                'loss_type': {'values': [str(x) for x in args.loss_type.split(',')]},
                'total_weight_loss': {'values': [float(x) for x in args.total_weight_loss.split(',')]}
                }
    }


    def sweep_train(config_defaults=None):
        ## tpu initialization
        strategy = training_utils.tf_tpu_initialize(args.tpu_name,args.tpu_zone)
        mixed_precision.set_global_policy('mixed_bfloat16')
        g = tf.random.Generator.from_seed(args.seed)

        with strategy.scope(): ## rest must be w/in strategy scope
            config_defaults = {"lr_base": 0.01 }### will be overwritten
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

            wandb.config.crop_size = (wandb.config.output_length - wandb.config.final_output_length) // 2

            gcs_path = wandb.config.gcs_path

            run_name = '_'.join([str(int(wandb.config.input_length) / 1000)[:4].rstrip('.') + 'k',
                                 'load-' + str(wandb.config.load_init),
                                 'LR-' + str(wandb.config.lr_base),
                                 'T-' + str(wandb.config.num_transformer_layers),
                                 'TF-' + str(wandb.config.use_tf_activity)])

            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = date_string.replace(' ','_')
            wandb.run.name = run_name + "_" + date_string
            base_name = wandb.config.model_save_basename + "_" + wandb.run.name

            ''' TPU init options '''

            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.DATA
            options.deterministic=False
            mixed_precision.set_global_policy('mixed_bfloat16')

            NUM_REPLICAS = strategy.num_replicas_in_sync
            BATCH_SIZE_PER_REPLICA=wandb.config.batch_size
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS
            print('global batch size:', GLOBAL_BATCH_SIZE)

            wandb.config.update({"train_steps": wandb.config.train_examples // (GLOBAL_BATCH_SIZE)},
                                allow_val_change=True)
            wandb.config.update({"val_steps_ho" : wandb.config.val_examples_ho // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"total_steps": wandb.config.train_examples // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)


            out_iterators = \
                    training_utils.return_distributed_iterators(wandb.config.gcs_path, wandb.config.gcs_path_holdout,
                                                                GLOBAL_BATCH_SIZE, wandb.config.input_length,
                                                                wandb.config.max_shift, wandb.config.output_length_ATAC,
                                                                wandb.config.output_length, wandb.config.crop_size,
                                                                wandb.config.output_res, args.num_parallel, args.num_epochs,
                                                                strategy, options, wandb.config.atac_mask_dropout,
                                                                wandb.config.atac_mask_dropout_val,
                                                                wandb.config.random_mask_size, wandb.config.log_atac,
                                                                wandb.config.use_atac, wandb.config.use_seq, wandb.config.seed,
                                                                wandb.config.seq_corrupt_rate, wandb.config.atac_corrupt_rate,
                                                                wandb.config.val_steps_ho, wandb.config.use_tf_activity, g)

            train_human, data_val_ho = out_iterators
            print('created dataset iterators')
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
                                    norm=True,
                                    BN_momentum=wandb.config.BN_momentum,
                                    normalize = True,
                                    num_transformer_layers=wandb.config.num_transformer_layers,
                                    final_point_scale=wandb.config.final_point_scale,
                                    filter_list_seq=wandb.config.filter_list_seq,
                                    filter_list_atac=wandb.config.filter_list_atac)

            print('initialized model')

            scheduler= tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=wandb.config.lr_base,
                decay_steps=wandb.config.total_steps*wandb.config.num_epochs, alpha=wandb.config.decay_frac)
            scheduler=optimizers.WarmUp(initial_learning_rate=wandb.config.lr_base,
                                         warmup_steps=wandb.config.warmup_frac*wandb.config.total_steps*wandb.config.num_epochs,
                                         decay_schedule_fn=scheduler)
            optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler,
                                                  epsilon=wandb.config.epsilon)

            metric_dict = {}

            train_step, val_step, \
                build_step, metric_dict = training_utils.return_train_val_functions(model,
                                                                                    wandb.config.train_steps,
                                                                                    optimizer,
                                                                                    strategy,
                                                                                    metric_dict,
                                                                                    GLOBAL_BATCH_SIZE,
                                                                                    wandb.config.gradient_clip,
                                                                                    wandb.config.loss_type,
                                                                                    wandb.config.total_weight_loss)


            global_step = 0
            val_losses = []
            val_pearsons = []
            val_R2 = []
            patience_counter = 0
            stop_criteria = False
            best_epoch = 0

            for epoch_i in range(1, wandb.config.num_epochs+1):
                step_num = (wandb.config.num_epochs_to_start+epoch_i) * \
                            wandb.config.train_steps * GLOBAL_BATCH_SIZE
                if epoch_i == 1:
                    print('building model')
                    build_step(data_val_ho)
                    if wandb.config.load_init:
                        model.load_weights(args.checkpoint_path + "/saved_model")
                        print('built and loaded model')
                    total_params = 0
                    for k in model.trainable_variables:
                        var = k.values[0]
                        total_params += tf.size(var)
                    print('total params: ' + str(total_params))

                ####### training steps #######################
                print('starting epoch_', str(epoch_i))
                start = time.time()
                for step in range(wandb.config.train_steps):
                    strategy.run(train_step, args=(next(train_human),))

                print('train_loss: ' + str(metric_dict['train_loss'].result().numpy()))
                wandb.log({'train_loss': metric_dict['train_loss'].result().numpy()},
                          step=step_num)

                wandb.log({'ATAC_pearsons_tr': metric_dict['ATAC_PearsonR_tr'].result()['PearsonR'].numpy(),
                           'ATAC_R2_tr': metric_dict['ATAC_R2_tr'].result()['R2'].numpy()},
                          step=step_num)
                duration = (time.time() - start) / 60.

                print('completed epoch ' + str(epoch_i) + ' - duration(mins): ' + str(duration))

                ####### validation steps #######################
                start = time.time()
                pred_list = []
                true_list = []
                for k in range(wandb.config.val_steps_ho):
                    true, pred = strategy.run(val_step, args=(next(data_val_ho),))
                    for x in strategy.experimental_local_results(true):
                        true_list.append(tf.reshape(x, [-1]))
                    for x in strategy.experimental_local_results(pred):
                        pred_list.append(tf.reshape(x, [-1]))

                figures,overall_corr,overall_corr_log= training_utils.make_plots(tf.concat(pred_list,0),
                                                                                 tf.concat(true_list,0),
                                                                                 5000)

                val_loss = metric_dict['val_loss'].result().numpy()
                print('val_loss: ' + str(val_loss))
                val_losses.append(val_loss)

                wandb.log({'val_loss': metric_dict['val_loss'].result().numpy()},
                           step=step_num)
                val_pearsons.append(metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy())
                print('ATAC_pearsons: ' + str(metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy()))
                print('ATAC_R2: ' + str(metric_dict['ATAC_R2'].result()['R2'].numpy()))
                wandb.log({'ATAC_pearsons': metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy(),
                           'ATAC_R2': metric_dict['ATAC_R2'].result()['R2'].numpy()},
                          step=step_num)

                print('ATAC_pearsons_overall: ' + str(overall_corr))
                print('ATAC_pearsons_overall_log: ' + str(overall_corr_log))
                wandb.log({'ATAC_pearsons_overall': overall_corr,
                           'ATAC_pearsons_overall_log': overall_corr_log},
                          step=step_num)

                wandb.log({'overall_predictions': figures},
                          step=step_num)

                duration = (time.time() - start) / 60.
                print('completed epoch ' + str(epoch_i) + ' validation - duration(mins): ' + str(duration))

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
