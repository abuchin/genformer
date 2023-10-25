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
import src.models.aformer_atac_rna as aformer
import src.metrics as metrics
import src.optimizers as optimizers
import src.schedulers as schedulers
import src.utils as utils

import training_utils_atac_rna as training_utils
import seaborn as sns
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy import stats

import src.load_weights_atac_rna as load_weights_atac_rna

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
                'input_length': {'values': [args.input_length]},
                'output_length': {'values': [args.output_length]},
                'output_length_ATAC': {'values': [args.output_length_ATAC]},
                'final_output_length': {'values': [args.final_output_length]},
                'output_res': {'values': [args.output_res]},
                'dropout_rate': {'values': [float(x) for x in args.dropout_rate.split(',')]},
                'pointwise_dropout_rate': {'values': [float(x) for x in args.pointwise_dropout_rate.split(',')]},
                'lr_base1': {'values':[float(x) for x in args.lr_base1.split(',')]},
                'lr_base2': {'values':[float(x) for x in args.lr_base2.split(',')]},
                'gradient_clip': {'values': [float(x) for x in args.gradient_clip.split(',')]},
                'atac_scale': {'values': [float(x) for x in args.atac_scale.split(',')]},
                'decay_frac': {'values': [float(x) for x in args.decay_frac.split(',')]},
                'num_transformer_layers': {'values': [int(x) for x in args.num_transformer_layers.split(',')]},
                'num_heads': {'values': [int(x) for x in args.num_heads.split(',')]},
                'num_random_features': {'values':[int(x) for x in args.num_random_features.split(',')]},
                'kernel_transformation': {'values':[args.kernel_transformation]},
                'epsilon': {'values':[args.epsilon]},
                'load_init_FT': {'values':[parse_bool_str(x) for x in args.load_init_FT.split(',')]},
                'load_init_FULL': {'values':[parse_bool_str(x) for x in args.load_init_FULL.split(',')]},
                'filter_list_seq': {'values': [[int(x) for x in args.filter_list_seq.split(',')]]},
                'filter_list_atac': {'values': [[int(x) for x in args.filter_list_atac.split(',')]]},
                'BN_momentum': {'values': [args.BN_momentum]},
                'use_seq': {'values':[parse_bool_str(x) for x in args.use_seq.split(',')]},
                'use_atac': {'values':[parse_bool_str(x) for x in args.use_atac.split(',')]},
                'atac_mask_dropout': {'values': [args.atac_mask_dropout]},
                'rectify': {'values':[parse_bool_str(x) for x in args.rectify.split(',')]},
                'log_atac': {'values':[parse_bool_str(x) for x in args.log_atac.split(',')]},
                'random_mask_size': {'values':[int(x) for x in args.random_mask_size.split(',')]},
                'final_point_scale': {'values':[int(x) for x in args.final_point_scale.split(',')]},
                'seed': {'values':[args.seed]},
                'atac_corrupt_rate': {'values': [int(x) for x in args.atac_corrupt_rate.split(',')]},
                'seq_corrupt_rate': {'values': [int(x) for x in args.seq_corrupt_rate.split(',')]},
                'use_tf_activity': {'values': [parse_bool_str(x) for x in args.use_tf_activity.split(',')]},
                'loss_type': {'values': [str(x) for x in args.loss_type.split(',')]},
                'freeze_conv_layers': {'values': [parse_bool_str(x) for x in args.freeze_conv_layers.split(',')]},
                'total_weight_loss': {'values': [float(x) for x in args.total_weight_loss.split(',')]}

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
            wandb.config.gcs_path_holdout=args.gcs_path_holdout
            wandb.config.num_epochs=args.num_epochs
            wandb.config.train_examples=args.train_examples
            wandb.config.val_examples=args.val_examples
            wandb.config.val_examples_ho=args.val_examples_ho
            wandb.config.val_examples_TSS=args.val_examples_TSS
            wandb.config.val_examples_TSS_ho=args.val_examples_TSS_ho
            wandb.config.batch_size=args.batch_size
            wandb.config.warmup_frac=args.warmup_frac
            wandb.config.patience=args.patience
            wandb.config.min_delta=args.min_delta
            wandb.config.model_save_dir=args.model_save_dir
            wandb.config.model_save_basename=args.model_save_basename
            wandb.config.max_shift=args.max_shift
            wandb.config.checkpoint_path=args.checkpoint_path
            wandb.config.crop_size = (wandb.config.output_length - wandb.config.final_output_length) // 2

            if (wandb.config.load_init_FULL and wandb.config.load_init_FT):
                raise ValueError('cannot load fine-tuning and FULL checkpoint at same time')

            run_name = '_'.join([str(int(wandb.config.input_length) / 1000)[:4].rstrip('.') + 'k',
                                 'load-FT-' + str(wandb.config.load_init_FT),
                                 'load-FULL-' + str(wandb.config.load_init_FULL),
                                 'LR1-' + str(wandb.config.lr_base1),
                                 'LR2-' + str(wandb.config.lr_base1),
                                 'T-' + str(wandb.config.num_transformer_layers),
                                 'TF-' + str(wandb.config.use_tf_activity)])

            date_string = f'{datetime.now():%Y-%m-%d %H:%M:%S%z}'
            date_string = date_string.replace(' ','_')
            wandb.run.name = run_name + "_" + date_string
            base_name = wandb.config.model_save_basename + "_" + wandb.run.name

            '''TPU init options'''
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy=\
                tf.data.experimental.AutoShardPolicy.DATA
            options.deterministic=False
            #options.experimental_threading.max_intra_op_parallelism=1
            mixed_precision.set_global_policy('mixed_bfloat16')

            NUM_REPLICAS = strategy.num_replicas_in_sync
            BATCH_SIZE_PER_REPLICA=wandb.config.batch_size
            GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA*NUM_REPLICAS
            print('global batch size:', GLOBAL_BATCH_SIZE)

            wandb.config.update({"train_steps": wandb.config.train_examples // (GLOBAL_BATCH_SIZE)},
                                allow_val_change=True)
            wandb.config.update({"val_steps" : wandb.config.val_examples // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)
            wandb.config.update({"total_steps": wandb.config.train_examples // GLOBAL_BATCH_SIZE},
                                allow_val_change=True)


            data_train,data_val = \
                    training_utils.return_distributed_iterators(wandb.config.gcs_path, wandb.config.gcs_path_holdout,
                                                                GLOBAL_BATCH_SIZE, wandb.config.input_length,
                                                                wandb.config.max_shift, wandb.config.output_length_ATAC,
                                                                wandb.config.output_length, wandb.config.crop_size,
                                                                wandb.config.output_res, args.num_parallel,
                                                                args.num_epochs, strategy, options,
                                                                wandb.config.atac_mask_dropout, wandb.config.random_mask_size,
                                                                wandb.config.log_atac, wandb.config.use_atac,
                                                                wandb.config.use_seq, wandb.config.seed,
                                                                wandb.config.seq_corrupt_rate, wandb.config.atac_corrupt_rate,
                                                                wandb.config.val_steps, wandb.config.use_tf_activity, g)

            inits=None
            print('created dataset iterators')
            if wandb.config.load_init_FT:
                print('loading fine-tuning weights')
                inits=load_weights_atac_rna.get_initializers_genformer_ft(wandb.config.checkpoint_path,
                                                                         wandb.config.num_transformer_layers,
                                                                         wandb.config.use_tf_activity)

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
                                    inits=inits,
                                    load_init_FT=wandb.config.load_init_FT,
                                    load_tf=wandb.config.use_tf_activity,
                                    final_point_scale=wandb.config.final_point_scale,
                                    filter_list_seq=wandb.config.filter_list_seq,
                                    freeze_conv_layers=wandb.config.freeze_conv_layers,
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

            optimizer1 = tf.keras.optimizers.Adam(learning_rate=scheduler1,
                                                  epsilon=wandb.config.epsilon)
            optimizer2 = tf.keras.optimizers.Adam(learning_rate=scheduler2,
                                                  epsilon=wandb.config.epsilon)

            optimizers_in = optimizer1,optimizer2

            metric_dict = {}

            train_step, val_step, build_step, metric_dict = training_utils.return_train_val_functions(model,
                                                                                            optimizers_in,
                                                                                            strategy,
                                                                                            metric_dict,
                                                                                            GLOBAL_BATCH_SIZE,
                                                                                            wandb.config.gradient_clip,
                                                                                            wandb.config.atac_scale,
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
                step_num = epoch_i * wandb.config.train_steps * GLOBAL_BATCH_SIZE
                if epoch_i == 1:
                    print('building model')
                    build_step(data_val)
                    print('built model')
                    if wandb.config.load_init_FULL:
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
                    strategy.run(train_step, args=(next(data_train),))

                print('train_loss: ' + str(metric_dict['train_loss'].result().numpy()))
                print('train_loss_rna: ' + str(metric_dict['train_loss_rna'].result().numpy()))
                print('train_loss_atac: ' + str(metric_dict['train_loss_atac'].result().numpy()))
                wandb.log({'train_loss': metric_dict['train_loss'].result().numpy(),
                           'train_loss_rna': metric_dict['train_loss_rna'].result().numpy(),
                           'train_loss_atac': metric_dict['train_loss_atac'].result().numpy()},
                          step=step_num)
                duration = (time.time() - start) / 60.
                print('completed epoch ' + str(epoch_i) + ' - duration(mins): ' + str(duration))


                ####### validation steps #######################
                start = time.time()
                pred_list = []
                true_list = []
                assay_list = []
                for k in range(wandb.config.val_steps):
                    true,pred,assay=strategy.run(val_step, args=(next(data_val),))
                    for x in strategy.experimental_local_results(true):
                        true_list.append(x)
                    for x in strategy.experimental_local_results(pred):
                        pred_list.append(x)
                    for x in strategy.experimental_local_results(assay):
                        assay_list.append(tf.reshape(x, [-1]))

                true_list = tf.concat(true_list,0).numpy()
                pred_list = tf.concat(pred_list,0).numpy()

                val_loss = metric_dict['val_loss'].result().numpy()
                print('val_loss: ' + str(metric_dict['val_loss'].result().numpy()))
                print('val_loss_rna: ' + str(metric_dict['val_loss_rna'].result().numpy()))
                print('val_loss_atac: ' + str(metric_dict['val_loss_atac'].result().numpy()))
                val_losses.append(val_loss)

                wandb.log({'val_loss': metric_dict['val_loss'].result().numpy(),
                           'val_loss_atac': metric_dict['val_loss_atac'].result().numpy(),
                           'val_loss_rna': metric_dict['val_loss_rna'].result().numpy()},
                           step=step_num)

                assay_list_concat = tf.concat(assay_list,0).numpy().astype(int)

                print(assay_list_concat.shape)

                rampage_100_idx = [i for i, val in enumerate(tf.concat(assay_list,0)) if val == 2]

                trues = tf.concat([true_list[i] for i in rampage_100_idx],0)
                print(tf.shape(trues))
                preds = tf.concat([pred_list[i] for i in rampage_100_idx],0)
                print(tf.shape(preds))
                rampage100_r,_ = pearsonr(trues,preds)

                val_pearsons.append(rampage100_r)

                print('ATAC_pearsons: ' + str(metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy()))
                print('ATAC_R2: ' + str(metric_dict['ATAC_R2'].result()['R2'].numpy()))
                print('RAMPAGE: ' + str(rampage100_r))
                wandb.log({'ATAC_pearsons': metric_dict['ATAC_PearsonR'].result()['PearsonR'].numpy(),
                           'ATAC_R2': metric_dict['ATAC_R2'].result()['R2'].numpy(),
                           'RAMPAGE_pearsons': rampage100_r},
                          step=step_num)

                end = time.time()
                duration = (end - start) / 60.
                print('completed epoch ' + str(epoch_i) + ' validation')
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
