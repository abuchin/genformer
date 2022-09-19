import argparse
import collections
import gzip
import math
import random
import shutil
import subprocess
import sys
import time
import matplotlib.pyplot as plt

import tensorflow as tf
import sonnet as snt
import tensorflow.experimental.numpy as tnp
import pandas as pd
import time
import os
import sys
import wandb
from wandb.keras import WandbCallback
import enformer_vanilla
import multiprocessing
import json
from tqdm import tqdm

from training_eval_fxns import define_dist_fxns
from metrics import CorrelationStats, PearsonR, R2, MetricDict
from data_loading import get_dataset

# ===========================================================================#


def main():
    parser = argparse.ArgumentParser(
        description='process input for enformer training loop')
    # ========================================================================
    # input files definition
    parser.add_argument('--tpu_name', dest = 'tpu_name',
                        help='tpu_name')
    parser.add_argument('--tpu_zone', dest = 'tpu_zone',
                        help='tpu_zone')
    parser.add_argument('--gcs_project', dest = 'gcs_project',
                        help='gcs_project')
    parser.add_argument('--num_epochs', dest = 'num_epochs',
                        type=int, help='num_epochs')
    parser.add_argument('--batch_size', dest = 'batch_size',
                        type=int, help='batch_size')
    parser.add_argument('--channels', dest = 'channels',
                        type=int, help='channels')
    parser.add_argument('--num_heads', dest = 'num_heads',
                        type=int,  help='num_heads')
    parser.add_argument('--num_transformer_layers', dest = 'num_transformer_layers',
                        type=int, help='num_transformer_layers')
    parser.add_argument('--pool_type', dest = 'pool_type',
                        help='pool_type')
    parser.add_argument('--target_learning_rate', dest = 'target_learning_rate',
                        type=float, help='target_learning_rate')
    parser.add_argument('--num_warmup_steps', dest = 'num_warmup_steps',
                        type=int, help='num_warmup_steps')
    parser.add_argument('--train_steps', dest = 'train_steps',
                        type=int, help='train_steps')
    parser.add_argument('--val_steps', dest = 'val_steps',
                        type=int, help='val_steps')
    parser.add_argument('--test_steps', dest = 'test_steps',
                        type=int, help='test_steps')
    parser.add_argument('--gradient_clip', dest = 'gradient_clip',
                        type=float, help='gradient_clip')
    parser.add_argument('--GCS_data_loc', dest = 'GCS_data_loc',
                        default='gs://basenji_barnyard/data',
                        help='GCS_data_loc')
    parser.add_argument('--num_parallel', dest = 'num_parallel',
                        type=int, default=multiprocessing.cpu_count(),
                        help='thread_count') 
    parser.add_argument('--wandb_project', 
                        dest='wandb_project',
                        help ='wandb_project')
    parser.add_argument('--wandb_user',
                        dest='wandb_user',
                        help ='wandb_user')
    parser.add_argument('--wandb_sweep_method',
                        dest='wandb_sweep_method',
                        help ='wandb_sweep_method')
    parser.add_argument('-data_splits_json',
                        required=True,
                        type=lambda f: open(f),
                        dest = 'data_splits_json',
                        help = 'data_splits_json')
    args = parser.parse_args()
    
    #================ init =======
    
    ### make sure gcloud auth set to picard-testing-176520
        
    ### make sure TPU started
    #command = 'gcloud compute tpus start ' + args.tpu_name + ' --zone=' + args.tpu_zone
    #subprocess.call(command,shell=True)
    subprocess.call("export CLOUDSDK_CORE_PROJECT=" + args.gcs_project, shell=True)
    subprocess.call("export CLOUDSDK_COMPUTE_ZONE=" + args.tpu_zone, shell=True)
    subprocess.call("export TPU_NAME=" + args.tpu_name, shell=True)
    #subprocess.call("os.environ['WANDB_MODE'] = 'offline'", shell=True)
    ## now start tpu

    #with strategy.scope():
    data_dict_list = []
    data_lookup = {}
    #data_splits_dict = json.loads(open(args.data_splits_json))
    for keys, values in json.load(args.data_splits_json).items():
        #key.split('_')[0] = assays
        human_indices = values[0]
        mouse_indices = values[1]

        human_len = len(human_indices)
        mouse_len = len(mouse_indices)

        if mouse_len == 0:
            data_dict_list.append({'human':human_len})
            data_lookup['human_' + str(human_len)] = human_indices

        else:
            data_dict_list.append({'human':human_len,
                                   'mouse':mouse_len})
            data_lookup['human_' + str(human_len)] = human_indices
            data_lookup['mouse_' + str(mouse_len)] = mouse_indices
    # ============== define sweep options ==================== #
    sweep_config = {
            "name" : "data_subset_sweep",
            'method': args.wandb_sweep_method,
            'metric': {
                'name': 'val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'data_subset':{
                    'values': data_dict_list
                },
            }
    }
    def sweep_train(config_defaults=None):
        # Set default values
        # Specify the other hyperparameters to the configuration, if any

        try: # detect TPUs
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(args.tpu_name) # TPU detection
            strategy = tf.distribute.TPUStrategy(tpu)
            #print('success')
        except ValueError: # no TPU found, detect GPUs
            #strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
            strategy = tf.distribute.get_strategy()
            
        with strategy.scope():
            config_defaults = {
                "data_subset": {'human':638, 'mouse':357}
            }

            wandb.init(config=config_defaults, project="data_subset_tests", entity="genformer")

            wandb.config.epochs=args.num_epochs
            wandb.config.train_steps=args.train_steps
            wandb.config.val_steps=args.val_steps
            wandb.config.batch_size=args.batch_size
            wandb.config.channels=args.channels
            wandb.config.num_heads=args.num_heads
            wandb.config.num_transformer_layers=args.num_transformer_layers    
            wandb.config.pool_type=args.pool_type
            wandb.config.target_learning_rate=args.target_learning_rate
            wandb.config.num_warmup_steps=args.num_warmup_steps
            wandb.config.gradient_clip=args.gradient_clip
            wandb.config.test_steps=args.test_steps
            out_heads = wandb.config.data_subset

            name = '_'.join([str(val) + str(key) for key,val in out_heads.items()])
            print(out_heads)
            datasets_tr = {}
            datasets_val = {}
            datasets_test={}

            metrics = {}

            if len(out_heads.keys()) == 2:
                h_len = out_heads['human']
                m_len = out_heads['mouse']
                h_indices = data_lookup['human_' + str(h_len)]
                m_indices = data_lookup['mouse_' + str(m_len)]
                batch = tf.constant(wandb.config.batch_size, dtype=tf.int64)
                train_dataset_h = get_dataset('human',
                                              'train',
                                              args.GCS_data_loc,
                                              h_indices,
                                              wandb.config.batch_size,
                                              args.num_parallel,
                                              wandb.config.epochs,
                                              tf.data.AUTOTUNE)
                dist_tr_h = iter(strategy.experimental_distribute_dataset(train_dataset_h))
                #print(next(dist_tr_h))
                train_dataset_m = get_dataset('mouse',
                                              'train',
                                              args.GCS_data_loc,
                                              m_indices,
                                              wandb.config.batch_size,
                                              args.num_parallel,
                                              wandb.config.epochs,
                                              tf.data.AUTOTUNE)
                dist_tr_m = iter(strategy.experimental_distribute_dataset(train_dataset_m))
                val_dataset_h = get_dataset('human',
                                            'valid',
                                            args.GCS_data_loc,
                                            h_indices,
                                            wandb.config.batch_size,
                                            args.num_parallel,
                                            wandb.config.epochs,
                                            tf.data.AUTOTUNE)
                dist_val_h = iter(strategy.experimental_distribute_dataset(val_dataset_h))
                val_dataset_m = get_dataset('mouse',
                                            'valid',
                                            args.GCS_data_loc,
                                            m_indices,
                                            wandb.config.batch_size,
                                            tf.data.AUTOTUNE,
                                            wandb.config.epochs,
                                            tf.data.AUTOTUNE)
                dist_val_m = iter(strategy.experimental_distribute_dataset(val_dataset_m))
                
                test_dataset_h = get_dataset('human',
                                            'test',
                                            args.GCS_data_loc,
                                            h_indices,
                                            wandb.config.batch_size,
                                            args.num_parallel,
                                            wandb.config.epochs,
                                            tf.data.AUTOTUNE)
                dist_test_h = iter(strategy.experimental_distribute_dataset(test_dataset_h))
                test_dataset_m = get_dataset('mouse',
                                            'test',
                                            args.GCS_data_loc,
                                            h_indices,
                                            wandb.config.batch_size,
                                            args.num_parallel,
                                            wandb.config.epochs,
                                            tf.data.AUTOTUNE)
                dist_test_m = iter(strategy.experimental_distribute_dataset(test_dataset_m))

                datasets_tr['human'] = dist_tr_h
                datasets_tr['mouse'] = dist_tr_m
                datasets_val['human'] = dist_val_h
                datasets_val['mouse'] = dist_val_m
                datasets_test['human'] = dist_test_h
                datasets_test['mouse'] = dist_test_m

                metrics['human'] = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1)), 'R2': R2(reduce_axis=(0,1))})
                metrics['mouse'] = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1)), 'R2': R2(reduce_axis=(0,1))})

            else:
                h_len = out_heads['human']
                h_indices = data_lookup['human_' + str(h_len)]
                #print(h_indices)
                train_dataset_h = get_dataset('human',
                                              'train',
                                              args.GCS_data_loc,
                                              h_indices,
                                              wandb.config.batch_size,
                                              tf.data.AUTOTUNE,
                                              wandb.config.epochs,
                                              tf.data.AUTOTUNE)

                dist_tr_h = iter(strategy.experimental_distribute_dataset(train_dataset_h))
                #print(next(dist_tr_h))
                val_dataset_h = get_dataset('human',
                                            'valid',
                                            args.GCS_data_loc,
                                            h_indices,
                                            wandb.config.batch_size,
                                            tf.data.AUTOTUNE,
                                            wandb.config.epochs,
                                            tf.data.AUTOTUNE)
                dist_val_h = iter(strategy.experimental_distribute_dataset(val_dataset_h))

                
                test_dataset_h = get_dataset('human',
                                            'test',
                                            args.GCS_data_loc,
                                            h_indices,
                                            wandb.config.batch_size,
                                            tf.data.AUTOTUNE,
                                            wandb.config.epochs,
                                            tf.data.AUTOTUNE)
                dist_test_h = iter(strategy.experimental_distribute_dataset(test_dataset_h))
                datasets_tr['human'] = dist_tr_h
                datasets_val['human'] = dist_val_h
                datasets_test['human'] = dist_test_h
                #print(next(dist_val_h))
                metrics['human'] = MetricDict({'PearsonR': PearsonR(reduce_axis=(0,1)), 'R2': R2(reduce_axis=(0,1))})

            # initialize model
            model = enformer_vanilla.Enformer(channels=wandb.config.channels,
                                              num_heads=wandb.config.num_heads,
                                              out_heads=out_heads,
                                              num_transformer_layers=wandb.config.num_transformer_layers,
                                              pooling_type = wandb.config.pool_type)
            print(out_heads)
            learning_rate = tf.Variable(0., trainable=False, name='learning_rate')
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


            
            @tf.function
            def dist_train_step_h(input_batch, model, gradient_clip):
                target=input_batch['target']
                sequence=input_batch['sequence']
                def train_step(target, sequence):
                    with tf.GradientTape() as tape:
                        outputs = model(sequence, is_training=True)['human']
                        loss = tf.reduce_mean(tf.keras.losses.poisson(target, outputs))
                    gradients = tape.gradient(loss, model.trainable_variables)
                    gradients, _ = tf.clip_by_global_norm(gradients, 0.2)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    return loss
                
                per_replica = strategy.run(train_step, args=(target, sequence))
                mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, 
                                            per_replica,axis=None)
                return mean_loss
            @tf.function
            def dist_train_step_m(input_batch, model, gradient_clip):
                target=input_batch['target']
                sequence=input_batch['sequence']
                def train_step(target, sequence):
                    with tf.GradientTape() as tape:
                        outputs = model(sequence, is_training=True)['mouse']
                        loss = tf.reduce_mean(tf.keras.losses.poisson(target, outputs))
                    gradients = tape.gradient(loss, model.trainable_variables)
                    gradients, _ = tf.clip_by_global_norm(gradients, 0.2)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    return loss
                
                per_replica = strategy.run(train_step, args=(target, sequence))
                mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, 
                                            per_replica,axis=None)
                return mean_loss

            @tf.function
            def dist_val_step_h(input_batch, model, in_metric):
                target=input_batch['target']
                sequence=input_batch['sequence']
                    
                def val_step(target, sequence):
                    outputs = model(sequence, is_training=False)['human']
                    loss = tf.reduce_mean(tf.keras.losses.poisson(target, outputs))
                    in_metric.update_state(target, outputs)
                    metrics_out = in_metric.result()

                    pearsonR = tf.reduce_mean(metrics_out['PearsonR'])
                    R2 = tf.reduce_mean(metrics_out['R2'])

                    return loss, pearsonR, R2

                per_rep_loss, per_rep_pearson, per_rep_R2 = strategy.run(val_step, 
                                                                         args=(target, sequence))

                mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, 
                                            per_rep_loss,axis=None)
                mean_pearson = strategy.reduce(tf.distribute.ReduceOp.MEAN, 
                                               per_rep_pearson, axis=None) 
                mean_R2 = strategy.reduce(tf.distribute.ReduceOp.MEAN, 
                                          per_rep_R2, axis=None)

                return mean_loss, mean_pearson, mean_R2
            @tf.function
            def dist_val_step_m(input_batch, model, in_metric):
                target=input_batch['target']
                sequence=input_batch['sequence']
                    
                def val_step(target, sequence):
                    outputs = model(sequence, is_training=False)['mouse']
                    loss = tf.reduce_mean(tf.keras.losses.poisson(target, outputs))
                    in_metric.update_state(target, outputs)
                    metrics_out = in_metric.result()

                    pearsonR = tf.reduce_mean(metrics_out['PearsonR'])
                    R2 = tf.reduce_mean(metrics_out['R2'])

                    return loss, pearsonR, R2

                per_rep_loss, per_rep_pearson, per_rep_R2 = strategy.run(val_step, 
                                                                         args=(target, sequence))

                mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, 
                                            per_rep_loss,axis=None)
                mean_pearson = strategy.reduce(tf.distribute.ReduceOp.MEAN, 
                                               per_rep_pearson, axis=None) 
                mean_R2 = strategy.reduce(tf.distribute.ReduceOp.MEAN, 
                                          per_rep_R2, axis=None)

                return mean_loss, mean_pearson, mean_R2

            global_step = 0

            for epoch_i in range(wandb.config.epochs):
                #for data_it in train_dist_dataset:
                train_loss = {}
                val_loss = {}
                val_pearsonR = {}
                val_R2 = {}
                for i in tqdm(range(wandb.config.train_steps)):
                        global_step += 1

                        if global_step > 1:
                            learning_rate_frac = tf.math.minimum(1.0, 
                                                                 global_step / tf.math.maximum(1.0, 
                                                                                               wandb.config.num_warmup_steps))      
                            learning_rate.assign(wandb.config.target_learning_rate * learning_rate_frac)

                        for organism, dataset in datasets_tr.items():
                            if organism == 'human':
                                loss = dist_train_step_h(next(dataset),
                                                       model,
                                                       wandb.config.gradient_clip)
                            else:
                                loss = dist_train_step_m(next(dataset),
                                                       model,
                                                       wandb.config.gradient_clip)
                            if organism not in train_loss.keys():
                                train_loss[organism] = []
                            else:
                                train_loss[organism].append(loss.numpy())

                for i in tqdm(range(wandb.config.val_steps)):
                    for organism, dataset in datasets_val.items():

                        in_metric = metrics[organism]
                        organism_tensor = tf.constant(organism)
                        if organism == 'human':
                            mean_loss, mean_pearson, mean_R2 = dist_val_step_h(next(dataset),
                                                                             model,
                                                                             in_metric)
                        else:
                            mean_loss, mean_pearson, mean_R2 = dist_val_step_m(next(dataset),
                                                                             model,
                                                                             in_metric)
                        if organism not in val_loss.keys():
                            val_loss[organism] = []
                            val_pearsonR[organism] = []
                            val_R2[organism] = []
                        else:
                            val_loss[organism].append(mean_loss.numpy())
                            val_pearsonR[organism].append(mean_pearson.numpy())
                            val_R2[organism].append(mean_R2.numpy())
                #print(val_pearsonR['human'].dtype)
                #print(val_R2['human'].dtype)
                if len(train_loss.keys()) == 2:
                    human_p = tnp.mean(val_pearsonR['human'])
                    wandb.log({'epochs': epoch_i,
                               'train_loss_human': tnp.mean(train_loss['human']).numpy(),
                               'train_loss_mouse': tnp.mean(train_loss['mouse']).numpy(),
                               'val_loss_human': tnp.mean(val_loss['human']).numpy(),
                               'val_loss_mouse': tnp.mean(val_loss['mouse']).numpy(),
                               'val_pearson_h': tnp.mean(val_pearsonR['human']).numpy(),
                               'val_pearson_m': tnp.mean(val_pearsonR['mouse']).numpy(),
                               'val_R2_h': tnp.mean(val_R2['human']).numpy(),
                               'val_R2_m': tnp.mean(val_R2['mouse']).numpy()})
                    print('epoch:', epoch_i)
                    print('train_loss_human:', tnp.mean(train_loss['human']).numpy())
                    print('train_loss_mouse:', tnp.mean(train_loss['mouse']).numpy())
                    print('val_loss_human:', tnp.mean(val_loss['human']).numpy())
                    print('val_loss_mouse:', tnp.mean(val_loss['mouse']).numpy())
                    print('val_pearson_h:', tnp.mean(val_pearsonR['human']).numpy())
                    print('val_pearson_m:', tnp.mean(val_pearsonR['mouse']).numpy())
                    print('val_R2_h:', tnp.mean(val_R2['human']).numpy())
                    print('val_R2_m:', tnp.mean(val_R2['mouse']).numpy())

                else:
                    wandb.log({'epochs': epoch_i,
                               'train_loss_human': tnp.mean(train_loss['human']).numpy(),
                               'val_loss_human': tnp.mean(val_loss['human']).numpy(),
                               'val_pearson_h': tnp.mean(val_pearsonR['human']).numpy(),
                               'val_R2_h': tnp.mean(val_R2['human']).numpy()})
                    print('epoch:', epoch_i)
                    print('train_loss_human:', tnp.mean(train_loss['human']).numpy())
                    print('val_loss_human:', tnp.mean(val_loss['human']).numpy())
                    print('val_pearson_h:', tnp.mean(val_pearsonR['human']).numpy())
                    print('val_R2_h:', tnp.mean(val_R2['human']).numpy())

                ### reset metric state
                for organism, metric in metrics.items():
                    metric.reset_state()
                if (epoch_i != 0 and epoch_i % 10 == 0):
                    #model.inference = inference
                    model.all_variables = list(model.variables)
                    path = "gs://picard-testing-176520/data_subset_sweep/" + name + "_" + str(epoch_i)
                
                    tf.saved_model.save(model, path)
                    test_pearsonR = {}
                    test_R2 = {}
                    
                    for i in tqdm(range(wandb.config.test_steps)):
                        for organism, dataset in datasets_test.items():
                            organism_tensor = tf.constant(organism)
                            in_metric = metrics[organism]
                            if organism == 'human':
                                mean_loss, mean_pearson, mean_R2 = dist_val_step_h(next(dataset),
                                                                                 model,
                                                                                 organism_tensor,
                                                                                 in_metric)
                            else:
                                mean_loss, mean_pearson, mean_R2 = dist_val_step_m(next(dataset),
                                                     model,
                                                     organism_tensor,
                                                     in_metric)
                            if organism not in test_pearsonR.keys():
                                test_pearsonR[organism] = []
                                test_R2[organism] = []
                            else:
                                test_pearsonR[organism].append(mean_pearson.numpy())
                                test_R2[organism].append(mean_R2.numpy())

                                
                    if len(test_pearsonR.keys()) == 2:
                        #human_p = tnp.mean(val_pearsonR['human'])
                        wandb.log({'test_pearsonR_human': tnp.mean(test_pearsonR['human']).numpy(),
                                   'test_R2_human': tnp.mean(test_R2['human']).numpy(),
                                   'test_pearsonR_mouse': tnp.mean(test_pearsonR['mouse']).numpy(),
                                   'test_R2_mouse': tnp.mean(test_R2['mouse']).numpy()})
                                   
                        print('test_pearsonR_human:', tnp.mean(test_pearsonR['human']).numpy())
                        print('test_R2_human:', tnp.mean(test_R2['human']).numpy())
                        print('test_pearsonR_mouse:', tnp.mean(test_pearsonR['mouse']).numpy())
                        print('test_R2_mouse:', tnp.mean(test_R2['mouse']).numpy())

                    else:
                        wandb.log({'test_pearsonR_human': tnp.mean(test_pearsonR['human']).numpy(),
                                   'test_R2_human': tnp.mean(test_R2['human']).numpy()})
                        print('test_pearsonR_human:', tnp.mean(test_pearsonR['human']).numpy())
                        print('test_R2_human:', tnp.mean(test_R2['human']).numpy())
                    for organism, metric in metrics.items():
                        metric.reset_state()
                    
    
    sweep_id = wandb.sweep(sweep_config, project="genformer")
    wandb.agent(sweep_id, function=sweep_train)
    #sweep_train()

##########################################################################
if __name__ == '__main__':
    main()
    
        