import argparse
import collections
import gzip
import heapq
import json
import math
import pdb
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import h5py

import tensorflow.io as tfio
import tensorflow as tf
from tensorflow import strings as tfs

import seqio
from seqio.scripts.cache_tasks_main import run_pipeline
import apache_beam as beam
from seqio import beam_utils
from apache_beam.options.pipeline_options import PipelineOptions
import re
import helpers
#from preprocessors import preprocessors



def seqIO_task_register(input_cell_types,
                        sweep_id,
                        gcs_path,
                        input_length,
                        output_length,
                        input_type,
                        seed):
    registered = []
    for cell_type in input_cell_types:
        task_name = re.sub("-", "_", cell_type) + re.sub("-", "_", sweep_id)
        add_task(cell_type,
                 sweep_id,
                 gcs_path,
                 input_length,
                 output_length,
                 input_type)
        registered.append(task_name)
    return registered

def seqIO_mix(registered_tasks,
              sweep_id,
              gcs_path,
              input_length,
              #input_length_t,
              output_length,
              #crop_size_i,
              crop_size,
              target_length,
              in_seed,
              cached):
    '''
    function which takes in a list of 
    GCS TFR paths and returns a mixed seqiodataset
    '''
    mix_list = []
    global input_length_g 
    input_length_g = input_length
    
    global output_length_g
    output_length_g = output_length
    
    #global crop_size_i_g
    #crop_size_i_g = crop_size_i
    #global target_length_i_g
    #target_length_i_g = input_length_t

    global crop_size_g
    crop_size_g = crop_size
    global target_length_g
    target_length_g = target_length
        
    seqio.MixtureRegistry.add(
        "mixture" + re.sub("-", "_", sweep_id),
        [(task, 1) for task in registered_tasks]
    )
    
    train = seqIO_get_dataset("train", "mixture" + re.sub("-", "_", sweep_id),
                              input_length_g, 
                              output_length_g,
                              in_seed)
    val = seqIO_get_dataset("val", "mixture" + re.sub("-", "_", sweep_id),
                             input_length_g, output_length_g,
                             in_seed)
    test = seqIO_get_dataset("test", "mixture" + re.sub("-", "_", sweep_id),
                             input_length_g, output_length_g,
                             in_seed)
    
    return train, val, test

def seqIO_get_dataset(split, mix_name,
                     input_length, output_length,
                     in_seed):
    return seqio.get_mixture_or_task(mix_name).get_dataset(
        sequence_length={"inputs": input_length_g, 
                         "target": output_length_g},
        split=split,
        shuffle=False,
        num_epochs=15,
        use_cached=False)

        
def add_task(input_cell_type,
             sweep_id,
             gcs_path,
             input_length,
             output_length,
             input_type):
    '''
    add a gcs tfrecord to seqio task
    '''

    input_file_tr = gcs_path + '/' + "train" + '/' + input_cell_type + '.tfr'
    input_file_val = gcs_path + '/' + "val" + '/' + input_cell_type + '.tfr'
    input_file_te = gcs_path + '/' + "test" + '/' + input_cell_type + '.tfr'
    #print(input_cell_type)
    #ds = ds.map(_parse_function,
    #                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    task_name = re.sub("-", "_", input_cell_type) + re.sub("-", "_", sweep_id)
    
    feature_description = {
        'sequence': tf.io.FixedLenFeature([], tf.string),
        'atac': tf.io.FixedLenFeature([], tf.string),
        #'exons': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([],tf.string),
        'tss_tokens': tf.io.FixedLenFeature([],tf.string)
        #'genes_list': tf.io.FixedLenFeature([], tf.string,default_value=''),
        #'tx_list': tf.io.FixedLenFeature([], tf.string,default_value=''),
        #'organism': tf.io.FixedLenFeature([], tf.string,default_value='')
    }
    
    if input_type == 'sequence_exon_atac':
        preprocessors_list = [parse_tfrecord_tss,
                              one_hot_concatenate_all]
    elif input_type == 'sequence_exon':
        preprocessors_list = [parse_tfrecord_tss,
                              one_hot_concatenate_seq_exon]
    else:
        raise ValueError('Please provide one o sequence_exon_atac or sequence exon as input_type')
    
    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TFExampleDataSource(
                        split_to_filepattern = {"train": input_file_tr,
                                                "val": input_file_val,
                                                "test": input_file_te},
                        feature_description = feature_description,
                        reader_cls = read_compressed),
        preprocessors=preprocessors_list,
        output_features={
            "inputs": seqio.Feature(
               vocabulary=seqio.PassThroughVocabulary,
               add_eos=False, dtype=tf.float32, rank=2
            ),
            "target": seqio.Feature(
               vocabulary=seqio.PassThroughVocabulary,
               add_eos=False, dtype=tf.float32, rank=1
            ),
            "tss_tokens": seqio.Feature(
               vocabulary=seqio.PassThroughVocabulary,
               add_eos=False, dtype=tf.int32, rank=1
            )},
        metric_fns=[])



def seqIO_cache(input_cell_types,
                gcs_path,
                input_length,
                output_length,
                in_seed):
    for cell_type in input_cell_types:
        add_task_tss(cell_type,
                 gcs_path,
                 input_length,
                 output_length)
        task_name = re.sub("-", "_", cell_type)
        job_name = re.sub("_", "", task_name).lower()
        beam_options = PipelineOptions(
            runner='DataflowRunner',
            project='picard-testing-176520',
            job_name=job_name,
            temp_location='gs://picard-testing-176520/temp',
            region='us-central1',
            max_num_workers='500',
            setup_file='/home/jupyter/models/genformer/setup.py')

        with beam.Pipeline(options=beam_options) as pipeline:
            tf.io.gfile.makedirs(gcs_path + '/cache_dir')
            run_pipeline(pipeline = pipeline,
                         task_names=[task_name],
                         cache_dir = gcs_path + '/cache_dir')
#@tf.function
def one_hot_concatenate_all(ds):
    '''
    function to one hot encode sequence and concatenate sequence
    '''
    @seqio.map_over_dataset
    def concatenate(ex):
        sequence = one_hot(ex['sequence']) #tf.slice(one_hot(ex['sequence']),
                            #[crop_size_i_g,0],
                            #[target_length_i_g,-1])
        atac =ex['atac']# tf.slice(ex['atac'],
                        #[crop_size_i_g],
                        #[target_length_i_g])
        tss_tokens = tf.slice(ex['tss_tokens'],
                              [crop_size_g],
                              [target_length_g])
        
        target = tf.slice(ex['target'],
                          [crop_size_g],
                          [target_length_g])

        
        return {
            'inputs':tf.concat([tf.expand_dims(atac, 1),
                                               sequence], axis=1),                     
            'target': tf.cast(target, dtype = tf.float32),
            'tss_tokens': tf.cast(tss_tokens, dtype = tf.int32)
        }

    return concatenate(ds)

#@tf.function
def one_hot_concatenate_seq_exon(ds):
    '''
    function to one hot encode sequence and concatenate sequence
    '''
    @seqio.map_over_dataset
    def concatenate(ex):
        sequence = tf.slice(ex['sequence']
                            [crop_size_i],
                            [target_length_i])
        #exons = tf.cast(ex['exons'], dtype = tf.float32)
        
        target = tf.slice(ex['target'],
                          [crop_size_g],
                          [target_length_g])
        tss_tokens = tf.slice(ex['tss_tokens'],
                              [crop_size_g],
                              [target_length_g])
        return {
            'inputs': tf.concat([tf.expand_dims(exons, 1),
                                 sequence], axis=1),                  
            'target': tf.cast(target, dtype = tf.float32),
            'tss_tokens': tf.cast(tss_tokens, dtype = tf.int32)
        }

    return concatenate(ds)


#@tf.function
def parse_tfrecord_tss(ds):
    
    @seqio.map_over_dataset
    def parse(data):
        atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                              out_type=tf.float32), [input_length_g,])
        target = tf.ensure_shape(tf.io.parse_tensor(data['target'],
                      out_type=tf.float32), [output_length_g,])
        tss_tokens = tf.ensure_shape(tf.io.parse_tensor(data['tss_tokens'],
                      out_type=tf.int32), [output_length_g,])
        sequence = data['sequence']

        return {"sequence" : sequence, 
                "atac": atac, 
               # "exons": exons,
                "target":target,
                "tss_tokens": tss_tokens}

    return parse(ds)



def read_compressed(dataset):
    return tf.data.TFRecordDataset(dataset,
                                   compression_type="ZLIB",
                                   num_parallel_reads=tf.data.experimental.AUTOTUNE)


def one_hot(sequence):
    '''
    convert input string tensor to one hot encoded
    will replace all N character with 0 0 0 0
    '''
    vocabulary = tf.constant(['A', 'T', 'C', 'G', 'N'])
    mapping = tf.constant([0, 1, 2, 3, 4])

    init = tf.lookup.KeyValueTensorInitializer(keys=vocabulary,
                                               values=mapping)
    table = tf.lookup.StaticHashTable(init, default_value=0)
    
    input_characters = tfs.upper(tfs.unicode_split(sequence, 'UTF-8'))

    out = tf.one_hot(table.lookup(input_characters), 
                      depth = 4, 
                      dtype=tf.float32)
    return out