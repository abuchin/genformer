import tensorflow.io as tfio
import tensorflow as tf
from tensorflow import strings as tfs

import seqio
import re

global input_length_g 
input_length_g = 256000
global output_length_g
output_length_g = 2000
    

def one_hot(sequence,
            replaceN='all_1'):
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
                      dtype=tf.bfloat16)
    return out

@seqio.map_over_dataset
def one_hot_concatenate(data):
    '''
    function to one hot encode sequence and concatenate sequence
    '''
    return {
    'inputs': tf.concat([one_hot(data['sequence']),
                         tf.expand_dims(tf.cast(data['atac'],dtype=tf.bfloat16),
                                        1)], axis=1),
    'targets': tf.cast(data['rna'], dtype = tf.bfloat16)
    }

@seqio.map_over_dataset
def parse_tfrecord(data):
    
    atac = tf.ensure_shape(tf.io.parse_tensor(data['atac'],
                          out_type=tf.uint32), [input_length_g,])
    rna = tf.ensure_shape(tf.io.parse_tensor(data['rna'],
                  out_type=tf.uint32), [output_length_g,])
    sequence = data['sequence']

    return {"atac" : atac, "rna": rna, "sequence": sequence}

def read_compressed(dataset):
    'Example function'
    return tf.data.TFRecordDataset(dataset,
                                   compression_type="ZLIB")