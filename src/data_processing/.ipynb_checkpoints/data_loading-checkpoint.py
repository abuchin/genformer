import tensorflow as tf
import tensorflow.experimental.numpy as tnp


'''
Data loading functions
'''


def deserialize(serialized_example,input_length,output_length):
    """Deserialize bytes stored in TFRecordFile."""
    feature_map = {
        'inputs': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([],tf.string),
        'tss_tokens': tf.io.FixedLenFeature([],tf.string)
    }

    data = tf.io.parse_example(serialized_example, feature_map)

    return {
        'inputs': tf.ensure_shape(tf.io.parse_tensor(data['inputs'],
                                                     out_type=tf.float32),
                                  [input_length,5]),
        'target': tf.ensure_shape(tf.io.parse_tensor(data['target'],
                                                     out_type=tf.float32),
                                  [output_length,]),
        'tss_tokens': tf.ensure_shape(tf.io.parse_tensor(data['tss_tokens'],
                                                     out_type=tf.int32),
                                  [output_length,])
    }



def return_dataset(gcs_path,
                   split,
                   organism,
                   prefetch,batch,
                   num_parallel=8):
    wc = str(organism) + "*.tfrecords"
    list_files = (tf.io.gfile.glob(os.path.join(gcs_path,
                                                split,
                                                wc)))
    random.shuffle(list_files)
    files = tf.data.Dataset.list_files(list_files)

    dataset = tf.data.TFRecordDataset(files,
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_parallel)


    dataset=dataset.map(lambda record: deserialize(record),num_parallel_calls=num_parallel)

    return dataset.repeat().batch(batch).prefetch(tf.data.experimental.AUTOTUNE)