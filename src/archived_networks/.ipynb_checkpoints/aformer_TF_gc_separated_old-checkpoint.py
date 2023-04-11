from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import tensorflow.experimental.numpy as tnp
import tensorflow as tf

from tensorflow.keras import layers as kl

from src.layers.layers_old import *
import tensorflow_addons as tfa
from tensorflow.keras import regularizers

from tensorflow.keras.layers.experimental import SyncBatchNormalization as syncbatchnorm

SEQUENCE_LENGTH=65536

@tf.keras.utils.register_keras_serializable()
class aformer(tf.keras.Model):
    def __init__(self,
                 kernel_transformation: 'softmax_kernel_transformation',
                 dropout_rate: float = 0.2,
                 positional_dropout_rate: float = 0.1,
                 input_length: int = 16384,
                 num_heads:int = 4,
                 numerical_stabilizer: float =0.001,
                 nb_random_features:int = 256,
                 hidden_size:int = 64,
                 widening:int = 2,
                 conv_filter_size_1_seq:int = 17,
                 conv_filter_size_2_seq:int = 5,
                 conv_filter_size_1_atac:int=50,
                 conv_filter_size_2_atac: int=5,
                 transformer_depth:int = 4,
                 momentum: float = 0.90,
                 channels_list: list = [36, 36, 48, 48, 64],
                 kernel_regularizer: float = 0.01,
                 d_model = 64,
                 bottleneck_units_tf=64,
                 bottleneck_units=64,
                 norm=True,
                 dim = 32, 
                 max_seq_length = 4096,
                 pooling_type='max',
                 # nb_random_features = 64, 
                 rel_pos_bins=128, 
                 kernel_size=None, 
                 use_rot_emb = True,
                 use_mask_pos = False, 
                 normalize = True,
                 seed = 3,
                 stride=1,
                 TF_inputs_hg=1637,
                 TF_inputs_mm=1366,
                 heads_dict: dict = {'hg':0,
                                     'mm':1},
                                     #'rm':2},
                 name: str = 'aformer',
                 **kwargs):
        """ 'aformer' model based on Enformer for predicting RNA-seq from atac + sequence
        Args: to do 
        
        
          name: model name
        """

        super(aformer, self).__init__(name=name,**kwargs)
        self.kernel_transformation=kernel_transformation
        self.dropout_rate=dropout_rate
        self.num_heads=num_heads
        self.input_length=input_length
        self.numerical_stabilizer=numerical_stabilizer
        self.nb_random_features=nb_random_features
        self.hidden_size=hidden_size
        self.widening=widening
        self.conv_filter_size_1_seq=conv_filter_size_1_seq
        self.conv_filter_size_2_seq=conv_filter_size_2_seq
        self.conv_filter_size_1_atac=conv_filter_size_1_atac
        self.conv_filter_size_2_atac=conv_filter_size_2_atac
        self.positional_dropout_rate=positional_dropout_rate
        self.pooling_type=pooling_type
        self.transformer_depth=transformer_depth
        self.momentum=momentum
        self.channels_list=channels_list
        self.kernel_regularizer=kernel_regularizer
        self.heads_dict=heads_dict
        self.bottleneck_units_tf=bottleneck_units_tf
        self.bottleneck_units=bottleneck_units
        self.norm=norm
        self.d_model = d_model
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.rel_pos_bins = rel_pos_bins
        self.kernel_size = kernel_size
        self.use_rot_emb = use_rot_emb
        self.use_mask_pos = use_mask_pos
        self.normalize = normalize
        self.seed = seed
        self.stride = stride
        
        ### conv stack for sequence inputs
        self.convstack_seq = convstackblock(initial_channels = self.hidden_size // 4,
                                            channels_list=self.channels_list,
                                            conv_filter_size_1=self.conv_filter_size_1_seq,
                                            conv_filter_size_2=self.conv_filter_size_2_seq,
                                            momentum=self.momentum,
                                            input_length=self.input_length,
                                            stride=self.stride,
                                            kernel_regularizer=self.kernel_regularizer,
                                            pooling_type=self.pooling_type, **kwargs)
        
        ### conv stack for atac inputs
        self.convstack_atac = convstackblock(initial_channels = self.hidden_size // 4,
                                            channels_list=self.channels_list,
                                            conv_filter_size_1=self.conv_filter_size_1_atac,
                                            conv_filter_size_2=self.conv_filter_size_2_atac,
                                            momentum=self.momentum,
                                            input_length=self.input_length,
                                            stride=self.stride,
                                            kernel_regularizer=self.kernel_regularizer,
                                            pooling_type=self.pooling_type, **kwargs)
                                        
        self.sin_pe = abs_sin_PE(self.positional_dropout_rate, **kwargs)

        self.transformer_stack = Performer_Encoder(num_layers=self.transformer_depth,
                                                   num_heads=self.num_heads, 
                                                   dim = self.dim,
                                                   d_model=self.d_model,
                                                   norm=self.norm,
                                                   max_seq_length=self.max_seq_length,
                                                   nb_random_features=self.nb_random_features,
                                                   widening=self.widening,
                                                   hidden_size=self.hidden_size,
                                                   numerical_stabilizer=self.numerical_stabilizer,
                                                   attention_dropout=self.dropout_rate / 5,
                                                   rel_pos_bins=self.rel_pos_bins,
                                                   kernel_size=self.kernel_size,
                                                   use_rot_emb=self.use_rot_emb,
                                                   use_mask_pos=self.use_mask_pos,
                                                   kernel_transformation=self.kernel_transformation,
                                                   normalize=self.normalize, seed = self.seed, **kwargs)

        
        ## final conv stack, organism specific
        self._heads = {head: headmodule_block(num_channels_in=2*self.hidden_size,
                                              momentum=self.momentum,
                                              dropout_rate=self.dropout_rate,
                                              kernel_regularizer=self.kernel_regularizer,
                                              bottleneck_units_tf=self.bottleneck_units_tf,
                                              bottleneck_units=self.bottleneck_units, **kwargs) for head in self.heads_dict.keys()}
        
    @property
    def heads(self):
        return self._heads

    #@tf.function(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.bfloat16),
    #                              tf.TensorSpec([None, 1572], tf.bfloat16)])
    def call(self, inputs, training:bool=True):
        
        sequence,atac,tf_inputs = inputs
        
        # sequence processing module
        x_seq = self.convstack_seq(sequence,training=training) ### here dimension is 131072 / 2, C = hidden/size / 2
        
        x_atac = self.convstack_atac(atac,training=training)

        transformer_input = tf.concat([x_seq,
                                       x_atac],axis=2)
        transformer_input = self.sin_pe(transformer_input)
        
        x,att_matrices = self.transformer_stack(transformer_input,training=training)

        org_spec_inputs = x,tf_inputs
        
        return [{head: head_module(org_spec_inputs,
                                   training=training)
                for head, head_module in self.heads.items()},att_matrices]
    

    def get_config(self):
        config = {
            "dropout_rate":self.dropout_rate,
            "input_length": self.input_length,
            "kernel_regularizer":self.kernel_regularizer,
            "final_out_length": self.final_out_length,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "numerical_stabilizer": self.numerical_stabilizer,
            "kernel_transformation": self.kernel_transformation,
            "nb_random_features": self.nb_random_features,
            "conv_filter_size_1_seq":self.conv_filter_size_1_seq,
            "conv_filter_size_2_seq":self.conv_filter_size_2_seq,
            "conv_filter_size_1_atac":self.conv_filter_size_1_atac,
            "conv_filter_size_2_atac":self.conv_filter_size_2_atac,
            "transformer_depth":self.transformer_depth,
            "widening":self.widening,
            "kernel_regularizer":self.kernel_regularizer,
            "momentum":self.momentum,
            "heads_dict":self.heads_dict,
            "channels_list":self.channels_list,
            "d_model":self.d_model,
            "norm":self.norm,
            "dim":self.dim,
            "bottleneck_units_tf":self.bottleneck_units_tf,
            "bottleneck_units":self.bottleneck_units,
            "human":self.human,
            "max_seq_length":self.max_seq_length,
            "rel_pos_bins":self.rel_pos_bins,
            "kernel_size":self.kernel_size,
            "use_rot_emb":self.use_rot_emb,
            "use_mask_pos":self.use_mask_pos,
            "normalize":self.normalize,
            "seed":self.seed,
            "TF_inputs_hg":self.TF_inputs_hg,
            "TF_inputs_mm":self.TF_inputs_mm,
            
        }
        
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    #@tf.function(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.float32),
    #                              tf.TensorSpec([None, 1572], tf.float32)])
    def predict_on_batch(self, inputs, training:bool=False):
        
        sequence,atac,tf_inputs = inputs
        
        # sequence processing module
        x_seq = self.convstack_seq(sequence,training=training) ### here dimension is 131072 / 2, C = hidden/size / 2
        
        x_atac = self.convstack_atac(atac,training=training)

        transformer_input = tf.concat([x_seq,
                                       x_atac],axis=2)
        transformer_input = self.sin_pe(transformer_input)
        
        x,att_matrices = self.transformer_stack(transformer_input,training=training)

        org_spec_inputs = x,tf_inputs
        
        return [{head: head_module(org_spec_inputs,
                                   training=training)
                for head, head_module in self.heads.items()},att_matrices]