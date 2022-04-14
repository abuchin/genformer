from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import tensorflow.experimental.numpy as tnp
import tensorflow as tf

from tensorflow.keras import layers as kl

import src.layers.fast_attention as fa

from src.layers.layers import Residual, FFN, conv1Dblock, TransformerBlock, abs_sin_PE, crop
import tensorflow_addons as tfa

SEQUENCE_LENGTH=409600

@tf.keras.utils.register_keras_serializable()
class genformer(tf.keras.Model):
    def __init__(self,
                 kernel_transformation: 'relu_kernel_transformation',
                 dropout_rate: float = 0.2,
                 final_out_length: int =50,
                 num_heads:int = 2,
                 numerical_stabilizer: float =0.001,
                 causal: bool = False,
                 nb_random_features:int = 256,
                 hidden_size:int = 64,
                 widening:int = 2,
                 conv_filter_size:int = 15,
                 transformer_depth:int = 8,
                 momentum: float = 0.90,
                 channels_list: list = [36, 36, 48, 48, 64],
                 kernel_regularizer: float = 0.001,
                 positional_encoding_type: str = 'abs_sin_PE',
                 positional_dropout_rate: float = None,
                 heads_dict: dict = {'hg':0,
                                     'mm':1,
                                     'rm':2},
                 name: str = 'genformer',
                 **kwargs):
        """ 'genformer' model based on Enformer for predicting RNA-seq from atac + sequence
        Args: to do 
        
        
          name: model name
        """

        super(genformer, self).__init__(name=name,**kwargs)
        self.kernel_transformation=kernel_transformation
        self.dropout_rate=dropout_rate
        self.final_out_length=final_out_length
        self.num_heads=num_heads
        self.numerical_stabilizer=numerical_stabilizer
        self.causal=causal
        self.nb_random_features=nb_random_features
        self.hidden_size=hidden_size
        self.widening=widening
        self.conv_filter_size=conv_filter_size
        self.transformer_depth=transformer_depth
        self.momentum=momentum
        self.channels_list=channels_list
        self.kernel_regularizer=kernel_regularizer
        self.positional_encoding_type = positional_encoding_type
        self.positional_dropout_rate=positional_dropout_rate
        self.heads_dict=heads_dict

        fast_attention_kwargs = {
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'attention_dropout': self.dropout_rate,
            'kernel_transformation': self.kernel_transformation,
            'numerical_stabilizer': self.numerical_stabilizer,
            'causal': self.causal,
            'nb_random_features': self.nb_random_features,
            'widening': self.widening,
            'dropout_rate': self.dropout_rate
        }

        
        ### initial stem 
        self.stem_initial_conv = kl.Conv1D(self.hidden_size // 2,
                                      kernel_size=self.conv_filter_size,
                                      strides=1,padding='same',
                                      kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.stem_residual_conv = conv1Dblock(num_channels=self.hidden_size // 2,
                                              conv_filter_size=1,stride=1,momentum=self.momentum,
                                              **kwargs,
                                              name = "stem_conv_block")
        self.stem_pool = kl.MaxPool1D(pool_size=2,strides=2,padding='valid') # todo: trial attention pooling
        
        
        
        ## conv tower
        self.conv_stack = tf.keras.Sequential()
        for k, channels in enumerate(self.channels_list):
            self.conv_stack.add(conv1Dblock(num_channels=channels,conv_filter_size=5,
                                            stride=1, momentum=self.momentum, **kwargs, 
                                            name = f'conv_stack_b_{k}'))
            self.conv_stack.add(Residual(conv1Dblock(num_channels=channels,conv_filter_size=1,
                                                     momentum=self.momentum, **kwargs, 
                                                     name = f'conv_stack_resb_{k}'),
                                         name = f'res_{k}'))
            self.conv_stack.add(kl.MaxPool1D(pool_size=2,strides=2,padding='valid')) # todo: trial attention pooling

        
        if positional_encoding_type == 'abs_sin_PE':
            self.sin_pe = abs_sin_PE(self.positional_dropout_rate, **kwargs)
        self.transformer_stack = tf.keras.Sequential()
        for k in range(self.transformer_depth):
            self.transformer_stack.add(TransformerBlock(**fast_attention_kwargs,
                                                        name = f'transformer_{k}'))
            
        ## at this stage have output binned at 128 bp resolution 
        ## only keep center 1/2 of input
        self.crop = crop(crop_frac=4)
        
        ## final conv stack, organism specific
        self._heads = {head: tf.keras.Sequential([
                                        conv1Dblock(num_channels=2*self.hidden_size,
                                                    conv_filter_size=1, stride=1,
                                                    momentum=self.momentum,name = 'final_conv'),
                                        kl.GlobalAveragePooling1D(data_format='channels_first'),
                                        tfa.layers.GELU(),
                                        kl.Dropout(self.dropout_rate),
                                        kl.Dense(self.final_out_length,
                                                 activation='softmax',
                                                 dtype=tf.float32,
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                 use_bias=True)
                                ])
                       
                       for head in self.heads_dict.keys()}
        """
                                        kl.Conv1D(1,
                                                  kernel_size=1,
                                                  strides=1,padding='same',
                                                  kernel_initializer=tf.keras.initializers.GlorotUniform()),
                                        tfa.layers.GELU(),
                                        kl.Dropout(self.dropout_rate,**kwargs),
                                        kl.Permute((2,1)),
                                        kl.Dense(self.final_out_length,
                                                 activation='softmax',
                                                 dtype=tf.float32,
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                 use_bias=True),
                                        kl.Reshape((self.final_out_length,))
                       ])
        """
        
    @property
    def trunk(self):
        return self._trunk
    @property
    def heads(self):
        return self._heads

    @tf.function#(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.float32)])
    def call(self, inputs: tf.Tensor, training:bool=None):
        x = self.stem_initial_conv(inputs,training=training)
        x = self.stem_residual_conv(x,training=training)
        x = self.stem_pool(x)
        x = self.conv_stack(x,training=training)
        if self.positional_encoding_type == 'abs_sin_PE':
            x = self.sin_pe(x,training=training)
        x = self.transformer_stack(x,training=training)
        x = self.crop(x)
        return {head: head_module(x)
                for head, head_module in self.heads.items()}
    
    @tf.function(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.float32)])
    def return_att_weights(self,inputs:tf.Tensor,training:bool=None):
        x = self.stem_initial_conv(inputs,training=training)
        x = self.stem_residual_conv(x,training=training)
        x = self.stem_pool(x)
        x = self.conv_stack(x,training=training)
        x = self.sin_pe(x,training=training)
        att_matrices = {}
        for idx, k in enumerate(self.transformer_stack.layers):
            att, k_prime, q_prime = k.return_attention_weights(x,training=False)
            att_matrices['layer_' + str(idx)] = (k_prime, q_prime)

        return att_matrices
                         
    def get_config(self):
        config = {
            "dropout_rate":self.dropout_rate,
            "kernel_regularizer":self.kernel_regularizer,
            "positional_encoding_type": self.positional_encoding_type,
            "positional_dropout_rate":self.positional_dropout_rate,
            "final_out_length": self.final_out_length,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "numerical_stabilizer": self.numerical_stabilizer,
            "kernel_transformation": self.kernel_transformation,
            "causal": self.causal,
            "nb_random_features": self.nb_random_features,
            "conv_filter_size":self.conv_filter_size,
            "transformer_depth":self.transformer_depth,
            "widening":self.widening,
            "kernel_regularizer":self.kernel_regularizer,
            "momentum":self.momentum,
            "heads_dict":self.heads_dict,
            "channels_list":self.channels_list,
            'final_conv_stack': self.final_conv_stack_channels_list
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    def from_config(cls, config):
        return cls(**config)
    
    @tf.function(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.float32)])
    def predict(self, x: tf.Tensor,training:bool=None):
        """Method for SavedModel."""
        return self(x, training=False)
