from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import tensorflow.experimental.numpy as tnp
import tensorflow as tf

from tensorflow.keras import layers as kl

from src.layers.layers import *
import tensorflow_addons as tfa
from tensorflow.keras import regularizers

from tensorflow.keras.layers.experimental import SyncBatchNormalization as syncbatchnorm

SEQUENCE_LENGTH=65536

@tf.keras.utils.register_keras_serializable()
class aformer(tf.keras.Model):
    def __init__(self,
                 kernel_transformation: 'relu_kernel_transformation',
                 dropout_rate: float = 0.2,
                 input_length: int = 16384,
                 num_heads:int = 2,
                 numerical_stabilizer: float =0.001,
                 nb_random_features:int = 256,
                 hidden_size:int = 64,
                 widening:int = 2,
                 conv_filter_size_1:int = 128,
                 conv_filter_size_2:int = 25,
                 transformer_depth:int = 8,
                 momentum: float = 0.90,
                 channels_list: list = [36, 36, 48, 48, 64],
                 kernel_regularizer: float = 0.01,
                 d_model = 64,
                 norm=True,
                 dim = 32, 
                 max_seq_length = 4096,
                 # nb_random_features = 64, 
                 rel_pos_bins=128, 
                 kernel_size=None, 
                 use_rot_emb = True,
                 use_mask_pos = False, 
                 normalize = True,
                 seed = 3,
                 TF_inputs=1572,
                 heads_dict: dict = {'hg':0},
                                     #'mm':1,
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
        self.conv_filter_size_1=conv_filter_size_1
        self.conv_filter_size_2=conv_filter_size_2
        self.transformer_depth=transformer_depth
        self.momentum=momentum
        self.channels_list=channels_list
        self.kernel_regularizer=kernel_regularizer
        self.heads_dict=heads_dict
        self.TF_inputs=TF_inputs
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
        
        ### initial stem 
        self.stem_initial_conv_seq = kl.Conv1D(self.hidden_size // 4,
                                           kernel_size=self.conv_filter_size_1,
                                           strides=1,padding='same',
                                           input_shape=(self.input_length, 5),
                                           kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                           kernel_regularizer=regularizers.L2(self.kernel_regularizer)
                                           )
        self.stem_gelu_seq = tfa.layers.GELU()
        self.stem_residual_conv_seq = Residual(conv1Dblock(num_channels=self.hidden_size // 4,
                                              conv_filter_size=1,stride=1,momentum=self.momentum,kernel_regularizer=self.kernel_regularizer,
                                              **kwargs,
                                              name = "stem_conv_block"), name = 'stem_res_conv')
        self.stem_pool_seq = kl.MaxPool1D(pool_size=2,strides=2,padding='valid') # todo: trial attention pooling
        #switch for softmax pooling 
        
        #self.dropout = kl.Dropout(rate=self.dropout_rate,**kwargs)
        ## conv tower
        """
        self.conv_stack_seq = tf.keras.Sequential()
        for k, channels in enumerate(self.channels_list):
            self.conv_stack_seq.add(conv1Dblock(num_channels=channels,conv_filter_size=5,
                                            stride=1, momentum=self.momentum,kernel_regularizer=self.kernel_regularizer, **kwargs, 
                                            name = f'conv_stack_seq_b_{k}'))
            self.conv_stack_seq.add(Residual(conv1Dblock(num_channels=channels,conv_filter_size=1,
                                                     momentum=self.momentum, kernel_regularizer=self.kernel_regularizer,**kwargs, 
                                                     name = f'conv_stack_seq_resb_{k}'),
                                         name = f'res_{k}'))
            self.conv_stack_seq.add(kl.MaxPool1D(pool_size=2,strides=2,padding='valid')) # todo: trial attention pooling
            
        """ #mbconv trial
        self.conv_stack_seq = tf.keras.Sequential()
        for k, channels in enumerate(self.channels_list):
            self.conv_stack_seq.add(conv1Dblock(num_channels=channels,conv_filter_size=5,
                                            stride=1, momentum=self.momentum,kernel_regularizer=self.kernel_regularizer, **kwargs, 
                                            name = f'conv_stack_seq_b_{k}'))
            self.conv_stack_seq.add(Residual(conv1Dblock(num_channels=channels,conv_filter_size=1,
                                                     momentum=self.momentum, kernel_regularizer=self.kernel_regularizer,**kwargs, 
                                                     name = f'conv_stack_seq_resb_{k}'),
                                         name = f'res_{k}'))
            self.conv_stack_seq.add(kl.MaxPool1D(pool_size=2,strides=2,padding='valid')) # todo: trial attention pooling
        
            
        self.stem_initial_conv_atac = kl.Conv1D(self.hidden_size // 4,
                                           kernel_size=self.conv_filter_size_1,
                                           strides=1,padding='same',
                                           input_shape=(self.input_length,1),
                                           kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                           kernel_regularizer=regularizers.L2(self.kernel_regularizer)
                                           )
        self.stem_gelu_atac = tfa.layers.GELU()
        self.stem_residual_conv_atac = Residual(conv1Dblock(num_channels=self.hidden_size // 4,
                                              conv_filter_size=1,stride=1,momentum=self.momentum,kernel_regularizer=self.kernel_regularizer,
                                              **kwargs,
                                              name = "stem_conv_block"), name = 'stem_res_conv')
        self.stem_pool_atac = kl.MaxPool1D(pool_size=2,strides=2,padding='valid') # todo: trial attention pooling
        #switch for softmax pooling 
        
        #self.dropout = kl.Dropout(rate=self.dropout_rate,**kwargs)
        ## conv tower
        self.conv_stack_atac = tf.keras.Sequential()
        for k, channels in enumerate(self.channels_list):
            self.conv_stack_atac.add(conv1Dblock(num_channels=channels,conv_filter_size=5,
                                            stride=1, momentum=self.momentum,kernel_regularizer=self.kernel_regularizer, **kwargs, 
                                            name = f'conv_stack_atac_b_{k}'))
            self.conv_stack_atac.add(Residual(conv1Dblock(num_channels=channels,conv_filter_size=1,
                                                     momentum=self.momentum, kernel_regularizer=self.kernel_regularizer,**kwargs, 
                                                     name = f'conv_stack_atac_resb_{k}'),
                                         name = f'res_{k}'))
            self.conv_stack_atac.add(kl.MaxPool1D(pool_size=2,strides=2,padding='valid')) # todo: trial attention pooling
            
            
        self.transformer_stack = Performer_Encoder(num_layers=self.transformer_depth, 
                                                  num_heads=self.num_heads, dim = self.dim,
                                                  d_model=self.d_model,
                                                   norm=self.norm,
                                                  max_seq_length=self.max_seq_length, 
                                                  nb_random_features=self.nb_random_features, 
                                                  widening=self.widening,
                                                  hidden_size=self.hidden_size,
                                                  numerical_stabilizer=self.numerical_stabilizer,
                                                  attention_dropout=self.dropout_rate,
                                                  rel_pos_bins=self.rel_pos_bins,  
                                                  kernel_size=self.kernel_size, 
                                                  use_rot_emb=self.use_rot_emb, 
                                                  use_mask_pos=self.use_mask_pos,
                                                  kernel_transformation=self.kernel_transformation,
                                                  normalize=self.normalize, seed = self.seed, **kwargs)

        #self.crop = crop(crop_frac=16)
        
        ## final conv stack, organism specific
        self._heads = {head: tf.keras.Sequential([
                                        conv1Dblock(num_channels=2*self.hidden_size,
                                                    conv_filter_size=1, stride=1,kernel_regularizer=self.kernel_regularizer,
                                                    momentum=self.momentum,name = 'final_conv', **kwargs),
                                        kl.GlobalAveragePooling1D(
                                            data_format='channels_last', **kwargs),
                                        kl.Dense(64,
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                 kernel_regularizer=regularizers.L2(self.kernel_regularizer),
                                                 use_bias=True),
                                        tfa.layers.GELU(),
                                        syncbatchnorm(axis=-1,
                                                momentum=self.momentum,
                                                center=True,
                                                scale=True,
                                                beta_initializer="zeros",
                                                gamma_initializer="ones",
                                                **kwargs),
                                        kl.Dense(1,
                                                 dtype=tf.float32,
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                 kernel_regularizer=regularizers.L2(self.kernel_regularizer),
                                                 use_bias=True),
                                       tf.keras.layers.Activation('softplus',
                                                                  dtype=tf.float32)
                                ]) for head in self.heads_dict.keys()}
        
        self.hg_TF_module = tf.keras.Sequential([
                                        kl.Dense(self.TF_inputs,
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                 kernel_regularizer=regularizers.L2(self.kernel_regularizer),
                                                 use_bias=True),
                                        tfa.layers.GELU(),
                                        syncbatchnorm(axis=-1,
                                                momentum=self.momentum,
                                                center=True,
                                                scale=True,
                                                beta_initializer="zeros",
                                                gamma_initializer="ones",
                                                **kwargs),
                                        kl.Dense(128,
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                 kernel_regularizer=regularizers.L2(self.kernel_regularizer),
                                                 use_bias=True),
                                        tfa.layers.GELU(),
                                        syncbatchnorm(axis=-1,
                                                momentum=self.momentum,
                                                center=True,
                                                scale=True,
                                                beta_initializer="zeros",
                                                gamma_initializer="ones",
                                                **kwargs)
                                        ])
                                    

    @property
    def heads(self):
        return self._heads

    #@tf.function(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.bfloat16),
    #                              tf.TensorSpec([None, 1572], tf.bfloat16)])
    def call(self, inputs, training:bool=True):
        
        sequence,atac,tf_inputs = inputs
        
        # TF processing module
        tf_out = self.hg_TF_module(tf_inputs,
                                   training=training)
        tf_out = tf.expand_dims(
            tf_out, axis=1, name=None
        )
        tf_out=tf.repeat(tf_out, (self.input_length // 2), axis=1)
        
        # sequence processing module
        x_seq = self.stem_initial_conv_seq(sequence,training=training)
        x_seq = self.stem_gelu_seq(x_seq)
        x_seq = self.stem_residual_conv_seq(x_seq,training=training)
        x_seq = self.stem_pool_seq(x_seq) ### here dimension is 131072 / 2, C = hidden/size / 2
        
        x_atac = self.stem_initial_conv_atac(atac,training=training)
        x_atac = self.stem_gelu_atac(x_atac)
        x_atac = self.stem_residual_conv_atac(x_atac,training=training)
        x_atac = self.stem_pool_atac(x_atac) ### here dimension is 131072 / 2, C = hidden/size / 2
        
        #### this has dimension sequence_length // 2 x 64 channels
        conv_stack_input_seq = tf.concat([x_seq,tf_out],axis=2)
        x_seq_conv_output = self.conv_stack_seq(conv_stack_input_seq,training=training)

        conv_stack_input_atac = tf.concat([x_atac,tf_out],axis=2)
        x_atac_conv_output = self.conv_stack_atac(conv_stack_input_atac,training=training)
        
        transformer_input = tf.concat([x_seq_conv_output,
                                       x_atac_conv_output],axis=2)
        
        x,att_matrices = self.transformer_stack(transformer_input,training=training)
        #x = self.crop(x)
        out = self.heads['hg'](x,training=training)
        #print(out.shape)
        
        return [{head: head_module(x,
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
            "conv_filter_size_1":self.conv_filter_size_1,
            "conv_filter_size_2":self.conv_filter_size_2,
            "transformer_depth":self.transformer_depth,
            "widening":self.widening,
            "kernel_regularizer":self.kernel_regularizer,
            "momentum":self.momentum,
            "heads_dict":self.heads_dict,
            "channels_list":self.channels_list,
            "d_model":self.d_model,
            "norm":self.norm,
            "dim":self.dim,
            "max_seq_length":self.max_seq_length,
            "rel_pos_bins":self.rel_pos_bins,
            "kernel_size":self.kernel_size,
            "use_rot_emb":self.use_rot_emb,
            "use_mask_pos":self.use_mask_pos,
            "normalize":self.normalize,
            "seed":self.seed
        }
        
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    #@tf.function(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.float32),
    #                              tf.TensorSpec([None, 1572], tf.float32)])
    def predict_on_batch(self, inputs):
        """Method for SavedModel."""
        sequence,atac,tf_inputs = inputs
        
        # TF processing module
        tf_out = self.hg_TF_module(tf_inputs,
                                   training=training)
        tf_out = tf.expand_dims(
            tf_out, axis=1, name=None
        )
        tf_out=tf.repeat(tf_out, (self.input_length // 2), axis=1)
        
        # sequence processing module
        x_seq = self.stem_initial_conv(sequence,training=False)
        x_seq = self.stem_gelu(x_seq)
        x_seq = self.stem_residual_conv(x_seq,training=False)
        x_seq = self.stem_pool(x_seq) ### here dimension is 131072 / 2, C = hidden/size / 2
        
        x_atac = self.stem_initial_conv(atac,training=False)
        x_atac = self.stem_gelu(x_atac)
        x_atac = self.stem_residual_conv(x_atac,training=False)
        x_atac = self.stem_pool(x_atac) ### here dimension is 131072 / 2, C = hidden/size / 2
        
        #### this has dimension sequence_length // 2 x 64 channels
        conv_stack_input_seq = tf.concat([x_seq,tf_out],axis=2)
        x_seq_conv_output = self.conv_stack_seq(conv_stack_input_seq,training=False)

        conv_stack_input_atac = tf.concat([x_atac,tf_out],axis=2)
        x_atac_conv_output = self.conv_stack_seq(conv_stack_input_atac,training=False)
        
        transformer_input = tf.concat([conv_stack_input_seq,
                                       conv_stack_input_atac],axis=2)
        
        x,att_matrices = self.transformer_stack(transformer_input,training=False)
        #x = self.crop(x)
        out = self.heads['hg'](x,training=False)
        #print(out.shape)
        
        return [{head: head_module(x,
                                   training=training)
                for head, head_module in self.heads.items()},att_matrices]


    
"""
                                        kl.Conv1D(1,
                                           kernel_size=1,
                                           strides=1,padding='same',
                                           kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                           kernel_regularizer=regularizers.L2(self.kernel_regularizer)
                                           ),
                                        tf.keras.layers.Activation('softplus')
"""