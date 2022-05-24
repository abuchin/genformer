from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import tensorflow.experimental.numpy as tnp
import tensorflow as tf

from tensorflow.keras import layers as kl

from src.layers.layers import *
import tensorflow_addons as tfa
from tensorflow.keras import regularizers

SEQUENCE_LENGTH=163840

@tf.keras.utils.register_keras_serializable()
class genformer(tf.keras.Model):
    def __init__(self,
                 kernel_transformation: 'relu_kernel_transformation',
                 dropout_rate: float = 0.2,
                 final_out_length:int=30,
                 num_heads:int = 2,
                 numerical_stabilizer: float =0.001,
                 nb_random_features:int = 256,
                 hidden_size:int = 64,
                 widening:int = 2,
                 conv_filter_size:int = 15,
                 transformer_depth:int = 8,
                 momentum: float = 0.90,
                 channels_list: list = [36, 36, 48, 48, 64],
                 kernel_regularizer: float = 0.01,
                 d_model = 64 ,
                 dim = 32, 
                 max_seq_length = 4096,
                 # nb_random_features = 64, 
                 rel_pos_bins=128, 
                 use_spe=False, 
                 spe_type=None, 
                 kernel_size=None, 
                 use_rot_emb = False,
                 use_mask_pos = True, 
                 normalize = False,
                 seed = 42,
                 heads_dict: dict = {'hg':0},
                                     #'mm':1,
                                     #'rm':2},
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
        self.nb_random_features=nb_random_features
        self.hidden_size=hidden_size
        self.widening=widening
        self.conv_filter_size=conv_filter_size
        self.transformer_depth=transformer_depth
        self.momentum=momentum
        self.channels_list=channels_list
        self.kernel_regularizer=kernel_regularizer
        self.heads_dict=heads_dict
        
        self.d_model = d_model
        self.dim = dim
        self.max_seq_length = max_seq_length
        # self.nb_random_features = nb_random_features
        self.rel_pos_bins = rel_pos_bins
        self.use_spe = use_spe
        self.spe_type = spe_type
        self.kernel_size = kernel_size
        self.use_rot_emb = use_rot_emb
        self.use_mask_pos = use_mask_pos
        self.normalize = normalize
        self.seed = seed
        
        ### initial stem 
        self.stem_initial_conv = kl.Conv1D(self.hidden_size // 2,
                                           kernel_size=self.conv_filter_size,
                                           strides=1,padding='same',
                                           input_shape=(SEQUENCE_LENGTH, 5),
                                           kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                           kernel_regularizer=regularizers.L2(self.kernel_regularizer)
                                           )
        self.stem_gelu = tfa.layers.GELU()
        self.stem_residual_conv = conv1Dblock(num_channels=self.hidden_size // 2,
                                              conv_filter_size=1,stride=1,momentum=self.momentum,kernel_regularizer=self.kernel_regularizer,
                                              **kwargs,
                                              name = "stem_conv_block")
        self.stem_pool = kl.MaxPool1D(pool_size=2,strides=2,padding='valid') # todo: trial attention pooling
        #switch for softmax pooling 
        
        
        ## conv tower
        self.conv_stack = tf.keras.Sequential()
        for k, channels in enumerate(self.channels_list):
            self.conv_stack.add(conv1Dblock(num_channels=channels,conv_filter_size=5,
                                            stride=1, momentum=self.momentum,kernel_regularizer=self.kernel_regularizer, **kwargs, 
                                            name = f'conv_stack_b_{k}'))
            self.conv_stack.add(Residual(conv1Dblock(num_channels=channels,conv_filter_size=1,
                                                     momentum=self.momentum, kernel_regularizer=self.kernel_regularizer,**kwargs, 
                                                     name = f'conv_stack_resb_{k}'),
                                         name = f'res_{k}'))
            self.conv_stack.add(kl.MaxPool1D(pool_size=2,strides=2,padding='valid')) # todo: trial attention pooling


        self.transformer_stack = Performer_Encoder(num_layers=self.transformer_depth, 
                                                  num_heads=self.num_heads, dim = self.dim,
                                                  d_model=self.d_model,
                                                  max_seq_length=self.max_seq_length, 
                                                  nb_random_features=self.nb_random_features, 
                                                  widening=self.widening,
                                                  hidden_size=self.hidden_size,
                                                  numerical_stabilizer=self.numerical_stabilizer,
                                                  attention_dropout=self.dropout_rate,
                                                  rel_pos_bins=self.rel_pos_bins, 
                                                  use_spe=self.use_spe, spe_type=self.spe_type, 
                                                  kernel_size=self.kernel_size, 
                                                  use_rot_emb=self.use_rot_emb, 
                                                  use_mask_pos=self.use_mask_pos,
                                                  kernel_transformation=self.kernel_transformation,
                                                  normalize=self.normalize, seed = self.seed, **kwargs)

        ## at this stage have output binned at 128 bp resolution 
        ## only keep center 1/2 of input
        self.crop = crop(crop_frac=4)
        
        ## final conv stack, organism specific
        self._heads = {head: tf.keras.Sequential([
                                        conv1Dblock(num_channels=2*self.hidden_size,
                                                    conv_filter_size=1, stride=1,kernel_regularizer=self.kernel_regularizer,
                                                    momentum=self.momentum,name = 'final_conv', **kwargs),
                                        #kl.GlobalAveragePooling1D(data_format='channels_first'),
                                        tfa.layers.GELU(),
                                        kl.Dropout(self.dropout_rate,**kwargs),
                                        kl.Dense(1,
                                                 activation='softplus',
                                                 dtype=tf.float32,
                                                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                                 use_bias=True)
                                ])
                       
                       for head in self.heads_dict.keys()}

    @property
    def trunk(self):
        return self._trunk
    @property
    def heads(self):
        return self._heads

    #@tf.function(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.float32)])
    def call(self, inputs: tf.Tensor, training:bool=None):
        x = self.stem_initial_conv(inputs,training=training)
        x = self.stem_gelu(x)
        x = self.stem_residual_conv(x,training=training)
        x = self.stem_pool(x)
        x = self.conv_stack(x,training=training)
        x,att_matrices = self.transformer_stack(x,training=training)
        x = self.crop(x)
        return [{head: head_module(x,training=training)
                for head, head_module in self.heads.items()},att_matrices]
    

    """
    @tf.function(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.float32)])
    def return_att_weights(self,inputs:tf.Tensor):
        x = self.stem_initial_conv(inputs,training=False)
        x = self.stem_gelu(x)
        x = self.stem_residual_conv(x,training=False)
        x = self.stem_pool(x)
        x = self.conv_stack(x,training=False)
        # x = self.sin_pe(x,training=training)
        return self.transformer_stack.return_attention_weights(x)
    """
    def get_config(self):
        config = {
            "dropout_rate":self.dropout_rate,
            "kernel_regularizer":self.kernel_regularizer,
            "final_out_length": self.final_out_length,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "numerical_stabilizer": self.numerical_stabilizer,
            "kernel_transformation": self.kernel_transformation,
            "nb_random_features": self.nb_random_features,
            "conv_filter_size":self.conv_filter_size,
            "transformer_depth":self.transformer_depth,
            "widening":self.widening,
            "kernel_regularizer":self.kernel_regularizer,
            "momentum":self.momentum,
            "heads_dict":self.heads_dict,
            "channels_list":self.channels_list,
            "d_model":self.d_model,
            "dim":self.dim,
            "max_seq_length":self.max_seq_length,
            "rel_pos_bins":self.rel_pos_bins,
            "use_spe":self.use_spe,
            "spe_type":self.spe_type,
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
    
    @tf.function(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.float32)])
    def predict_on_batch(self, inputs: tf.Tensor):
        """Method for SavedModel."""
        x = self.stem_initial_conv(inputs,training=False)
        x = self.stem_gelu(x)
        x = self.stem_residual_conv(x,training=False)
        x = self.stem_pool(x)
        x = self.conv_stack(x,training=False)
        x,att_matrices = self.transformer_stack(x,training=False)
        x = self.crop(x)
        return [{head: head_module(x,training=False)
                for head, head_module in self.heads.items()},att_matrices]
