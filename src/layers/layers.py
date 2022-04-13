from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import tensorflow.experimental.numpy as tnp
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers as kl
import src.layers.fast_attention as fa
import src.utils as utils

@tf.keras.utils.register_keras_serializable()
class Residual(kl.Layer):
    def __init__(self, 
                 layer :  kl.Layer,
                 name : str = 'residual',
                 **kwargs):
        """Simple Residual block
        Args:
          name: Module name.
        """
        super().__init__(**kwargs,name=name)
        self._layer=layer
        self._add = kl.Add()
    
    def get_config(self):
        config = {
            "layer": tf.keras.layers.serialize(self._layer)
        }
        base_config = super().get_config()
        return {**base_config, **config}
    def call(self, inputs, training=None,**kwargs):
        return inputs + self._layer(inputs, training=training, **kwargs)


@tf.keras.utils.register_keras_serializable()
class crop(kl.Layer):
    def __init__(self, crop_frac: int = 4, name: str = 'cropping'):
        super().__init__(name=name)
        """Simple cropping layer
        Args:
          crop_frac: what fraction of input spatial dimension to crop from each end
                      e.g. crop_frac = 4 means 1/4 of input cropped from each end.
          name: Module name.
        """
        self._crop_frac = crop_frac
    
    def get_config(self):
        config = {"crop_frac":self._crop_frac}
        base_config = super().get_config()
        return {**base_config, **config}
    
    def call(self, inputs):
        crop_size = inputs.shape[1] // self._crop_frac
        return inputs[..., crop_size:-crop_size, :]


############################ conv 1D block #####################################
@tf.keras.utils.register_keras_serializable()
class conv1Dblock(kl.Layer):
    def __init__(self,
                 num_channels: int , 
                 conv_filter_size: int,
                 momentum: float,
                 stride: int = 1,
                 name: str = 'conv1Dblock',
                 **kwargs):
        """Enformer style conv block
        Args:
            num_channels
            conv_filter_size
            momentum: batch norm momentum
            stride: default 1 for no dim reduction
            name: Module name.
        """
        super().__init__(name=name, **kwargs)
        self.num_channels = num_channels
        self.conv_filter_size = conv_filter_size
        self.momentum = momentum
        self.stride=stride
        
        self.conv = kl.Conv1D(filters = self.num_channels,
                              kernel_size = self.conv_filter_size,
                              strides=self.stride,
                              padding='same',
                              kernel_initializer=tf.keras.initializers.GlorotUniform())
        self.gelu = tfa.layers.GELU()
        self.batch_norm = kl.BatchNormalization(axis=-1,
                                                momentum=self.momentum,
                                                center=True,
                                                scale=True,
                                                beta_initializer="zeros",
                                                gamma_initializer="ones",
                                                **kwargs)

    def get_config(self):
        config = {
            "num_channels":self.num_channels,
            "conv_filter_size":self.conv_filter_size,
            "momentum":self.momentum,
            "stride":self.stride
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    def call(self, inputs, training=None):
        x = self.batch_norm(inputs, training=training) 
        # todo: try switch order conv/batch norm for conventional conv block style
        x = self.gelu(x)
        x = self.conv(x)
        return x

@tf.keras.utils.register_keras_serializable()
class FFN(kl.Layer):
    def __init__(self, 
                 num_channels: int, 
                 widening: int, 
                 dropout_rate: float,
                 name: str = 'FFN',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        """FFN/MLP layer for transformer block
        Args:
            num_channels: num output channels
            widening: scaling factor for how many channels to start w/
                      e.g. widening = 2, num_channels = 12 means start w/ 24
            dropout_rate: dropout rate used throughout network
            name: Module name.
        """
        self.ffn_channels = num_channels
        self.ffn_widening = widening
        self.ffn_dropout = dropout_rate
            
        self.FFN_layer_norm = kl.LayerNormalization(axis=-1,
                                                  scale=True,
                                                  center=True,
                                                  beta_initializer="zeros",
                                                  gamma_initializer="ones")
        self.FFN_dense_wide = kl.Dense(self.ffn_channels*self.ffn_widening,
                                  activation='linear',
                                  use_bias=True)
        self.dropout = kl.Dropout(rate=self.ffn_dropout,**kwargs)
        self.relu = kl.ReLU()
        self.FFN_dense_narrow = kl.Dense(self.ffn_channels,
                                     activation='linear',
                                     use_bias=True)
    
    def get_config(self):
        config = {
            "ffn_channels":self.ffn_channels,
            "ffn_widening":self.ffn_widening,
            "ffn_dropout":self.ffn_dropout
        }
        base_config = super().get_config()
        return {**base_config,**config}
    
    def call(self, inputs, training=None):
        x = self.FFN_layer_norm(inputs)
        x = self.FFN_dense_wide(x)
        x = self.dropout(x,training=training)
        x = self.relu(x)
        x = self.FFN_dense_narrow(x)
        x = self.dropout(x,training=training)
        return x

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(kl.Layer):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 attention_dropout: float,
                 numerical_stabilizer: float,
                 causal: bool,
                 nb_random_features: int,
                 widening: int,
                 dropout_rate: float,
                 kernel_transformation: str = 'relu_kernel_transformation',
                 positional_dropout_rate: float = 0.10,
                 name = 'transformer_layer',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        """Transformer block w/ performer attention
        Args:
            hidden size: ~channel dimension for transformer input
            num_heads: num attention heads
            attention_dropout: post attention layer dropout rate
            numerical_stabilizer: small float for stability
            causal: causal masking desired, default to no for our case
            nb_random_features: dim for projection matrix
            widening: scaling factor for how many channels to start w/
                      e.g. widening = 2, num_channels = 12 means start w/ 24
            dropout_rate: transformer MLP dropout rate
            widening: scaling factor for how many channels to start w/
                      e.g. widening = 2, num_channels = 12 means start w/ 24
            dropout_rate: dropout rate used throughout network
            kernel_transformation: softmax or relu kernel transform for fast att.
            positional_encoding_type: absolute sinusoidal or relative(rotary)
            name: Module name.
        """
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.attention_dropout=attention_dropout
        self.kernel_transformation=kernel_transformation 
        self.numerical_stabilizer=numerical_stabilizer
        self.causal=causal
        self.nb_random_features=nb_random_features
        self.widening=widening
        self.dropout_rate=dropout_rate
        
        self.layer_norm = kl.LayerNormalization(axis=-1,
                                                  scale=True,
                                                  center=True,
                                                  beta_initializer="zeros",
                                                  gamma_initializer="ones")
        self.self_attention = fa.SelfAttention(hidden_size=self.hidden_size,
                                               num_heads=self.num_heads,
                                               attention_dropout=self.attention_dropout,
                                               kernel_transformation=self.kernel_transformation,
                                               numerical_stabilizer=self.numerical_stabilizer,
                                               causal=self.causal,
                                               nb_random_features=self.nb_random_features)
        self.dropout = kl.Dropout(rate=self.attention_dropout,**kwargs)
        self.FFN = FFN(num_channels=self.hidden_size,
                       widening=self.widening,
                       dropout_rate=self.dropout_rate,
                       name='FFN')         
    
    def get_config(self):
        config = {
            "hidden_size":self.hidden_size,
            "num_heads":self.num_heads,
            "attention_dropout":self.attention_dropout,
            "numerical_stabilizer":self.numerical_stabilizer,
            "causal": self.causal,
            "nb_random_features":self.nb_random_features,
            "widening":self.widening,
            "gen_dropout_rate":self.gen_dropout_rate,
            "kernel_transformation":self.kernel_transformation
        }
        base_config = super().get_config()
        return{**base_config, **config}
    
    def call(self, inputs, training=None):
        ## mha
        x = self.layer_norm(inputs)
        x, k_prime, q_prime = self.self_attention(x, training=training)
        x = self.dropout(x, training=training)
        mha_output = x + inputs
        
        ## ffn
        FFN_out = self.FFN(mha_output,training=training)
        return (FFN_out + mha_output)
    
    @tf.function
    def return_attention_weights(self,inputs,training=False):
        """ Method to return attention weights for saved model
            Returns: q_prime, k_prime from fast attention which 
            can be used to compute full approximated att. matrix
        """
        x = self.layer_norm(inputs)
        return self.self_attention(x, training=training)
    
    
@tf.keras.utils.register_keras_serializable()
class abs_sin_PE(kl.Layer):
    def __init__(self, 
                 positional_dropout_rate: float, 
                 name: str='sinusoidal_pos_encoding', 
                 **kwargs):
        """basic absolute sinusoidal PE layer
        Args:
            positional_dropout_rate: dropout rate for positional embeddings
        """
        super().__init__(name=name,**kwargs)
        self._positional_dropout_rate = positional_dropout_rate
        self._dropout = kl.Dropout(rate=self._positional_dropout_rate,**kwargs)
        
    def build(self, input_shape):
        self._pe = utils.sinusoidal(input_shape)
        super(abs_sin_PE,self).build(input_shape)

    def get_config(self):
        config = {
            "dropout":self._positional_dropout_rate
        }
        base_config = super().get_config()
        return{**base_config, **config}

    def call(self, inputs, training=None):
        return self._dropout(self._pe + inputs,
                             training=training)
    

    
@tf.keras.utils.register_keras_serializable()
class attention_pool(kl.Layer):
    def __init__(self, pool_size: int = 2, 
                 w_init_scale: float = 2.0,  
                 per_channel: bool = True,
                 name: str='attention_pool'):
        """Softmax pooling from enformer
        Args:
          pool_size: Pooling size, same as in Max/AvgPooling.
          per_channel: If True, the logits/softmax weights will be computed for
            each channel separately. If False, same weights will be used across all
            channels.
          w_init_scale: When 0.0 is equivalent to avg pooling, and when
            ~2.0 and `per_channel=False` it's equivalent to max pooling.
          name: Module name.
        """
        super().__init__(name=name)
        self._pool_size = pool_size
        self._per_channel=per_channel
        self._w_init_scale = w_init_scale
        self._logit_linear = None

    def build(self, input_shape):
        num_features = input_shape[-1]
        if self._per_channel:
            units=num_features
        else:
            units=1
        self._logit_linear = kl.Dense(units=units,
                                      use_bias=False,
                                      kernel_initializer=tf.keras.initializers.Identity(gain=self._w_init_scale))
        super(attention_pool, self).build(input_shape)
                                            
    ### revisit 
    def call(self, inputs):
        _, length, num_features = inputs.shape
        print(inputs.shape)
        inputs = tf.reshape(inputs, (-1, length // self._pool_size, 
                                     self._pool_size,  num_features))
        out = tf.reduce_sum(inputs * tf.nn.softmax(self._logit_linear(inputs), 
                                                   axis=-2), 
                            axis=-2)
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update ({
            "pool_size": self._pool_size,
            "w_init_scale": self._w_init_scale,
            "per_channel":self._per_channel
        })
        return config
    
    def cast_inputs(self, inputs):
        # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)