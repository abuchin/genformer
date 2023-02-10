from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import tensorflow.experimental.numpy as tnp
import tensorflow as tf

from tensorflow.keras import layers as kl

import tensorflow_addons as tfa
from tensorflow.keras import regularizers

from tensorflow.keras.layers.experimental import SyncBatchNormalization as syncbatchnorm


@tf.keras.utils.register_keras_serializable()
class enformer_convs(tf.keras.Model):
    def __init__(self,
                 load_init=False,
                 model_inits=None,
                 filter_list_seq=None,
                 use_stem_first_only=False,
                 name: str = 'enformer_convs',
                 **kwargs):
        """ 'aformer' model based on Enformer for predicting RNA-seq from atac + sequence
        Args: to do 
        
        
          name: model name
        """

        super(enformer_convs, self).__init__(name=name,**kwargs)
        self.load_init=load_init
        self.model_inits=model_inits
        self.filter_list_seq = [768, 896, 1024, 1152, 1280, 1536] if self.load_init else filter_list_seq
        #self.filter_list_atac = filter_list_atac
        
        
        def enf_conv_block(filters, 
                           width=1, 
                           w_init='glorot_uniform', 
                           padding='same', 
                           name='conv_block',
                           beta_init=None,
                           gamma_init=None,
                           mean_init=None,
                           var_init=None,
                           kernel_init=None,
                           bias_init=None,
                           train=True,
                           **kwargs):
            return tf.keras.Sequential([
              syncbatchnorm(axis=-1,
                            center=True,
                            scale=True,
                            beta_initializer=beta_init if self.load_init else "zeros",
                            gamma_initializer=gamma_init if self.load_init else "ones",
                            trainable=train,
                            moving_mean_initializer=mean_init if self.load_init else "zeros",
                            moving_variance_initializer=var_init if self.load_init else "ones",
                            **kwargs),
              tfa.layers.GELU(),
              tf.keras.layers.Conv1D(filters,
                                     width, 
                                     kernel_initializer=kernel_init if self.load_init else w_init,
                                     bias_initializer=bias_init if self.load_init else bias_init,
                                     trainable=train,
                                     padding=padding, **kwargs)
            ], name=name)
        
        ### conv stack for sequence inputs
        self.stem_conv = tf.keras.layers.Conv1D(filters= int(self.filter_list_seq[-1]) // 2,
                                   kernel_size=15,
                                   kernel_initializer=self.model_inits['stem_conv_k'] if self.load_init else 'glorot_uniform',
                                   bias_initializer=self.model_inits['stem_conv_b'] if self.load_init else 'zeros',
                                   strides=1,
                                   trainable=False,
                                   padding='same')
                                   #data_format='channels_last')
        self.stem_res_conv=Residual(enf_conv_block(int(self.filter_list_seq[-1]) // 2, 1,
                                                   beta_init=self.model_inits['stem_res_conv_BN_b'] if self.load_init else None,
                                                   gamma_init=self.model_inits['stem_res_conv_BN_g'] if self.load_init else None,
                                                   mean_init=self.model_inits['stem_res_conv_BN_m'] if self.load_init else None,
                                                   var_init=self.model_inits['stem_res_conv_BN_v'] if self.load_init else None,
                                                   kernel_init=self.model_inits['stem_res_conv_k'] if self.load_init else None,
                                                   bias_init=self.model_inits['stem_res_conv_b'] if self.load_init else None,
                                                   train=False,
                                                   name='pointwise_conv_block'))
        self.stem_pool = SoftmaxPooling1D(per_channel=True,
                                          w_init_scale=2.0,
                                          pool_size=2,
                                          k_init=self.model_inits['stem_pool'] if self.load_init else None,
                                          train=False,
                                          name ='stem_pool')


        self.conv_tower_seq = tf.keras.Sequential([
            tf.keras.Sequential([
                enf_conv_block(num_filters, 
                               5, 
                               beta_init=self.model_inits['BN1_b_' + str(i)] if self.load_init else None,
                               gamma_init=self.model_inits['BN1_g_' + str(i)] if self.load_init else None,
                               mean_init=self.model_inits['BN1_b_' + str(i)] if self.load_init else None,
                               var_init=self.model_inits['BN1_v_' + str(i)] if self.load_init else None,
                               kernel_init=self.model_inits['conv1_k_' + str(i)] if self.load_init else None,
                               bias_init=self.model_inits['conv1_b_' + str(i)] if self.load_init else None,
                               train=False,
                               padding='same'),
                Residual(enf_conv_block(num_filters, 1, 
                                       beta_init=self.model_inits['BN2_b_' + str(i)] if self.load_init else None,
                                       gamma_init=self.model_inits['BN2_g_' + str(i)] if self.load_init else None,
                                       mean_init=self.model_inits['BN2_b_' + str(i)] if self.load_init else None,
                                       var_init=self.model_inits['BN2_v_' + str(i)] if self.load_init else None,
                                       kernel_init=self.model_inits['conv2_k_' + str(i)] if self.load_init else None,
                                       bias_init=self.model_inits['conv2_b_' + str(i)] if self.load_init else None,
                                        train=False,
                                        name='pointwise_conv_block')),
                kl.MaxPooling1D(pool_size=2,
                                    strides=2,
                                    name=f'pooling_block_{i}')
                ],
                       name=f'conv_tower_block_{i}')
            for i, num_filters in enumerate(self.filter_list_seq)], name='conv_tower')
        

        self.gap = kl.GlobalAveragePooling1D()
        
        
    def call(self, peaks, use_stem_first_only=False, training:bool=False):
        
        if use_stem_first_only:
            x = self.stem_conv(peaks,training=False)
            return self.gap(x)#self.gap(x)
        else:
            x = self.stem_conv(peaks,training=False)
            x = self.stem_res_conv(x,training=False)
            x = self.stem_pool(x,training=False)
            x = self.conv_tower_seq(x,training=False)
            return self.gap(x)#self.gap(x)
    

    def get_config(self):
        config = {
            "filter_list_seq":self.filter_list_seq
            
        }
        
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)



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
    
    def get_config(self):
        config = {
            "layer": self._layer
        }
        base_config = super().get_config()
        return {**base_config, **config}
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    def call(self, inputs, training=None,**kwargs):
        return inputs + self._layer(inputs, training=training, **kwargs)
    
    
    
    
class SoftmaxPooling1D(kl.Layer):
    def __init__(self, pool_size: int = 2, 
                 w_init_scale: float = 2.0,
                 k_init=None,
                 train=True,
                 per_channel: bool = True,
                 name: str='SoftmaxPooling1D'):
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
        self.train=train
        self._k_init=k_init
        
    def build(self, input_shape):
        num_features = input_shape[-1]
        if self._per_channel:
            units=num_features
        else:
            units=1
        self._logit_linear = kl.Dense(units=units,
                                      use_bias=False,
                                      trainable=self.train,
                                      kernel_initializer=self._k_init if (self._k_init is not None) else tf.keras.initializers.Identity(gain=self._w_init_scale))
        super(SoftmaxPooling1D,self).build(input_shape)
                                            
    ### revisit 
    def call(self, inputs):
        _, length, num_features = inputs.shape
        #print(inputs.shape)
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
