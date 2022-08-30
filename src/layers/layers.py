from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import tensorflow.experimental.numpy as tnp
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers as kl
#import src.layers.fast_attention as fa
import src.layers.fast_attention_rpe_genformer1 as fa_rpe
import src.utils as utils
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental import SyncBatchNormalization as syncbatchnorm

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
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs):
        crop_size = inputs.shape[1] // self._crop_frac
        out = inputs[..., crop_size:-crop_size, :]
        return out

############################ conv 1D block #####################################
@tf.keras.utils.register_keras_serializable()
class conv1Dblock(kl.Layer):
    def __init__(self,
                 num_channels: int , 
                 conv_filter_size: int,
                 momentum: float,
                 stride: int = 1,
                 kernel_regularizer: float = 0.01,
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
        self.kernel_regularizer=kernel_regularizer
        
        self.conv = kl.Conv1D(filters = self.num_channels,
                              kernel_size = self.conv_filter_size,
                              strides=self.stride,
                              padding='same',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              kernel_regularizer=tf.keras.regularizers.L2(self.kernel_regularizer))
        self.gelu = tfa.layers.GELU()
        self.batch_norm = syncbatchnorm(axis=-1,
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
            "stride":self.stride,
            "kernel_regularizer":self.kernel_regularizer
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.gelu(x)
        x = self.batch_norm(x, training=training) 
        
        return x

@tf.keras.utils.register_keras_serializable()
class seperable_conv1Dblock(kl.Layer):
    def __init__(self,
                 num_channels: int , 
                 conv_filter_size: int,
                 momentum: float,
                 stride: int = 1,
                 kernel_regularizer: float = 0.01,
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
        self.kernel_regularizer=kernel_regularizer
        
        self.conv = kl.SeparableConv1D(filters = self.num_channels,
                              kernel_size = self.conv_filter_size,
                              strides=self.stride,
                              padding='same',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              kernel_regularizer=tf.keras.regularizers.L2(self.kernel_regularizer))
        self.gelu = tfa.layers.GELU()
        self.batch_norm = syncbatchnorm(axis=-1,
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
            "stride":self.stride,
            "kernel_regularizer":self.kernel_regularizer
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.gelu(x)
        x = self.batch_norm(x, training=training) 
        # todo: try switch order conv/batch norm for conventional conv block style
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
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
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
                 nb_random_features: int,
                 widening: int,
                 dropout_rate: float,
                 kernel_transformation: str = 'relu_kernel_transformation',
                 name = 'transformer_layer',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        """Transformer block w/ performer attention
        Args:
            hidden size: ~channel dimension for transformer input
            num_heads: num attention heads
            attention_dropout: post attention layer dropout rate
            numerical_stabilizer: small float for stability
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
        self.nb_random_features=nb_random_features
        self.widening=widening
        self.dropout_rate=dropout_rate
        
        self.layer_norm = kl.LayerNormalization(axis=-1,
                                                  scale=True,
                                                  center=True,
                                                  beta_initializer="zeros",
                                                  gamma_initializer="ones")
        self.self_attention = fa.Attention(hidden_size=self.hidden_size,
                                               num_heads=self.num_heads,
                                               attention_dropout=self.attention_dropout,
                                               kernel_transformation=self.kernel_transformation,
                                               numerical_stabilizer=self.numerical_stabilizer,
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
            "nb_random_features":self.nb_random_features,
            "widening":self.widening,
            "gen_dropout_rate":self.gen_dropout_rate,
            "kernel_transformation":self.kernel_transformation
        }
        base_config = super().get_config()
        return{**base_config, **config}
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, training=None):
        ## mha
        x = self.layer_norm(inputs)
        x, k_prime, q_prime = self.self_attention(x, training=training,**kwargs)
        x = self.dropout(x, training=training)
        mha_output = x + inputs
        
        ## ffn
        FFN_out = self.FFN(mha_output,training=training)
        return (FFN_out + mha_output)
    
    @tf.function
    def return_attention_weights(self,inputs,**kwargs):
        """ Method to return attention weights for saved model
            Returns: q_prime, k_prime from fast attention which 
            can be used to compute full approximated att. matrix
        """
        x = self.layer_norm(inputs)
        return self.self_attention(x, training=False,**kwargs)
    
    
@tf.keras.utils.register_keras_serializable()
class Performer(kl.Layer):
    def __init__(self,
                 d_model,
                 normalize,
                 hidden_size: int,
                 num_heads: int,
                 seed: int,
                 attention_dropout: float,
                 numerical_stabilizer: float,
                 nb_random_features: int,
                 max_seq_length: int,
                 widening: int,
                 rel_pos_bins=None,
                 kernel_transformation: str = 'relu_kernel_transformation',
                 use_mask_pos: bool = False,
                 use_rot_emb: bool = True,
                 name = 'transformer_layer',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        """Transformer block w/ performer attention
        Args:
            hidden size: ~channel dimension for transformer input
            num_heads: num attention heads
            attention_dropout: post attention layer dropout rate
            numerical_stabilizer: small float for stability
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
        self.max_seq_length = max_seq_length
        self.nb_random_features=nb_random_features
        self.widening=widening
        self.rel_pos_bins = rel_pos_bins
        self.use_rot_emb=use_rot_emb
        self.use_mask_pos=use_mask_pos
        self.d_model=d_model
        self.normalize=normalize
        self.seed=seed
        
        
        self.layer_norm = kl.LayerNormalization(axis=-1,
                                                  scale=True,
                                                  center=True,
                                                  beta_initializer="zeros",
                                                  gamma_initializer="ones")
        self.self_attention = fa_rpe.Attention(hidden_size=self.d_model,
                                                   num_heads=self.num_heads,
                                                   nb_random_features=self.nb_random_features,
                                                   attention_dropout=self.attention_dropout,
                                                   use_rot_emb=self.use_rot_emb,
                                                   use_mask_pos=self.use_mask_pos,
                                                   max_seq_length=self.max_seq_length,
                                                   normalize=self.normalize,
                                                   kernel_transformation=self.kernel_transformation,
                                                   numerical_stabilizer=self.numerical_stabilizer,
                                                   seed=self.seed,
                                                   **kwargs)
        self.dropout = kl.Dropout(rate=self.attention_dropout,**kwargs)
        self.FFN = FFN(num_channels=self.hidden_size,
                       widening=self.widening,
                       dropout_rate=self.attention_dropout,
                       name='FFN',
                       **kwargs)         
    
    def get_config(self):
        config = {
            "hidden_size":self.hidden_size,
            "num_heads":self.num_heads,
            "attention_dropout":self.attention_dropout,
            "numerical_stabilizer":self.numerical_stabilizer,
            "nb_random_features":self.nb_random_features,
            "widening":self.widening,
            "kernel_transformation":self.kernel_transformation,
            "max_seq_length":self.max_seq_length,
            "rel_pos_bins":self.rel_pos_bins,
            "use_rot_emb":self.use_rot_emb,
            "use_mask_pos":self.use_mask_pos,
            "d_model":self.d_model,
            "normalize":self.normalize,
            "seed":self.seed
        }
        base_config = super().get_config()
        return{**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, rpe=None, training=None, **kwargs):
        x = self.layer_norm(inputs)
        x, k_prime, q_prime = self.self_attention(tf.cast(x,dtype=tf.float32),
                                                  tf.cast(x,dtype=tf.float32),
                                                  rpe=tf.cast(rpe,dtype=tf.float32),
                                                  **kwargs)

        x = self.dropout(x, training=training)

        mha_output = x + inputs
        ## ffn
        FFN_out = self.FFN(mha_output,training=training,**kwargs)
        return self.layer_norm(FFN_out + mha_output), k_prime, q_prime
    
    """
    @tf.function
    def return_attention(self,inputs,rpe,**kwargs):
         Method to return attention weights for saved model
            Returns: q_prime, k_prime from fast attention which 
            can be used to compute full approximated att. matrix
        
        x = self.layer_norm(inputs)
        return self.self_attention(x,x,rpe=rpe,**kwargs)
    """
    
@tf.keras.utils.register_keras_serializable()
class Performer_Encoder(kl.Layer):
    def __init__(self,
                 num_layers,
                 num_heads,
                 dim,
                 d_model,
                 max_seq_length,
                 nb_random_features,
                 widening,
                 hidden_size,
                 numerical_stabilizer,
                 attention_dropout = .1,
                 num_realization=1,
                 rel_pos_bins=None,
                 kernel_size=None,
                 use_rot_emb=False,
                 use_mask_pos=False,
                 normalize=False,
                 norm=True,
                 seed=42,
                 kernel_transformation: str = 'softmax_kernel_transformation',
                 name = 'performer_stack',
                 **kwargs):
        
        
        super().__init__(name=name, **kwargs)
        """Performer Encoder block
        Args:
            hidden size: ~channel dimension for transformer input
            num_heads: num attention heads
            attention_dropout: post attention layer dropout rate
            numerical_stabilizer: small float for stability
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
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.dim=dim
        self.hidden_size=hidden_size
        self.d_model=d_model
        self.max_seq_length=max_seq_length
        self.nb_random_features=nb_random_features
        self.attention_dropout=attention_dropout
        self.num_realization=num_realization
        self.numerical_stabilizer=numerical_stabilizer
        self.rel_pos_bins=rel_pos_bins#None#rel_pos_bins
        self.use_rot_emb=use_rot_emb
        self.use_mask_pos=use_mask_pos
        self.normalize=normalize
        self.norm=norm
        self.widening=widening
        self.kernel_transformation=kernel_transformation
        self.seed=seed
        
        self.layers = [Performer(d_model=self.d_model, 
                                 normalize=self.normalize,
                                 hidden_size=self.hidden_size,
                                 num_heads=self.num_heads, 
                                 attention_dropout=self.attention_dropout, 
                                 numerical_stabilizer=self.numerical_stabilizer,
                                 nb_random_features=self.nb_random_features,
                                 widening=self.widening,
                                 max_seq_length=self.max_seq_length,
                                 rel_pos_bins=self.rel_pos_bins,
                                 kernel_transformation=self.kernel_transformation,
                                 use_mask_pos=self.use_mask_pos,
                                 seed=self.seed,
                                 use_rot_emb=self.use_rot_emb,
                                 **kwargs) for i in range(self.num_layers)]
        
        self.layer_norm = kl.LayerNormalization(axis=-1,
                                                  scale=True,
                                                  center=True,
                                                  beta_initializer="zeros",
                                                  gamma_initializer="ones")
        
        
    def build(self, input_shape):
        N = input_shape[0]
        L = input_shape[1]
        
        if self.use_mask_pos:
            self.relative_positional_bias = tf.constant(tf.random.uniform((self.num_heads, 
                                                                           2 * self.rel_pos_bins - 1)))
            
        if self.use_rot_emb:
            self.pos_emb = FixedPositionalEmbedding(self.d_model, self.max_seq_length)
            self.layer_pos_emb = FixedPositionalEmbedding(self.dim, self.max_seq_length)       
        
        if self.use_mask_pos:
            if L <= self.rel_pos_bins:
                self.rpe = tf.concat((tf.expand_dims(self.relative_positional_bias[:,0], axis=1), 
                            self.relative_positional_bias[:,self.rel_pos_bins-L: self.rel_pos_bins+L-1]), axis=1)
            else:
                self.rpe = tf.concat([tf.repeat(tf.expand_dims(self.relative_positional_bias[:,0], axis=1), repeats= L-self.rel_pos_bins+1, axis=1), 
                        self.relative_positional_bias,
                        tf.repeat(tf.expand_dims(self.relative_positional_bias[:,-1], axis=1), repeats=L-self.rel_pos_bins, axis=1)], axis=1)

        super(Performer_Encoder,self).build(input_shape)
    
    def get_config(self):
        config = {
            "hidden_size":self.hidden_size,
            "num_heads":self.num_heads,
            "attention_dropout":self.attention_dropout,
            "numerical_stabilizer":self.numerical_stabilizer,
            "nb_random_features":self.nb_random_features,
            "widening":self.widening,
            "kernel_transformation":self.kernel_transformation,
            "num_layers":self.num_layers,
            "dim":self.dim,
            "d_model":self.d_model,
            "max_seq_length":self.max_seq_length,
            "num_realization":self.num_realization,
            "rel_pos_bins":self.rel_pos_bins,
            "use_rot_emb":self.use_rot_emb,
            "use_mask_pos":self.use_mask_pos,
            "normalize":self.normalize,
            "norm":self.norm,
            "seed":self.seed
        }

        base_config = super().get_config()
        return{**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, x, training=None, **kwargs):
        att_matrices={}
        for idx,layer in enumerate(self.layers):
            if self.use_rot_emb is True:
                x += self.pos_emb(x)
                rpe = self.layer_pos_emb(x)
                x,k_prime,q_prime = layer(x, rpe=rpe, training=training)
                att_matrices['layer_' + str(idx)] = (k_prime,q_prime)
                
            if self.use_mask_pos is True:
                x,k_prime,q_prime = layer(x, rpe=self.rpe, training=training)
                att_matrices['layer_' + str(idx)] = (k_prime,q_prime)
            
        if self.norm:
            x = self.layer_norm(x)
            
        return x,att_matrices
    
    """
    @tf.function
    def return_attention_weights(self,inputs):
         Method to return attention weights for saved model
            Returns: q_prime, k_prime from fast attention which 
            can be used to compute full approximated att. matrix
        
        att_matrices = {}
        ## to do, just call build here but for some reason wasn't showing up as model attribute
        N = inputs.shape[0]
        L = inputs.shape[1]
        
        self.relative_positional_bias = tf.constant(tf.random.uniform((self.num_heads, 2 * self.rel_pos_bins - 1)))
        
        if L <= self.rel_pos_bins:
            self.rpe = tf.concat((tf.expand_dims(self.relative_positional_bias[:,0], axis=1), 
                        self.relative_positional_bias[:,self.rel_pos_bins-L: self.rel_pos_bins+L-1]), axis=1)
        else:
            self.rpe = tf.concat([tf.repeat(tf.expand_dims(self.relative_positional_bias[:,0], axis=1), repeats= L-self.rel_pos_bins+1, axis=1), 
                    self.relative_positional_bias,
                    tf.repeat(tf.expand_dims(self.relative_positional_bias[:,-1], axis=1), repeats=L-self.rel_pos_bins, axis=1)], axis=1)
            
        for idx, k in enumerate(self.layers):
            att,k_prime,q_prime = k.return_attention(inputs,
                                                     rpe=self.rpe,
                                                     **kwargs)
            att_matrices['layer_' + str(idx)] = (k_prime,q_prime)
        return att_matrices
    """
    

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
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=None):
        return self._dropout(self._pe + inputs,
                             training=training)
    

@tf.keras.utils.register_keras_serializable()
class rotary_PE(kl.Layer):
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
    
    def cast_inputs(self, inputs):
        # Casts to float16, the policy's lowest-precision dtype
        return self._mixed_precision_policy.cast_to_lowest(inputs)

    

class FixedPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

    def build(self, input_shape):
        self.inv_freq = 1. / (10000 ** (tf.range(start=0, limit=self.dim, delta=2, dtype='float32') / self.dim))
        self.position = tf.range(start=0, limit=self.max_seq_len, delta=1, dtype='float32')
        self.sinusoid_inp = tf.einsum("i,j->ij", self.position, self.inv_freq)
        self.emb = tf.concat((tf.math.sin(self.sinusoid_inp), 
                              tf.math.cos(self.sinusoid_inp)), axis=-1)

    def call(self, x):
        return tf.cast(self.emb[None, :x.shape[1], :],
                       dtype=tf.bfloat16)
    

############################ conv stack block #####################################
@tf.keras.utils.register_keras_serializable()
class convstackblock(kl.Layer):
    def __init__(self,
                 initial_channels: int,
                 channels_list: list , 
                 conv_filter_size_1: int,
                 conv_filter_size_2: float,
                 momentum: float,
                 input_length:int,
                 stride: int = 1,
                 kernel_regularizer: float = 0.01,
                 pooling_type='max',
                 name: str = 'convstackblock',
                 **kwargs):
        """Enformer style conv stack block
        Args:
            num_channels
            conv_filter_size
            momentum: batch norm momentum
            stride: default 1 for no dim reduction
            name: Module name.
        """
        super().__init__(name=name, **kwargs)
        self.initial_channels=initial_channels
        self.channels_list=channels_list
        self.conv_filter_size_1=conv_filter_size_1
        self.conv_filter_size_2=conv_filter_size_2
        self.stride=stride
        self.input_length=input_length
        self.momentum=momentum
        self.kernel_regularizer=kernel_regularizer
        self.pooling_type=pooling_type
        
        
        self.stem_initial_conv = kl.Conv1D(filters=self.initial_channels,
                                           kernel_size=self.conv_filter_size_1,
                                           strides=self.stride,
                                           padding='same',
                                           input_shape=(self.input_length, 5),
                                           kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                           kernel_regularizer=regularizers.L2(self.kernel_regularizer)
                                           )
        self.stem_gelu = tfa.layers.GELU()
        
        self.stem_residual_conv_seq = Residual(conv1Dblock(num_channels=self.initial_channels,
                                              conv_filter_size=1,stride=1,momentum=self.momentum,
                                                           kernel_regularizer=self.kernel_regularizer,
                                              **kwargs,
                                              name = "stem_conv_block"), name = 'stem_res_conv')
        
        self.maxpool = kl.MaxPool1D(pool_size=2,strides=2,padding='valid')
        
        
        self.conv_stack = tf.keras.Sequential()
        for k, channels in enumerate(self.channels_list):
            self.conv_stack.add(conv1Dblock(num_channels=channels,
                                                conv_filter_size=self.conv_filter_size_2,
                                                stride=1, 
                                                momentum=self.momentum,
                                                kernel_regularizer=self.kernel_regularizer, 
                                                **kwargs, 
                                            name = f'conv_stack_seq_b_{k}'))
            self.conv_stack.add(Residual(conv1Dblock(num_channels=channels,
                                                         conv_filter_size=1, 
                                                         momentum=self.momentum, 
                                                         kernel_regularizer=self.kernel_regularizer,
                                                         **kwargs, 
                                                     name = f'conv_stack_seq_resb_{k}'),
                                         name = f'res_{k}'))
            self.conv_stack.add(kl.MaxPool1D(pool_size=2,strides=2,padding='valid')) # todo: trial attention pooling
        

    def get_config(self):
        config = {
            "initial_channels":self.initial_channels,
            "channels_list":self.channels_list,
            "conv_filter_size_1":self.conv_filter_size_1,
            "conv_filter_size_2":self.conv_filter_size_2,
            "stride":self.stride,
            "momentum":self.momentum,
            "input_length":self.input_length,
            "kernel_regularizer":self.kernel_regularizer,
            "pooling_type":self.pooling_type
            
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, training=None):
        seq_or_atac = inputs

        x = self.stem_initial_conv(seq_or_atac,training=training)
        x = self.stem_gelu(x)
        x = self.stem_residual_conv_seq(x,training=training)
        x = self.maxpool(x, training=training)
        
        x = self.conv_stack(x,training=training)
        
        return x
    
    
############################ output head module #####################################
@tf.keras.utils.register_keras_serializable()
class headmodule_block(kl.Layer):
    def __init__(self,
                 num_channels_in: int,
                 momentum: float,
                 kernel_regularizer: float = 0.01,
                 bottleneck_units_tf: int = 64,
                 bottleneck_units: int = 64,
                 dropout_rate=0.10,
                 use_tf_acc=False,
                 name: str = 'headmodule_block',
                 **kwargs):
        """Enformer style conv stack block
        Args:
            num_channels
            conv_filter_size
            momentum: batch norm momentum
            stride: default 1 for no dim reduction
            name: Module name.
        """
        super().__init__(name=name, **kwargs)
        self.num_channels_in=num_channels_in
        self.momentum=momentum
        self.kernel_regularizer=kernel_regularizer
        self.bottleneck_units=bottleneck_units
        self.bottleneck_units_tf=bottleneck_units_tf
        self.dropout_rate=dropout_rate
        self.use_tf_acc=use_tf_acc
        
        if self.use_tf_acc:
            self.tf_bottleneck = kl.Dense(self.bottleneck_units_tf,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        kernel_regularizer=regularizers.L2(self.kernel_regularizer),
                                        use_bias=False)
            self.dropout_tf = kl.Dropout(rate=self.dropout_rate / 2,**kwargs)
        
        self.gelu = tfa.layers.GELU()
        self.syncbatch_norm_2 = syncbatchnorm(axis=-1,
                                            momentum=self.momentum,
                                            center=True,
                                            scale=True,
                                            beta_initializer="zeros",
                                            gamma_initializer="ones",
                                            **kwargs)
        
        
        
        self.final_conv = conv1Dblock(num_channels=num_channels_in,
                                        conv_filter_size=1, 
                                        stride=1, 
                                        kernel_regularizer=self.kernel_regularizer,
                                        momentum=self.momentum,
                                      name = 'final_conv', **kwargs)
        self.gelu = tfa.layers.GELU()
        self.syncbatch_norm_1 = syncbatchnorm(axis=-1,
                                            momentum=self.momentum,
                                            center=True,
                                            scale=True,
                                            beta_initializer="zeros",
                                            gamma_initializer="ones",
                                            **kwargs)
        
        self.flatten_layer = tf.keras.layers.Flatten()
    
        self.bottleneck_units = kl.Dense(self.bottleneck_units,
                                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                 kernel_regularizer=regularizers.L2(self.kernel_regularizer),
                                 use_bias=False)
        
        self.syncbatch_norm_3 = syncbatchnorm(axis=-1,
                                            momentum=self.momentum,
                                            center=True,
                                            scale=True,
                                            beta_initializer="zeros",
                                            gamma_initializer="ones",
                                            **kwargs)

        self.final_unit = kl.Dense(1,
                                     kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                     kernel_regularizer=regularizers.L2(self.kernel_regularizer),
                                     use_bias=True)
        self.final_softplus = tf.keras.layers.Activation('softplus',
                                                        dtype=tf.float32)
        
        self.dropout = kl.Dropout(rate=self.dropout_rate,**kwargs)

    def get_config(self):
        config = {
            "num_channels_in":self.num_channels_in,
            "momentum":self.momentum,
            "kernel_regularizer":self.kernel_regularizer,
            "bottleneck_units":self.bottleneck_units,
            "dropout_rate":self.dropout_rate,
            "bottleneck_units_tf": self.bottleneck_units_tf,
            "use_tf_acc": self.use_tf_acc
            
        }
        
                    
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, training=None):
        seq_and_atac, tf_inputs = inputs
        
        x = self.final_conv(seq_and_atac,training=training)
        x = self.gelu(x)
        x = self.syncbatch_norm_1(x,training=training)
        x = self.flatten_layer(x, training=training)
        
        if self.use_tf_acc:
            tf_out = self.tf_bottleneck(tf_inputs,training=training)
            tf_out = self.gelu(tf_out)
            tf_out = self.syncbatch_norm_2(tf_out,training=training)
            tf_out = self.dropout_tf(tf_out,training=training)
            
            final_out = tf.concat([x,tf_out],axis=1)
        else:
            final_out = x
        
        final_out = self.bottleneck_units(final_out,training=training)
        final_out = self.gelu(final_out)
        final_out = self.syncbatch_norm_3(final_out,training=training)
        final_out = self.dropout(final_out,training=training)
        final_out = self.final_unit(final_out,training=training)
        final_out = self.final_softplus(final_out,training=training)
        
        return final_out
    
    
    
    
############################ tf_module module #####################################
@tf.keras.utils.register_keras_serializable()
class tf_module(kl.Layer):
    def __init__(self,
                 TF_inputs: int,
                 momentum: float,
                 kernel_regularizer: float = 0.01,
                 bottleneck_units: int = 64,
                 dropout_rate: float = 0.1,
                 name: str = 'headmodule_block',
                 **kwargs):
        """Enformer style conv stack block
        Args:
            num_channels
            conv_filter_size
            momentum: batch norm momentum
            stride: default 1 for no dim reduction
            name: Module name.
        """
        super().__init__(name=name, **kwargs)
        self.TF_inputs=TF_inputs
        self.momentum=momentum
        self.kernel_regularizer=kernel_regularizer
        self.bottleneck_units=bottleneck_units
        self.dropout_rate=dropout_rate
        
        self.dense = kl.Dense(self.TF_inputs,
                                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                 kernel_regularizer=regularizers.L2(self.kernel_regularizer),
                                 use_bias=False)
        
        self.gelu = tfa.layers.GELU()
        self.syncbatch_norm_1 = syncbatchnorm(axis=-1,
                                            momentum=self.momentum,
                                            center=True,
                                            scale=True,
                                            beta_initializer="zeros",
                                            gamma_initializer="ones",
                                            **kwargs)
        self.syncbatch_norm_2 = syncbatchnorm(axis=-1,
                                            momentum=self.momentum,
                                            center=True,
                                            scale=True,
                                            beta_initializer="zeros",
                                            gamma_initializer="ones",
                                            **kwargs)
        
        self.dropout = kl.Dropout(rate=self.dropout_rate,**kwargs)

        self.bottleneck = kl.Dense(bottleneck_units,
                                 kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                 kernel_regularizer=regularizers.L2(self.kernel_regularizer),
                                 use_bias=False)

    def get_config(self):
        config = {
            "TF_inputs":self.TF_inputs,
            "momentum":self.momentum,
            "kernel_regularizer":self.kernel_regularizer,
            "bottleneck_units":self.bottleneck_units,
            "dropout_rate":self.dropout_rate
            
        }
        base_config = super().get_config()
        return {**base_config, **config}
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def call(self, inputs, training=None):
        x = self.dense(inputs,training=training)
        x = self.gelu(x)
        x = self.syncbatch_norm_1(x,training=training)
        x = self.dropout(x,training=training)
        x = self.bottleneck(inputs,training=training)
        x = self.gelu(x)
        x = self.syncbatch_norm_2(x,training=training)
        x = self.dropout(x,training=training)
        return x