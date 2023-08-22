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
class pt_init(tf.keras.initializers.Initializer):
    def __init__(self, input_arr):
        self.input_arr = input_arr

    def __call__(self):
        return self.input_arr

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

@tf.keras.utils.register_keras_serializable()
class layer_norm_fp32(kl.Layer):
    def __init__(self,
                 epsilon=1e-05,
                 beta_initializer="zeros",
                 gamma_initializer="ones",
                 name: str = 'layer_norm_fp32',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self.epsilon=epsilon
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.layer_norm = kl.LayerNormalization(axis=-1,
                                                  scale=True,
                                                  center=True,
                                                    epsilon=self.epsilon,
                                                  beta_initializer=self.beta_initializer,
                                                  gamma_initializer=self.gamma_initializer)

    def get_config(self):
        config = {
            "epsilon":self.epsilon
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=None):
        x = tf.cast(inputs,dtype=tf.float32)
        x = self.layer_norm(x)
        return tf.cast(x, dtype=tf.bfloat16)


@tf.keras.utils.register_keras_serializable()
class sync_batch_norm_fp32(kl.Layer):
    def __init__(self,
                 beta_init=None,
                 gamma_init=None,
                 mean_init=None,
                 var_init=None,
                 epsilon=1e-05,
                 train=True,
                 momentum = 0.90,
                 load_init=True,
                 name: str = 'sync_batch_norm_fp32',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        self.epsilon=epsilon
        self.load_init=load_init
        self.beta_init=beta_init
        self.gamma_init=gamma_init
        self.mean_init=mean_init
        self.var_init=var_init
        self.train=train
        self.momentum=momentum
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-1,
                                            synchronized=True,
                                          center=True,
                                          scale=True,
                                          beta_initializer=self.beta_init if self.load_init else "zeros",
                                          gamma_initializer=self.gamma_init if self.load_init else "ones",
                                          trainable=self.train,
                                          momentum=self.momentum,
                                          epsilon=self.epsilon,
                                          moving_mean_initializer=self.mean_init if self.load_init else "zeros",
                                          moving_variance_initializer=self.var_init if self.load_init else "ones",
                                          **kwargs)

    def get_config(self):
        config = {
            "epsilon":self.epsilon,
            "load_init":self.load_init,
            "beta_init":self.beta_init,
            "gamma_init":self.gamma_init,
            "mean_init":self.mean_init,
            "var_init":self.var_init
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=None):
        x = tf.cast(inputs, dtype=tf.float32)
        x = self.batch_norm(x,training=training)
        return tf.cast(x, dtype=tf.bfloat16)


@tf.keras.utils.register_keras_serializable()
class FFN(kl.Layer):
    def __init__(self,
                 num_channels: int,
                 dropout_rate: float,
                   FFN_LN_gamma_init=None,
                   FFN_LN_beta_init=None,
                   FFN_kernel1_init=None,
                   FFN_bias1_init=None,
                   FFN_kernel2_init=None,
                   FFN_bias2_init=None,
                   load_init = True,
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
        self.ffn_widening = 2
        self.ffn_dropout = dropout_rate
        self.load_init=load_init
        self.FFN_LN_gamma_init=None,
        self.FFN_LN_beta_init=FFN_LN_beta_init
        self.FFN_kernel1_init=FFN_kernel1_init
        self.FFN_bias1_init=FFN_bias1_init
        self.FFN_kernel2_init=FFN_kernel2_init
        self.FFN_bias2_init=FFN_bias2_init


        self.FFN_layer_norm = layer_norm_fp32(epsilon=1e-05,
                                                  beta_initializer="zeros",
                                                  gamma_initializer="ones")
        self.FFN_dense_wide = kl.Dense(self.ffn_channels*self.ffn_widening,
                                       activation='linear',
                                       kernel_initializer=FFN_kernel1_init if self.load_init else 'lecun_normal',
                                       bias_initializer=FFN_bias1_init if self.load_init else 'lecun_normal',
                                       use_bias=True)
        self.dropout = kl.Dropout(rate=self.ffn_dropout,**kwargs)
        self.relu = kl.ReLU()
        self.FFN_dense_narrow = kl.Dense(self.ffn_channels,
                                         activation='linear',
                                         kernel_initializer=FFN_kernel2_init if self.load_init else 'lecun_normal',
                                         bias_initializer=FFN_bias2_init if self.load_init else 'lecun_normal',
                                         use_bias=True)

    def get_config(self):
        config = {
            "ffn_channels":self.ffn_channels,
            "ffn_widening":self.ffn_widening,
            "ffn_dropout":self.ffn_dropout,
            "load_init": self.load_init,
            "FFN_LN_gamma_init":self.FFN_LN_gamma_init,
            "FFN_LN_beta_init":self.FFN_LN_beta_init,
            "FFN_kernel1_init":self.FFN_kernel1_init,
            "FFN_bias1_init":self.FFN_bias1_init,
            "FFN_kernel2_init":self.FFN_kernel2_init,
            "FFN_bias2_init":self.FFN_bias2_init
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
class Performer(kl.Layer):
    def __init__(self,
                 d_model,
                 normalize,
                 hidden_size: int,
                 num_heads: int,
                 seed: int,
                 dropout_rate: float,
                 numerical_stabilizer: float,
                 nb_random_features: int,
                 max_seq_length: int,
                 rel_pos_bins=None,
                 kernel_transformation: str = 'relu_kernel_transformation',
                 use_mask_pos: bool = False,
                 use_rot_emb: bool = True,
                 LN_gamma_init = None,
                 LN_beta_init= None,
                 q_init=None,
                 k_init=None,
                 v_init=None,
                 att_output=None,
                 FFN_LN_gamma_init=None,
                 FFN_LN_beta_init=None,
                 FFN_kernel1_init=None,
                 FFN_bias1_init=None,
                 FFN_kernel2_init=None,
                 FFN_bias2_init=None,
                 load_init: bool = False,
                 name = 'transformer_layer',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        """Transformer block w/ performer attention
        Args:
            hidden size: ~channel dimension for transformer input
            num_heads: num attention heads
            numerical_stabilizer: small float for stability
            nb_random_features: dim for projection matrix
            widening: scaling factor for how many channels to start w/
                      e.g. widening = 2, num_channels = 12 means start w/ 24
            dropout_rate: transformer MLP dropout rate
            kernel_transformation: softmax or relu kernel transform for fast att.
            positional_encoding_type: absolute sinusoidal or relative(rotary)
            name: Module name.
        """
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.dropout_rate=dropout_rate
        self.kernel_transformation=kernel_transformation
        self.numerical_stabilizer=numerical_stabilizer
        self.max_seq_length = max_seq_length
        self.nb_random_features=nb_random_features
        self.rel_pos_bins = rel_pos_bins
        self.use_rot_emb=use_rot_emb
        self.use_mask_pos=use_mask_pos
        self.d_model=d_model
        self.normalize=normalize
        self.seed=seed
        self.load_init=load_init
        self.FFN_LN_gamma_init=None,
        self.FFN_LN_beta_init=FFN_LN_beta_init
        self.FFN_kernel1_init=FFN_kernel1_init
        self.FFN_bias1_init=FFN_bias1_init
        self.FFN_kernel2_init=FFN_kernel2_init
        self.FFN_bias2_init=FFN_bias2_init

        self.layer_norm = layer_norm_fp32(epsilon=1e-05,
                                                  beta_initializer="zeros",
                                                  gamma_initializer="ones")


        self.self_attention = fa_rpe.Attention(hidden_size=self.d_model,
                                               num_heads=self.num_heads,
                                               nb_random_features=self.nb_random_features,
                                               use_rot_emb=self.use_rot_emb,
                                               use_mask_pos=self.use_mask_pos,
                                               normalize=self.normalize,
                                               kernel_transformation=self.kernel_transformation,
                                               numerical_stabilizer=self.numerical_stabilizer,
                                               seed=self.seed,
                                               q_init=q_init,
                                               k_init=k_init,
                                               v_init=v_init,
                                               att_output=att_output,
                                               load_init = self.load_init,
                                               **kwargs)
        self.dropout = kl.Dropout(rate=self.dropout_rate,**kwargs)
        self.FFN = FFN(num_channels=self.hidden_size,
                       dropout_rate=self.dropout_rate,
                       FFN_LN_gamma_init=FFN_LN_gamma_init,
                       FFN_LN_beta_init=FFN_LN_beta_init,
                       FFN_kernel1_init=FFN_kernel1_init,
                       FFN_bias1_init=FFN_bias1_init,
                       FFN_kernel2_init=FFN_kernel2_init,
                       FFN_bias2_init=FFN_bias2_init,
                       load_init = self.load_init,
                       name='FFN',
                       **kwargs)

    def get_config(self):
        config = {
            "hidden_size":self.hidden_size,
            "num_heads":self.num_heads,
            "numerical_stabilizer":self.numerical_stabilizer,
            "nb_random_features":self.nb_random_features,
            "kernel_transformation":self.kernel_transformation,
            "max_seq_length":self.max_seq_length,
            "rel_pos_bins":self.rel_pos_bins,
            "use_rot_emb":self.use_rot_emb,
            "use_mask_pos":self.use_mask_pos,
            "d_model":self.d_model,
            "normalize":self.normalize,
            "seed":self.seed,
            "load_init": self.load_init,
            "FFN_LN_gamma_init":self.FFN_LN_gamma_init,
            "FFN_LN_beta_init":self.FFN_LN_beta_init,
            "FFN_kernel1_init":self.FFN_kernel1_init,
            "FFN_bias1_init":self.FFN_bias1_init,
            "FFN_kernel2_init":self.FFN_kernel2_init,
            "FFN_bias2_init":self.FFN_bias2_init
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
        #return self.layer_norm(FFN_out + mha_output), k_prime, q_prime
        return (FFN_out + mha_output), k_prime, q_prime

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
class Performer_stable(kl.Layer):
    def __init__(self,
                 d_model,
                 normalize,
                 hidden_size: int,
                 num_heads: int,
                 seed: int,
                 dropout_rate: float,
                 numerical_stabilizer: float,
                 nb_random_features: int,
                 max_seq_length: int,
                 rel_pos_bins=None,
                 kernel_transformation: str = 'relu_kernel_transformation',
                 use_mask_pos: bool = False,
                 use_rot_emb: bool = True,
                 q_init=None,
                 k_init=None,
                 v_init=None,
                 att_output=None,
                 FFN_LN_gamma_init=None,
                 FFN_LN_beta_init=None,
                 FFN_kernel1_init=None,
                 FFN_bias1_init=None,
                 FFN_kernel2_init=None,
                 FFN_bias2_init=None,
                 load_init: bool = False,
                 name = 'transformer_layer',
                 **kwargs):
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
            kernel_transformation: softmax or relu kernel transform for fast att.
            positional_encoding_type: absolute sinusoidal or relative(rotary)
            name: Module name.
        """
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.dropout_rate=dropout_rate
        self.kernel_transformation=kernel_transformation
        self.numerical_stabilizer=numerical_stabilizer
        self.max_seq_length = max_seq_length
        self.nb_random_features=nb_random_features
        self.rel_pos_bins = rel_pos_bins
        self.use_rot_emb=use_rot_emb
        self.use_mask_pos=use_mask_pos
        self.d_model=d_model
        self.normalize=normalize
        self.seed=seed
        self.load_init=load_init
        self.FFN_LN_gamma_init=None,
        self.FFN_LN_beta_init=FFN_LN_beta_init
        self.FFN_kernel1_init=FFN_kernel1_init
        self.FFN_bias1_init=FFN_bias1_init
        self.FFN_kernel2_init=FFN_kernel2_init
        self.FFN_bias2_init=FFN_bias2_init


        self.layer_norm = ScaleNorm(self.d_model ** 0.50)
        self.self_attention = fa_rpe.Attention(hidden_size=self.d_model,
                                               num_heads=self.num_heads,
                                               nb_random_features=self.nb_random_features,
                                               use_rot_emb=self.use_rot_emb,
                                               use_mask_pos=self.use_mask_pos,
                                               normalize=self.normalize,
                                               kernel_transformation=self.kernel_transformation,
                                               numerical_stabilizer=self.numerical_stabilizer,
                                               seed=self.seed,
                                               q_init=q_init,
                                               k_init=k_init,
                                               v_init=v_init,
                                               att_output=att_output,
                                               load_init = self.load_init,
                                               **kwargs)
        self.dropout = kl.Dropout(rate=self.dropout_rate,**kwargs)
        self.FFN = FFN(num_channels=self.hidden_size,
                       dropout_rate=self.dropout_rate,
                       FFN_LN_gamma_init=FFN_LN_gamma_init,
                       FFN_LN_beta_init=FFN_LN_beta_init,
                       FFN_kernel1_init=FFN_kernel1_init,
                       FFN_bias1_init=FFN_bias1_init,
                       FFN_kernel2_init=FFN_kernel2_init,
                       FFN_bias2_init=FFN_bias2_init,
                       load_init = self.load_init,
                       name='FFN',
                       **kwargs)

    def get_config(self):
        config = {
            "hidden_size":self.hidden_size,
            "num_heads":self.num_heads,
            "dropout":self.dropout,
            "numerical_stabilizer":self.numerical_stabilizer,
            "nb_random_features":self.nb_random_features,
            "kernel_transformation":self.kernel_transformation,
            "max_seq_length":self.max_seq_length,
            "rel_pos_bins":self.rel_pos_bins,
            "use_rot_emb":self.use_rot_emb,
            "use_mask_pos":self.use_mask_pos,
            "d_model":self.d_model,
            "normalize":self.normalize,
            "seed":self.seed,
            "load_init": self.load_init,
            "FFN_LN_gamma_init":self.FFN_LN_gamma_init,
            "FFN_LN_beta_init":self.FFN_LN_beta_init,
            "FFN_kernel1_init":self.FFN_kernel1_init,
            "FFN_bias1_init":self.FFN_bias1_init,
            "FFN_kernel2_init":self.FFN_kernel2_init,
            "FFN_bias2_init":self.FFN_bias2_init
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
        return (FFN_out + mha_output), k_prime, q_prime

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
                 hidden_size,
                 numerical_stabilizer,
                 dropout_rate = 0.40,
                 rel_pos_bins=None,
                 use_rot_emb=True,
                 use_mask_pos=False,
                 normalize=True,
                 norm=True,
                 seed=42,
                 load_init=True,
                 inits=None,
                 kernel_transformation: str = 'relu_kernel_transformation',
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
        self.numerical_stabilizer=numerical_stabilizer
        self.rel_pos_bins=rel_pos_bins#None#rel_pos_bins
        self.use_rot_emb=use_rot_emb
        self.use_mask_pos=use_mask_pos
        self.normalize=normalize
        self.norm=norm
        self.kernel_transformation=kernel_transformation
        self.seed=seed
        self.dropout_rate=dropout_rate
        self.load_init=load_init
        self.inits=inits

        self.layers = [Performer(d_model=self.d_model,
                                 normalize=self.normalize,
                                 hidden_size=self.hidden_size,
                                 num_heads=self.num_heads,
                                 dropout_rate=self.dropout_rate,
                                 numerical_stabilizer=self.numerical_stabilizer,
                                 nb_random_features=self.nb_random_features,
                                 max_seq_length=self.max_seq_length,
                                 rel_pos_bins=self.rel_pos_bins,
                                 kernel_transformation=self.kernel_transformation,
                                 use_mask_pos=self.use_mask_pos,
                                 seed=self.seed,
                                 use_rot_emb=self.use_rot_emb,
                                 load_init=self.load_init,
                                 LN_gamma_init = inits["LN_g" + str(i)] if self.load_init else None,
                                 LN_beta_init=  inits["LN_b" + str(i)] if self.load_init else None,
                                 q_init= inits["SA_q" + str(i)] if self.load_init else None,
                                 k_init= inits["SA_k" + str(i)] if self.load_init else None,
                                 v_init= inits["SA_v" + str(i)] if self.load_init else None,
                                 att_output= inits["SA_O" + str(i)] if self.load_init else None,
                                 FFN_LN_gamma_init= inits["FFN_LN_g" + str(i)] if self.load_init else None,
                                 FFN_LN_beta_init= inits["FFN_LN_b" + str(i)] if self.load_init else None,
                                 FFN_kernel1_init= inits["FFN_wide_k" + str(i)] if self.load_init else None,
                                 FFN_bias1_init= inits["FFN_wide_b" + str(i)] if self.load_init else None,
                                 FFN_kernel2_init= inits["FFN_narr_k" + str(i)] if self.load_init else None,
                                 FFN_bias2_init= inits["FFN_narr_b" + str(i)] if self.load_init else None,
                                 **kwargs) for i in range(self.num_layers)]

        self.layer_norm = layer_norm_fp32(epsilon=1e-05,
                                                  beta_initializer=self.inits["performer_encoder_LN_b"] if self.load_init else "zeros",
                                                  gamma_initializer=self.inits["performer_encoder_LN_g"] if self.load_init else "ones")


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
            "numerical_stabilizer":self.numerical_stabilizer,
            "nb_random_features":self.nb_random_features,
            "kernel_transformation":self.kernel_transformation,
            "num_layers":self.num_layers,
            "dim":self.dim,
            "d_model":self.d_model,
            "max_seq_length":self.max_seq_length,
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


@tf.keras.utils.register_keras_serializable()
class Performer_Encoder_stable(kl.Layer):
    def __init__(self,
                 num_layers,
                 num_heads,
                 dim,
                 d_model,
                 max_seq_length,
                 nb_random_features,
                 hidden_size,
                 numerical_stabilizer,
                 dropout_rate = 0.40,
                 rel_pos_bins=None,
                 use_rot_emb=True,
                 use_mask_pos=False,
                 normalize=True,
                 norm=True,
                 seed=42,
                 load_init=True,
                 inits=None,
                 kernel_transformation: str = 'relu_kernel_transformation',
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
        self.numerical_stabilizer=numerical_stabilizer
        self.rel_pos_bins=rel_pos_bins#None#rel_pos_bins
        self.use_rot_emb=use_rot_emb
        self.use_mask_pos=use_mask_pos
        self.normalize=normalize
        self.norm=norm
        self.kernel_transformation=kernel_transformation
        self.seed=seed
        self.dropout_rate=dropout_rate
        self.load_init=load_init
        self.inits=inits

        self.layers = [Performer_stable(d_model=self.d_model,
                                 normalize=self.normalize,
                                 hidden_size=self.hidden_size,
                                 num_heads=self.num_heads,
                                 dropout_rate=self.dropout_rate,
                                 numerical_stabilizer=self.numerical_stabilizer,
                                 nb_random_features=self.nb_random_features,
                                 max_seq_length=self.max_seq_length,
                                 rel_pos_bins=self.rel_pos_bins,
                                 kernel_transformation=self.kernel_transformation,
                                 use_mask_pos=self.use_mask_pos,
                                 seed=self.seed,
                                 use_rot_emb=self.use_rot_emb,
                                 load_init=self.load_init,
                                 q_init= inits["SA_q" + str(i)] if self.load_init else None,
                                 k_init= inits["SA_k" + str(i)] if self.load_init else None,
                                 v_init= inits["SA_v" + str(i)] if self.load_init else None,
                                 att_output= inits["SA_O" + str(i)] if self.load_init else None,
                                 FFN_LN_gamma_init= inits["FFN_LN_g" + str(i)] if self.load_init else None,
                                 FFN_LN_beta_init= inits["FFN_LN_b" + str(i)] if self.load_init else None,
                                 FFN_kernel1_init= inits["FFN_wide_k" + str(i)] if self.load_init else None,
                                 FFN_bias1_init= inits["FFN_wide_b" + str(i)] if self.load_init else None,
                                 FFN_kernel2_init= inits["FFN_narr_k" + str(i)] if self.load_init else None,
                                 FFN_bias2_init= inits["FFN_narr_b" + str(i)] if self.load_init else None,
                                 **kwargs) for i in range(self.num_layers)]

        self.layer_norm = layer_norm_fp32(epsilon=1e-05,
                                                  beta_initializer=self.inits["performer_encoder_LN_b"] if self.load_init else "zeros",
                                                  gamma_initializer=self.inits["performer_encoder_LN_g"] if self.load_init else "ones")
        #kl.LayerNormalization(axis=-1,
                          #                        scale=True,
                          #                        center=True,
                          #                      epsilon=1e-05,
                          #                        beta_initializer=self.inits["performer_encoder_LN_b"] if self.load_init else "zeros",
                          #                        gamma_initializer=self.inits["performer_encoder_LN_g"] if self.load_init else "ones")



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

        super(Performer_Encoder_stable,self).build(input_shape)

    def get_config(self):
        config = {
            "hidden_size":self.hidden_size,
            "num_heads":self.num_heads,
            "attention_dropout":self.attention_dropout,
            "numerical_stabilizer":self.numerical_stabilizer,
            "nb_random_features":self.nb_random_features,
            "kernel_transformation":self.kernel_transformation,
            "num_layers":self.num_layers,
            "dim":self.dim,
            "d_model":self.d_model,
            "max_seq_length":self.max_seq_length,
            "rel_pos_bins":self.rel_pos_bins,
            "use_rot_emb":self.use_rot_emb,
            "use_mask_pos":self.use_mask_pos,
            "normalize":self.normalize,
            "norm":self.norm,
            "seed":self.seed,
            "inits":self.inits
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

@tf.keras.utils.register_keras_serializable()
class abs_sin_PE(kl.Layer):
    def __init__(self,
                 name: str='sinusoidal_pos_encoding',
                 **kwargs):
        """basic absolute sinusoidal PE layer
        Args:
            positional_dropout_rate: dropout rate for positional embeddings
        """
        super().__init__(name=name,**kwargs)

    def build(self, input_shape):
        self._pe = utils.sinusoidal(input_shape)
        super(abs_sin_PE,self).build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        return{**base_config, **config}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=None):
        return tf.cast(self._pe,dtype=tf.bfloat16) + tf.cast(inputs,dtype=tf.bfloat16)


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
@tf.keras.utils.register_keras_serializable()
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

@tf.keras.utils.register_keras_serializable()
class TargetLengthCrop1D(kl.Layer):
    """Crop sequence to match the desired target length."""
    def __init__(self,
               uncropped_length: int = 768,
               target_length: int = 448,
               name: str = 'target_length_crop'):
        super().__init__(name=name)
        self._target_length = target_length
        self._uncropped_length = uncropped_length

    def call(self, inputs):
        if self._target_length is None:
            return inputs
        trim = (self._uncropped_length - self._target_length) // 2
        if trim < 0:
            raise ValueError('inputs longer than target length')
        elif trim == 0:
            return inputs
        else:
            return inputs[..., trim:-trim, :]



@tf.keras.utils.register_keras_serializable()
class ScaleNorm(kl.Layer):
    """ScaleNorm"""
    def __init__(self,
                 scale,
                 eps=1e-5,
                 name: str = 'scalenorm'):
        super(ScaleNorm, self).__init__()
        self.scale = scale
        self.eps = eps


    def call(self, x):
        norm = self.scale / tf.maximum(self.eps,tf.norm(x, axis=-1, keepdim=True))
        return x * norm
