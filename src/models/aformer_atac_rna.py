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
                 kernel_transformation = 'relu_kernel_transformation',
                 dropout_rate: float = 0.2,
                 pointwise_dropout_rate: float = 0.2,
                 input_length: int = 196608,
                 output_length: int = 1536,
                 final_output_length: int = 896,
                 num_heads:int = 4,
                 numerical_stabilizer: float =0.001,
                 nb_random_features:int = 256,
                 num_transformer_layers:int = 6,
                 norm=True,
                 max_seq_length = 1536,
                 BN_momentum = 0.90,
                 use_rot_emb = True,
                 normalize = True,
                 seed = 3,
                 load_init_FT=False,
                 inits=None,
                 filter_list_seq=[768, 896, 1024, 1152, 1280, 1536],
                 filter_list_atac=[32, 64],
                 final_point_scale=6,
                 freeze_conv_layers=False,
                 load_tf=False,
                 num_tfs=1629,
                 tf_dropout_rate=0.01,
                 name: str = 'aformer',
                 **kwargs):
        """ 'aformer' model based on Enformer for predicting RNA-seq from atac + sequence
        Args: to do


          name: model name
        """

        super(aformer, self).__init__(name=name,**kwargs)
        self.kernel_transformation=kernel_transformation
        self.dropout_rate=dropout_rate
        self.pointwise_dropout_rate=pointwise_dropout_rate
        self.num_heads=num_heads
        self.input_length=input_length
        self.numerical_stabilizer=numerical_stabilizer
        self.nb_random_features=nb_random_features
        self.num_transformer_layers=num_transformer_layers
        self.output_length=output_length
        self.final_output_length=final_output_length
        self.norm=norm
        self.max_seq_length = max_seq_length + 1
        self.use_rot_emb = use_rot_emb
        self.normalize = normalize
        self.seed = seed
        self.inits=inits
        self.filter_list_seq = filter_list_seq
        self.filter_list_atac=filter_list_atac
        self.load_init_FT=load_init_FT
        self.load_tf=load_tf
        self.BN_momentum=BN_momentum
        self.final_point_scale=final_point_scale
        self.num_tfs=num_tfs
        self.tf_dropout_rate=tf_dropout_rate
        self.freeze_conv_layers=freeze_conv_layers

        self.hidden_size=self.filter_list_seq[-1] + self.filter_list_atac[-1] #+ self.global_acc_size
        self.d_model = self.filter_list_seq[-1] + self.filter_list_atac[-1] #+ self.global_acc_size

        self.dim = self.hidden_size  // self.num_heads

        def enf_conv_block(filters,
                           width=1,
                           w_init='lecun_normal',
                           b_init='zeros',
                           padding='same',
                           name='conv_block',
                           beta_init=None,
                           gamma_init=None,
                           mean_init=None,
                           var_init=None,
                           kernel_init=None,
                           bias_init=None,
                           train=True,
                           dilation_rate=1,
                           stride=1,
                           **kwargs):

            return tf.keras.Sequential([
                tf.keras.layers.BatchNormalization(axis=-1,
                                                    synchronized=True,
                                                    center=True,
                                                    scale=True,
                                                    beta_initializer=beta_init if self.load_init_FT else "zeros",
                                                    gamma_initializer=gamma_init if self.load_init_FT else "ones",
                                                     trainable=train,
                                                     momentum=self.BN_momentum,
                                                     epsilon=1.0e-05,
                                                     moving_mean_initializer=mean_init if self.load_init_FT else "zeros",
                                                     moving_variance_initializer=var_init if self.load_init_FT else "ones",
                                                     **kwargs),
                tfa.layers.GELU(),
                tf.keras.layers.Conv1D(filters,
                                     width,
                                     kernel_initializer=kernel_init if self.load_init_FT else w_init,
                                     bias_initializer=bias_init if self.load_init_FT else bias_init,
                                     trainable=train,
                                     strides=stride,
                                     dilation_rate=dilation_rate,
                                     padding=padding, **kwargs)], name=name)

        ### conv stack for sequence inputs
        self.stem_conv = tf.keras.layers.Conv1D(filters= int(self.filter_list_seq[0]),
                                   kernel_size=15,
                                   kernel_initializer=self.inits['stem_conv_k'] if self.load_init_FT else 'lecun_normal',
                                   bias_initializer=self.inits['stem_conv_b'] if self.load_init_FT else 'zeros',
                                   strides=1,
                                   trainable=False if self.freeze_conv_layers else True,
                                   padding='same')

        self.stem_res_conv=Residual(enf_conv_block(int(self.filter_list_seq[0]), 1,
                                                   beta_init=self.inits['stem_res_conv_BN_b'] if self.load_init_FT else None,
                                                   gamma_init=self.inits['stem_res_conv_BN_g'] if self.load_init_FT else None,
                                                   mean_init=self.inits['stem_res_conv_BN_m'] if self.load_init_FT else None,
                                                   var_init=self.inits['stem_res_conv_BN_v'] if self.load_init_FT else None,
                                                   kernel_init=self.inits['stem_res_conv_k'] if self.load_init_FT else None,
                                                   bias_init=self.inits['stem_res_conv_b'] if self.load_init_FT else None,
                                                   train=False if self.freeze_conv_layers else True,
                                                   #strides=1,
                                                   name='pointwise_conv_block'))

        ### conv stack for sequence inputs
        self.stem_conv_atac = tf.keras.layers.Conv1D(filters=32,
                                                     kernel_size=50,
                                                     kernel_initializer=self.inits['stem_conv_atac_k'] if self.load_init_FT else 'lecun_normal',
                                                     bias_initializer=self.inits['stem_conv_atac_b'] if self.load_init_FT else 'zeros',
                                                     strides=1,
                                                     dilation_rate=1,
                                                     trainable=False if self.freeze_conv_layers else True,
                                                     padding='same')

        self.stem_res_conv_atac =Residual(enf_conv_block(32,
                                                         1,
                                                         beta_init=self.inits['stem_res_conv_atac_BN_b'] if self.load_init_FT else None,
                                                         gamma_init=self.inits['stem_res_conv_atac_BN_g'] if self.load_init_FT else None,
                                                         mean_init=self.inits['stem_res_conv_atac_BN_m'] if self.load_init_FT else None,
                                                         var_init=self.inits['stem_res_conv_atac_BN_v'] if self.load_init_FT else None,
                                                         kernel_init=self.inits['stem_res_conv_atac_k'] if self.load_init_FT else None,
                                                         bias_init=self.inits['stem_res_conv_atac_b'] if self.load_init_FT else None,
                                                         train=False if self.freeze_conv_layers else True,
                                                         name='pointwise_conv_block_atac'))
        self.stem_pool_atac = tf.keras.layers.MaxPool1D(pool_size=2)

        self.stem_pool = tf.keras.layers.MaxPool1D(pool_size=2)


        self.conv_tower = tf.keras.Sequential([
            tf.keras.Sequential([
                enf_conv_block(filters=num_filters,
                               width=5,
                               beta_init=self.inits['BN1_b_' + str(i)] if self.load_init_FT else None,
                               gamma_init=self.inits['BN1_g_' + str(i)] if self.load_init_FT else None,
                               mean_init=self.inits['BN1_m_' + str(i)] if self.load_init_FT else None,
                               var_init=self.inits['BN1_v_' + str(i)] if self.load_init_FT else None,
                               kernel_init=self.inits['conv1_k_' + str(i)] if self.load_init_FT else None,
                               bias_init=self.inits['conv1_b_' + str(i)] if self.load_init_FT else None,
                               train=False if self.freeze_conv_layers else True,
                               stride=1,
                               padding='same'),
                tf.keras.layers.MaxPool1D(pool_size=2)],
                       name=f'conv_tower_block_{i}')
            for i, num_filters in enumerate(self.filter_list_seq)], name='conv_tower')


        self.conv_tower_atac = tf.keras.Sequential([
            tf.keras.Sequential([
                enf_conv_block(filters=num_filters,
                               width=5,
                               beta_init=self.inits['BN_at1_b_' + str(i)] if self.load_init_FT else None,
                               gamma_init=self.inits['BN_at1_g_' + str(i)] if self.load_init_FT else None,
                               mean_init=self.inits['BN_at1_m_' + str(i)] if self.load_init_FT else None,
                               var_init=self.inits['BN_at1_v_' + str(i)] if self.load_init_FT else None,
                               kernel_init=self.inits['conv_at1_k_' + str(i)] if self.load_init_FT else None,
                               bias_init=self.inits['conv_at1_b_' + str(i)] if self.load_init_FT else None,
                               train=False if self.freeze_conv_layers else True,
                               dilation_rate=1,
                               stride=1,
                               padding='same'),
                tf.keras.layers.MaxPool1D(pool_size=4)],
                       name=f'conv_tower_block_atac_{i}')
            for i, num_filters in enumerate(self.filter_list_atac)], name='conv_tower_atac')

        self.tf_dropout=kl.Dropout(rate=self.tf_dropout_rate,
                                    **kwargs)

        self.tf_activity_fc = kl.Dense(self.hidden_size,
                                        activation='gelu',
                                        kernel_initializer=self.inits['tf_activity_fc_k'] if (self.load_init_FT and self.load_tf) else 'lecun_normal',
                                        bias_initializer=self.inits['tf_activity_fc_b'] if (self.load_init_FT and self.load_tf) else 'lecun_normal',
                                        use_bias=True)




        self.performer = Performer_Encoder(num_layers=self.num_transformer_layers,
                                                    num_heads=self.num_heads,
                                                    dim = self.dim,
                                                    d_model=self.d_model,
                                                    norm=self.norm,
                                                    max_seq_length=self.max_seq_length,
                                                    nb_random_features=self.nb_random_features,
                                                    hidden_size=self.hidden_size,
                                                    numerical_stabilizer=self.numerical_stabilizer,
                                                    dropout_rate=self.dropout_rate,
                                                    use_rot_emb=self.use_rot_emb,
                                                    kernel_transformation=self.kernel_transformation,
                                                    normalize=self.normalize,
                                                    seed = self.seed,
                                                    load_init= True if self.load_init_FT else False,
                                                    inits=inits if self.load_init_FT else None,
                                                    name = 'shared_transformer',
                                                    **kwargs)


        self.crop_final = TargetLengthCrop1D(uncropped_length=self.output_length,
                                             target_length=self.final_output_length,
                                             name='target_input')

        self.final_pointwise_conv = enf_conv_block(filters=self.filter_list_seq[-1] // self.final_point_scale,
                                                   beta_init=self.inits['final_point_BN_b'] if self.load_init_FT else None,
                                                   gamma_init=self.inits['final_point_BN_g'] if self.load_init_FT else None,
                                                   mean_init=self.inits['final_point_BN_m'] if self.load_init_FT else None,
                                                   var_init=self.inits['final_point_BN_v'] if self.load_init_FT else None,
                                                   kernel_init=self.inits['final_point_k'] if self.load_init_FT else None,
                                                   bias_init=self.inits['final_point_b'] if self.load_init_FT else None,
                                                  **kwargs,
                                                  name = 'final_pointwise')

        self.final_dense_profile_atac = kl.Dense(1, ## atac is the first, cage/RNA is the second dim
                                            activation='softplus',
                                            kernel_initializer=self.inits['final_dense_k'] if self.load_init_FT else 'lecun_normal',
                                            bias_initializer=self.inits['final_dense_b'] if self.load_init_FT else 'lecun_normal',
                                            use_bias=True)
        self.final_dense_profile_rna = kl.Dense(1, ## atac is the first, cage/RNA is the second dim
                                            activation='softplus',
                                            kernel_initializer='lecun_normal',
                                            bias_initializer='lecun_normal',
                                            use_bias=True)
        self.assay_type = kl.Dense(1, ## atac is the first, cage/RNA is the second dim
                                   activation='softplus',
                                   kernel_initializer='lecun_normal',
                                   bias_initializer='lecun_normal',
                                   use_bias=True)

        self.dropout = kl.Dropout(rate=self.pointwise_dropout_rate,
                                  **kwargs)
        self.gelu = tfa.layers.GELU()


    def call(self, inputs, training:bool=True):

        sequence,atac,tf_activity,assay_type = inputs

        x = self.stem_conv(sequence,
                           training=training)
        x = self.stem_res_conv(x,
                               training=training)

        x = self.stem_pool(x,
                           training=training)

        x = self.conv_tower(x,
                            training=training)

        atac_x = self.stem_conv_atac(atac,
                                     training=training)

        atac_x = self.stem_res_conv_atac(atac_x,
                                         training=training)

        atac_x = self.stem_pool_atac(atac_x,
                                     training=training)

        atac_x = self.conv_tower_atac(atac_x,training=training)


        transformer_input = tf.concat([x,atac_x],
                                      axis=2)

        ### tf activity
        tf_activity = self.tf_dropout(tf_activity,training=training)
        tf_activity = self.tf_activity_fc(tf_activity)
        transformer_input_x = tf.concat([transformer_input,tf_activity],
                                        axis=1)
        out,att_matrices = self.performer(transformer_input_x,
                                          training=training)
        out = out[:, :-1, :]
        out = self.crop_final(out)
        out = self.final_pointwise_conv(out,
                                       training=training)
        out = self.dropout(out,
                        training=training)
        out = self.gelu(out)
        ### atac prediction
        out_atac = self.final_dense_profile_atac(out, training=training)

        ### rna prediction
        assay_type = self.assay_type(assay_type,training=training)
        assay_type = tf.tile(assay_type, [1, final_output_length,1])
        out = tf.concat([out,assay_type],axis=2)
        out_rna = self.final_dense_profile_rna(out, training=training)
        return out_atac, out_rna


    def get_config(self):
        config = {
            "kernel_transformation":self.kernel_transformation,
            "dropout_rate": self.dropout_rate,
            "pointwise_dropout_rate": self.pointwise_dropout_rate,
            "num_heads": self.num_heads,
            "input_length": self.input_length,
            "numerical_stabilizer": self.numerical_stabilizer,
            "nb_random_features": self.nb_random_features,
            "num_transformer_layers": self.num_transformer_layers,
            "output_length" : self.output_length,
            "final_output_length":self.final_output_length,
            "norm":self.norm,
            "max_seq_length":self.max_seq_length,
            "use_rot_emb":self.use_rot_emb,
            "normalize":self.normalize,
            "seed":self.seed,
            "inits":self.inits,
            "filter_list_seq":self.filter_list_seq,
            "filter_list_atac":self.filter_list_atac,
            "load_init_FT":self.load_init_FT

        }



        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def predict_on_batch(self, inputs, training:bool=False):

        sequence,atac,tf_activity,assay_type = inputs

        x = self.stem_conv(sequence,
                           training=training)
        x = self.stem_res_conv(x,
                               training=training)

        x = self.stem_pool(x,
                           training=training)

        x = self.conv_tower(x,
                            training=training)

        atac_x = self.stem_conv_atac(atac,
                                     training=training)

        atac_x = self.stem_res_conv_atac(atac_x,
                                         training=training)

        atac_x = self.stem_pool_atac(atac_x,
                                     training=training)

        atac_x = self.conv_tower_atac(atac_x,training=training)


        transformer_input = tf.concat([x,atac_x],
                                      axis=2)

        ### tf activity
        tf_activity = self.tf_dropout(tf_activity,training=training)
        tf_activity = self.tf_activity_fc(tf_activity)
        transformer_input_x = tf.concat([transformer_input,tf_activity],
                                        axis=1)
        out,att_matrices = self.performer(transformer_input_x,
                                          training=training)
        out = out[:, :-1, :]
        out = self.crop_final(out)
        out = self.final_pointwise_conv(out,
                                       training=training)
        out = self.dropout(out,
                        training=training)
        out = self.gelu(out)
        ### atac prediction
        out_atac = self.final_dense_profile_atac(out, training=training)

        ### rna prediction
        assay_type = self.assay_type(assay_type,training=training)
        assay_type = tf.tile(assay_type, [1, final_output_length,1])
        out = tf.concat([out,assay_type],axis=2)
        out_rna = self.final_dense_profile_rna(out, training=training)

        return out_atac, out_rna, att_matrices
