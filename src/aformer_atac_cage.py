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
                 hidden_size:int = 1536,
                 num_transformer_layers:int = 6,
                 d_model = 192,
                 norm=True,
                 dim = 1536, 
                 max_seq_length = 1536,
                 BN_momentum = 0.80,
                 rel_pos_bins=1536, 
                 use_rot_emb = True,
                 use_mask_pos = False, 
                 normalize = True,
                 seed = 3,
                 load_init=False,
                 inits=None,
                 filter_list_seq=None,
                 filter_list_atac=None,
                 freeze_conv_layers=False,
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
        self.hidden_size=hidden_size
        self.num_transformer_layers=num_transformer_layers
        self.output_length=output_length
        self.final_output_length=final_output_length
        self.norm=norm
        self.d_model = d_model
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.rel_pos_bins = rel_pos_bins
        self.use_rot_emb = use_rot_emb
        self.use_mask_pos = use_mask_pos
        self.normalize = normalize
        self.seed = seed
        self.inits=inits
        self.filter_list_atac = filter_list_atac
        self.freeze_conv_layers = freeze_conv_layers
        
        if inits is None:
            self.load_init=False
        else:
            self.load_init=load_init
            
        self.filter_list_seq = [768, 896, 1024, 1152, 1280, 1536] if self.load_init else filter_list_seq
        
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
                           #strides=2,
                           train=True,
                           **kwargs):
            return tf.keras.Sequential([
              syncbatchnorm(axis=-1,
                            center=True,
                            scale=True,
                            momentum=BN_momentum,
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
                                     strides=1,
                                     padding=padding, **kwargs)
            ], name=name)
        
        ### conv stack for sequence inputs
        self.stem_conv = tf.keras.layers.Conv1D(filters= int(self.filter_list_seq[-1]) // 2,
                                   kernel_size=15,
                                   kernel_initializer=self.inits['stem_conv_k'] if self.load_init else 'glorot_uniform',
                                   bias_initializer=self.inits['stem_conv_b'] if self.load_init else 'zeros',
                                   #strides=2,
                                   trainable=False if self.freeze_conv_layers else True,
                                   padding='same')
        """
        self.stem_conv_atac = tf.keras.layers.Conv1D(filters= 16,
                                   kernel_size=3,
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer='zeros',
                                   strides=1,
                                   padding='same')
        """
        
        self.stem_res_conv=Residual(enf_conv_block(int(self.filter_list_seq[-1]) // 2, 1,
                                                   beta_init=self.inits['stem_res_conv_BN_b'] if self.load_init else None,
                                                   gamma_init=self.inits['stem_res_conv_BN_g'] if self.load_init else None,
                                                   mean_init=self.inits['stem_res_conv_BN_m'] if self.load_init else None,
                                                   var_init=self.inits['stem_res_conv_BN_v'] if self.load_init else None,
                                                   kernel_init=self.inits['stem_res_conv_k'] if self.load_init else None,
                                                   bias_init=self.inits['stem_res_conv_b'] if self.load_init else None,
                                                   train=False if self.freeze_conv_layers else True,
                                                   #strides=1,
                                                   name='pointwise_conv_block'))
        """
        self.stem_res_conv_atac=Residual(tf.keras.layers.Conv1D(filters= 16,
                                   kernel_size=3,
                                   kernel_initializer='glorot_uniform',
                                   bias_initializer='zeros',
                                   strides=1,
                                   padding='same'))
        """
        
        self.stem_pool = SoftmaxPooling1D(per_channel=True,
                                          w_init_scale=2.0,
                                          pool_size=2,
                                          k_init=self.inits['stem_pool'] if self.load_init else None,
                                          train=False if self.freeze_conv_layers else True,
                                          name ='stem_pool')
        
        
        self.conv_tower_seq = tf.keras.Sequential([
            tf.keras.Sequential([
                enf_conv_block(num_filters, 
                               5, 
                               beta_init=self.inits['BN1_b_' + str(i)] if self.load_init else None,
                               gamma_init=self.inits['BN1_g_' + str(i)] if self.load_init else None,
                               mean_init=self.inits['BN1_b_' + str(i)] if self.load_init else None,
                               var_init=self.inits['BN1_v_' + str(i)] if self.load_init else None,
                               kernel_init=self.inits['conv1_k_' + str(i)] if self.load_init else None,
                               bias_init=self.inits['conv1_b_' + str(i)] if self.load_init else None,
                               train=False if self.freeze_conv_layers else True,
                               #strides=1,
                               padding='same'),
                Residual(enf_conv_block(num_filters, 1, 
                                       beta_init=self.inits['BN2_b_' + str(i)] if self.load_init else None,
                                       gamma_init=self.inits['BN2_g_' + str(i)] if self.load_init else None,
                                       mean_init=self.inits['BN2_b_' + str(i)] if self.load_init else None,
                                       var_init=self.inits['BN2_v_' + str(i)] if self.load_init else None,
                                       kernel_init=self.inits['conv2_k_' + str(i)] if self.load_init else None,
                                       bias_init=self.inits['conv2_b_' + str(i)] if self.load_init else None,
                                        train=False if self.freeze_conv_layers else True,
                                        #strides=2,
                                        name='pointwise_conv_block')),
                SoftmaxPooling1D(per_channel=True,
                                 w_init_scale=2.0,
                                 k_init=self.inits['pool_'+str(i)] if self.load_init else None,
                                 train=False if self.freeze_conv_layers else True,
                                 pool_size=2),
                ],
                       name=f'conv_tower_block_{i}')
            for i, num_filters in enumerate(self.filter_list_seq)], name='conv_tower')
        
        """
        self.conv_tower_atac = tf.keras.Sequential([
            tf.keras.Sequential([
                enf_conv_block(num_filters, 
                               5,
                               padding='same'),
                Residual(enf_conv_block(num_filters, 1, 
                                        name='pointwise_conv_block')),
                kl.MaxPooling1D(pool_size=2,
                                 strides=2)
                ],
                       name=f'atac_conv_tower_block_{i}')
            for i, num_filters in enumerate(self.filter_list_atac)], name='conv_tower_atac')
        """
        
        #self.conv_mix = enf_conv_block(filters=self.filter_list_seq[-1],
        #                               **kwargs,
        #                               name = 'final_pointwise_1')

        self.sin_pe = abs_sin_PE(name='sin_pe',
                                  **kwargs)
        

        self.shared_transformer = Performer_Encoder(num_layers=self.num_transformer_layers,
                                                    num_heads=self.num_heads, 
                                                    dim = self.dim,
                                                    d_model=self.d_model,
                                                    norm=self.norm,
                                                    max_seq_length=self.max_seq_length,
                                                    nb_random_features=self.nb_random_features,
                                                    hidden_size=self.hidden_size,
                                                    numerical_stabilizer=self.numerical_stabilizer,
                                                    dropout_rate=self.dropout_rate,
                                                    rel_pos_bins=self.rel_pos_bins,
                                                    use_rot_emb=self.use_rot_emb,
                                                    use_mask_pos=self.use_mask_pos,
                                                    kernel_transformation=self.kernel_transformation,
                                                    normalize=self.normalize, 
                                                    seed = self.seed,
                                                    name = 'shared_transformer',
                                                    **kwargs)
        
        self.crop_final = TargetLengthCrop1D(uncropped_length=self.output_length, 
                                             target_length=self.final_output_length,
                                             name='target_input')
        
        self.final_pointwise = enf_conv_block(filters=2*1536,
                                               **kwargs)

        self.final_dense = kl.Dense(1,
                                   activation='softplus',
                                   use_bias=True)

        self.dropout = kl.Dropout(rate=self.pointwise_dropout_rate,
                                  **kwargs)
        self.gelu = tfa.layers.GELU()

        
    def call(self, inputs, training:bool=True):

        sequence = inputs
        

        ### seq convs
        seq_x = self.stem_conv(sequence,
                           training=training)
        seq_x = self.stem_res_conv(seq_x,
                               training=training)
        seq_x = self.dropout(seq_x,training=training)
        seq_x = self.gelu(seq_x)
        
        seq_x = self.stem_pool(seq_x,
                           training=training)
        enformer_conv_out = self.conv_tower_seq(seq_x,
                                            training=training)
        
        ### peak convs
        #print(atac)
        """
        atac_x = self.stem_conv_atac(atac,
                                     training=training)
        #print(atac_x)
        atac_conv_out = self.stem_res_conv_atac(atac_x,
                                                training=training)
        #print(atac_conv_out)
        conv_concat = tf.concat([enformer_conv_out,
                                 atac_conv_out],axis=2)
        """
        
        #conv_out = self.conv_mix(conv_concat,training=training)
        conv_out = self.sin_pe(enformer_conv_out)
        
        shared_transformer_out,att_matrices_shared = self.shared_transformer(conv_out,
                                                                             training=training)
        out = self.crop_final(shared_transformer_out)
        out = self.final_pointwise(out,training=training)
        out = self.dropout(out,
                           training=training)
        out = self.gelu(out)
        
        out = self.final_dense(out,
                               training=training)
        
        return out
    

    def get_config(self):
        config = {
            "dropout_rate":self.dropout_rate,
            "input_length": self.input_length,
            "dim_reduce_length_seq": self.dim_reduce_length_seq,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "numerical_stabilizer": self.numerical_stabilizer,
            "kernel_transformation": self.kernel_transformation,
            "nb_random_features": self.nb_random_features,
            "shared_transformer_depth" : self.shared_transformer_depth,
            "pre_transf_channels":self.pre_transf_channels,
            "d_model":self.d_model,
            "norm":self.norm,
            "dim":self.dim,
            "max_seq_length":self.max_seq_length,
            "rel_pos_bins":self.rel_pos_bins,
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
    def predict_on_batch(self, inputs, training:bool=False):
        
        sequence, TSSs, exons, peaks, atac = inputs
        

        ### seq convs
        x = self.stem_conv(sequence,
                           training=training)
        x = self.stem_res_conv(x,
                               training=training)
        x = self.stem_pool(x,
                           training=training)
        enformer_conv_out = self.conv_tower(x,
                                            training=training)

        ### atac convs
        x_atac = self.stem_conv(atac,
                           training=training)
        x_atac = self.stem_res_conv(x_atac,
                               training=training)
        x_atac = self.stem_pool(x_atac,
                           training=training)
        atac_conv_out = self.conv_tower(x_atac,
                                            training=training)

        tf_processed = self.tf_module(tf_inputs, 
                                      training=training)
        tf_processed = tf.expand_dims(tf_processed, 
                                      axis=1)
        tf_processed = tf.tile(tf_processed, 
                               [1,self.dim_reduce_length,1])
        
        enformer_conv_out = tf.concat([enformer_conv_out,
                                        atac_conv_out,
                                        tf_processed,
                                        TSSs,
                                        exons],axis=2)

        enformer_conv_out = self.conv_mix_block(enformer_conv_out,
                                                training=training)
        enformer_conv_out = self.sin_pe(enformer_conv_out)
        shared_transformer_out,att_matrices_shared = self.shared_transformer(enformer_conv_out,
                                                                             training=training)

        rna_output = self.final_pointwise_rna(shared_transformer_out)
        rna_output = self.dropout(rna_output,
                                  training=training)
        rna_output = self.gelu(rna_output)
        rna_output = self.rna_head(rna_output,
                                   training=training)

        return rna_output, att_matrices_shared
    

