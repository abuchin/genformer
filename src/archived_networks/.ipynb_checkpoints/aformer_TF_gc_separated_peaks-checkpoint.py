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
                 kernel_transformation = 'softmax_kernel_transformation',
                 dropout_rate: float = 0.2,
                 tf_dropout_rate: float = 0.2,
                 pointwise_dropout_rate: float = 0.2,
                 attention_dropout_rate: float = 0.05,
                 #peaks_dropout_rate: float = 0.15,
                 input_length: int = 196608,
                 dim_reduce_length_seq: int = 768,
                 dim_reduce_length_seq_crop: int = 448,
                 dim_reduce_length_class: int = 96,
                 dim_reduce_length_class_crop: int = 56,
                 peaks_reduce_dim: int= 64,
                 num_heads:int = 4,
                 numerical_stabilizer: float =0.001,
                 nb_random_features:int = 256,
                 hidden_size:int = 128,
                 shared_transformer_depth:int = 4,
                 transformer_depth_rna: int = 4,
                 pre_transf_channels: int = 128,
                 d_model = 128,
                 norm=True,
                 dim = 32, 
                 max_seq_length = 1536,
                 rel_pos_bins=1536, 
                 use_rot_emb = True,
                 use_mask_pos = False, 
                 normalize = True,
                 seed = 3,
                 load_init=False,
                 inits=None,
                 filter_list_seq=None,
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
        self.pre_transf_channels=pre_transf_channels
        self.shared_transformer_depth=shared_transformer_depth
        self.transformer_depth_rna=transformer_depth_rna
        self.norm=norm
        self.d_model = d_model
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.rel_pos_bins = rel_pos_bins
        self.use_rot_emb = use_rot_emb
        self.use_mask_pos = use_mask_pos
        self.normalize = normalize
        self.seed = seed
        #self.peaks_dropout_rate=peaks_dropout_rate
        self.attention_dropout_rate=attention_dropout_rate
        self.peaks_reduce_dim=peaks_reduce_dim
        self.load_init=load_init
        self.inits=inits
        self.filter_list_seq = [768, 896, 1024, 1152, 1280, 1536] if self.load_init else filter_list_seq
        #self.filter_list_atac = filter_list_atac
        self.freeze_conv_layers = freeze_conv_layers
        self.dim_reduce_length_seq=dim_reduce_length_seq
        self.dim_reduce_length_seq_crop=dim_reduce_length_seq_crop
        
        
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
                                   kernel_initializer=self.inits['stem_conv_k'] if self.load_init else 'glorot_uniform',
                                   bias_initializer=self.inits['stem_conv_b'] if self.load_init else 'zeros',
                                   strides=1,
                                   trainable=False if self.freeze_conv_layers else True,
                                   padding='same')
                                   #data_format='channels_last')
        self.stem_res_conv=Residual(enf_conv_block(int(self.filter_list_seq[-1]) // 2, 1,
                                                   beta_init=self.inits['stem_res_conv_BN_b'] if self.load_init else None,
                                                   gamma_init=self.inits['stem_res_conv_BN_g'] if self.load_init else None,
                                                   mean_init=self.inits['stem_res_conv_BN_m'] if self.load_init else None,
                                                   var_init=self.inits['stem_res_conv_BN_v'] if self.load_init else None,
                                                   kernel_init=self.inits['stem_res_conv_k'] if self.load_init else None,
                                                   bias_init=self.inits['stem_res_conv_b'] if self.load_init else None,
                                                   train=False if self.freeze_conv_layers else True,
                                                   name='pointwise_conv_block'))
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
                               padding='same'),
                Residual(enf_conv_block(num_filters, 1, 
                                       beta_init=self.inits['BN2_b_' + str(i)] if self.load_init else None,
                                       gamma_init=self.inits['BN2_g_' + str(i)] if self.load_init else None,
                                       mean_init=self.inits['BN2_b_' + str(i)] if self.load_init else None,
                                       var_init=self.inits['BN2_v_' + str(i)] if self.load_init else None,
                                       kernel_init=self.inits['conv2_k_' + str(i)] if self.load_init else None,
                                       bias_init=self.inits['conv2_b_' + str(i)] if self.load_init else None,
                                        train=False if self.freeze_conv_layers else True,
                                        name='pointwise_conv_block')),
                SoftmaxPooling1D(per_channel=True,
                                 w_init_scale=2.0,
                                 k_init=self.inits['pool_'+str(i)] if self.load_init else None,
                                 train=False if self.freeze_conv_layers else True,
                                 pool_size=2),
                ],
                       name=f'conv_tower_block_{i}')
            for i, num_filters in enumerate(self.filter_list_seq)], name='conv_tower')
        
        #self.conv_tower_peaks_pool = tf.keras.Sequential([
        #                                SoftmaxPooling1D(per_channel=True,
        #                                                 w_init_scale=2.0,
        #                                                 pool_size=4,
        #                                                 name=f'pooling_block_{i}')
        #    for i, num_filters in enumerate(self.filter_list_seq)], name='conv_tower_peaks_pool')
        self.peaks_pool = kl.MaxPool1D(pool_size=6,
                                       strides=4)

        self.conv_tower_peaks = tf.keras.Sequential([
            tf.keras.Sequential([
                self.conv_tower_seq.layers[i].layers[0],
                self.conv_tower_seq.layers[i].layers[1],
                #SoftmaxPooling1D(per_channel=True,
                #                 w_init_scale=2.0,
                #                 pool_size=4,
                kl.MaxPool1D(pool_size=4,
                             strides=4,
                             name=f'pooling_block_{i}')
                ],
                       name=f'conv_tower_peaks_block{i}')
            for i, num_filters in enumerate(self.filter_list_seq)], name='conv_tower_peaks')
        

        self.peaks_module = peaks_module(reduce_channels=self.peaks_reduce_dim,
                                   name='tf_module',
                                   **kwargs)
        
        self.conv_mix_block = conv_mix_block(num_channels_out=self.pre_transf_channels,
                                             name='conv_mix_block',
                                             **kwargs)
    

        self.sin_pe = abs_sin_PE(name='sin_pe',
                                  **kwargs)
        self.sin_pe2 = abs_sin_PE(name='sin_pe2',
                                  **kwargs)
        
        self.shared_transformer = Performer_Encoder(num_layers=self.shared_transformer_depth,
                                                    num_heads=self.num_heads, 
                                                    dim = self.dim,
                                                    d_model=self.d_model,
                                                    norm=self.norm,
                                                    max_seq_length=self.max_seq_length,
                                                    nb_random_features=self.nb_random_features,
                                                    hidden_size=self.hidden_size,
                                                    numerical_stabilizer=self.numerical_stabilizer,
                                                    attention_dropout=self.attention_dropout_rate,
                                                    rel_pos_bins=self.rel_pos_bins,
                                                    use_rot_emb=self.use_rot_emb,
                                                    use_mask_pos=self.use_mask_pos,
                                                    kernel_transformation=self.kernel_transformation,
                                                    normalize=self.normalize, 
                                                    seed = self.seed,
                                                    name = 'shared_transformer',
                                                    **kwargs)
        
        self.transformer_stack_rna = Performer_Encoder(num_layers=self.transformer_depth_rna,
                                                       num_heads=self.num_heads, 
                                                       dim = (self.pre_transf_channels // self.num_heads),
                                                       d_model=self.pre_transf_channels,
                                                       norm=self.norm,
                                                       max_seq_length=self.max_seq_length,
                                                       nb_random_features=self.nb_random_features,
                                                       hidden_size=self.pre_transf_channels,
                                                       numerical_stabilizer=self.numerical_stabilizer,
                                                       attention_dropout=self.attention_dropout_rate,
                                                       rel_pos_bins=self.rel_pos_bins,
                                                       use_rot_emb=self.use_rot_emb,
                                                       use_mask_pos=self.use_mask_pos,
                                                       kernel_transformation=self.kernel_transformation,
                                                       normalize=self.normalize, 
                                                       seed = self.seed,
                                                       name = 'transformer_stack_rna',
                                                       **kwargs)



        self.dropout = kl.Dropout(rate=self.pointwise_dropout_rate,
                                  **kwargs)
        self.gelu = tfa.layers.GELU()

        self.final_pointwise_rna = enf_conv_block(filters=self.pre_transf_channels // 8,
                                                  **kwargs,
                                                  name = 'final_pointwise_rna')
        self.rna_head = output_head_rna(num_channels_out = self.pre_transf_channels // 16,
                                        name = 'rna_out_head',
                                        **kwargs)
        
        self.final_pointwise_atac = enf_conv_block(filters=int(self.filter_list_seq[-1]) // 8,
                                                   **kwargs,
                                                  name = 'final_pointwise_atac')
        
        self.atac_head = output_head_atac(dropout_rate=self.pointwise_dropout_rate,
                                          name = 'atac_out_head',
                                          **kwargs)
        self.crop_reg = TargetLengthCrop1D(uncropped_length = self.dim_reduce_length_seq,
                                           target_length = self.dim_reduce_length_seq_cropped)
        self.crop_class = TargetLengthCrop1D(uncropped_length = 96,
                                             target_length = 56)
        
        
    def call(self, inputs, training:bool=True):

        sequence, TSSs, exons, peaks = inputs
        

        ### seq convs
        seq_x = self.stem_conv(sequence,
                           training=training)
        seq_x = self.stem_res_conv(seq_x,
                               training=training)
        seq_x = self.stem_pool(seq_x,
                           training=training)
        enformer_conv_out = self.conv_tower_seq(seq_x,
                                            training=training)
        
        ### peak convs
        peaks_x = self.stem_conv(peaks,
                           training=training)
        peaks_x = self.stem_res_conv(peaks_x,
                               training=training)
        peaks_x = self.peaks_pool(peaks_x,
                           training=training)
        peaks_x = self.conv_tower_peaks(peaks_x,
                                  training=training)

        peaks_processed = self.peaks_module(peaks_x,
                                      training=training)
        peaks_processed = tf.expand_dims(peaks_processed, 
                                         axis=1)
        peaks_processed = tf.tile(peaks_processed, 
                                  [1,self.dim_reduce_length_seq,1])
        
        
        enformer_conv_out = tf.concat([enformer_conv_out,
                                       atac,
                                       peaks_processed,
                                       TSSs,
                                       exons],axis=2)

        enformer_conv_out = self.conv_mix_block(enformer_conv_out,
                                                training=training)
        enformer_conv_out = self.sin_pe(enformer_conv_out)
        shared_transformer_out,att_matrices_shared = self.shared_transformer(enformer_conv_out,
                                                                             training=training)
        
        atac_output = self.final_pointwise_atac(shared_transformer_out,
                                                training=training)
        atac_output = self.dropout(atac_output,
                                   training=training)
        atac_output = self.gelu(atac_output)
        atac_output_reg,atac_output_class = self.atac_head(atac_output,
                                                           training=training)
        
        
        transformer_input_2 = self.sin_pe2(shared_transformer_out)
        transformer_out_2,att_matrices_2 = self.transformer_stack_rna(transformer_input_2,
                                                                      training=training)

        rna_output = self.final_pointwise_rna(transformer_out_2,
                                             training=training)
        rna_output = self.dropout(rna_output,
                                  training=training)
        rna_output = self.gelu(rna_output)
        rna_output = self.rna_head(rna_output,
                                   training=training)

        return self.crop_reg(atac_output_reg), self.crop_class(atac_out_class),rna_output

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
    

