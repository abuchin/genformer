from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import tensorflow.experimental.numpy as tnp
import tensorflow as tf

from tensorflow.keras import layers as kl

from src.layers.layers_dual import *
import tensorflow_addons as tfa
from tensorflow.keras import regularizers

from tensorflow.keras.layers.experimental import SyncBatchNormalization as syncbatchnorm

SEQUENCE_LENGTH=65536

@tf.keras.utils.register_keras_serializable()
class aformer(tf.keras.Model):
    def __init__(self,
                 kernel_transformation = 'relu_kernel_transformation',
                 dropout_rate: float = 0.2,
                 attention_dropout_rate: float = 0.05,
                 input_length: int = 196608,
                 atac_output_length: int = 896,
                 num_heads:int = 4,
                 numerical_stabilizer: float =0.001,
                 nb_random_features:int = 256,
                 hidden_size:int = 128,
                 transformer_depth_1:int = 4,
                 transformer_depth_2:int = 4,
                 shared_transformer_depth:int = 4,
                 pre_transf_channels: int = 128,
                 d_model = 128,
                 TF_inputs=128,
                 norm=True,
                 dim = 32, 
                 max_seq_length = 1536,
                 rel_pos_bins=1536, 
                 kernel_size=None, 
                 use_rot_emb = True,
                 use_mask_pos = False, 
                 normalize = True,
                 seed = 3,
                 load_init=False,
                 inits=None,
                 filter_list=None,
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
        self.num_heads=num_heads
        self.input_length=input_length
        self.numerical_stabilizer=numerical_stabilizer
        self.nb_random_features=nb_random_features
        self.hidden_size=hidden_size
        self.pre_transf_channels=pre_transf_channels
        self.transformer_depth_1=transformer_depth_1
        self.transformer_depth_2=transformer_depth_2
        self.shared_transformer_depth=shared_transformer_depth
        self.norm=norm
        self.d_model = d_model
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.rel_pos_bins = rel_pos_bins
        self.use_rot_emb = use_rot_emb
        self.use_mask_pos = use_mask_pos
        self.normalize = normalize
        self.seed = seed
        self.TF_inputs=TF_inputs
        self.attention_dropout_rate=attention_dropout_rate
        self.atac_output_length=atac_output_length
        self.load_init=load_init
        self.inits=inits
        self.filter_list = [768, 896, 1024, 1152, 1280, 1536] if self.load_init else filter_list
        self.freeze_conv_layers = freeze_conv_layers
        
        print(self.filter_list)
        
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
        self.stem_conv = tf.keras.layers.Conv1D(filters= int(self.filter_list[-1]) // 2,
                                   kernel_size=15,
                                   kernel_initializer=self.inits['stem_conv_k'] if self.load_init else 'glorot_uniform',
                                   bias_initializer=self.inits['stem_conv_b'] if self.load_init else 'zeros',
                                   strides=1,
                                   trainable=False if self.freeze_conv_layers else True,
                                   padding='same')
                                   #data_format='channels_last')
        self.stem_res_conv=Residual(enf_conv_block(int(self.filter_list[-1]) // 2, 1,
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
        

        self.conv_tower = tf.keras.Sequential([
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
            for i, num_filters in enumerate(self.filter_list)], name='conv_tower')
        
        ### conv stack for atac inputs
        self.tf_module = tf_module(TF_inputs=self.TF_inputs,
                                   dropout_rate=self.dropout_rate,
                                   name='tf_module',
                                   **kwargs)
        
        self.dim_reduce_block = conv1d_block_dim_reduce(num_channels_out=self.pre_transf_channels,
                                                         name='dim_reduce_block',
                                                        **kwargs)
        
        self.dim_reduce_block2 = conv1d_block_dim_reduce(num_channels_out=self.pre_transf_channels,
                                                         name='dim_reduce_block',
                                                        **kwargs)

        self.sin_pe1 = abs_sin_PE(name='sin_pe1',
                                  **kwargs)
        self.sin_pe2 = abs_sin_PE(name='sin_pe2',
                                  **kwargs)
        self.sin_pe3 = abs_sin_PE(name='sin_pe3',
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
                                                   normalize=self.normalize, seed = self.seed,
                                                     name = 'shared_transformer',
                                                     **kwargs)


        
        self.transformer_stack_1 = Performer_Encoder(num_layers=self.transformer_depth_1,
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
                                                   normalize=self.normalize, seed = self.seed,
                                                     name = 'transformer_stack1',
                                                     **kwargs)
        
        self.transformer_stack_2 = Performer_Encoder(num_layers=self.transformer_depth_2,
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
                                                   normalize=self.normalize, seed = self.seed, 
                                                     name = 'transformer_stack2',
                                                     **kwargs)

        
        self.final_pointwise_atac = enf_conv_block(filters=2*int(self.filter_list[-1]),
                                                  name = 'final_pointwise_atac')
        self.dropout = kl.Dropout(rate=0.05,**kwargs)
        self.gelu = tfa.layers.GELU()
        self.atac_head = output_head_atac(name = 'atac_out_head',
                                          **kwargs)
        
        self.final_pointwise_rna = enf_conv_block(filters=2*self.pre_transf_channels,
                                                  name = 'final_pointwise_rna')
        self.rna_head = output_head_rna(name = 'rna_out_head',
                                        dropout_rate = self.dropout_rate,
                                        **kwargs)
        self.crop = TargetLengthCrop1D(target_length=atac_output_length)
        

    def call(self, inputs, atac_train, rna_train, use_tf_module, training:bool=True):

        sequence, TSSs, exons, tf_inputs, atac, target = inputs
        x = self.stem_conv(sequence,
                           training=training)
        x = self.stem_res_conv(x,training=training)
        x = self.stem_pool(x,training=training)
        enformer_conv_out = self.conv_tower(x,training=training)

        tf_processed = self.tf_module(tf_inputs, 
                                      training=training)
        tf_processed = tf.expand_dims(tf_processed, 
                                      axis=1)
        tf_processed = tf.tile(tf_processed, 
                               [1,1536,1])
        if not use_tf_module:
            tf_processed = tf.ones_like(tf_processed)
        
        enformer_conv_out = tf.concat([enformer_conv_out,
                                       tf_processed],axis=2)
        enformer_conv_out = self.dim_reduce_block(enformer_conv_out,
                                                   training=training)
        enformer_conv_out = self.sin_pe1(enformer_conv_out)
        shared_transformer_out,att_matrices_shared = self.shared_transformer(enformer_conv_out,
                                                                             training=training)

        ### transformer 1 is atac output
        if atac_train:
            ## add on absolute PEs here, will also add on RPE within transformer stack
            transformer_input_1 = self.sin_pe2(shared_transformer_out)
            transformer_out_1, att_matrices_1 = self.transformer_stack_1(shared_transformer_out,
                                                                        training=training)
        
            atac_output = self.final_pointwise_atac(transformer_out_1,
                                                    training=training)
            atac_output = self.dropout(atac_output,
                                       training=training)
            atac_output = self.gelu(atac_output,
                                    training=training)
            atac_output = self.atac_head(atac_output,
                                         training=training)
        else:
            atac_output = atac

        ### now feed out transformer_out_1 into the RNA transformer after appending w/ TSSs, exons, introns, ATAC
        ## transformer_out_1 dimension is [B x 1536 x 132]
        if rna_train: 
            transformer_input_2 = tf.concat([shared_transformer_out,
                                             atac_output,
                                             TSSs,
                                             exons],
                                            axis=2)
            
            transformer_input_2 = self.dim_reduce_block2(transformer_input_2,
                                                         training=training)
            
            transformer_input_2 = self.sin_pe3(transformer_input_2)
            transformer_out_2,att_matrices_2 = self.transformer_stack_2(transformer_input_2,
                                                                        training=training)

            rna_output = self.final_pointwise_rna(transformer_out_2)
            rna_output = self.dropout(rna_output,
                                      training=training)
            rna_output = self.gelu(rna_output,
                                   training=training)
            rna_output = self.rna_head(rna_output,
                                       training=training)
        else:
            rna_output = target

        return self.crop(atac_output), rna_output
    

    def get_config(self):
        config = {
            "dropout_rate":self.dropout_rate,
            "input_length": self.input_length,
            "final_out_length": self.final_out_length,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "numerical_stabilizer": self.numerical_stabilizer,
            "kernel_transformation": self.kernel_transformation,
            "nb_random_features": self.nb_random_features,
            "transformer_depth_1":self.transformer_depth_1,
            "transformer_depth_2":self.transformer_depth_2,
            "pre_transf_channels":self.pre_transf_channels,
            "d_model":self.d_model,
            "norm":self.norm,
            "dim":self.dim,
            "max_seq_length":self.max_seq_length,
            "rel_pos_bins":self.rel_pos_bins,
            "kernel_size":self.kernel_size,
            "use_rot_emb":self.use_rot_emb,
            "use_mask_pos":self.use_mask_pos,
            "normalize":self.normalize,
            "seed":self.seed,
            "TF_inputs":self.TF_inputs
            
        }
        
        base_config = super().get_config()
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    #@tf.function(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.float32),
    #                              tf.TensorSpec([None, 1572], tf.float32)])
    def predict_on_batch(self, inputs, training:bool=False):
        
        sequence,atac,tf_inputs = inputs
        
        # sequence processing module
        x_seq = self.convstack_seq(sequence,training=training) ### here dimension is 131072 / 2, C = hidden/size / 2
        
        x_atac = self.convstack_atac(atac,training=training)

        transformer_input = tf.concat([x_seq,
                                       x_atac],axis=2)
        transformer_input = self.sin_pe(transformer_input)
        
        x,att_matrices = self.transformer_stack(transformer_input,training=training)

        org_spec_inputs = x,tf_inputs
        
        return [{head: head_module(org_spec_inputs,
                                   training=training)
                for head, head_module in self.heads.items()},att_matrices]

