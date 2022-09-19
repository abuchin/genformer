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
                 kernel_transformation = 'softmax_kernel_transformation',
                 dropout_rate: float = 0.2,
                 attention_dropout_rate: float = 0.05,
                 input_length: int = 196608,
                 atac_output_length: int = 1536,
                 num_heads:int = 4,
                 numerical_stabilizer: float =0.001,
                 nb_random_features:int = 256,
                 hidden_size:int = 64,
                 transformer_depth_1:int = 4,
                 transformer_depth_2:int = 4,
                 pre_transf1_channels: int = 128,
                 pre_transf2_channels: int = 32,
                 d_model = 64,
                 TF_inputs=128,
                 norm=True,
                 dim = 32, 
                 max_seq_length = 1536,
                 # nb_random_features = 64, 
                 rel_pos_bins=128, 
                 kernel_size=None, 
                 use_rot_emb = True,
                 use_mask_pos = False, 
                 normalize = True,
                 seed = 3,
                 stride=1,
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
        self.pre_transf1_channels=pre_transf1_channels
        self.pre_transf2_channels=pre_transf2_channels
        self.transformer_depth_1=transformer_depth_1
        self.transformer_depth_2=transformer_depth_2
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
        self.stride = stride
        
        ### conv stack for sequence inputs
        self.enformer_seq_module = enformer_pretrained(**kwargs)
        
        ### conv stack for atac inputs
        self.tf_module = tf_module(TF_inputs = self.TF_inputs,
                                   dropout=self.dropout,
                                   **kwargs)
        
        self.dim_reduce_block1 = conv1d_block_dim_reduce(num_channels_out=self.pre_transf1_channels,
                                                        **kwargs)
        self.dim_reduce_block2 = conv1d_block_dim_reduce(num_channels_out=self.pre_transf2_channels,
                                                        **kwargs)
                                        
        self.sin_pe = abs_sin_PE(self.positional_dropout_rate, **kwargs)

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
                                                   kernel_size=self.kernel_size,
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
                                                   widening=self.widening,
                                                   hidden_size=self.hidden_size,
                                                   numerical_stabilizer=self.numerical_stabilizer,
                                                   attention_dropout=self.attention_dropout_rate,
                                                   rel_pos_bins=self.rel_pos_bins,
                                                   kernel_size=self.kernel_size,
                                                   use_rot_emb=self.use_rot_emb,
                                                   use_mask_pos=self.use_mask_pos,
                                                   kernel_transformation=self.kernel_transformation,
                                                   normalize=self.normalize, seed = self.seed, 
                                                     name = 'transformer_stack2',
                                                     **kwargs)

        
        ## final conv stack, organism specific
        self.atac_head = output_head(output_length=self.atac_output_length,
                                     dropout_rate=self.dropout_rate / 5,
                                     name = 'atac_out_head',
                                     **kwargs)
        
        ## final conv stack, organism specific
        self.rna_head = output_head(output_length=1,
                                     dropout_rate=self.dropout_rate / 10,
                                    name = 'rna_out_head',
                                     **kwargs)

    #@tf.function(input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 5], tf.bfloat16),
    #                              tf.TensorSpec([None, 1572], tf.bfloat16)])
    def call(self, inputs, training:bool=True):
        
        sequence,TSSs,exons,introns,tf_inputs = inputs
        
        # sequence processing module
        enformer_conv_out = self.enformer_seq_module(sequence,training=training) 
        ## output dimension of [B x L x C] -> [B x 1536 x 1536]
        
        # process the TF expression data
        ## can make this organism specific
        tf_processed = self.tf_module(tf_inputs,training=training)
        
        ## now we want to append the TF expression data in the channel dimension
        ## so we get an input to pre transformer reduction layer of [B x 1536 x 1600]

        transformer_input_1 = tf.concat([enformer_conv_out,
                                         tf_processed],axis=2)
        
        # pre transformer channel reduction block
        # now dimension will go from [B x 1536 x 128] using 1x1 convolutions
        transformer_input_1 = self.dim_reduce_block1(transformer_input_1,
                                                    training=training)
        
        ## add on absolute PEs here, will also add on RPE within transformer stack
        transformer_input_1 = self.sin_pe(transformer_input_1)
        transformer_out_1,att_matrices_1 = self.transformer_stack_1(transformer_input_1,
                                                                    training=training)
        
        
        atac_output = self.atac_head(transformer_out_1,training=training)

        ### now feed out transformer_out_1 into the RNA transformer after appending w/ TSSs, exons, introns, ATAC
        ## transformer_out_1 dimension is [B x 1536 x 132]
        transformer_input_2 = tf.concat([transformer_input_1,
                                         atac_output,
                                         TSSs,
                                         exons,
                                         introns],
                                        axis=2)
        transformer_input_2 = self.dim_reduce_block2(transformer_input_2,
                                                    training=training)
        # transformer_input_2 dimension is [B x 1536 x 32]
        transformer_input_2 = self.sin_pe(transformer_input_2)
        transformer_out_2,att_matrices_2 = self.transformer_stack_2(transformer_input_2,
                                                                    training=training)

        # transformer_out_2 dimension is [B x 1536 x 32]
        
        rna_output = self.rna_head(transformer_out_2,training=training)
        
        
        return atac_output, rna_output, att_matrices_2
    

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
            "pre_transf1_channels":self.pre_transf1_channels,
            "pre_transf2_channels":self.pre_transf2_channels,
            "d_model":self.d_model,
            "norm":self.norm,
            "dim":self.dim,
            "human":self.human,
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

