#!/bin/bash -l

python train_model_aformer_TF_genecentered_separated.py \
            --tpu_name="node-22" \
            --tpu_zone="us-east1-d" \
            --wandb_project="aformer_TF_gene_centered" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_TF_gene_centered" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/65k_genecentered_blacklist0.50_atacnormalized/preprocessed" \
            --output_heads="hg,mm;hg" \
            --input_length="65536" \
            --max_shift=300 \
            --target_unit="logTPM" \
            --batch_size=8 \
            --num_epochs=9 \
            --train_steps=2676 \
            --warmup_frac=0.2 \
            --total_steps=26760 \
            --val_steps_h=421 \
            --val_steps_m=58 \
            --patience=10 \
            --min_delta=0.001 \
            --model_save_dir="gs://picard-testing-176520/65k_genecentered_blacklist0.50_atacnormalized/models" \
            --model_save_basename="aformer_TF_gene_centered" \
            --lr_base="5.0e-04" \
            --min_lr="5.0e-07" \
            --optimizer="adamw" \
            --gradient_clip="0.2" \
            --precision="mixed_bfloat16" \
            --weight_decay_frac="1.0e-02" \
            --epsilon=1.0e-10 \
            --rectify=True \
            --conv_channel_list="96,96,112,112,128,128" \
            --conv_filter_size_1_atac="15" \
            --conv_filter_size_2_atac="5" \
            --conv_filter_size_1_seq="15" \
            --conv_filter_size_2_seq="5" \
            --dropout="0.20" \
            --num_transformer_layers="2" \
            --num_heads="4" \
            --momentum="0.90" \
            --num_random_features="256" \
            --hidden_size="256" \
            --dim=64 \
            --slow_step_frac=0.5 \
            --sync_period=6 \
            --rel_pos_bins=512 \
            --kernel_transformation="softmax_kernel_transformation" \
            --kernel_regularizer="0.00001" \
            --savefreq=5 \
            --use_rot_emb="True" \
            --use_mask_pos="False" \
            --use_fft_prior="True" \
            --freq_limit="5000" \
            --fft_prior_scale="0.5" \
            --bottleneck_units="64" \
            --bottleneck_units_tf="64" \
            --use_tf_acc="True,False"
