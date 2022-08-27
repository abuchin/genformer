#!/bin/bash -l

python3 train_model_aformer_TF_genecentered_separated.py \
            --tpu_name="node-24" \
            --tpu_zone="us-east1-d" \
            --wandb_project="aformer_initial_run" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_initial_run" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/16k_genecentered_blacklist0.50_atacnormalized/preprocessed" \
            --gcs_path_val_ho="gs://picard-testing-176520/16k_genecentered_blacklist0.50_atacnormalized/val_holdout/preprocessed/val" \
            --output_heads="hg" \
            --input_length="32768,65536,131072,196608" \
            --max_shift=300 \
            --target_unit="logTPM" \
            --batch_size=72 \
            --num_epochs=25 \
            --train_steps=298 \
            --warmup_frac=0.08 \
            --total_steps=14900 \
            --val_steps_h=47 \
            --val_steps_ho=6 \
            --patience=7 \
            --min_delta=0.001 \
            --model_save_dir="gs://picard-testing-176520/16k_genecentered_blacklist0.50_atacnormalized/models" \
            --model_save_basename="aformer_initial_tests" \
            --lr_base="1.0e-07" \
            --min_lr="5.0e-12" \
            --optimizer="adamw" \
            --gradient_clip="0.2" \
            --precision="mixed_bfloat16" \
            --weight_decay_frac="1.0e-02" \
            --epsilon=1.0e-10 \
            --rectify=True \
            --conv_channel_list="48,48,56,56,64,64" \
            --conv_filter_size_1_atac="15" \
            --conv_filter_size_2_atac="5" \
            --conv_filter_size_1_seq="15" \
            --conv_filter_size_2_seq="5" \
            --dropout="0.25" \
            --num_transformer_layers="2" \
            --num_heads="4" \
            --momentum="0.90" \
            --num_random_features="128" \
            --hidden_size="128" \
            --dim=32 \
            --slow_step_frac=0.5 \
            --sync_period=6 \
            --num_parallel= 768 \
            --rel_pos_bins=512 \
            --kernel_transformation="relu_kernel_transformation" \
            --kernel_regularizer="0" \
            --savefreq=8 \
            --use_rot_emb="True" \
            --use_mask_pos="False" \
            --use_fft_prior="True" \
            --freq_limit_scale="0.07" \
            --fft_prior_scale="0.20" \
            --bottleneck_units="32" \
            --bottleneck_units_tf="32" \
            --use_tf_acc="False"
