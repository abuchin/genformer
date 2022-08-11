#!/bin/bash -l

python train_model_aformer_TF_genecentered_separated.py \
            --tpu_name="node-20" \
            --tpu_zone="us-east1-d" \
            --wandb_project="aformer_TF_gene_centered_test" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_TF_gene_centered_test" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/65k_genecentered_blacklist0.50_atacnormalized/preprocessed" \
            --output_heads="hg;hg,mm" \
            --input_length=65536 \
            --max_shift=300 \
            --target_unit="logTPM" \
            --batch_size=72 \
            --num_epochs=20 \
            --train_steps=298 \
            --warmup_frac=0.025 \
            --val_steps_h=47 \
            --val_steps_m=7 \
            --patience=10 \
            --min_delta=0.001 \
            --model_save_dir="gs://picard-testing-176520/65k_genecentered_blacklist0.50_atacnormalized/models" \
            --model_save_basename="aformer_TF_gene_centered" \
            --lr_schedule="cosine_decay_w_warmup" \
            --lr_base="0.0005" \
            --min_lr="0.000001" \
            --optimizer="adafactor" \
            --gradient_clip="0.2" \
            --precision="mixed_bfloat16" \
            --weight_decay_frac="5.0e-04" \
            --epsilon=1.0e-10 \
            --rectify=True \
            --conv_channel_list="96,96,112,112,128,128" \
            --conv_filter_size_1_atac="20" \
            --conv_filter_size_2_atac="5" \
            --conv_filter_size_1_seq="15" \
            --conv_filter_size_2_seq="5" \
            --dropout="0.15,0.35" \
            --num_transformer_layers="2,4" \
            --num_heads="4" \
            --momentum="0.90" \
            --num_random_features="256" \
            --hidden_size="256" \
            --dim=64 \
            --slow_step_frac=0.5 \
            --sync_period=6 \
            --max_seq_length=512 \
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
            --bottleneck_units_tf="64"
