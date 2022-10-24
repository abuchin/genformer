#!/bin/bash -l

python3 train_model_aformer_TF_expression_peaks.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="aformer_TF_ATAC" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_TF_ATAC" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/seqtoatac_98k_57kstride_blacklist0.25_peaks/preprocessed" \
            --gcs_path_val_ho="gs://picard-testing-176520/seqtoatac_98k_57kstride_blacklist0.25_peaks/val_holdout/preprocessed" \
            --input_length=98304 \
            --atac_length_uncropped=768 \
            --atac_output_length=448 \
            --max_shift=20 \
            --batch_size=8 \
            --num_epochs=30 \
            --train_examples=3338355 \
            --warmup_frac=0.02 \
            --val_examples=952508 \
            --val_examples_ho=56772 \
            --patience=20 \
            --min_delta=0.000005 \
            --model_save_dir="gs://picard-testing-176520/seqtoatac_98k_57kstride_blacklist0.25_peaks/models" \
            --model_save_basename="aformer_TF_ATAC" \
            --lr_base1="2.5e-05" \
            --lr_base2="2.0e-04" \
            --lr_base3="9.0e-05" \
            --decay_frac="0." \
            --weight_decay_frac="0.30" \
            --gradient_clip="5.0" \
            --epsilon=1.0e-14 \
            --transformer_depth_rna="4" \
            --shared_transformer_depth="6" \
            --pre_transf_channels="800" \
            --dropout_rate="0.40" \
            --tf_dropout_rate="0.25" \
            --attention_dropout_rate="0.15" \
            --pointwise_dropout_rate="0.15" \
            --num_heads="8" \
            --num_random_features="256" \
            --hidden_size="800" \
            --dim=100 \
            --kernel_transformation="softmax_kernel_transformation" \
            --savefreq=2 \
            --TF_inputs=256 \
            --train_mode="atac_only" \
            --use_tf_module="True" \
            --rna_loss_scale="0.50" \
            --lambda1="0.50" \
            --lambda2="1.0" \
            --lambda3="0.50" \
            --freeze_conv_layers="False" \
            --load_init="True" \
            --atac_peaks_cropped=56 \
            --loss_type="mse" \
            --enformer_checkpoint_path="sonnet_weights"