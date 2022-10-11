#!/bin/bash -l

python3 train_model_aformer_TF_expression.py \
            --tpu_name="node-2" \
            --tpu_zone="us-central1-a" \
            --wandb_project="aformer_TF_ATAC" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_TF_ATAC" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/seqtoatac_98k_73kstride_blacklist0.25/preprocessed" \
            --input_length=98304 \
            --atac_length_uncropped=768 \
            --atac_output_length=448 \
            --max_shift=20 \
            --batch_size=16 \
            --num_epochs=30 \
            --train_examples=3587877 \
            --warmup_frac=0.001 \
            --val_examples=634035 \
            --val_examples_ho=634035 \
            --patience=6 \
            --min_delta=0.001 \
            --model_save_dir="gs://picard-testing-176520/seqtoatac_98k_73kstride_blacklist0.25/models" \
            --model_save_basename="aformer_TF_ATAC" \
            --lr_base1="5.0e-06" \
            --lr_base2="1.0e-04,2.50e-05" \
            --decay_frac="1.0" \
            --weight_decay_frac="0.2" \
            --gradient_clip="5.0" \
            --epsilon=1.0e-14 \
            --transformer_depth_1="2" \
            --transformer_depth_2="2" \
            --shared_transformer_depth="5" \
            --pre_transf_channels="1600" \
            --dropout_rate="0.45" \
            --tf_dropout_rate="0.30" \
            --attention_dropout_rate="0.20" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="8" \
            --num_random_features="256" \
            --hidden_size="1600" \
            --dim=200 \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=10 \
            --TF_inputs=256 \
            --train_mode="atac_only" \
            --load_init="True" \
            --freeze_conv_layers="True" \
            --use_tf_module="True" \
            --rna_loss_scale="0.50" \
            --checkpoint_path="sonnet_weights"