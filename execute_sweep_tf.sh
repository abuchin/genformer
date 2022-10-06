#!/bin/bash -l

python3 train_model_aformer_TF_expression.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="aformer_TF_ATAC" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_TF_ATAC" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/seqtoatac_98k_73kstride_blacklist0.25/preprocessed" \
            --input_length=98304 \
            --atac_length_uncropped=768 \
            --atac_output_length=448 \
            --max_shift=20 \
            --batch_size=24 \
            --num_epochs=60 \
            --train_examples=3587877 \
            --warmup_frac=0.01 \
            --val_examples=634035 \
            --val_examples_ho=634035 \
            --patience=15 \
            --min_delta=0.001 \
            --model_save_dir="gs://picard-testing-176520/seqtoatac_98k_73kstride_blacklist0.25/models" \
            --model_save_basename="aformer_TF_ATAC" \
            --lr_base="1.0e-04" \
            --gradient_clip="1.0" \
            --epsilon=1.0e-14 \
            --transformer_depth_1="4" \
            --transformer_depth_2="4" \
            --shared_transformer_depth="4" \
            --pre_transf_channels="384" \
            --dropout_rate="0.30" \
            --attention_dropout_rate="0.30" \
            --num_heads="8" \
            --num_random_features="384" \
            --hidden_size="384" \
            --dim=48 \
            --kernel_transformation="relu_kernel_transformation" \
            --tf_module_kernel="relu_kernel_transformation" \
            --savefreq=10 \
            --TF_inputs=96 \
            --train_mode="atac_only" \
            --load_init="False" \
            --freeze_conv_layers="False" \
            --use_tf_module="True,False" \
            --rna_loss_scale="0.50" \
            --filter_list="192,224,256,288,320,384" \
            --checkpoint_path="sonnet_weights" \
            --tf_transformer_layers="2" \
            --tf_heads="8" \
            --tf_hidden_size="64"