#!/bin/bash -l

python3 train_model_atac.py \
            --tpu_name="node-2" \
            --tpu_zone="us-east1-d" \
            --wandb_project="atac_pretraining" \
            --wandb_user="njaved" \
            --wandb_sweep_name="atac_pretraining" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/genformer_atac_pretrain/196k/genformer_atac_pretrain_globalacc_conv_fpm_human" \
            --gcs_path_holdout="gs://picard-testing-176520/genformer_atac_pretrain/196k/genformer_atac_pretrain_globalacc_conv_fpm_human_valid" \
            --input_length=196608 \
            --output_length=1536 \
            --output_length_ATAC=49152 \
            --final_output_length=896 \
            --max_shift=10 \
            --batch_size=4 \
            --num_epochs=60 \
            --train_examples=5000 \
            --val_examples_ho=15491 \
            --BN_momentum=0.90 \
            --warmup_frac=0.001 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.0000005 \
            --model_save_dir="gs://picard-testing-176520/genformer_atac_pretrain/models" \
            --model_save_basename="aformer" \
            --lr_base1="1.0e-04" \
            --lr_base2="1.0e-04" \
            --decay_frac="0.005" \
            --gradient_clip="5.0" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="6" \
            --dropout_rate="0.20" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="8" \
            --num_random_features="256" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=50 \
            --freeze_conv_layers="False" \
            --load_init="False" \
            --rectify="True" \
            --filter_list_seq="768,896,1024,1152,1280,1536" \
            --filter_list_atac="32,64" \
            --atac_mask_dropout=0.05 \
            --log_atac="False" \
            --sonnet_weights_bool="False" \
            --random_mask_size="2048" \
            --use_atac="True" \
            --final_point_scale="6" \
            --use_seq="True" \
            --use_pooling="True" \
            --seed=15 \
            --seq_corrupt_rate="20" \
            --atac_corrupt_rate="20" \
            --use_tf_activity="True"
