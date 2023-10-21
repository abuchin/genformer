#!/bin/bash -l

python3 train_model_atac_rna.py \
            --tpu_name="pod2" \
            --tpu_zone="us-east1-d" \
            --wandb_project="paired_rna_atac" \
            --wandb_user="njaved" \
            --wandb_sweep_name="paired_rna_atac" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/paired_rna_atac/524k/paired_atac_rna_global_acc_fpm" \
            --gcs_path_holdout="gs://picard-testing-176520/paired_rna_atac/524k/paired_atac_rna_global_acc_fpm_holdout" \
            --input_length=524288 \
            --output_length=4096 \
            --output_length_ATAC=131072 \
            --final_output_length=896 \
            --max_shift=10 \
            --batch_size=4 \
            --num_epochs=60 \
            --train_examples=1000000 \
            --val_examples=64177  \
            --BN_momentum=0.90 \
            --warmup_frac=0.0005 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.000005 \
            --model_save_dir="gs://picard-testing-176520/paired_rna_atac/models" \
            --model_save_basename="aformer_baseline" \
            --lr_base1="7.5e-05" \
            --lr_base2="1.0e-04" \
            --decay_frac="0.005" \
            --gradient_clip="5.0" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="7" \
            --dropout_rate="0.20" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="8" \
            --num_random_features="256" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=50 \
            --load_init_FT="True" \
            --load_init_FULL="False" \
            --rectify="True" \
            --checkpoint_path="gs://picard-testing-176520/genformer_atac_pretrain/models/aformer_524k_load-True_LR1-6e-05_LR2-6e-05_T-7_TF-False_2023-10-05_18:58:19/iteration_26" \
            --filter_list_seq="768,896,1024,1152,1280,1536" \
            --atac_scale="0.01" \
            --atac_mask_dropout=0.05 \
            --random_mask_size="512" \
            --log_atac="False" \
            --final_point_scale="6" \
            --seed=5 \
            --seq_corrupt_rate="5" \
            --atac_corrupt_rate="10" \
            --use_tf_activity="False" \
            --use_atac="True" \
            --use_seq="True" \
            --freeze_conv_layers="False" \
            --loss_type="poisson"
