#!/bin/bash -l

python3 train_model_atac_cage.py \
            --tpu_name="pod3" \
            --tpu_zone="us-central1-a" \
            --wandb_project="paired_rampage_atac" \
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
            --num_epochs=25 \
            --train_examples=250000 \
            --val_examples=59751  \
            --BN_momentum=0.90 \
            --warmup_frac=0.02 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.000005 \
            --model_save_dir="gs://picard-testing-176520/paired_rna_atac/genformer/models" \
            --model_save_basename="aformer_baseline" \
            --lr_base1="1.0e-04" \
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
            --savefreq=1 \
            --load_init_FT="True" \
            --load_init_FULL="True" \
            --rectify="True" \
            --checkpoint_path="gs://picard-testing-176520/genformer_atac_pretrain/models/aformer_524k_load-False_LR1-0.0001_LR2-0.0001_T-7_TF-False_2023-10-01_15:07:58/iteration_26" \
            --filter_list_seq="768,896,1024,1152,1280,1536" \
            --rna_scale="0.95" \
            --atac_mask_dropout=0.025 \
            --random_mask_size="256" \
            --log_atac="False" \
            --final_point_scale="6" \
            --seed=5 \
            --seq_corrupt_rate="20" \
            --atac_corrupt_rate="20" \
            --use_tf_activity="False"