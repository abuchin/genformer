#!/bin/bash -l

python3 train_model_atac.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="atac_pretraining" \
            --wandb_user="njaved" \
            --wandb_sweep_name="atac_pretraining" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/genformer_atac_pretrain/524/genformer_atac_pretrain_globalacc_conv_rpgc_human" \
            --gcs_path_mm="gs://picard-testing-176520/genformer_atac_pretrain/262k/genformer_atac_pretrain_globalacc_conv_rpgc_mouse" \
            --gcs_path_rm="gs://picard-testing-176520/genformer_atac_pretrain/262k/genformer_atac_pretrain_globalacc_conv_rpgc_rhesus" \
            --gcs_path_rat="gs://picard-testing-176520/genformer_atac_pretrain/262k/genformer_atac_pretrain_globalacc_conv_rpgc_rat" \
            --gcs_path_holdout="gs://picard-testing-176520/genformer_atac_pretrain/524/genformer_atac_pretrain_globalacc_conv_rpgc_human" \
            --training_type="hg_mm_rm_rat" \
            --input_length=524288 \
            --output_length=4096 \
            --output_length_ATAC=131072 \
            --final_output_length=2048 \
            --max_shift=10 \
            --batch_size=4 \
            --num_epochs=50 \
            --train_examples=100000 \
            --val_examples_ho=28769 \
            --BN_momentum=0.90 \
            --warmup_frac=0.005 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.0000005 \
            --model_save_dir="gs://picard-testing-176520/genformer_atac_pretrain/models" \
            --model_save_basename="aformer" \
            --lr_base1="2.0e-04" \
            --lr_base2="2.0e-04" \
            --decay_frac="0.50" \
            --gradient_clip="2.5" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="7" \
            --dropout_rate="0.20" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="8" \
            --num_random_features="256" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=1 \
            --freeze_conv_layers="False" \
            --load_init="False" \
            --rectify="True" \
            --filter_list_seq="512,564,616,668,720,768" \
            --filter_list_atac="64,128" \
            --optimizer="adam" \
            --stable_variant="False" \
            --atac_mask_dropout=0.15 \
            --log_atac="True" \
            --learnable_PE="False" \
            --sonnet_weights_bool="False" \
            --random_mask_size="768" \
            --use_atac="True" \
            --final_point_scale="4" \
            --use_seq="True" \
            --bce_loss_scale="0.95" \
            --use_pooling="False" \
            --seed=12 \
            --seq_corrupt_rate="15" \
            --atac_corrupt_rate="20"
                        
            
