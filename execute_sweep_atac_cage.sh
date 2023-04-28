#!/bin/bash -l

python3 train_model_atac_cage.py \
            --tpu_name="pod1" \
            --tpu_zone="us-east1-d" \
            --wandb_project="paired_rampage_atac" \
            --wandb_user="njaved" \
            --wandb_sweep_name="paired_rampage_atac" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/paired_rampage_atac/262k/genformer_atac_rampage_globalacc_conv_rpgc_5prime" \
            --gcs_path_holdout="gs://picard-testing-176520/paired_rampage_atac/262k/genformer_atac_rampage_globalacc_conv_rpgc_5prime_holdout" \
            --gcs_path_TSS="gs://picard-testing-176520/paired_rampage_atac/262k/genformer_atac_rampage_globalacc_conv_rpgc_TSS_5prime" \
            --gcs_path_TSS_holdout="gs://picard-testing-176520/paired_rampage_atac/262k/genformer_atac_rampage_globalacc_conv_rpgc_TSS_5prime_holdout" \
            --input_length=262144 \
            --output_length=2048 \
            --output_length_ATAC=65536 \
            --final_output_length=1536 \
            --max_shift=10 \
            --batch_size=4 \
            --num_epochs=100 \
            --train_examples=150000 \
            --val_examples=64177  \
            --val_examples_ho=6639  \
            --val_examples_TSS=97208 \
            --val_examples_TSS_ho=10056 \
            --BN_momentum=0.90 \
            --warmup_frac=0.05 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.000005 \
            --model_save_dir="gs://picard-testing-176520/paired_rampage_atac/genformer/models" \
            --model_save_basename="aformer_baseline" \
            --lr_base1="5.0e-06" \
            --lr_base2="5.0e-05" \
            --lr_base3="2.0e-04" \
            --decay_frac="0.50" \
            --gradient_clip="5.0" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="6" \
            --dropout_rate="0.30" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="8" \
            --num_random_features="256" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=1 \
            --freeze_conv_layers="False" \
            --freeze_BN_layers="False" \
            --load_init="True" \
            --rectify="True" \
            --multitask_checkpoint_path="gs://picard-testing-176520/genformer_atac_pretrain/models/aformer_hg_262k_load-True_LR-0.01_T-6_D-0.3_2023-04-26_00:40:44/iteration_84" \
            --filter_list_seq="768,896,1024,1152,1280,1536" \
            --inits_type="enformer_performer_full" \
            --cage_scale="0.90" \
            --optimizer="adam" \
            --stable_variant="False" \
            --atac_mask_dropout=0.05 \
            --random_mask_size="512" \
            --log_atac="True" \
            --learnable_PE="True" \
            --sonnet_weights_bool="True" \
            --predict_atac="True"
                        
            
