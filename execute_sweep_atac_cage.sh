#!/bin/bash -l

python3 train_model_atac_cage_early.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="paired_rampage_atac" \
            --wandb_user="njaved" \
            --wandb_sweep_name="paired_rampage_atac" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/genformer_atac_rampage_globalacc_conv_rpgc_5prime_65k" \
            --gcs_path_holdout="gs://picard-testing-176520/genformer_atac_rampage_globalacc_conv_rpgc_5prime_65k_holdout" \
            --gcs_path_TSS="gs://picard-testing-176520/genformer_atac_rampage_globalacc_conv_rpgc_TSS_5prime_65k" \
            --gcs_path_TSS_holdout="gs://picard-testing-176520/genformer_atac_rampage_globalacc_conv_rpgc_TSS_5prime_65k_holdout" \
            --input_length=65536 \
            --output_length=512 \
            --output_length_ATAC=16384 \
            --final_output_length=312 \
            --max_shift=10 \
            --batch_size=4 \
            --num_epochs=100 \
            --train_examples=250000 \
            --val_examples=59751  \
            --val_examples_ho=11065  \
            --val_examples_TSS=57618 \
            --val_examples_TSS_ho=4775 \
            --BN_momentum=0.90 \
            --warmup_frac=0.005 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.000005 \
            --model_save_dir="gs://picard-testing-176520/paired_rampage_atac/genformer/models" \
            --model_save_basename="aformer_baseline" \
            --lr_base1="5.0e-06" \
            --lr_base2="1.0e-04" \
            --decay_frac="1.0" \
            --gradient_clip="5.0" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="2" \
            --dropout_rate="0.05" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="4" \
            --num_random_features="256" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=15 \
            --freeze_conv_layers="False" \
            --load_init="True" \
            --wd_1_frac=0.0 \
            --wd_2_frac=0.0 \
            --rectify="True" \
            --multitask_checkpoint_path="gs://picard-testing-176520/sonnet_weights/sonnet_weights" \
            --filter_list_seq="768,896,1024,1152,1280,1536" \
            --inits_type="enformer_conv" \
            --predict_masked_atac_bool="True" \
            --cage_scale="0.50" \
            --optimizer="adamw" \
            --stable_variant="False" \
            --atac_mask_dropout=0.10 \
            --loss_fn="poisson" \
            --use_global="True" \
            --use_atac="True" \
            --log_atac="True" \
            --learnable_PE="True" \
            --global_acc_size=128 \
            --sonnet_weights_bool="True"
                        
            
