#!/bin/bash -l

python3 train_model_atac_cage.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="enformer_baseline" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_baseline" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/genformer_atac_rampage_globalacc_conv" \
            --gcs_path_TSS="gs://picard-testing-176520/genformer_atac_rampage_globalacc_conv_TSS" \
            --input_length=196608 \
            --output_length=1536 \
            --final_output_length=896 \
            --max_shift=10 \
            --batch_size=2 \
            --num_epochs=100 \
            --train_examples=500 \
            --val_examples=250 \
            --val_examples_TSS=250 \
            --BN_momentum=0.90 \
            --warmup_frac=0.025 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.000005 \
            --model_save_dir="gs://picard-testing-176520/aformer_baseline" \
            --model_save_basename="aformer_baseline" \
            --lr_base1="1.0e-06" \
            --lr_base2="7.5e-05" \
            --decay_frac="0.50" \
            --gradient_clip="5.0" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="6" \
            --dropout_rate="0.05" \
            --pointwise_dropout_rate="0.025" \
            --num_heads="4" \
            --num_random_features="256" \
            --hidden_size="1552" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=40 \
            --freeze_conv_layers="True" \
            --load_init="True" \
            --wd1_frac=0.0 \
            --wd2_frac=0.0 \
            --rectify="True" \
            --multitask_checkpoint_path="gs://picard-testing-176520/enformer_performer_FULL_atac_mean/models/enformer_performer_FULL_atac_mean_196k_load_init-True_freeze-True_LR1-1e-06_LR2-7.5e-05_T-6_F-1536_D-0.4_K-relu_kernel_transformation_AD-0.05/iteration_30" \
            --inits_type="enformer_performer" \
            --predict_masked_atac_bool="True"
            
