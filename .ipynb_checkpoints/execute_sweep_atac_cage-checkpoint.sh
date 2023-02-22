#!/bin/bash -l

python3 train_model_atac_cage_early.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="enformer_baseline" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_baseline" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/genformer_atac_rampage_globalacc_conv/preprocessed" \
            --gcs_path_TSS="gs://picard-testing-176520/genformer_atac_rampage_globalacc_conv_TSS/preprocessed" \
            --input_length=196608 \
            --output_length=1536 \
            --final_output_length=896 \
            --max_shift=10 \
            --batch_size=1 \
            --num_epochs=100 \
            --train_examples=250000 \
            --val_examples=68603 \
            --val_examples_TSS=64759 \
            --BN_momentum=0.90 \
            --warmup_frac=0.0025 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.000005 \
            --model_save_dir="gs://picard-testing-176520/aformer_baseline" \
            --model_save_basename="aformer_baseline" \
            --lr_base1="2.0e-06" \
            --lr_base2="1.0e-04" \
            --decay_frac="0.50" \
            --gradient_clip="5.0" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="6" \
            --dropout_rate="0.05" \
            --pointwise_dropout_rate="0.05" \
            --num_heads="8" \
            --num_random_features="256" \
            --hidden_size="1600" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=40 \
            --freeze_conv_layers="True" \
            --load_init="True" \
            --wd_1=0.0 \
            --wd_2=0.0 \
            --rectify="True" \
            --multitask_checkpoint_path="/home/jupyter/dev/BE_CD69_paper_2022/enformer_fine_tuning/checkpoint/sonnet_weights" \
            --inits_type="enformer_conv" \
            --predict_masked_atac_bool="True" \
            --cage_scale="5.0" \
            --optimizer="adamw" \
            --atac_mask_dropout=0.05
            
