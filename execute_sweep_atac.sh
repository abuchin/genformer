#!/bin/bash -l

python3 train_model_atac_early.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="atac_pretraining" \
            --wandb_user="njaved" \
            --wandb_sweep_name="atac_pretraining" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/genformer_atac_pretrain_globalacc_conv_rpgc" \
            --gcs_path_holdout="gs://picard-testing-176520/genformer_atac_pretrain_globalacc_conv_rpgc_holdout" \
            --input_length=196608 \
            --output_length=1536 \
            --output_length_ATAC=49152 \
            --final_output_length=896 \
            --max_shift=10 \
            --batch_size=4 \
            --num_epochs=100 \
            --train_examples=250000 \
            --val_examples=50000  \
            --val_examples_ho=22130  \
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
            --num_transformer_layers="7" \
            --dropout_rate="0.05" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="8" \
            --num_random_features="256" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=2 \
            --freeze_conv_layers="False" \
            --load_init="True" \
            --wd_1_frac=0.0 \
            --wd_2_frac=0.0 \
            --rectify="True" \
            --multitask_checkpoint_path="gs://picard-testing-176520/sonnet_weights/sonnet_weights" \
            --filter_list_seq="768,896,1024,1152,1280,1536" \
            --inits_type="enformer_conv" \
            --optimizer="adamw" \
            --stable_variant="False" \
            --atac_mask_dropout=0.15 \
            --seq_mask_dropout=0.10 \
            --use_global="True" \
            --log_atac="True" \
            --learnable_PE="True" \
            --global_acc_size=64 \
            --sonnet_weights_bool="True" 
                        
            
