#!/bin/bash -l

python3 train_model_atac_cage_early.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="paired_rampage_atac" \
            --wandb_user="njaved" \
            --wandb_sweep_name="paired_rampage_atac" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/genformer_atac_rampage_globalacc_conv_rpgc" \
            --gcs_path_TSS="gs://picard-testing-176520/genformer_atac_rampage_globalacc_conv_rpgc_TSS" \
            --input_length=196608 \
            --output_length=1536 \
            --output_length_ATAC=49152 \
            --final_output_length=896 \
            --max_shift=10 \
            --batch_size=8 \
            --num_epochs=100 \
            --train_examples=900000 \
            --val_examples=59751  \
            --val_examples_TSS= 57618 \
            --BN_momentum=0.90 \
            --warmup_frac=0.005 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.000005 \
            --model_save_dir="gs://picard-testing-176520/paired_rampage_atac/genformer/models" \
            --model_save_basename="aformer_baseline" \
            --lr_base1="1.0e-05" \
            --lr_base2="1.0e-04" \
            --decay_frac="1.0" \
            --gradient_clip="5.0" \
            --epsilon=1.0e-14 \
            --num_transformer_layers="6" \
            --dropout_rate="0.05" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="8" \
            --num_random_features="256" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=2 \
            --freeze_conv_layers="False" \
            --load_init="True" \
            --wd_1_frac=1.0e-03 \
            --wd_2_frac=1.0e-03 \
            --rectify="True" \
            --multitask_checkpoint_path="gs://picard-testing-176520/enformer_performer/models/enformer_performer_230303_E-P-_5313_enformer_196k_load_init-False_freeze-False_LR1-0.0001_LR2-0.0001_T-6_F-1024_K-relu_kernel_transformation/iteration_21" \
            --filter_list_seq="512,640,768,896,1024,1024" \
            --inits_type="enformer_conv" \
            --predict_masked_atac_bool="True" \
            --cage_scale="0.50" \
            --optimizer="adabelief" \
            --stable_variant="False" \
            --atac_mask_dropout=0.15 \
            --loss_fn="poisson" \
            --use_global="False" \
            --use_atac="False" \
            --log_atac="True" \
            --learnable_PE="True" \
            --global_acc_size=128 \
            --sonnet_weights_bool="False"
                        
            
