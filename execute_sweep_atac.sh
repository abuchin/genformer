#!/bin/bash -l

python3 train_model_atac.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="atac_pretraining" \
            --wandb_user="njaved" \
            --wandb_sweep_name="atac_pretraining" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/genformer_atac_pretrain/196k/genformer_atac_pretrain_globalacc_conv_rpgc" \
            --gcs_path_holdout="gs://picard-testing-176520/genformer_atac_pretrain/196k/genformer_atac_pretrain_globalacc_conv_rpgc_holdout" \
            --input_length=196608 \
            --output_length=1536 \
            --output_length_ATAC=49152 \
            --final_output_length=896 \
            --max_shift=10 \
            --batch_size=4 \
            --num_epochs=150 \
            --train_examples=500000 \
            --val_examples_ho=28769 \
            --BN_momentum=0.90 \
            --warmup_frac=0.005 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.000005 \
            --model_save_dir="gs://picard-testing-176520/genformer_atac_pretrain/models" \
            --model_save_basename="aformer_baseline" \
            --lr_base1="2.0e-04" \
            --lr_base2="2.0e-04" \
            --decay_frac="0.40" \
            --gradient_clip="5.0" \
            --epsilon=1.0e-14 \
            --num_transformer_layers="1" \
            --dropout_rate="0.10" \
            --pointwise_dropout_rate="0.10" \
            --num_heads="8" \
            --num_random_features="256" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=2 \
            --freeze_conv_layers="False" \
            --load_init="False" \
            --wd_1_frac=0.0 \
            --wd_2_frac=0.0 \
            --rectify="True" \
            --multitask_checkpoint_path="gs://picard-testing-176520/sonnet_weights/sonnet_weights" \
            --filter_list_seq="384,512,640,768,896,1024" \
            --inits_type="enformer_conv" \
            --optimizer="adabelief" \
            --stable_variant="False" \
            --atac_mask_dropout=0.30 \
            --log_atac="True" \
            --learnable_PE="True" \
            --sonnet_weights_bool="True" \
            --random_mask_size="2048" \
            --bce_loss_scale=0.90
                        
            