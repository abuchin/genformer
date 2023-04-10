#!/bin/bash -l

python3 train_model_atac.py \
            --tpu_name="node-3" \
            --tpu_zone="us-central1-a" \
            --wandb_project="atac_pretraining" \
            --wandb_user="njaved" \
            --wandb_sweep_name="atac_pretraining" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/genformer_atac_pretrain/262k/genformer_atac_pretrain_globalacc_conv_rpgc_human" \
            --gcs_path_mm="gs://picard-testing-176520/genformer_atac_pretrain/262k/genformer_atac_pretrain_globalacc_conv_rpgc_mouse" \
            --gcs_path_rm="gs://picard-testing-176520/genformer_atac_pretrain/262k/genformer_atac_pretrain_globalacc_conv_rpgc_rhesus" \
            --gcs_path_rat="gs://picard-testing-176520/genformer_atac_pretrain/262k/genformer_atac_pretrain_globalacc_conv_rpgc_rat" \
            --gcs_path_holdout="gs://picard-testing-176520/genformer_atac_pretrain/262k/genformer_atac_pretrain_globalacc_conv_rpgc_val_holdout" \
            --input_length=262144 \
            --output_length=2048 \
            --output_length_ATAC=65536 \
            --final_output_length=1536 \
            --max_shift=10 \
            --batch_size=4 \
            --num_epochs=50 \
            --train_examples=100000 \
            --val_examples_ho=19917 \
            --BN_momentum=0.90 \
            --warmup_frac=0.05 \
            --patience=50 \
            --output_res=128 \
            --min_delta=0.0000005 \
            --model_save_dir="gs://picard-testing-176520/genformer_atac_pretrain/models" \
            --model_save_basename="aformer" \
            --lr_base1="1.0e-04" \
            --decay_frac="0.25" \
            --gradient_clip="5.0" \
            --epsilon=1.0e-8 \
            --num_transformer_layers="6" \
            --dropout_rate="0.30" \
            --pointwise_dropout_rate="0.15" \
            --num_heads="8" \
            --num_random_features="256" \
            --kernel_transformation="relu_kernel_transformation" \
            --savefreq=1 \
            --freeze_conv_layers="False" \
            --load_init="False" \
            --rectify="True" \
            --filter_list_seq="384,512,640,768,896,1024" \
            --filter_list_atac="32,64" \
            --inits_type="enformer_performer_full" \
            --optimizer="adam" \
            --stable_variant="False" \
            --atac_mask_dropout=0.20 \
            --log_atac="True" \
            --learnable_PE="False" \
            --sonnet_weights_bool="True" \
            --random_mask_size="1536" \
            --bce_loss_scale=0.95 \
            --use_atac="True" \
            --use_seq="True"
                        
            
