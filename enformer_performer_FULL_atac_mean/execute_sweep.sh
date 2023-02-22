#!/bin/bash -l

python3 train_model_batchnorm_experiments.py \
            --tpu_name="node-5" \
            --tpu_zone="us-central1-a" \
            --wandb_project="enformer_baseline" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_baseline" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://genformer_data/expanded_originals/196k" \
            --gcs_path_TSS="gs://genformer_data/expanded_originals/196k/human/tfrecords_tss" \
            --num_epochs=100 \
            --warmup_frac=0.01 \
            --patience=50 \
            --min_delta=0.00001 \
            --model_save_dir="gs://picard-testing-176520/enformer_performer_FULL_atac_mean/models" \
            --model_save_basename="enformer_performer_FULL_atac_mean" \
            --lr_base1="1.0e-04" \
            --lr_base2="1.0e-04" \
            --wd_1="1.0e-08" \
            --wd_2="1.0e-08" \
            --decay_frac="0.90" \
            --gradient_clip="5.0" \
            --BN_momentum="0.90" \
            --hidden_size="1040" \
            --epsilon=1.0e-8 \
            --num_parallel=8 \
            --dropout_rate=0.40 \
            --attention_dropout_rate=0.05 \
            --savefreq=5 \
            --val_examples_TSS=2134 \
            --load_init="False" \
            --freeze_conv_layers="False" \
            --stable_variant="True" \
            --num_examples_dict="human:250,250;mouse:250,250" \
            --num_transformer_layers=8 \
            --num_heads=8 \
            --optimizer="adamw" \
            --heads_channels="human:5313;mouse:1643" \
            --kernel_transformation="relu_kernel_transformation" \
            --use_mask_pos="False" \
            --use_rot_emb="True" \
            --filter_list_seq="512,512,768,768,1024,1024"
