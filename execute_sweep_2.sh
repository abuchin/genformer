#!/bin/bash -l

python train_model_aformer_TF.py \
            --tpu_name="node-14" \
            --tpu_zone="us-central1-a" \
            --wandb_project="aformer" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_CAGE" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/CAGEseq_131k_65kstride_blacklist0.35_128outputres_TF/preprocessed" \
            --output_heads="hg" \
            --input_length=131072 \
            --output_length=896 \
            --batch_size=16 \
            --num_epochs=60 \
            --train_steps=5450 \
            --warmup_frac=0.025 \
            --val_steps=842 \
            --patience=30 \
            --min_delta=0.001 \
            --model_save_dir="gs://picard-testing-176520/CAGEseq_131k_65kstride_blacklist0.35_128outputres_TF/models" \
            --model_save_basename="aformer_CAGE" \
            --lr_schedule="cosine_decay_w_warmup" \
            --lr_base="0.0001" \
            --min_lr="0.000001" \
            --optimizer="adabelief" \
            --gradient_clip="0.2" \
            --precision="mixed_bfloat16" \
            --weight_decay_frac="5.0e-05" \
            --epsilon=1.0e-10 \
            --rectify=True \
            --conv_channel_list="192,192,224,224,256,256" \
            --conv_filter_size_1="15" \
            --conv_filter_size_2="5" \
            --dropout="0.45" \
            --num_transformer_layers="4" \
            --num_heads="8" \
            --momentum="0.90" \
            --num_random_features="256" \
            --hidden_size="256" \
            --dim=32 \
            --slow_step_frac=0.5 \
            --sync_period=6 \
            --max_seq_length=1024 \
            --rel_pos_bins=1024 \
            --kernel_transformation="softmax_kernel_transformation" \
            --kernel_regularizer=0.001 \
            --savefreq=5 
