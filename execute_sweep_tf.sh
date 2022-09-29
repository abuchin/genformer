#!/bin/bash -l

python3 train_model_aformer_TF_expression.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="aformer_TF_gene_centered_test" \
            --wandb_user="njaved" \
            --wandb_sweep_name="aformer_TF_gene_centered_test" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/196k_genecentered_blacklist0.30_atacnormalized/preprocessed" \
            --input_length=196608 \
            --max_shift=20 \
            --batch_size=6 \
            --num_epochs=100 \
            --train_steps=50 \
            --warmup_frac=0.025 \
            --val_steps=25 \
            --val_steps_ho=25 \
            --patience=30 \
            --min_delta=0.001 \
            --model_save_dir="gs://picard-testing-176520/196k_genecentered_blacklist0.30/models" \
            --model_save_basename="aformer_TF_gene_centered" \
            --lr_base="5.0e-04" \
            --gradient_clip="1.0" \
            --weight_decay_frac="5.0e-06" \
            --epsilon=1.0e-14 \
            --transformer_depth_1="2" \
            --transformer_depth_2="2" \
            --shared_transformer_depth="3" \
            --pre_transf_channels="768" \
            --dropout_rate="0.20" \
            --attention_dropout_rate="0.05" \
            --num_heads="4" \
            --num_random_features="256" \
            --hidden_size="768" \
            --dim=192 \
            --kernel_transformation="softmax_kernel_transformation" \
            --savefreq=10 \
            --TF_inputs=128 \
            --train_mode="atac_only" \
            --load_init="True" \
            --freeze_conv_layers="True" \
            --use_tf_module="True" \
            --rna_loss_scale="0.50" \
            --checkpoint_path="sonnet_weights"