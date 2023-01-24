#!/bin/bash -l

python3 train_model.py \
            --tpu_name="node-6" \
            --tpu_zone="us-central1-a" \
            --wandb_project="enformer_baseline" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_baseline" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/enformer_baseline_paired_ATAC_RAMPAGE/tfrecords"\
            --gcs_path_TSS="gs://picard-testing-176520/enformer_baseline_paired_ATAC_RAMPAGE/tss_sequences/tfrecords" \
            --num_epochs=50 \
            --warmup_frac=0.146 \
            --patience=30\
            --min_delta=0.00001 \
            --model_save_dir="gs://picard-testing-176520/enformer_baseline_paired_ATAC_RAMPAGE/models" \
            --model_save_basename="enformer_baseline" \
            --lr_base1="1.0e-06" \
            --lr_base2="2.5e-04" \
            --epsilon=1.0e-10 \
            --num_parallel=8 \
            --savefreq=25 \
            --train_examples=34021 \
            --val_examples=2213 \
            --val_examples_TSS=3352 \
            --num_targets=62 \
            --use_enformer_weights="True" \
            --enformer_checkpoint_path="sonnet_weights"
