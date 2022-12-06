#!/bin/bash -l

python3 train_model.py \
            --tpu_name="pod" \
            --tpu_zone="us-east1-d" \
            --wandb_project="enformer_baseline" \
            --wandb_user="njaved" \
            --wandb_sweep_name="enformer_baseline" \
            --gcs_project="picard-testing-176520" \
            --gcs_path="gs://picard-testing-176520/genformer_baseline/tfrecords" \
            --gcs_path_TSS="gs://picard-testing-176520/genformer_baseline/tss_centered/tfrecords" \
            --num_epochs=120 \
            --warmup_frac=0.146 \
            --patience=30\
            --min_delta=0.00001 \
            --model_save_dir="gs://picard-testing-176520/enformer_baseline/models" \
            --model_save_basename="enformer_baseline" \
            --lr_base1="1.0e-04" \
            --lr_base2="1.0e-04" \
            --epsilon=1.0e-10 \
            --num_parallel=8 \
            --savefreq=8 \
            --train_examples=3420 \
            --val_examples=2213 \
            --val_examples_TSS=1645 \
            --num_targets=98
