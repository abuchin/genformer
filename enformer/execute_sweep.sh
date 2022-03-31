#!/bin/bash -l

python /home/jupyter/models/enformer/Enformer_tests/train_model.py \
            --tpu_name="node-1" \
            --tpu_zone="us-central1-a" \
            --gcs_project="picard-testing-176520" \
            --num_epochs=11 \
            --batch_size=16 \
            --channels=384 \
            --num_heads=8 \
            --num_transformer_layers=11 \
            --pool_type="attention" \
            --target_learning_rate=0.0005 \
            --num_warmup_steps=5000 \
            --train_steps=2500 \
            --val_steps=125 \
            --test_steps=120 \
            --gradient_clip=0.2 \
            --GCS_data_loc="gs://basenji_barnyard/data" \
            --num_parallel=8 \
            --wandb_project="genformer" \
            --wandb_user="njaved" \
            --wandb_sweep_method="grid" \
            -data_splits_json="/home/jupyter/models/enformer/Enformer_tests/data_subsets_HM.json"
