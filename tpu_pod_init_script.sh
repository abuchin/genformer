#!/bin/bash -l

sudo pip3 install silence_tensorflow --no-deps
sudo pip3 install tensorflow-addons --no-deps
sudo pip3 install matplotlib==3.4.3 --no-deps
sudo pip3 install pandas==1.3.3 --no-deps
sudo pip3 install seaborn==0.11.2 --no-deps
sudo pip3 install einops==0.4.0 --no-deps
sudo pip3 install tqdm==4.63.0 --no-deps
sudo pip3 install scipy==1.7.1 --no-deps
sudo pip3 install wandb==0.13.1 --no-deps
sudo pip3 install plotly==5.8.2 --no-deps 
sudo pip3 install scikit-learn==1.0 --no-deps
sudo pip3 install tensorboard-plugin-profile==2.4.0 --no-deps
sudo pip3 install dm-sonnet==2.0.0 --no-deps
sudo pip3 install -U --no-deps numpy
export TPU_NAME=pod
export ZONE=us-east1-d 
export TPU_LOAD_LIBRARY=0

gsutil cp gs://picard-testing-176520/sonnet_weights.tar.gz .
tar -xzvf sonnet_weights.tar.gz

wandb login
