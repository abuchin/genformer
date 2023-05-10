#!/bin/bash -l

sudo pip3 install silence_tensorflow
sudo pip3 install tensorflow-addons
sudo pip3 install matplotlib==3.4.3
sudo pip3 install pandas==1.3.3
sudo pip3 install tensorflow-probability
sudo pip3 install seaborn==0.11.2
sudo pip3 install einops==0.4.0
sudo pip3 install tqdm==4.63.0
sudo pip3 install scipy
sudo pip3 install wandb
sudo pip3 install plotly
sudo pip3 install scikit-learn
sudo pip3 install numpy --upgrade


gsutil cp gs://picard-testing-176520/sonnet_weights.tar.gz .
tar -xzvf sonnet_weights.tar.gz

wandb login
