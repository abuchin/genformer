#!/bin/bash -l

pip3 install -r /usr/share/tpu/models/official/requirements.txt
sudo pip3 install tensorflow-addons
sudo pip3 install tensorflow-model-optimization>=0.1.3
sudo pip3 install matplotlib
sudo pip3 install pandas
sudo pip3 install seaborn
sudo pip3 install einops
sudo pip3 install tqdm
sudo pip3 install wandb


export PYTHONPATH="${PYTHONPATH}:/usr/share/tpu/models"
