#!/bin/bash -l

pip3 install -r /usr/share/tpu/models/official/requirements.txt
sudo pip3 install tensorflow-addons
sudo pip3 install matplotlib
sudo pip3 install pandas
sudo pip3 install seaborn
sudo pip3 install einops
sudo pip3 install tqdm
sudo pip3 install wandb
sudo pip3 install plotly

export TPU_NAME=javed_tpu_pod2
export PYTHONPATH="${PYTHONPATH}:/usr/share/tpu/models"
export TPU_LOAD_LIBRARY=0
chmod a+x execute_sweep_tpupod.sh