#!/bin/bash

# set your path
HOME_ROOT=/root/MooER
cd $HOME_ROOT || exit 0

export PYTHONPATH=${HOME_ROOT}/src:$PYTHONPATH
VISIBLE_DEVICES=0,1,2,3,4,5,6,7
################### For MUSA User #############################
# export MUSA_VISIBLE_DEVICES=$VISIBLE_DEVICES
# export DS_ACCELERATOR=musa
###############################################################
# For CUDA User
export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES
###############################################################
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

export NCCL_PROTOS=2
export DS_ENV_FILE=./ds_env

# use hostfile
#deepspeed \
#    --hostfile hostfile \
#    --master_port=10086 \
#    ${HOME_ROOT}/src/mooer/training.py \
#    --training_config ${HOME_ROOT}/src/mooer/configs/asr_config_training.py
#    # you can modifiy training_config path to yourself! :D
#    # modify the ip & cards in hostfile

# use localhost
deepspeed \
    --include localhost:$VISIBLE_DEVICES \
    --master_port=10086 \
    ${HOME_ROOT}/src/mooer/training.py \
    --training_config ${HOME_ROOT}/src/mooer/configs/asr_config_training.py
    # you can modifiy training_config path to yourself! :D

