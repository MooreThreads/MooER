#!/bin/bash

# set your path
HOME_ROOT=/root/MooER
cd $HOME_ROOT || exit 0

test_data_dir=YOUR/testsets/root
test_sets=test-clean/test-other/aishell
decode_path=YOUR/decode/dir

export PYTHONPATH=${HOME_ROOT}/src:$PYTHONPATH
VISIBLE_DEVICES=0
################### For MUSA User #############################
# export MUSA_VISIBLE_DEVICES=$VISIBLE_DEVICES
# export DS_ACCELERATOR=musa
###############################################################
# For CUDA User
export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES
###############################################################

python ${HOME_ROOT}/src/mooer/testing.py \
    --test_config ${HOME_ROOT}/src/mooer/configs/asr_config_inference.py \
    --test_data_dir $test_data_dir \
    --test_sets $test_sets \
    --decode_path $decode_path

# compute CER
for testset in `echo $test_sets | sed "s|/| |g"`; do
  echo $testset
  python ${HOME_ROOT}/src/tools/compute-wer.py --char=1 --v=1 ${test_data_dir}/${testset}/text \
    ${decode_path}/${testset}/text > ${decode_path}/${testset}/wer 2>&1
done
