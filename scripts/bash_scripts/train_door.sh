#!/usr/bin/env bash

python scripts/door_task/finetune_cmkf.py \
--original-experiment door_cmekf --checkpoint-label phase2 \
--experiment-name 0727_door_cmekf_0

#python scripts/door_task/finetune_cmkf.py \
#--original-experiment door_cmekf --checkpoint-label phase2 \
#--experiment-name 0727_door_umekf_0 --unimodal

python scripts/door_task/finetune_cmkf.py \
--original-experiment 0727_sm_door0.4_cmekf_0 --checkpoint-label phase3-force \
--experiment-name 0727_door0.4_umekf_1 --unimodal


python scripts/door_task/finetune_cmkf.py \
--original-experiment 0727_sg_door0.8_cmekf_0 --checkpoint-label phase3-force \
--experiment-name 0727_door0.8_umekf_0 --unimodal