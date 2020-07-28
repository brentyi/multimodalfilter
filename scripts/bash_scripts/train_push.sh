#!/usr/bin/env bash


#python scripts/push_task/finetune_ekf.py \
#--original-experiment 727_push0.4_cmekf_0 --checkpoint-label phase3-force \
#--experiment-name 0727_sig_push0.4_cmekf_1

python scripts/push_task/finetune_ekf.py \
--original-experiment 727_push0.8_cmekf_0 --checkpoint-label phase3-force \
--experiment-name 0727_sig_push0.8_cmekf_1

python scripts/push_task/finetune_ekf.py \
--original-experiment 727_push0.4_cmekf_0 --checkpoint-label phase3-force \
--experiment-name 0727_sig_push0.4_umekf_0 --unimodal

python scripts/push_task/finetune_ekf.py \
--original-experiment 727_push0.8_cmekf_0 --checkpoint-label phase3-force \
--experiment-name 0727_sig_push0.8_umekf_0  --unimodal

