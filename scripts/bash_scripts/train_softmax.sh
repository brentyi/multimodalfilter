#!/usr/bin/env bash

python scripts/door_task/finetune_cmkf.py \
--original-experiment door_cmekf_0.4_f --checkpoint-label phase2 \
--experiment-name 0727_sm_door0.4_cmekf_0


python scripts/door_task/finetune_cmkf.py \
--original-experiment door_cmekf_0.8_f --checkpoint-label phase2 \
--experiment-name 0727_sm_door0.8_cmekf_0