#!/usr/bin/env bash

python scripts/door_task/train_door.py \
--model-type DoorKalmanFilter \
--experiment-name door_ekf_0.4_f_meas_init --image_blackout_ratio 0.4

python scripts/door_task/train_door.py \
--model-type DoorCrossmodalKalmanFilter \
--experiment-name door_cmekf_0.4_f_meas_init2 --image_blackout_ratio 0.4

python scripts/door_task/train_door.py \
--model-type DoorUnimodalKalmanFilter \
--experiment-name door_umekf_0.4_f_meas_init --image_blackout_ratio 0.4

#python scripts/door_task/finetune_cmkf.py --original-experiment door_cmekf_0.8_f_meas_init --checkpoint-label phase3 --experiment-name door_cmekf_0.8_f_meas_init_finetune
#python scripts/door_task/finetune_cmkf.py --original-experiment door_cmekf_0.4_f_meas_init --checkpoint-label phase3 --experiment-name door_cmekf_0.4_f_meas_init_finetune