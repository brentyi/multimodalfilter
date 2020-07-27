#!/usr/bin/env bash

python scripts/door_task/train_door.py \
--model-type DoorKalmanFilter \
--experiment-name door_ekf_measinit

python scripts/door_task/train_door.py \
--model-type DoorCrossmodalKalmanFilter \
--experiment-name door_cmekf_measinit

python scripts/door_task/train_door.py \
--model-type DoorUnimodalKalmanFilter \
--experiment-name door_umekf_measinit

