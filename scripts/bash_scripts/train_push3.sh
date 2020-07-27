#!/usr/bin/env bash

python scripts/push_task/train_push.py \
--model-type PushCrossmodalKalmanFilter \
--experiment-name push_cmekf_measinit__0.8_2 --image_blackout_ratio 0.8

python scripts/push_task/train_push.py \
--model-type PushKalmanFilter \
--experiment-name push_ekf_measinit__0.8 --image_blackout_ratio 0.8

python scripts/push_task/train_push.py \
--model-type PushUnimodalKalmanFilter \
--experiment-name push_umekf_measinit__0.8 --image_blackout_ratio 0.8

